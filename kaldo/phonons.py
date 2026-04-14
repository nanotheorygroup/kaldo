"""
kaldo
Anharmonic Lattice Dynamics

"""
import functools
import os

from kaldo.storable import is_calculated
from kaldo.storable import lazy_property, Storable
from kaldo.helpers.logger import log_size
from kaldo.storable import FOLDER_NAME
from kaldo.grid import Grid
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.observables.harmonic_with_q_temp import HarmonicWithQTemp
from kaldo.forceconstants import ForceConstants
import kaldo.controllers.anharmonic as aha
import tensorflow as tf
import kaldo.controllers.isotopic as isotopic
from concurrent.futures import as_completed
from kaldo.parallel import get_executor
from scipy import stats
import numpy as np
from numpy.typing import ArrayLike
import ase.units as units
from kaldo.helpers.tools import timeit
from kaldo.helpers.logger import get_logger

logging = get_logger()

# Constants
GAMMA_TO_THZ = 1e11 * units.mol * (units.mol / (10 * units.J)) ** 2
THZ_TO_MEV = units.J * units._hbar * 2 * np.pi * 1e15


def _get_ir_kgrid_data(atoms, kpts, grid_type='C'):
    """
    Compute the IBZ mapping and per-equivalent-kpoint column-permutation tables
    using spglib, aligned with kALDo's k-point ordering.

    Parameters
    ----------
    atoms : ASE Atoms
        Primitive unit cell.
    kpts : array-like (3,)
        k-point mesh dimensions.
    grid_type : str
        kALDo grid ordering ('C' or 'F').

    Returns
    -------
    ir_mapping : np.ndarray (n_k_points,)
        ir_mapping[ik] = index of the IBZ representative for k-point ik.
    krot_perm : list of (np.ndarray or None), length n_k_points
        For IBZ representatives: None.
        For non-IBZ ik: int array (n_k_points,) where krot_perm[ik][ik']
        is the index of S^{-1}·q_{ik'}, where S is the rotation mapping
        q_{ir_mapping[ik]} to q_{ik}.
    ibz_indices : list of int
        Sorted list of IBZ representative k-point indices.
    krot_cart : list of (np.ndarray or None), length n_k_points
        For IBZ representatives: None.
        For non-IBZ ik: (3, 3) float array, the Cartesian rotation matrix R
        such that v(q_ik) = R @ v(q_irr) for any vector quantity (e.g. velocity).
    """
    try:
        import spglib
    except ImportError:
        raise ImportError(
            "spglib is required for q-space symmetry reduction. "
            "Install with: pip install spglib"
        )

    kpts_arr = np.asarray(kpts, dtype=int)
    n_k_points = int(np.prod(kpts_arr))
    grid = Grid(kpts_arr, order=grid_type)

    # Integer grid coordinates (0..Ni-1) and fractional equivalents.
    k_indices = grid.generate_index_grid()   # (n_k_points, 3) int
    q_fracs = k_indices / kpts_arr.astype(float)  # (n_k_points, 3) in [0,1)

    # Build spglib structure tuple.
    cell = atoms.cell[:]
    scaled_pos = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    spg_struct = (cell, scaled_pos, numbers)

    # --- IBZ mapping via spglib -------------------------------------------
    spg_mapping, spg_grid_address = spglib.get_ir_reciprocal_mesh(
        kpts_arr.tolist(), spg_struct, is_shift=[0, 0, 0], is_time_reversal=True)

    # spg_grid_address[i] contains integer coords in [0, Ni).
    # Build a map from spglib's linear index to kALDo's linear index.
    spg_to_kaldo = np.empty(n_k_points, dtype=int)
    for i_spg, addr in enumerate(spg_grid_address):
        addr_w = np.asarray(addr, dtype=int) % kpts_arr
        spg_to_kaldo[i_spg] = int(np.ravel_multi_index(addr_w, kpts_arr.tolist(), order=grid_type))

    ir_mapping = np.empty(n_k_points, dtype=int)
    for i_spg in range(n_k_points):
        ik = spg_to_kaldo[i_spg]
        ik_irr = spg_to_kaldo[int(spg_mapping[i_spg])]
        ir_mapping[ik] = ik_irr

    ibz_indices = sorted(set(ir_mapping.tolist()))

    # --- Rotation permutation tables for gamma-tensor replication -----------
    # Symmetry operations: R is a (3,3) integer matrix in fractional real-space.
    # Action on fractional reciprocal-space coords: k' = (R^{-1})^T k.
    # Inverse of that action: k = R^T k'.  (since ((R^{-1})^T)^{-1} = R^T)
    sym = spglib.get_symmetry(spg_struct)
    rotations_frac = sym['rotations']   # (n_ops, 3, 3) integer

    def _apply_rot_to_mesh(R_mat):
        """
        Apply integer rotation matrix R_mat (3,3) to every fractional k-point
        and return the array of resulting mesh indices.
        """
        # (R_mat @ q.T).T = q @ R_mat.T for each row-vector q.
        q_rot = q_fracs @ R_mat.T        # (n_k_points, 3)
        q_rot = q_rot % 1.0
        # Collapse floating-point values sitting just below 1.0 to 0.
        q_rot[np.abs(q_rot - 1.0) < 1e-9] = 0.0
        q_int = np.round(q_rot * kpts_arr).astype(int) % kpts_arr
        return np.ravel_multi_index(q_int.T, kpts_arr.tolist(), order=grid_type)

    # For each non-IBZ k-point ik, find the rotation S in reciprocal space
    # (= (R^{-1})^T) that maps q_{ir_mapping[ik]} to q_{ik}, then store:
    #   krot_perm : permutation of the full mesh by S^{-1} = R^T (for gamma tensor)
    #   krot_cart : 3x3 Cartesian rotation matrix R_cart = A @ R @ A^{-1}
    #               where A = cell.T, for rotating vector quantities (e.g. velocity).
    krot_perm = [None] * n_k_points
    krot_cart = [None] * n_k_points
    A = cell.T                      # columns are lattice vectors
    A_inv = np.linalg.inv(A)

    for ik in range(n_k_points):
        ik_irr = ir_mapping[ik]
        if ik == ik_irr:
            continue

        q_irr = q_fracs[ik_irr]
        q_target = q_fracs[ik]

        found = False
        for R in rotations_frac:
            # Compute (R^{-1})^T = round(inv(R))^T acting on q_irr.
            R_inv = np.round(np.linalg.inv(R)).astype(int)
            R_recip = R_inv.T  # (R^{-1})^T
            q_test = R_recip @ q_irr
            q_test = q_test % 1.0
            q_test[np.abs(q_test - 1.0) < 1e-9] = 0.0
            if np.allclose(q_test, q_target, atol=1e-9):
                # S^{-1} = R^T applied to the full mesh gives the permutation.
                krot_perm[ik] = _apply_rot_to_mesh(R.T)
                # Cartesian rotation: R acts on fractional real-space coords,
                # so R_cart = A @ R @ A^{-1} transforms Cartesian vectors.
                krot_cart[ik] = A @ R @ A_inv
                found = True
                break

        if not found:
            # Time-reversal fallback: q_ik = -q_irr (mod 1).
            # This arises for non-centrosymmetric crystals where -I is not a
            # point group rotation but is_time_reversal=True in get_ir_reciprocal_mesh
            # still identifies q and -q as equivalent.
            # Γ(-q,μ;q',μ') = Γ(q,μ;-q',μ'), so permuting the k'-axis by
            # q'→-q' (R_mat=-I) correctly replicates the gamma tensor.
            # Velocity is odd under inversion: v(-q) = -v(q), so R_cart = -I.
            q_neg = (-q_irr) % 1.0
            q_neg[np.abs(q_neg - 1.0) < 1e-9] = 0.0
            if np.allclose(q_neg, q_target, atol=1e-9):
                krot_perm[ik] = _apply_rot_to_mesh(-np.eye(3, dtype=int))
                krot_cart[ik] = -np.eye(3, dtype=float)
                found = True

        if not found:
            raise RuntimeError(
                f"No spglib symmetry operation found mapping IBZ k-point "
                f"{ik_irr} to k-point {ik}. "
                "Ensure the k-mesh is commensurate with the crystal symmetry."
            )

    return ir_mapping, krot_perm, ibz_indices, krot_cart

def _sparse_tensor_to_numpy(st):
    """Convert a tf.SparseTensor to a picklable (indices, values, dense_shape) tuple."""
    if st is None:
        return None
    return (st.indices.numpy(), st.values.numpy(), tuple(st.dense_shape.numpy()))


def _numpy_to_sparse_tensor(data):
    """Reconstruct a tf.SparseTensor from a (indices, values, dense_shape) tuple."""
    if data is None:
        return None
    indices, values, dense_shape = data
    return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)


def _compute_kpoint_projection(index_k, n_modes, n_k_points, omega, physical_mode,
                                evect_np, third_sparse_data, chi_k_np, velocity_np,
                                cell_inv, kpts, broadening_shape, third_bandwidth,
                                hbar, n_replicas, is_sparse, kpoint_maps):
    """Compute sparse_phase and sparse_potential for all modes at one k-point.

    All inputs are numpy arrays (picklable). TF tensors are created internally.
    Returns a list of n_modes entries, each being:
        (phase_data_list, potential_data_list)
    where each *_data_list is [is_plus_0, is_plus_1] with entries as
    (indices, values, dense_shape) numpy tuples or None.
    """
    # Pin BLAS threads in worker processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # Reconstruct TF tensors from numpy
    evect_tf = tf.cast(tf.convert_to_tensor(evect_np), dtype=tf.complex128)
    second_minus = tf.math.conj(evect_tf)
    _chi_k = tf.cast(tf.convert_to_tensor(chi_k_np), dtype=tf.complex128)
    second_minus_chi = tf.math.conj(_chi_k)
    velocity_tf = tf.convert_to_tensor(velocity_np)

    if is_sparse:
        coords, data, shape = third_sparse_data
        third_tf = tf.SparseTensor(
            tf.cast(coords, dtype=tf.int64), data, shape
        )
    else:
        third_tf = tf.convert_to_tensor(third_sparse_data)
    third_tf = tf.cast(third_tf, dtype=tf.complex128)

    results = []
    for mu in range(n_modes):
        nu_single = index_k * n_modes + mu
        if not physical_mode[index_k, mu]:
            results.append(([None, None], [None, None]))
            continue

        phase_list = []
        potential_list = []
        for is_plus in (0, 1):
            index_kpp_full = tf.cast(kpoint_maps[(index_k, is_plus)], dtype=tf.int32)
            if third_bandwidth:
                sigma_tf = tf.constant(third_bandwidth, dtype=tf.float64)
            else:
                sigma_tf = aha.calculate_broadening(velocity_tf, cell_inv, kpts, index_kpp_full)

            dirac_delta_result = aha.calculate_dirac_delta_crystal(
                omega, physical_mode, sigma_tf, broadening_shape,
                index_kpp_full, index_k, mu, is_plus, n_k_points, n_modes,
            )
            if not dirac_delta_result:
                phase_list.append(None)
                potential_list.append(None)
                continue

            potential_result = aha.sparse_potential_mu(
                nu_single, evect_tf, dirac_delta_result, index_k, mu,
                n_k_points, n_modes, is_plus, is_sparse, index_kpp_full,
                _chi_k, second_minus, second_minus_chi, third_tf,
                n_replicas, omega, hbar,
            )
            phase_list.append(_sparse_tensor_to_numpy(dirac_delta_result))
            potential_list.append(_sparse_tensor_to_numpy(potential_result))

        results.append((phase_list, potential_list))
    return results


def _save_kpoint_projection(output_dir, index_k, results):
    """Save per-k-point projection results to disk."""
    save_dict = {}
    for mu, (phase_list, pot_list) in enumerate(results):
        for ip, (phase, pot) in enumerate(zip(phase_list, pot_list)):
            prefix = f'mu{mu}_ip{ip}'
            if phase is not None:
                indices, values, dense_shape = phase
                save_dict[f'{prefix}_phase_indices'] = indices
                save_dict[f'{prefix}_phase_values'] = values
                save_dict[f'{prefix}_phase_shape'] = np.array(dense_shape)
            if pot is not None:
                indices, values, dense_shape = pot
                save_dict[f'{prefix}_pot_indices'] = indices
                save_dict[f'{prefix}_pot_values'] = values
                save_dict[f'{prefix}_pot_shape'] = np.array(dense_shape)
    save_dict['n_modes'] = np.array(len(results))
    np.savez(os.path.join(output_dir, f'kpt_{index_k:05d}.npz'), **save_dict)
    open(os.path.join(output_dir, f'kpt_{index_k:05d}.done'), 'w').close()


def _load_kpoint_projection(output_dir, index_k, n_modes):
    """Load per-k-point projection results from disk."""
    path = os.path.join(output_dir, f'kpt_{index_k:05d}.npz')
    data = np.load(path, allow_pickle=False)
    results = []
    for mu in range(n_modes):
        phase_list = []
        pot_list = []
        for ip in range(2):
            prefix = f'mu{mu}_ip{ip}'
            phase_key = f'{prefix}_phase_indices'
            if phase_key in data:
                phase_list.append((
                    data[f'{prefix}_phase_indices'],
                    data[f'{prefix}_phase_values'],
                    tuple(data[f'{prefix}_phase_shape']),
                ))
            else:
                phase_list.append(None)
            pot_key = f'{prefix}_pot_indices'
            if pot_key in data:
                pot_list.append((
                    data[f'{prefix}_pot_indices'],
                    data[f'{prefix}_pot_values'],
                    tuple(data[f'{prefix}_pot_shape']),
                ))
            else:
                pot_list.append(None)
        results.append((phase_list, pot_list))
    return results


class Phonons(Storable):
    """
    The Phonons object exposes all the phononic properties of a system by manipulation
    of the quantities passed into the ForceConstant object. The arguments passed in here
    reflect assumptions to be made about the macroscopic system e.g. the temperature, or
    whether the system is amorphous or a nanowire.
    The ForceConstants, and temperature are the only two required parameters, though we
    highly recommend the switch controlling whether to use quantum/classical statistics
    (``is_classic``) and the number of k-points to consider (``kpts``).
    For most users, you will not need to access any Phonon object functions directly
    , but only reference an attribute (e.g. Phonons.frequency). Please check out the
    examples for details on our recommendations for retrieving, and plotting data.

    Parameters
    ----------
    forceconstants : ForceConstants
        Contains all the information about the system and the derivatives
        of the potential energy.
    temperature : float
        Defines the temperature of the simulation
        Units: K
    is_classic : bool
        Specifies if the system is treated with classical or quantum
        statistics.
        Default: False
    kpts : (int, int, int)
        Defines the number of k points to use to create the k mesh
        Default: (1, 1, 1)
    min_frequency : float
        Ignores all phonons with frequency below ``min_frequency``
        Units: Thz
        Default: None
    max_frequency : float
        Ignores all phonons with frequency above ``max_frequency``
        Units: THz
        Default: None
    third_bandwidth : float
        Defines the width of the energy conservation smearing in the phonons
        scattering calculation. If ``None`` the width is calculated
        dynamically. Otherwise the input value corresponds to the width.
        Units: THz
        Default: None
    broadening_shape : string
        Defines the algorithm to use for line-broadening when enforcing
        energy conservation rules for three-phonon scattering.
        Options: 'gauss', 'lorentz' and 'triangle'.
        Default: ``'gauss'``
    folder : string
        Specifies where to store the data files.
        Default: ``'output'``.
    storage : string
        Defines the strategy used to store observables. The ``default`` strategy
        stores formatted text files for most harmonic properties but relies on
        numpy arrays for large arrays like the gamma tensor. The ``memory`` option
        doesn't generate any output except what is printed in your script.
        Options: ``'default'``, ``'formatted'``, ``'numpy'``, ``'memory'``, ``'hdf5'``
        Default: ``'formatted'``
    grid_type : string
        Specifies whether the atoms in the replicated system were repeated using
        a C-like index ordering which changes the last axis the fastest or
        FORTRAN-like index ordering which changes the first index fastest.
        Options: ``'C'``, ``'F'``
        Default: ``'C'``
    is_balanced : bool
        Enforce detailed balance when calculating anharmonic properties. Useful for
        simulations where it may be difficult to get a sufficiently dense k-point grid.
        Default: False
    is_unfolding : bool
        If the second order force constants need to be unfolded like in P. B. Allen
        et al., Phys. Rev. B 87, 085322 (2013) set this to True.
        Default: False
    g_factor : (n_atoms) array , optional
        It contains the isotopic g factor for each atom of the unit cell. 
        g factor is the natural isotopic distributions of each element. 
        More reference can be found: M. Berglund, M.E. Wieser, Isotopic compositions of the elements 2009 (IUPAC technical report), Pure Appl. Chem. 83 (2011) 397–410.
        Default: None
    is_symmetrizing_frequency : bool, optional
        TODO: add more doc here
        Default: False
    is_antisymmetrizing_velocity : bool, optional
        TODO: add more doc here
        Default: False
    include_isotopes: bool, optional.
        Defines if you want to include isotopic scattering bandwidths.
        Default: False.
    iso_speed_up: bool, optional.
        Defines if you want to truncate the energy-conservation delta
        in the isotopic scattering computation. Default is True.
    is_nw: bool, optional
        Defines if you would like to assume the system is a nanowire. 
        Default: False

    Returns
    -------
    Phonons Object
    """
    
    # Define storage formats for phonon properties
    # This encapsulates the storage strategy within the class
    _store_formats = {
        'physical_mode': 'formatted',
        'frequency': 'formatted',
        'participation_ratio': 'formatted',
        'velocity': 'formatted',
        'heat_capacity': 'formatted',
        'population': 'formatted',
        'bandwidth': 'formatted',
        'phase_space': 'formatted',
        'diffusivity': 'numpy',
        'flux': 'numpy',
        '_dynmat_derivatives': 'numpy',
        '_eigensystem': 'numpy',
        '_ps_and_gamma': 'numpy',
        '_ps_gamma_and_gamma_tensor': 'numpy',
        '_generalized_diffusivity': 'numpy'
    }
    def __init__(self,
                 forceconstants: ForceConstants,
                 temperature: float | None = None,
                 *,
                 is_classic: bool = False,
                 kpts: tuple[int, int, int] = (1, 1, 1),
                 min_frequency: float = 0.,
                 max_frequency: float | None = None,
                 third_bandwidth: float | None = None,
                 broadening_shape: str = "gauss",
                 folder: str = FOLDER_NAME,
                 storage: str = "formatted",
                 grid_type: str = "C",
                 is_balanced: bool = False,
                 is_unfolding: bool = False,
                 g_factor: ArrayLike = None,
                 is_symmetrizing_frequency: bool = False, 
                 is_antisymmetrizing_velocity: bool = False,
                 include_isotopes: bool = False,
                 iso_speed_up: bool = True,
                 is_nw: bool = False,
                 n_workers: int = 1,
                 projection_output_dir: str | None = None,
                 use_q_symmetry: bool = False,
                 **kwargs):
        self.forceconstants = forceconstants
        if n_workers is not None and n_workers < 1:
            raise ValueError(f"n_workers must be >= 1 or None, got {n_workers}")
        self.n_workers = n_workers
        self.projection_output_dir = projection_output_dir
        self.is_classic = is_classic
        if temperature is not None:
            self.temperature = float(temperature)
        self.folder = folder
        self.kpts = np.array(kpts)
        self._grid_type = grid_type
        self._reciprocal_grid = Grid(self.kpts, order=self._grid_type)
        self.is_unfolding = is_unfolding
        if self.is_unfolding:
            logging.info('Using unfolding.')
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.broadening_shape = broadening_shape
        self.is_nw = is_nw
        self.third_bandwidth = third_bandwidth
        self.storage = storage
        self.is_symmetrizing_frequency = is_symmetrizing_frequency
        self.is_antisymmetrizing_velocity = is_antisymmetrizing_velocity
        self.is_balanced = is_balanced
        self.atoms = self.forceconstants.atoms
        self.supercell = np.array(self.forceconstants.supercell)
        self.n_k_points = int(np.prod(self.kpts))
        self.n_atoms = self.forceconstants.n_atoms
        self.n_modes = self.forceconstants.n_modes
        self.n_phonons = self.n_k_points * self.n_modes
        self.hbar = units._hbar
        if self.is_classic:
            self.hbar = self.hbar * 1e-6
        self.g_factor = g_factor
        self.include_isotopes = include_isotopes
        self.iso_speed_up = iso_speed_up
        self.use_q_symmetry = use_q_symmetry

    def _get_folder_path_components(self, label):
        """Get folder path components for Phonons-specific attributes."""
        components = []
        
        if '<temperature>' in label and hasattr(self, 'temperature'):
            components.append(str(int(self.temperature)))
            
        if '<statistics>' in label:
            if self.is_classic:
                components.append('classic')
            else:
                components.append('quantum')
                
        if '<third_bandwidth>' in label and self.third_bandwidth is not None:
            components.append('tb_' + str(np.mean(self.third_bandwidth)))
            
        if '<include_isotopes>' in label and self.include_isotopes:
            components.append('isotopes')
            
        return components

    def _load_formatted_property(self, property_name, name):
        """Override formatted loading for Phonons-specific properties"""
        if property_name == 'physical_mode':
            loaded = np.loadtxt(name + '.dat', skiprows=1)
            return np.round(loaded, 0).astype(bool)
        elif property_name == 'velocity':
            loaded = []
            for alpha in range(3):
                loaded.append(np.loadtxt(name + '_' + str(alpha) + '.dat', skiprows=1))
            return np.array(loaded).transpose(1, 2, 0)
        else:
            # Use default implementation for other properties
            return super()._load_formatted_property(property_name, name)
    
    def _save_formatted_property(self, property_name, name, data):
        """Override formatted saving for Phonons-specific properties"""
        if property_name == 'physical_mode':
            fmt = '%d'
            np.savetxt(name + '.dat', data, fmt=fmt, header=str(data.shape))
        elif property_name == 'velocity':
            fmt = '%.18e'
            for alpha in range(3):
                np.savetxt(name + '_' + str(alpha) + '.dat', data[..., alpha], fmt=fmt, 
                          header=str(data[..., 0].shape))
        else:
            # Use default implementation for other properties
            super()._save_formatted_property(property_name, name, data)



    @lazy_property(label='')
    def physical_mode(self):
        """
        Calculate physical modes. Non physical modes are the first 3 modes of q=(0, 0, 0) and, if defined, all the
        modes outside the frequency range min_frequency and max_frequency.

        Returns
        -------
        physical_mode : np array(n_k_points, n_modes)
            bool
        """
        q_points = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        physical_mode = np.zeros((self.n_k_points, self.n_modes), dtype=bool)

        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point=q_point,
                                   second=self.forceconstants.second,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage,
                                   is_nw=self.is_nw,
                                   is_unfolding=self.is_unfolding,
                                   is_amorphous=self._is_amorphous)

            physical_mode[ik] = phonon.physical_mode
        if self.min_frequency is not None:
            physical_mode[self.frequency < self.min_frequency] = False
        if self.max_frequency is not None:
            physical_mode[self.frequency > self.max_frequency] = False
        return physical_mode


    @lazy_property(label='')
    def frequency(self):
        """Calculate phonons frequency

        Returns
        -------
        frequency : np array(n_k_points, n_modes)
             frequency in THz
        """
        q_points = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        frequency = np.zeros((self.n_k_points, self.n_modes))

        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point=q_point,
                                   second=self.forceconstants.second,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage,
                                   is_nw=self.is_nw,
                                   is_unfolding=self.is_unfolding,
                                   is_amorphous=self._is_amorphous)
            frequency[ik] = phonon.frequency

        return frequency


    @lazy_property(label='')
    def participation_ratio(self):
        """
        Calculates the participation ratio of each normal mode. Participation ratio's
        represent the fraction of atoms that are displaced meaning a value of 1 corresponds
        to translation. Defined by equations in DOI: 10.1103/PhysRevB.53.11469

        Returns
        -------
        participation_ratio : np array(n_k_points, n_modes)
             atomic participation
        """
        q_points = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        participation_ratio = np.zeros((self.n_k_points, self.n_modes))
        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point=q_point,
                                   second=self.forceconstants.second,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage,
                                   is_nw=self.is_nw,
                                   is_unfolding=self.is_unfolding,
                                   is_amorphous=self._is_amorphous)

            participation_ratio[ik] = phonon.participation_ratio

        return participation_ratio


    @lazy_property(label='')
    def velocity(self):
        """
        Calculates the velocity using Hellmann-Feynman theorem.

        Returns
        -------
        velocity : np.array(n_k_points, n_unit_cell * 3, 3)
             velocity in 100m/s or A/ps
        """

        q_points = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        velocity = np.zeros((self.n_k_points, self.n_modes, 3))

        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point=q_point,
                                   second=self.forceconstants.second,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage,
                                   is_nw=self.is_nw,
                                   is_unfolding=self.is_unfolding,
                                   is_amorphous=self._is_amorphous)
            velocity[ik] = phonon.velocity

        return velocity


    @lazy_property(label='')
    def _eigensystem(self):
        """
        Calculate the eigensystems, for each k point in k_points.

        Returns
        -------
        _eigensystem : np.array(n_k_points, n_unit_cell * 3 + 1, n_unit_cell * 3)
            eigensystem is calculated for each k point, the three dimensional array
            records the eigenvalues in the first column of the second dimension.

            If the system is not amorphous, these values are stored as complex numbers.
        """
        type = complex if (not self._is_amorphous) else float
        q_points = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        shape = (self.n_k_points, self.n_modes + 1, self.n_modes)
        log_size(shape, name='eigensystem', type=type)
        eigensystem = np.zeros(shape, dtype=type)
        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point=q_point,
                                   second=self.forceconstants.second,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage,
                                   is_nw=self.is_nw,
                                   is_unfolding=self.is_unfolding,
                                   is_amorphous=self._is_amorphous)

            eigensystem[ik] = phonon._eigensystem

        return eigensystem


    @lazy_property(label='<temperature>/<statistics>')
    def heat_capacity(self):
        """
        Calculate the heat capacity for each k point in k_points and each mode.
        If classical, it returns the Boltzmann constant in J/K. If quantum it returns the derivative of the
        Bose-Einstein weighted by each phonons energy
        :math:`c_\\mu = k_B \\frac{\\nu_\\mu^2}{ \\tilde T^2} n_\\mu (n_\\mu + 1)`, 
        where the frequency :math:`\\nu` and the temperature :math:`\\tilde T` are in THz.

        Returns
        -------
        c_v : np.array(n_k_points, n_modes)
            heat capacity in J/K for each k point and each mode
        """
        q_points = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        c_v = np.zeros((self.n_k_points, self.n_modes))
        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQTemp(q_point=q_point,
                                       second=self.forceconstants.second,
                                       distance_threshold=self.forceconstants.distance_threshold,
                                       folder=self.folder,
                                       storage=self.storage,
                                       temperature=self.temperature,
                                       is_classic=self.is_classic,
                                       is_nw=self.is_nw,
                                       is_unfolding=self.is_unfolding,
                                       is_amorphous=self._is_amorphous)
            c_v[ik] = phonon.heat_capacity
        return c_v


    @lazy_property(label='<temperature>/<statistics>')
    def heat_capacity_2d(self):
        """
        Calculate the generalized 2d heat capacity for each k point in k_points and each mode.
        If classical, it returns the Boltzmann constant in W/m/K.

        Returns
        -------
        heat_capacity_2d : np.array(n_k_points, n_modes, n_modes)
            heat capacity in W/m/K for each k point and each modes couple.
        """
        q_points = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        shape = (self.n_k_points, self.n_modes, self.n_modes)
        log_size(shape, name='heat_capacity_2d', type=float)
        heat_capacity_2d = np.zeros(shape)
        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQTemp(q_point=q_point,
                                       second=self.forceconstants.second,
                                       distance_threshold=self.forceconstants.distance_threshold,
                                       folder=self.folder,
                                       storage=self.storage,
                                       temperature=self.temperature,
                                       is_classic=self.is_classic,
                                       is_nw=self.is_nw,
                                       is_unfolding=self.is_unfolding,
                                       is_amorphous=self._is_amorphous)

            heat_capacity_2d[ik] = phonon.heat_capacity_2d
        return heat_capacity_2d


    @lazy_property(label='<temperature>/<statistics>')
    def population(self):
        """
        Calculate the phonons population for each k point in k_points and each mode.
        If classical, it returns the temperature divided by each frequency, using equipartition theorem.
        If quantum it returns the Bose-Einstein distribution

        Returns
        -------
        population : np.array(n_k_points, n_modes)
            population for each k point and each mode
        """
        q_points = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        population = np.zeros((self.n_k_points, self.n_modes))
        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQTemp(q_point=q_point,
                                       second=self.forceconstants.second,
                                       distance_threshold=self.forceconstants.distance_threshold,
                                       folder=self.folder,
                                       storage=self.storage,
                                       temperature=self.temperature,
                                       is_classic=self.is_classic,
                                       is_nw=self.is_nw,
                                       is_unfolding=self.is_unfolding,
                                       is_amorphous=self._is_amorphous)

            population[ik] = phonon.population
        return population

    @lazy_property(label='<temperature>')
    def free_energy(self):
        """
        Harmonic free energy, already Brillouin-zone averaged,
        returned in eV per mode (including zero-point energy).

        Formula: F = k_B * T * ln(1 - exp(-hbar*omega/(k_B*T))) + hbar*omega/2

        where:
            k_B: Boltzmann constant
            T: temperature
            hbar: reduced Planck constant
            omega: angular frequency (2*pi*frequency)

        Returns
        -------
        ndarray
            Free energy array with shape matching frequency array. Non-physical modes
            (including acoustic modes at Gamma point and modes with imaginary frequencies)
            have zero free energy in the returned array.

        Notes
        -----
        - At T=0, only the zero-point energy contributes (thermal term vanishes)
        - Modes with imaginary frequencies (negative frequency values) are automatically
          excluded and will trigger a warning, as they may indicate structural instability
        """
        # At T=0, F = ZPE only (thermal contribution vanishes)
        if self.temperature == 0:
            return self.zero_point_harmonic_energy

        # Check for imaginary frequencies (negative values)
        has_imaginary = np.any(self.frequency < 0)
        if has_imaginary:
            n_imaginary = np.sum(self.frequency < 0)
            logging.warning(
                f"Found {n_imaginary} modes with imaginary frequencies (negative values). "
                f"These modes will be excluded from free energy calculation. "
                f"This may indicate structural instability."
            )

        x_vals = units._hbar * self.frequency * 2.0 * np.pi * 1.0e12 / (units._k * self.temperature)
        ln_term = np.zeros_like(x_vals)

        # Only calculate for physical modes with positive frequencies
        # This avoids both log(0) for low frequencies and log(negative) for imaginary frequencies
        physical = self.physical_mode.reshape(self.frequency.shape)
        valid_modes = physical & (self.frequency > 0)
        ln_term[valid_modes] = np.log1p(-np.exp(-x_vals[valid_modes]))  # ln(1 − e^{-x})

        # Thermal contribution: k_B*T*ln(1 - exp(-hbar*omega/(k_B*T))) in eV
        thermal_part_eV = 1.0 / units._e * units._k * self.temperature * ln_term

        # Zero-point energy: use the existing method to avoid code duplication (in eV)
        zpe_part = self.zero_point_harmonic_energy * self.n_k_points

        # Combine thermal and zero-point energy contributions
        f_cell = thermal_part_eV + zpe_part
        return f_cell / self.n_k_points

    @lazy_property(label='')
    def zero_point_harmonic_energy(self):
        """
        Harmonic zero-point energy, Brillouin-zone averaged,
        returned in eV per mode.
        """
        zpe_cell = 0.5 * units._hbar * self.frequency * 2.0 * np.pi * 1.0e12 / units._e
        return zpe_cell / self.n_k_points


    @lazy_property(label='<temperature>/<statistics>/<third_bandwidth>/<include_isotopes>')
    def bandwidth(self):
        """
        Calculate the phonons bandwidth, the inverse of the lifetime, for each k point in k_points and each mode.

        Returns
        -------
        bandwidth : np.array(n_k_points, n_modes)
            bandwidth for each k point and each mode
        """
        gamma = self.anharmonic_bandwidth
        if self.include_isotopes:
            gamma += self.isotopic_bandwidth
        return gamma


    @lazy_property(label='<third_bandwidth>')
    def isotopic_bandwidth(self):
        """ 
        Calculate the isotopic bandwidth with Tamura perturbative formula.
        Defined by equations in DOI:https://doi.org/10.1103/PhysRevB.27.858

        Returns
        -------
        isotopic_bw : np array(n_k_points, n_modes)
             atomic participation
        """
        if self._is_amorphous:
            logging.warning('isotopic scattering not implemented for amorphous systems')
            return np.zeros(self.n_k_points, self.n_modes)
        else:
            if self.g_factor is not None:
                isotopic_bw=isotopic.compute_isotopic_bw(self)
            else:
                atoms=self.atoms
                logging.warning('input isotopic gfactors are missing, using isotopic concentrations from ase database (NIST)')
                self.g_factor=isotopic.compute_gfactor(atoms.get_atomic_numbers() )
                logging.info('g factors='+str(self.g_factor))
                isotopic_bw = isotopic.compute_isotopic_bw(self)

            return isotopic_bw


    @lazy_property(label='<temperature>/<statistics>/<third_bandwidth>')
    def anharmonic_bandwidth(self):
        """
        Calculate the phonons bandwidth, the inverse of the lifetime, for each k point in k_points and each mode.

        Returns
        -------
        bandwidth : np.array(n_k_points, n_modes)
            bandwidth for each k point and each mode
        """
        gamma = self._ps_and_gamma[:, 1].reshape(self.n_k_points, self.n_modes)
        return gamma


    @lazy_property(label='<temperature>/<statistics>/<third_bandwidth>')
    def phase_space(self):
        """
        Calculate the 3-phonons-processes phase_space, for each k point in k_points and each mode.

        Returns
        -------
        phase_space : np.array(n_k_points, n_modes)
            phase_space for each k point and each mode
        """
        ps = self._ps_and_gamma[:, 0].reshape(self.n_k_points, self.n_modes)
        return ps


    @lazy_property(label='')
    def eigenvalues(self):
        """
        Calculates the eigenvalues of the dynamical matrix in Thz^2.

        Returns
        -------
        eigenvalues : np array(n_phonons)
             Eigenvalues of the dynamical matrix
        """
        eigenvalues = self._eigensystem[:, 0, :]
        return eigenvalues


    @property
    def eigenvectors(self):
        """
        Calculates the eigenvectors of the dynamical matrix.

        Returns
        -------
        eigenvectors : np array(n_phonons, n_phonons)
             Eigenvectors of the dynamical matrix
        """
        eigenvectors = self._eigensystem[:, 1:, :]
        return eigenvectors


    @lazy_property(label='<temperature>/<statistics>/<third_bandwidth>')
    def _ps_and_gamma(self):
        store_format = self._store_formats.get('_ps_gamma_and_gamma_tensor', 'numpy') \
            if self.storage == 'formatted' else self.storage
        if is_calculated('_ps_gamma_and_gamma_tensor', self, '<temperature>/<statistics>/<third_bandwidth>', \
                         format=store_format):
            ps_and_gamma = self._ps_gamma_and_gamma_tensor[:, :2]
        else:
            ps_and_gamma = self._select_algorithm_for_phase_space_and_gamma(is_gamma_tensor_enabled=False)
        return ps_and_gamma


    @lazy_property(label='<temperature>/<statistics>/<third_bandwidth>')
    def _ps_gamma_and_gamma_tensor(self):
        ps_gamma_and_gamma_tensor = self._select_algorithm_for_phase_space_and_gamma(is_gamma_tensor_enabled=True)
        return ps_gamma_and_gamma_tensor

    @lazy_property(label='<third_bandwidth>')
    def _sparse_phase_and_potential(self):
        """
        Calculate both sparse phase and potential tensors for anharmonic interactions.
        
        Returns
        -------
        tuple : (sparse_phase, sparse_potential)
            Both sparse tensor data structures
        """
        # Calculate from scratch
        if self._is_amorphous:
            return self._project_amorphous()
        else:
            return self._project_crystal()

    def _convert_sparse_tensors_to_per_mu_arrays(self, sparse_phase, sparse_potential):
        """
        Convert sparse tensors to per-mu arrays for numpy storage.
        
        Returns a list where each element contains the data for one mu (phonon mode).
        Each mu can have 0, 1, or 2 non-None tensors (for is_plus=0,1).
        
        Returns
        -------
        list : List of per-mu data dictionaries
        """
        per_mu_data = []
        
        for nu_single in range(len(sparse_phase)):
            mu_data = {'exists': False, 'tensors': []}
            
            for is_plus in range(2):
                if (nu_single < len(sparse_phase) and 
                    is_plus < len(sparse_phase[nu_single]) and
                    sparse_phase[nu_single][is_plus] is not None):
                    
                    phase_tensor = sparse_phase[nu_single][is_plus]
                    potential_tensor = sparse_potential[nu_single][is_plus]
                    
                    # Get tensor components
                    indices = phase_tensor.indices.numpy()
                    phase_values = phase_tensor.values.numpy()  
                    potential_values = potential_tensor.values.numpy()
                    dense_shape = phase_tensor.dense_shape.numpy()
                    
                    tensor_data = {
                        'is_plus': is_plus,
                        'indices': indices,
                        'phase_values': phase_values,
                        'potential_values': potential_values,
                        'dense_shape': dense_shape
                    }
                    
                    mu_data['tensors'].append(tensor_data)
                    mu_data['exists'] = True
            
            per_mu_data.append(mu_data)
        
        return per_mu_data
    
    def _convert_per_mu_arrays_to_sparse_tensors(self, per_mu_data):
        """
        Convert per-mu arrays back to sparse tensor format.
        
        Parameters
        ----------
        per_mu_data : list
            List of per-mu data dictionaries
            
        Returns
        -------
        tuple : (sparse_phase, sparse_potential)
        """
        import tensorflow as tf
        
        sparse_phase = []
        sparse_potential = []
        
        for nu_single, mu_data in enumerate(per_mu_data):
            sparse_phase.append([None, None])  # Initialize with None for both is_plus
            sparse_potential.append([None, None])
            
            if mu_data['exists']:
                for tensor_data in mu_data['tensors']:
                    is_plus = tensor_data['is_plus']
                    
                    # Reconstruct sparse tensors
                    phase_tensor = tf.SparseTensor(
                        indices=tensor_data['indices'],
                        values=tensor_data['phase_values'],
                        dense_shape=tensor_data['dense_shape']
                    )
                    potential_tensor = tf.SparseTensor(
                        indices=tensor_data['indices'],
                        values=tensor_data['potential_values'],
                        dense_shape=tensor_data['dense_shape']
                    )
                    
                    sparse_phase[nu_single][is_plus] = phase_tensor
                    sparse_potential[nu_single][is_plus] = potential_tensor
        
        return sparse_phase, sparse_potential

    def _load_property(self, property_name, folder, format='formatted'):
        """
        Override to handle custom loading for sparse tensors.
        """
        if property_name == '_sparse_phase_and_potential' and format == 'numpy':
            # Custom loading for sparse tensors from per-mu numpy arrays
            base_name = folder + '/' + property_name
            
            # Load list of which mus exist
            saved_mus = np.load(f'{base_name}_mu_list.npy')
            
            # Determine total number of mus needed
            n_phonons = self.n_phonons
            
            # Initialize per_mu_data with correct size
            per_mu_data = [{'exists': False, 'tensors': []} for _ in range(n_phonons)]
            
            # Load existing mu data
            for nu_single in saved_mus:
                mu_filename = f'{base_name}_mu_{nu_single}.npy'
                mu_data = np.load(mu_filename, allow_pickle=True).item()
                per_mu_data[nu_single] = mu_data
            
            return self._convert_per_mu_arrays_to_sparse_tensors(per_mu_data)
        else:
            # Use parent method for other properties
            return super()._load_property(property_name, folder, format)

    def _save_property(self, property_name, folder, data, format='formatted'):
        """
        Override to handle custom storage for sparse tensors.
        """
        if property_name == '_sparse_phase_and_potential':
            if format == 'numpy':
                # Custom storage for sparse tensors as per-mu numpy arrays
                sparse_phase, sparse_potential = data
                per_mu_data = self._convert_sparse_tensors_to_per_mu_arrays(sparse_phase, sparse_potential)
                
                # Save each mu separately
                import os
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    
                base_name = folder + '/' + property_name
                
                # Save only mus that have data
                saved_mus = []
                for nu_single, mu_data in enumerate(per_mu_data):
                    if mu_data['exists']:
                        mu_filename = f'{base_name}_mu_{nu_single}.npy'
                        np.save(mu_filename, mu_data, allow_pickle=True)
                        saved_mus.append(nu_single)
                
                # Save list of which mus exist
                np.save(f'{base_name}_mu_list.npy', np.array(saved_mus, dtype=np.int32))
                
                logging.info(f'{base_name} stored as {len(saved_mus)} per-mu arrays')
            else:
                # Sparse tensors cannot be saved in formatted text format
                # Force numpy format for this property type
                logging.warning(f'Sparse tensors cannot be saved in {format} format. Using numpy format instead.')
                self._save_property(property_name, folder, data, format='numpy')
        else:
            # Use parent method for other properties
            super()._save_property(property_name, folder, data, format)

    @property
    def sparse_phase(self):
        """
        Sparse phase space tensor for anharmonic interactions.
        
        Returns
        -------
        sparse_phase : list
            List of sparse tensors containing phase space information
        """
        return self._sparse_phase_and_potential[0]

    @property
    def sparse_potential(self):
        """
        Sparse potential tensor for anharmonic interactions.
        
        Returns
        -------
        sparse_potential : list
            List of sparse tensors containing potential information
        """
        return self._sparse_phase_and_potential[1]


    @property
    def omega(self):
        """
        Calculates the angular frequencies from the diagonalized dynamical matrix.

        Returns
        -------
        frequency : np.array(n_k_points, n_modes)
            frequency in rad
        """
        return self.frequency * 2 * np.pi


    @property
    def _rescaled_eigenvectors(self):
        n_atoms = self.n_atoms
        n_modes = self.n_modes
        masses = self.atoms.get_masses()
        rescaled_eigenvectors = self.eigenvectors[:, :, :].reshape(
            (self.n_k_points, n_atoms, 3, n_modes)) / np.sqrt(
            masses[np.newaxis, :, np.newaxis, np.newaxis])
        rescaled_eigenvectors = rescaled_eigenvectors.reshape((self.n_k_points, n_modes, n_modes))
        return rescaled_eigenvectors


    @property
    def _is_amorphous(self):
        is_amorphous = np.array_equal(self.kpts, (1, 1, 1)) and np.array_equal(self.supercell, (1, 1, 1))
        return is_amorphous


    def pdos(self, p_atoms=None, direction=None, bandwidth=0.05, n_points=200):
        """Calculate the atom projected phonon density of states.
        Total density of states can be computed by specifying all atom indices in p_atoms.
        p_atoms input format is flexible:
        - Providing a list of atom indices will return the single pdos summed over those atoms
        - Providing a list of lists of atom indices will return one pdos for each set of indices

        Returns
        -------
            frequency : np array(n_points)
                Frequencies
            pdos : np.array(n_projections, n_points)
                pdos for each set of projected atoms and directions
        """
        if p_atoms is None:
            p_atoms = list(range(self.n_atoms))

        n_proj = len(p_atoms)

        if n_proj == 0:
            logging.error('No atoms provided for projection.')
            raise IndexError('Cannot project on an empty set of atoms.')

        else:
            try:
                _ = iter(p_atoms[0])

            except TypeError as e:
                n_proj = 1
                p_atoms = [p_atoms]

        n_modes = self.n_modes
        eigensystem = self._eigensystem
        eigenvals = eigensystem[:, 0, :]
        normal_modes = eigensystem[:, 1:, :]
        frequency = np.real(np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.))

        # Get physical mode mask to filter out acoustic modes at Gamma and other non-physical modes
        physical_mode = self.physical_mode

        fmin, fmax = frequency.min(), frequency.max()

        f_grid = np.linspace(0.9 * fmin, 1.1 * fmax, n_points)

        p_dos = np.zeros((n_proj,n_points), dtype=float)
        for ip in range(n_proj):

            n_atoms = len(p_atoms[ip])
            atom_mask = np.zeros(n_modes, dtype=bool)
            for p in p_atoms[ip]:
                i0 = 3 * p
                atom_mask[i0:i0+3] = True

            masked_modes = normal_modes[:, atom_mask, :]

            if isinstance(direction, str):
                logging.error('Direction type not implemented.')
                raise NotImplementedError('Direction type not implemented.')

            else:
                ix = 3 * np.arange(n_atoms, dtype=int)
                iy,iz = ix + 1, ix + 2

                proj = None
                if direction is None:
                    proj = np.abs(masked_modes[:, ix, :]) ** 2
                    proj += np.abs(masked_modes[:, iy, :]) ** 2
                    proj += np.abs(masked_modes[:, iz, :]) ** 2

                else:
                    direction = np.array(direction, dtype=float)
                    direction /= np.linalg.norm(direction)

                    proj = masked_modes[:, ix, :] * direction[0]
                    proj += masked_modes[:, iy, :] * direction[1]
                    proj += masked_modes[:, iz, :] * direction[2]
                    proj = np.abs(proj) ** 2

            for i in range(n_points):
                x = (frequency - f_grid[i]) / np.sqrt(bandwidth)
                amp = stats.norm.pdf(x)
                # Apply physical mode mask when summing
                for j in range(proj.shape[1]):
                    p_dos[ip, i] += np.sum(amp * proj[:, j, :] * physical_mode)

            # TODO: np.trapz is deprecated in numpy 2.0+, but np.trapezoid does not exist before numpy 2.0. migrate it after this function gets removed. 
            p_dos[ip] *= 3 * n_atoms / np.trapz(p_dos[ip], f_grid)

        return f_grid, p_dos


    def _allowed_third_phonons_index(self, index_q, is_plus):
        q_vec = self._reciprocal_grid.id_to_unitary_grid_index(index_q)
        qp_vec = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        qpp_vec = q_vec[np.newaxis, :] + (int(is_plus) * 2 - 1) * qp_vec[:, :]
        rescaled_qpp = np.round((qpp_vec * self._reciprocal_grid.grid_shape), 0).astype(int)
        rescaled_qpp = np.mod(rescaled_qpp, self._reciprocal_grid.grid_shape)
        index_qpp_full = np.ravel_multi_index(rescaled_qpp.T, self._reciprocal_grid.grid_shape, mode='raise',
                                              order=self._grid_type)
        return index_qpp_full


    @property
    def _ir_kgrid_data(self):
        """Cached (ir_mapping, krot_perm, ibz_indices) from spglib."""
        try:
            return self.__ir_kgrid_data
        except AttributeError:
            self.__ir_kgrid_data = _get_ir_kgrid_data(
                self.atoms, self.kpts, self._grid_type)
            n_irr = len(self.__ir_kgrid_data[2])
            logging.info(
                f'q-symmetry: IBZ has {n_irr} of {self.n_k_points} k-points '
                f'(reduction factor {self.n_k_points / n_irr:.1f}x)'
            )
            return self.__ir_kgrid_data

    def _select_algorithm_for_phase_space_and_gamma(self, is_gamma_tensor_enabled=True):
        self.n_k_points = np.prod(self.kpts)
        self.n_phonons = self.n_k_points * self.n_modes
        self.is_gamma_tensor_enabled = is_gamma_tensor_enabled
        # Reshape population to 1D for unified indexing
        population_flat = self.population.flatten()

        # Apply hbar scaling factor for classical vs quantum
        hbar_factor = 1e-6 if self.is_classic else 1

        ps_and_gamma = aha.calculate_ps_and_gamma(
            self.sparse_phase,
            self.sparse_potential,
            population_flat,
            self.is_balanced,
            self.n_phonons,
            self._is_amorphous,
            self.is_gamma_tensor_enabled,
            hbar_factor
        )
        if not self._is_amorphous:
            ps_and_gamma[:, 0] /= self.n_k_points

        # Replicate IBZ results to symmetry-equivalent k-points.
        if self.use_q_symmetry and not self._is_amorphous:
            ir_mapping, krot_perm, _, _ = self._ir_kgrid_data
            n_k = self.n_k_points
            n_m = self.n_modes
            for ik in range(n_k):
                ik_irr = ir_mapping[ik]
                if ik == ik_irr:
                    continue
                # Scalar columns (phase space, bandwidth): direct copy.
                ps_and_gamma[ik * n_m:(ik + 1) * n_m, :2] = \
                    ps_and_gamma[ik_irr * n_m:(ik_irr + 1) * n_m, :2]
                # Gamma-tensor columns: replicate with k-index permutation.
                if is_gamma_tensor_enabled:
                    perm = krot_perm[ik]   # (n_k,) int array
                    for mu in range(n_m):
                        row_irr = ps_and_gamma[ik_irr * n_m + mu, 2:].reshape(n_k, n_m)
                        ps_and_gamma[ik * n_m + mu, 2:] = row_irr[perm, :].reshape(n_k * n_m)

        return ps_and_gamma


    @timeit
    def _project_amorphous(self):
        """
        Project anharmonic properties for amorphous materials.

        Args:
            self: Phonon object containing material properties

        Returns:
            np.ndarray: Array containing projected properties
        """
        frequency = self.frequency
        omega = 2 * np.pi * frequency
        n_replicas = self.forceconstants.n_replicas
        rescaled_eigenvectors = self._rescaled_eigenvectors.astype(float)
        evect_tf = tf.convert_to_tensor(rescaled_eigenvectors[0])
        coords = self.forceconstants.third.value.coords
        coords = np.vstack([coords[1], coords[2], coords[0]])
        coords = tf.cast(coords.T, dtype=tf.int64)
        data = self.forceconstants.third.value.data
        third_tf = tf.SparseTensor(
            coords, data, (self.n_modes * n_replicas, self.n_modes * n_replicas, self.n_modes)
        )
        third_tf = tf.sparse.reshape(third_tf, ((self.n_modes * n_replicas) ** 2, self.n_modes))
        physical_mode = self.physical_mode.reshape((self.n_k_points, self.n_modes))
        logging.info("Projection started")
        hbar = units._hbar
        sigma_tf = tf.constant(self.third_bandwidth, dtype=tf.float64)
        n_modes = self.n_modes
        broadening_shape = self.broadening_shape
        n_phonons = self.n_phonons
        sparse_phase = []
        sparse_potential = []
        for nu_single in range(self.n_phonons):
            if nu_single % 200 == 0:
                logging.info("calculating third " + f"{nu_single}" + ": " + \
                             f"{100 * nu_single / self.n_phonons:.2f}%")

            sparse_phase.append([])
            sparse_potential.append([])
            for is_plus in (0, 1):

                # ps_and_gamma = np.zeros(2)
                if not physical_mode[0, nu_single]:
                    sparse_phase[nu_single].extend([None, None])
                    sparse_potential[nu_single].extend([None, None])
                    continue

                dirac_delta_result = aha.calculate_dirac_delta_amorphous(is_plus, nu_single, omega, physical_mode, sigma_tf,
                                                                     broadening_shape, n_phonons)
                if not dirac_delta_result:
                    sparse_phase[nu_single].append(None)
                    sparse_potential[nu_single].append(None)
                    continue
                sparse_phase[nu_single].append(dirac_delta_result)
                mup_vec, mupp_vec = tf.unstack(dirac_delta_result.indices, axis=1)

                third_nu_tf = tf.sparse.sparse_dense_matmul(third_tf, tf.reshape(evect_tf[:, nu_single], (n_modes, 1)))
                third_nu_tf = tf.reshape(third_nu_tf, (n_modes * n_replicas, n_modes * n_replicas))
                scaled_potential_tf = tf.einsum("ij,in,jm->nm", third_nu_tf, evect_tf, evect_tf)
                coords = tf.stack((mup_vec, mupp_vec), axis=-1)
                pot_times_dirac = tf.gather_nd(scaled_potential_tf, coords) ** 2
                pot_times_dirac /= tf.gather(omega[0], mup_vec) * tf.gather(omega[0], mupp_vec)
                pot_times_dirac *= np.pi * hbar / 4.0 * GAMMA_TO_THZ / omega.flatten()[nu_single]
                
                # Convert to sparse tensor using the same indices as sparse_phase
                sparse_potential_tensor = tf.SparseTensor(
                    indices=dirac_delta_result.indices,
                    values=pot_times_dirac,
                    dense_shape=dirac_delta_result.dense_shape
                )
                sparse_potential[nu_single].append(sparse_potential_tensor)
        return sparse_phase, sparse_potential


    @timeit
    def _project_crystal(self):
        n_replicas = self.forceconstants.third.n_replicas
        try:
            sparse_third = self.forceconstants.third.value.reshape((self.n_modes, -1))
            sparse_coords = tf.stack([sparse_third.coords[1], sparse_third.coords[0]], -1)
            sparse_coords = tf.cast(sparse_coords, dtype=tf.int64)
            third_sparse_data = (
                sparse_coords.numpy(),
                sparse_third.data.astype(np.complex128) if sparse_third.data.dtype != np.complex128 else sparse_third.data,
                ((self.n_modes * n_replicas) ** 2, self.n_modes),
            )
            is_sparse = True
        except AttributeError:
            third_sparse_data = np.array(self.forceconstants.third.value)
            is_sparse = False

        k_mesh = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        n_k_points = k_mesh.shape[0]
        chi_k_np = np.array(self.forceconstants.third._chi_k(k_mesh))
        evect_np = np.array(self._rescaled_eigenvectors)
        physical_mode = self.physical_mode.reshape((self.n_k_points, self.n_modes))
        omega = self.omega
        velocity_np = np.array(self.velocity)
        cell_inv = self.forceconstants.cell_inv
        kpts = self.kpts
        hbar = units._hbar
        n_modes = self.n_modes

        # Determine which k-points to compute.  When q-symmetry is enabled
        # only IBZ representatives need full projection; equivalents are
        # skipped here and replicated in _select_algorithm_for_phase_space_and_gamma.
        if self.use_q_symmetry:
            _, _, ibz_compute, _ = self._ir_kgrid_data
        else:
            ibz_compute = list(range(n_k_points))

        # Pre-compute k-point mappings (avoids passing self to workers)
        kpoint_maps = {}
        for ik in ibz_compute:
            for is_plus in (0, 1):
                kpoint_maps[(ik, is_plus)] = self._allowed_third_phonons_index(ik, is_plus)

        logging.info("Projection started")

        # Shared config for all k-point workers (all numpy, picklable)
        shared = dict(
            n_modes=n_modes, n_k_points=n_k_points, omega=omega,
            physical_mode=physical_mode, evect_np=evect_np,
            third_sparse_data=third_sparse_data, chi_k_np=chi_k_np,
            velocity_np=velocity_np, cell_inv=cell_inv, kpts=kpts,
            broadening_shape=self.broadening_shape,
            third_bandwidth=self.third_bandwidth, hbar=hbar,
            n_replicas=n_replicas, is_sparse=is_sparse,
            kpoint_maps=kpoint_maps,
        )

        use_parallel = self.n_workers is None or self.n_workers > 1
        backend = 'process' if use_parallel else 'serial'
        n_workers = self.n_workers if use_parallel else None

        # Determine which k-points to compute (resume support + q-symmetry).
        # When use_q_symmetry is True, ibz_compute already limits to IBZ reps;
        # non-IBZ k-points are assembled as all-None tensors below.
        output_dir = self.projection_output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            kpoints_to_compute = [
                ik for ik in ibz_compute
                if not os.path.exists(os.path.join(output_dir, f'kpt_{ik:05d}.done'))
            ]
            n_resumed = len(ibz_compute) - len(kpoints_to_compute)
            if n_resumed:
                logging.info(f'Resuming projection: skipping {n_resumed} already-computed k-point(s)')
        else:
            kpoints_to_compute = list(ibz_compute)

        worker_fn = functools.partial(_compute_kpoint_projection, **shared)

        # Collect results per k-point
        kpoint_results = {}
        with get_executor(backend=backend, n_workers=n_workers) as executor:
            futures = {
                executor.submit(worker_fn, ik): ik
                for ik in kpoints_to_compute
            }
            for future in as_completed(futures):
                ik = futures[future]
                kpoint_results[ik] = future.result()
                logging.info(f'Completed k-point {ik}/{n_k_points}: '
                             f'{100 * len(kpoint_results) / len(kpoints_to_compute):.0f}%')

                # Write to disk if output_dir is set
                if output_dir:
                    _save_kpoint_projection(output_dir, ik, kpoint_results[ik])

        # All-None placeholder used for non-IBZ k-points under q-symmetry.
        _null_results = [([None, None], [None, None])] * n_modes

        # Assemble into the flat sparse_phase / sparse_potential lists.
        # Non-IBZ k-points get all-None tensors; their ps_and_gamma rows are
        # filled by replication in _select_algorithm_for_phase_space_and_gamma.
        sparse_phase = []
        sparse_potential = []
        ibz_set = set(ibz_compute)
        for ik in range(n_k_points):
            if ik in kpoint_results:
                results = kpoint_results[ik]
            elif output_dir and ik in ibz_set:
                results = _load_kpoint_projection(output_dir, ik, n_modes)
            elif ik not in ibz_set:
                # Non-IBZ k-point: placeholder (replicated later)
                results = _null_results
            else:
                raise RuntimeError(f'Missing results for k-point {ik}')

            for mu in range(n_modes):
                phase_list, pot_list = results[mu]
                sparse_phase.append([
                    _numpy_to_sparse_tensor(phase_list[0]),
                    _numpy_to_sparse_tensor(phase_list[1]),
                ])
                sparse_potential.append([
                    _numpy_to_sparse_tensor(pot_list[0]),
                    _numpy_to_sparse_tensor(pot_list[1]),
                ])

        return sparse_phase, sparse_potential

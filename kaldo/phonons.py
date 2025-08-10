"""
kaldo
Anharmonic Lattice Dynamics

"""
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
from scipy import stats
import numpy as np
from numpy.typing import ArrayLike
import ase.units as units
from kaldo.helpers.tools import timeit
from kaldo.helpers.logger import get_logger

logging = get_logger()

# Constants
GAMMA_TO_THZ = 1e11 * units.mol * (units.mol / (10 * units.J)) ** 2
HBAR = units._hbar
THZ_TO_MEV = units.J * HBAR * 2 * np.pi * 1e15

class Phonons(Storable):
    """
    The Phonons object exposes all the phononic properties of a system by manipulation
    of the quantities passed into the ForceConstant object. The arguments passed in here
    reflect assumptions to be made about the macroscopic system e.g. the temperature, or
    whether the system is amorphous or a nanowire.
    The ForceConstants, and temperature are the only two required parameters, though we
    highly recommend the switch controlling whether to use quantum/classical statistics
    (`is_classic`) and the number of k-points to consider (`kpts`).
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
        Default: `False`
    kpts : (int, int, int)
        Defines the number of k points to use to create the k mesh
        Default: (1, 1, 1)
    min_frequency : float
        Ignores all phonons with frequency below `min_frequency`
        Units: Thz
        Default: `None`
    max_frequency : float
        Ignores all phonons with frequency above `max_frequency`
        Units: THz
        Default: `None`
    third_bandwidth : float
        Defines the width of the energy conservation smearing in the phonons
        scattering calculation. If `None` the width is calculated
        dynamically. Otherwise the input value corresponds to the width.
        Units: THz
        Default: `None`
    broadening_shape : string
        Defines the algorithm to use for line-broadening when enforcing
        energy conservation rules for three-phonon scattering.
        Options: `gauss`, `lorentz` and `triangle`.
        Default: `gauss`
    folder : string
        Specifies where to store the data files.
        Default: `output`.
    storage : string
        Defines the strategy used to store observables. The `default` strategy
        stores formatted text files for most harmonic properties but relies on
        numpy arrays for large arrays like the gamma tensor. The `memory` option
        doesn't generate any output except what is printed in your script.
        Options: `default`, `formatted`, `numpy`, `memory`, `hdf5`
        Default: 'formatted'
    grid_type : string
        Specifies whether the atoms in the replicated system were repeated using
        a C-like index ordering which changes the last axis the fastest or
        FORTRAN-like index ordering which changes the first index fastest.
        Options: 'C', 'F'
        Default: 'C'
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
                 **kwargs):
        self.forceconstants = forceconstants
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
        velocity : np array(n_k_points, n_unit_cell * 3, 3)
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
        Bose-Einstein weighted by each phonons energy.
        .. math::

            c_\\mu = k_B \\frac{\\nu_\\mu^2}{ \\tilde T^2} n_\\mu (n_\\mu + 1)

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
        Harmonic **thermal** free energy, already Brillouin-zone averaged,
        returned in eV per mode (ZPE not included).
        """
        x_vals = units._hbar * self.frequency * 2.0 * np.pi * 1.0e12 / (units._k * self.temperature)
        ln_term = np.log1p(-np.exp(-x_vals))  # ln(1 − e^{-x})
        f_cell = 1000.0 / units._e * units._k * self.temperature * ln_term
        return f_cell / self.n_k_points

    @lazy_property(label='')
    def zero_point_harmonic_energy(self):
        """
        Harmonic zero-point energy, Brillouin-zone averaged,
        returned in eV per mode.
        """
        zpe_cell = 0.5 * units._hbar * self.frequency * 2.0 * np.pi * 1.0e15 / units._e
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
        if property_name == '_sparse_phase_and_potential' and format == 'numpy':
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
                for j in range(proj.shape[1]):
                    p_dos[ip, i] += np.sum(amp * proj[:, j, :])

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


    def _select_algorithm_for_phase_space_and_gamma(self, is_gamma_tensor_enabled=True):
        self.n_k_points = np.prod(self.kpts)
        self.n_phonons = self.n_k_points * self.n_modes
        self.is_gamma_tensor_enabled = is_gamma_tensor_enabled
        # Reshape population to 1D for unified indexing
        population_flat = self.population.flatten()
        
        ps_and_gamma = aha.calculate_ps_and_gamma(
            self.sparse_phase,
            self.sparse_potential,
            population_flat,
            self.is_balanced,
            self.n_phonons,
            self._is_amorphous,
            self.is_gamma_tensor_enabled
        )
        if not self._is_amorphous:
            ps_and_gamma[:, 0] /= self.n_k_points

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
        hbar = HBAR * (1e-6 if self.is_classic else 1)
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
            third_tf = tf.SparseTensor(
                sparse_coords, sparse_third.data, ((self.n_modes * n_replicas) ** 2, self.n_modes)
            )
            is_sparse = True
        except AttributeError:
            third_tf = tf.convert_to_tensor(self.forceconstants.third.value)
            is_sparse = False
        third_tf = tf.cast(third_tf, dtype=tf.complex128)
        k_mesh = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        n_k_points = k_mesh.shape[0]
        _chi_k = tf.convert_to_tensor(self.forceconstants.third._chi_k(k_mesh))
        _chi_k = tf.cast(_chi_k, dtype=tf.complex128)
        evect_tf = tf.convert_to_tensor(self._rescaled_eigenvectors)
        evect_tf = tf.cast(evect_tf, dtype=tf.complex128)
        second_minus = tf.math.conj(evect_tf)
        second_minus_chi = tf.math.conj(_chi_k)
        logging.info("Projection started")
        broadening_shape = self.broadening_shape
        physical_mode = self.physical_mode.reshape((self.n_k_points, self.n_modes))
        omega = self.omega
        velocity_tf = tf.convert_to_tensor(self.velocity)
        cell_inv = self.forceconstants.cell_inv
        kpts = self.kpts
        hbar = HBAR * (1e-6 if self.is_classic else 1)
        n_modes = self.n_modes
        n_phonons = self.n_phonons
        sparse_phase = []
        sparse_potential = []
        for nu_single in range(n_phonons):
            if nu_single % 200 == 0:
                logging.info(f"calculating third {nu_single}: {100 * nu_single / self.n_phonons:.2f}%")
            sparse_phase.append([])
            sparse_potential.append([])
            index_k, mu = np.unravel_index(nu_single, (n_k_points, self.n_modes))
            if not physical_mode[index_k, mu]:
                sparse_phase[nu_single].extend([None, None])
                sparse_potential[nu_single].extend([None, None])
                continue
            for is_plus in (0, 1):
                index_kpp_full = tf.cast(self._allowed_third_phonons_index(index_k, is_plus), dtype=tf.int32)
                if self.third_bandwidth:
                    sigma_tf = tf.constant(self.third_bandwidth, dtype=tf.float64)
                else:
                    sigma_tf = aha.calculate_broadening(velocity_tf, cell_inv, kpts, index_kpp_full)
                dirac_delta_result = aha.calculate_dirac_delta_crystal(
                    omega,
                    physical_mode,
                    sigma_tf,
                    broadening_shape,
                    index_kpp_full,
                    index_k,
                    mu,
                    is_plus,
                    n_k_points,
                    n_modes,
                )
                if not dirac_delta_result:
                    sparse_phase[nu_single].append(None)
                    sparse_potential[nu_single].append(None)
                    continue
                sparse_phase[nu_single].append(dirac_delta_result)
                sparse_potential[nu_single].append(
                    aha.sparse_potential_mu(
                        nu_single,
                        evect_tf,
                        dirac_delta_result,
                        index_k,
                        mu,
                        n_k_points,
                        n_modes,
                        is_plus,
                        is_sparse,
                        index_kpp_full,
                        _chi_k,
                        second_minus,
                        second_minus_chi,
                        third_tf,
                        n_replicas,
                        omega,
                        hbar,
                    )
                )
        return sparse_phase, sparse_potential
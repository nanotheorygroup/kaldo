"""
kaldo
Anharmonic Lattice Dynamics

"""
from kaldo.helpers.storage import is_calculated
from kaldo.helpers.storage import lazy_property
from kaldo.helpers.logger import log_size
from kaldo.helpers.storage import DEFAULT_STORE_FORMATS, FOLDER_NAME
from kaldo.grid import Grid
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.observables.harmonic_with_q_temp import HarmonicWithQTemp
from kaldo.forceconstants import ForceConstants
import kaldo.controllers.anharmonic as aha
import kaldo.controllers.isotopic as isotopic
from scipy import stats
import numpy as np
from numpy.typing import ArrayLike
import ase.units as units
from kaldo.helpers.logger import get_logger
logging = get_logger()


class Phonons:
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

    # @lazy_property(label="<temperature>/<statistics>")
    @lazy_property(label='<temperature>')
    def free_energy(self):  # type: ignore
        """Return F(T) (meV pephonons.free_energyr H₂O) from an explicit list of mode frequencies.

        Parameters
        ----------
        freqs_thz : 1-D ndarray
            Flattened array of **physical-mode** frequencies (THz) for *all*
            q-points.
        t_kelvin : float
            Temperature in kelvin.
        n_qpoints : int
            Number of distinct q-points in the mesh.
        n_atoms_cell : int
            Atoms in the primitive cell (12 for ice-Ih).
        include_zpe : bool, optional
            Add the zero-point contribution (default: ``False``).

        Returns
        -------
        float
            Harmonic free energy in meV per atom/molecule.
        """
        freqs = self.frequency
        t_kelvin = self.temperature
        hbar_omega = units._hbar * freqs * 2.0 * np.pi * 1.0e12
        x_vals = hbar_omega / (units._k * t_kelvin)

        ln_term = np.log1p(-np.exp(-x_vals))  # ln(1 − e^{-x})

        f_cell = 1000.0 / units._e * units._k * t_kelvin * ln_term
        include_zpe = True
        if include_zpe:
            zpe_cell_J = 0.5 * hbar_omega  # J / cell
            F0_offset_mev_zpe = zpe_cell_J * 1000 / units._e
            f_cell += F0_offset_mev_zpe
        return f_cell

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
        store_format = DEFAULT_STORE_FORMATS['_ps_gamma_and_gamma_tensor'] \
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


# Helpers properties

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
        n_kpts = self.n_k_points
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
        if self._is_amorphous:
            ps_and_gamma = aha.project_amorphous(self)
        else:
            ps_and_gamma = aha.project_crystal(self)
        return ps_and_gamma


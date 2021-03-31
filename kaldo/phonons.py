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
import kaldo.controllers.anharmonic as aha
import numpy as np
import ase.units as units
from kaldo.helpers.logger import get_logger
logging = get_logger()


class Phonons:
    """The Phonons object exposes all the phononic properties of a system.
    It's can be fed into a Conductivity object and must be built with a
    ForceConstant object.

    Parameters
    ----------
    forceconstants : ForceConstants
        contains all the information about the system and the derivatives
        of the potential.
    is_classic : bool
        specifies if the system is classic, `True` or quantum, `False`
        Default is `False`
    kpts : (3) tuple, optional
        defines the number of k points to use to create the k mesh
        Default is (1, 1, 1)
    temperature : float
        defines the temperature of the simulation. Units: K.
    min_frequency : float, optional
        ignores all phonons with frequency below `min_frequency` THz,
        Default is `None`
    max_frequency : float, optional
        ignores all phonons with frequency above `max_frequency` THz
        Default is `None`
    third_bandwidth : float, optional
        Defines the width of the energy conservation smearing in the phonons
        scattering calculation. If `None` the width is calculated
        dynamically. Otherwise the input value corresponds to the width.
        Units: THz.
    broadening_shape : string, optional
        Defines the algorithm to use for the broadening of the conservation
        of the energy for third irder interactions. Available broadenings
        are `gauss`, `lorentz` and `triangle`.
        Default is `gauss`.
    folder : string, optional
        Specifies where to store the data files. Default is `output`.
    storage : `default`, `formatted`, `numpy`, `memory`, `hdf5`, optional
        Defines the storing strategy used to store the observables. The
        `default` strategy stores formatted output and numpy arrays.
        `memory` storage doesn't generate any output.
    grid_type : 'F' or 'C, optional
        Specify if to use 'C" style atoms replica grid of fortran
        style 'F',
        Default 'C'
    is_balanced : Enforce detailed balance when calculating anharmonic properties,
        Default: False

    Returns
    -------
    Phonons Object
    """
    def __init__(self, **kwargs):
        self.forceconstants = kwargs.pop('forceconstants')
        self.is_classic = bool(kwargs.pop('is_classic', False))
        if 'temperature' in kwargs:
            self.temperature = float(kwargs['temperature'])
        self.folder = kwargs.pop('folder', FOLDER_NAME)
        self.kpts = kwargs.pop('kpts', (1, 1, 1))
        self._grid_type = kwargs.pop('grid_type', 'C')
        self._reciprocal_grid = Grid(self.kpts, order=self._grid_type)
        self.is_unfolding = kwargs.pop('is_unfolding', False)
        if self.is_unfolding:
            logging.info('Using unfolding.')
        self.kpts = np.array(self.kpts)
        self.min_frequency = kwargs.pop('min_frequency', 0)
        self.max_frequency = kwargs.pop('max_frequency', None)
        self.broadening_shape = kwargs.pop('broadening_shape', 'gauss')
        self.is_nw = kwargs.pop('is_nw', False)
        self.third_bandwidth = kwargs.pop('third_bandwidth', None)
        self.storage = kwargs.pop('storage', 'formatted')
        self.is_symmetrizing_frequency = kwargs.pop('is_symmetrizing_frequency', False)
        self.is_antisymmetrizing_velocity = kwargs.pop('is_antisymmetrizing_velocity', False)
        self.is_balanced = kwargs.pop('is_balanced', False)
        self.atoms = self.forceconstants.atoms
        self.supercell = np.array(self.forceconstants.supercell)
        self.n_k_points = int(np.prod(self.kpts))
        self.n_atoms = self.forceconstants.n_atoms
        self.n_modes = self.forceconstants.n_modes
        self.n_phonons = self.n_k_points * self.n_modes
        self.is_able_to_calculate = True
        self.hbar = units._hbar
        if self.is_classic:
            self.hbar = self.hbar * 1e-6



    @lazy_property(label='')
    def physical_mode(self):
        """Calculate physical modes. Non physical modes are the first 3 modes of q=(0, 0, 0) and, if defined, all the
        modes outside the frequency range min_frequency and max_frequency.
        Returns
        -------
        physical_mode : np array
            (n_k_points, n_modes) bool
        """
        q_points = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        physical_mode = np.zeros((self.n_k_points, self.n_modes), dtype=np.bool)

        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point=q_point,
                                   second=self.forceconstants.second,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage,
                                   is_nw=self.is_nw,
                                   is_unfolding=self.is_unfolding)

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
        frequency : np array
            (n_k_points, n_modes) frequency in THz
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
                                   is_unfolding=self.is_unfolding)

            frequency[ik] = phonon.frequency
        return frequency


    @lazy_property(label='')
    def participation_ratio(self):
        """Calculate phonons participation ratio. A value of 1 corresponds to translation.
        Returns
        -------
        participation_ratio : np array
            (n_k_points, n_modes) atomic participation
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
                                   is_unfolding=self.is_unfolding)

            participation_ratio[ik] = phonon.participation_ratio

        return participation_ratio


    @lazy_property(label='')
    def velocity(self):
        """Calculates the velocity using Hellmann-Feynman theorem.

        Returns
        -------
        velocity : np array
            (n_k_points, n_unit_cell * 3, 3) velocity in 100m/s or A/ps
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
                                   is_unfolding=self.is_unfolding)
            velocity[ik] = phonon.velocity
        return velocity

    @lazy_property(label='')
    def _eigensystem(self):
        """Calculate the eigensystems, for each k point in k_points.

        Returns
        -------
        _eigensystem : np.array(n_k_points, n_unit_cell * 3, n_unit_cell * 3 + 1)
            eigensystem is calculated for each k point, the three dimensional array
            records the eigenvalues in the last column of the last dimension.

            If the system is not amorphous, these values are stored as complex numbers.
        """
        q_points = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        shape = (self.n_k_points, self.n_modes + 1, self.n_modes)
        log_size(shape, name='eigensystem', type=np.complex)
        eigensystem = np.zeros(shape, dtype=np.complex)
        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point=q_point,
                                   second=self.forceconstants.second,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage,
                                   is_nw=self.is_nw,
                                   is_unfolding=self.is_unfolding)

            eigensystem[ik] = phonon._eigensystem

        return eigensystem


    @lazy_property(label='<temperature>/<statistics>')
    def heat_capacity(self):
        """Calculate the heat capacity for each k point in k_points and each mode.
        If classical, it returns the Boltzmann constant in W/m/K. If quantum it returns the derivative of the
        Bose-Einstein weighted by each phonons energy.
        .. math::

            c_\\mu = k_B \\frac{\\nu_\\mu^2}{ \\tilde T^2} n_\\mu (n_\\mu + 1)

        where the frequency :math:`\\nu` and the temperature :math:`\\tilde T` are in THz.

        Returns
        -------
        c_v : np.array(n_k_points, n_modes)
            heat capacity in W/m/K for each k point and each mode
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
                                       is_unfolding=self.is_unfolding)
            c_v[ik] = phonon.heat_capacity
        return c_v


    @lazy_property(label='<temperature>/<statistics>')
    def heat_capacity_2d(self):
        """Calculate the generalized 2d heat capacity for each k point in k_points and each mode.
        If classical, it returns the Boltzmann constant in W/m/K.

        Returns
        -------
        heat_capacity_2d : np.array(n_k_points, n_modes, n_modes)
            heat capacity in W/m/K for each k point and each modes couple.
        """
        q_points = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        shape = (self.n_k_points, self.n_modes, self.n_modes)
        log_size(shape, name='heat_capacity_2d', type=np.float)
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
                                       is_unfolding=self.is_unfolding)
            heat_capacity_2d[ik] = phonon.heat_capacity_2d
        return heat_capacity_2d


    @lazy_property(label='<temperature>/<statistics>')
    def population(self):
        """Calculate the phonons population for each k point in k_points and each mode.
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
                                       is_unfolding=self.is_unfolding)
            population[ik] = phonon.population
        return population


    @lazy_property(label='<temperature>/<statistics>/<third_bandwidth>')
    def bandwidth(self):
        """Calculate the phonons bandwidth, the inverse of the lifetime, for each k point in k_points and each mode.

        Returns
        -------
        bandwidth : np.array(n_k_points, n_modes)
            bandwidth for each k point and each mode
        """
        gamma = self._ps_and_gamma[:, 1].reshape(self.n_k_points, self.n_modes)
        return gamma


    @lazy_property(label='<temperature>/<statistics>/<third_bandwidth>')
    def phase_space(self):
        """Calculate the 3-phonons-processes phase_space, for each k point in k_points and each mode.

        Returns
        -------
        phase_space : np.array(n_k_points, n_modes)
            phase_space for each k point and each mode
        """
        ps = self._ps_and_gamma[:, 0].reshape(self.n_k_points, self.n_modes)
        return ps


    @lazy_property(label='')
    def eigenvalues(self):
        """Calculates the eigenvalues of the dynamical matrix in Thz^2.

        Returns
        -------
        eigenvalues : np array
            (n_phonons) Eigenvalues of the dynamical matrix
        """
        eigenvalues = self._eigensystem[:, 0, :]
        return eigenvalues


    @property
    def eigenvectors(self):
        """Calculates the eigenvectors of the dynamical matrix.

        Returns
        -------
        eigenvectors : np array
            (n_phonons, n_phonons) Eigenvectors of the dynamical matrix
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
        """Calculates the angular frequencies from the diagonalized dynamical matrix.

        Returns
        -------
        frequency : np array
            (n_k_points, n_modes) frequency in rad
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
        is_amorphous = (self.kpts == (1, 1, 1)).all()
        return is_amorphous


    def _allowed_third_phonons_index(self, index_q, is_plus):
        q_vec = self._reciprocal_grid.id_to_unitary_grid_index(index_q)
        qp_vec = self._reciprocal_grid.unitary_grid(is_wrapping=False)
        qpp_vec = q_vec[np.newaxis, :] + (int(is_plus) * 2 - 1) * qp_vec[:, :]
        rescaled_qpp = np.round((qpp_vec * self._reciprocal_grid.grid_shape), 0).astype(np.int)
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


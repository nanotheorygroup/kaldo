"""
kaldo
Anharmonic Lattice Dynamics

"""
from kaldo.helpers.storage import is_calculated
from kaldo.helpers.storage import lazy_property
from kaldo.observables.physical_mode import PhysicalMode
from kaldo.helpers.storage import DEFAULT_STORE_FORMATS, FOLDER_NAME
from kaldo.grid import Grid
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import numpy as np
import ase.units as units

from kaldo.helpers.logger import get_logger
logging = get_logger()

KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J




class Phonons:
    def __init__(self, **kwargs):
        """The phonons object exposes all the phononic properties of a system,

        Parameters
        ----------
        forceconstants : ForceConstant
            contains all the information about the system and the derivatives of the potential.
        is_classic : bool
            specifies if the system is classic, `True` or quantum, `False`
        kpts (optional) : (3) tuple
            defines the number of k points to use to create the k mesh. Default is [1, 1, 1].
        temperature : float
            defines the temperature of the simulation. Units: K.
        min_frequency (optional) : float
            ignores all phonons with frequency below `min_frequency` THz, Default is None..
        max_frequency (optional) : float
            ignores all phonons with frequency above `max_frequency` THz, Default is None.
        third_bandwidth (optional) : float
            defines the width of the energy conservation smearing in the phonons scattering calculation.
            If `None` the width is calculated dynamically. Otherwise the input value corresponds to the
            width. Units: THz.
        broadening_shape (optional) : string
            defines the algorithm to use for the broadening of the conservation of the energy for third irder interactions
            . Available broadenings are `gauss`, `lorentz` and `triangle`. Default is `gauss`.
        is_tf_backend (optional) : bool
            defines if the third order phonons scattering calculations should be performed on tensorflow (True) or
            numpy (False). Default is True.
        folder (optional) : string
            specifies where to store the data files. Default is `output`.
        storage (optional) : 'formatted', 'numpy', 'memory', 'hdf5'
            defines the storing strategy used to store the observables. The `default` strategy stores formatted output
            and numpy arrays. `memory` storage doesn't generate any output.
        grid_type: 'F' or 'C
            specify if to use 'C" style atoms replica grid of fortran style 'F', default 'C'

        Returns
        -------
        Phonons
            An instance of the `Phonons` class.

        """
        self.forceconstants = kwargs.pop('forceconstants')
        if 'is_classic' in kwargs:
            self.is_classic = bool(kwargs['is_classic'])
        if 'temperature' in kwargs:
            self.temperature = float(kwargs['temperature'])
        self.folder = kwargs.pop('folder', FOLDER_NAME)
        self.kpts = kwargs.pop('kpts', (1, 1, 1))
        grid_type = kwargs.pop('grid_type', 'C')
        self._reciprocal_grid = Grid(self.kpts, order=grid_type)

        self.kpts = np.array(self.kpts)

        self.min_frequency = kwargs.pop('min_frequency', 0)
        self.max_frequency = kwargs.pop('max_frequency', None)
        self.broadening_shape = kwargs.pop('broadening_shape', 'gauss')
        self.is_tf_backend = kwargs.pop('is_tf_backend', True)
        self.is_nw = kwargs.pop('is_nw', False)
        self.third_bandwidth = kwargs.pop('third_bandwidth', None)
        self.storage = kwargs.pop('storage', 'formatted')
        self.is_symmetrizing_frequency = kwargs.pop('is_symmetrizing_frequency', False)
        self.is_antisymmetrizing_velocity = kwargs.pop('is_antisymmetrizing_velocity', False)
        self.atoms = self.forceconstants.atoms
        self.supercell = np.array(self.forceconstants.supercell)
        self.n_k_points = int(np.prod(self.kpts))
        self.n_atoms = self.forceconstants.n_atoms
        self.n_modes = self.forceconstants.n_modes
        self.n_phonons = self.n_k_points * self.n_modes
        self.is_able_to_calculate = True



    @lazy_property(label='')
    def physical_mode(self):
        """
        Calculate physical modes. Non physical modes are the first 3 modes of q=(0, 0, 0) and, if defined, all the
        modes outside the frequency range min_frequency and max_frequency.
        Returns
        -------
        physical_mode : np array
            (n_k_points, n_modes) bool
        """
        physical_mode = PhysicalMode(self.frequency, self.min_frequency, self.max_frequency).calculate()
        return physical_mode.reshape(self.n_k_points, self.n_modes)


    @lazy_property(label='')
    def frequency(self):
        """
        Calculate phonons frequency
        Returns
        -------
        frequency : np array
            (n_k_points, n_modes) frequency in THz
        """
        q_points = self._main_q_mesh
        frequency = np.zeros((self.n_k_points, self.n_modes))
        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point, self.forceconstants.second_order,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage)

            frequency[ik] = phonon.frequency

        return frequency


    @lazy_property(label='')
    def velocity(self):
        """Calculates the velocity using Hellmann-Feynman theorem.
        Returns
        -------
        velocity : np array
            (n_k_points, n_unit_cell * 3, 3) velocity in 100m/s or A/ps
        """

        q_points = self._main_q_mesh

        velocity = np.zeros((self.n_k_points, self.n_modes, 3))
        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point, self.forceconstants.second_order,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage)
            velocity[ik] = phonon.velocity
        return velocity


    @lazy_property(label='')
    def _dynmat_derivatives(self):
        q_points = self._main_q_mesh

        dynmat_derivatives = np.zeros((self.n_k_points, self.n_modes, self.n_modes, 3), dtype=np.complex)
        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point, self.forceconstants.second_order,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage)
            dynmat_derivatives[ik] = phonon._dynmat_derivatives
        return dynmat_derivatives


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
        q_points = self._main_q_mesh

        eigensystem = np.zeros((self.n_k_points, self.n_modes + 1, self.n_modes), dtype=np.complex)

        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point, self.forceconstants.second_order,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage)

            eigensystem[ik] = phonon._eigensystem

        return eigensystem


    @property
    def _velocity_af(self):
        q_points = self._main_q_mesh

        velocity_AF = np.zeros((self.n_k_points, self.n_modes, self.n_modes, self.n_modes), dtype=np.complex)
        for ik in range(len(q_points)):
            q_point = q_points[ik]
            phonon = HarmonicWithQ(q_point,self.forceconstants.second_order,
                                   distance_threshold=self.forceconstants.distance_threshold,
                                   folder=self.folder,
                                   storage=self.storage)

            velocity_AF[ik] = phonon._velocity_af

        return velocity_AF


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
        c_v = self.calculate_heat_capacity().reshape(self.n_k_points, self.n_modes)
        return c_v


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
        population =  self.calculate_population()
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
            ps_and_gamma = self.select_backend_for_phase_space_and_gamma(is_gamma_tensor_enabled=False)
        return ps_and_gamma


    @lazy_property(label='<temperature>/<statistics>/<third_bandwidth>')
    def _ps_gamma_and_gamma_tensor(self):
        ps_gamma_and_gamma_tensor = self.select_backend_for_phase_space_and_gamma(is_gamma_tensor_enabled=True)
        return ps_gamma_and_gamma_tensor

# Helpers properties

    @property
    def omega(self):
        return self.frequency * 2 * np.pi


    @property
    def _main_q_mesh(self):
        return self._reciprocal_grid.unitary_grid()

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
        qp_vec = self._reciprocal_grid.unitary_grid()
        qpp_vec = q_vec[np.newaxis, :] + (int(is_plus) * 2 - 1) * qp_vec[:, :]
        rescaled_qpp = np.round((qpp_vec * self._reciprocal_grid.grid_shape), 0).astype(np.int)
        rescaled_qpp = np.mod(rescaled_qpp, self._reciprocal_grid.grid_shape)
        index_qpp_full = np.ravel_multi_index(rescaled_qpp.T, self._reciprocal_grid.grid_shape, mode='raise')
        return index_qpp_full


    def select_backend_for_phase_space_and_gamma(self, is_gamma_tensor_enabled=True):
        logging.info('Projection started')
        if self.is_tf_backend:
            try:
                import kaldo.controllers.anharmonic_tf as aha
            except ImportError as err:
                logging.info(err)
                logging.warning('In order to run accelerated algoritgms, tensorflow>=2.0 is required. \
                Please consider installing tensorflow>=2.0. More info here: \
                https://www.tensorflow.org/install/pip')
                logging.info('Using numpy engine instead.')
                import kaldo.controllers.anharmonic as aha
        else:
            import kaldo.controllers.anharmonic as aha
        self.n_k_points = np.prod(self.kpts)
        self.n_phonons = self.n_k_points * self.n_modes
        self.is_gamma_tensor_enabled = is_gamma_tensor_enabled
        if self._is_amorphous:
            ps_and_gamma = aha.project_amorphous(self)
        else:
            ps_and_gamma = aha.project_crystal(self)
        return ps_and_gamma


    def calculate_population(self):
        frequency = self.frequency.reshape((self.n_k_points, self.n_modes))
        temp = self.temperature * KELVINTOTHZ
        density = np.zeros((self.n_k_points, self.n_modes))
        physical_mode = self.physical_mode.reshape((self.n_k_points, self.n_modes))
        if self.is_classic is False:
            density[physical_mode] = 1. / (np.exp(frequency[physical_mode] / temp) - 1.)
        else:
            density[physical_mode] = temp / frequency[physical_mode]
        return density


    def calculate_heat_capacity(self):
        frequency = self.frequency
        c_v = np.zeros_like(frequency)
        physical_mode = self.physical_mode
        temperature = self.temperature * KELVINTOTHZ
        if (self.is_classic):
            c_v[physical_mode] = KELVINTOJOULE
        else:
            f_be = self.population
            c_v[physical_mode] = KELVINTOJOULE * f_be[physical_mode] * (f_be[physical_mode] + 1) * self.frequency[
                physical_mode] ** 2 / \
                                 (temperature ** 2)
        return c_v

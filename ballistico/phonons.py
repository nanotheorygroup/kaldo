"""
Ballistico
Anharmonic Lattice Dynamics
"""
from ballistico.helpers.storage import is_calculated
from ballistico.helpers.storage import lazy_property
from ballistico.helpers.tools import q_vec_from_q_index
from ballistico.helpers.storage import DEFAULT_STORE_FORMATS
from ballistico.controllers.harmonic import calculate_physical_modes, calculate_frequency, calculate_velocity, \
    calculate_heat_capacity, calculate_occupations, calculate_dynmat_derivatives, calculate_eigensystem, \
    calculate_velocity_af, calculate_sij, calculate_sij_sparse, calculate_generalized_diffusivity
import numpy as np
import ase.units as units
from opt_einsum import contract

from ballistico.helpers.logger import get_logger
logging = get_logger()

KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J
THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
EVTOTENJOVERMOL = units.mol / (10 * units.J)

DELTA_DOS = 1
NUM_DOS = 100
FOLDER_NAME = 'ald-output'


class Phonons:
    def __init__(self, **kwargs):
        """The phonons object exposes all the phononic properties of a system,

        Parameters
        ----------
        finite_difference : FiniteDifference
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
        diffusivity_bandwidth (optional) : float
            Specifies the bandwidth to use in the calculation of the flux operator in the Allen-Feldman model of the
            thermal conductivity in amorphous systems. Units: rad/ps
        diffusivity_threshold (optional) : float
            This option is off by default. In such case the flux operator in the QHGK and AF models is calculated
        broadening_shape (optional) : string
            defines the algorithm to use to calculate the broadening. Available broadenings are `gauss`, `lorentz` and `triangle`.
            Default is `gauss`.
        is_tf_backend (optional) : bool
            defines if the third order phonons scattering calculations should be performed on tensorflow (True) or
            numpy (False). Default is True.
        folder (optional) : string
            specifies where to store the data files. Default is `output`.
        storage (optional) : 'formatted', 'numpy', 'memory', 'hdf5'
            defines the storing strategy used to store the observables. The `default` strategy stores formatted output
            and numpy arrays. `memory` storage doesn't generate any output.

        Returns
        -------
        Phonons
            An instance of the `Phonons` class.

        """
        self.finite_difference = kwargs.pop('finite_difference')
        if 'is_classic' in kwargs:
            self.is_classic = bool(kwargs['is_classic'])
        if 'temperature' in kwargs:
            self.temperature = float(kwargs['temperature'])
        self.folder = kwargs.pop('folder', FOLDER_NAME)
        self.kpts = kwargs.pop('kpts', (1, 1, 1))
        self.kpts = np.array(self.kpts)
        self.min_frequency = kwargs.pop('min_frequency', None)
        self.max_frequency = kwargs.pop('max_frequency', None)
        self.broadening_shape = kwargs.pop('broadening_shape', 'gauss')
        self.is_tf_backend = kwargs.pop('is_tf_backend', True)
        self.is_nw = kwargs.pop('is_nw', False)
        self.third_bandwidth = kwargs.pop('third_bandwidth', None)
        self.diffusivity_bandwidth = kwargs.pop('diffusivity_bandwidth', None)
        self.diffusivity_threshold = kwargs.pop('diffusivity_threshold', None)
        self.diffusivity_threshold = kwargs.pop('diffusivity_shape', 'lorentz')
        self.storage = kwargs.pop('storage', 'numpy')
        self.is_symmetrizing_frequency = kwargs.pop('is_symmetrizing_frequency', False)
        self.is_antisymmetrizing_velocity = kwargs.pop('is_antisymmetrizing_velocity', False)
        self.atoms = self.finite_difference.atoms
        self.supercell = np.array(self.finite_difference.supercell)
        self.n_k_points = int(np.prod(self.kpts))
        self.n_atoms = self.finite_difference.n_atoms
        self.n_modes = self.finite_difference.n_modes
        self.n_phonons = self.n_k_points * self.n_modes
        self.is_able_to_calculate = True

        self.store_format = {}
        for observable in DEFAULT_STORE_FORMATS:
            self.store_format[observable] = DEFAULT_STORE_FORMATS[observable] \
                if self.storage == 'formatted' else self.storage


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
        physical_mode = calculate_physical_modes(self)
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
        frequency = calculate_frequency(self)
        return frequency.reshape(self.n_k_points, self.n_modes)


    @lazy_property(label='')
    def velocity(self):
        """Calculates the velocity using Hellmann-Feynman theorem.
        Returns
        -------
        velocity : np array
            (n_k_points, n_unit_cell * 3, 3) velocity in 100m/s or A/ps
        """
        velocity = calculate_velocity(self)
        return velocity


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
        c_v = calculate_heat_capacity(self).reshape(self.n_k_points, self.n_modes)
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
        occupations =  calculate_occupations(self)
        return occupations


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
    def diffusivity(self):
        """Calculate the diffusivity, for each k point in k_points and each mode.

        Returns
        -------
        diffusivity : np.array(n_k_points, n_modes)
            diffusivity in mm^2/s
        """
        generalized_diffusivity = self._generalized_diffusivity
        diffusivity = 1 / 3 * 1 / 100 * contract('knaa->kn', generalized_diffusivity)
        return diffusivity


    @property
    def flux(self):
        """Calculate the flux, for each couple of k point in k_points/modes.

        Returns
        -------
        flux : np.array(n_k_points, n_modes, n_k_points, n_modes, 3)
        """
        if self.diffusivity_threshold is not None:
            sij = self.flux_sparse
        else:
            sij = self.flux_dense
        return sij


    @lazy_property(label='<diffusivity_bandwidth>')
    def flux_dense(self):
        """Calculate the flux, for each couple of k point in k_points/modes.

        Returns
        -------
        flux : np.array(n_k_points, n_modes, n_k_points, n_modes, 3)
        """
        sij = calculate_sij(self)
        return sij


    @lazy_property(label='<diffusivity_bandwidth>/<diffusivity_threshold>')
    def flux_sparse(self):
        """Calculate the flux, for each couple of k point in k_points/modes.

        Returns
        -------
        flux : np.array(n_k_points, n_modes, n_k_points, n_modes, 3)
        """
        sij = calculate_sij_sparse(self)
        return sij


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



    @lazy_property(label='')
    def _dynmat_derivatives(self):
        dynmat_derivatives = calculate_dynmat_derivatives(self)
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
        eigensystem = calculate_eigensystem(self)
        return eigensystem


    @lazy_property(label='<diffusivity_bandwidth>/<diffusivity_threshold>/<temperature>/<statistics>/<third_bandwidth>')
    def _ps_and_gamma(self):
        if is_calculated('_ps_gamma_and_gamma_tensor', self, '<temperature>/<statistics>/<third_bandwidth>', \
                         format=self.store_format['_ps_gamma_and_gamma_tensor']):
            ps_and_gamma = self._ps_gamma_and_gamma_tensor[:, :2]
        else:
            ps_and_gamma = self.calculate_phase_space_and_gamma(is_gamma_tensor_enabled=False)
        return ps_and_gamma


    @lazy_property(label='<diffusivity_bandwidth>/<diffusivity_threshold>/<temperature>/<statistics>/<third_bandwidth>')
    def _ps_gamma_and_gamma_tensor(self):
        ps_gamma_and_gamma_tensor = self.calculate_phase_space_and_gamma(is_gamma_tensor_enabled=True)
        return ps_gamma_and_gamma_tensor


    @lazy_property(label='<diffusivity_bandwidth>/<diffusivity_threshold>')
    def _generalized_diffusivity(self):
        generalized_diffusivity = calculate_generalized_diffusivity(self)
        return generalized_diffusivity

# Helpers properties

    @property
    def _chi_k(self):
        chi = np.zeros((self.n_k_points, self.finite_difference.n_replicas), dtype=np.complex)
        for index_q in range(self.n_k_points):
            k_point = q_vec_from_q_index(index_q, self.kpts)
            chi[index_q] = self.chi(k_point)
        return chi


    @property
    def _omegas(self):
        return self.frequency * 2 * np.pi


    @property
    def _main_q_mesh(self):
        q_mesh = q_vec_from_q_index(np.arange(self.n_k_points), self.kpts)
        return q_mesh


    @property
    def _velocity_af(self):
        velocity_AF = calculate_velocity_af(self)
        return velocity_AF


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


    def chi(self, qvec):
        dxij = self.finite_difference.list_of_replicas
        cell_inv = self.finite_difference.cell_inv
        chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
        return chi_k


    def calculate_phase_space_and_gamma(self, is_gamma_tensor_enabled=True):
        print('Projection started')
        if self.is_tf_backend:
            try:
                import ballistico.controllers.anharmonic_tf as aha
            except ImportError as err:
                logging.info(err)
                logging.warning('In order to run accelerated algoritgms, tensorflow>=2.0 is required. \
                Please consider installing tensorflow>=2.0. More info here: \
                https://www.tensorflow.org/install/pip')
                logging.info('Using numpy engine instead.')
                import ballistico.controllers.anharmonic as aha
        else:
            import ballistico.controllers.anharmonic as aha
        self.n_k_points = np.prod(self.kpts)
        self.n_phonons = self.n_k_points * self.n_modes
        self.is_gamma_tensor_enabled = is_gamma_tensor_enabled
        if self._is_amorphous:
            ps_and_gamma = aha.project_amorphous(self)
        else:
            ps_and_gamma = aha.project_crystal(self)
        return ps_and_gamma


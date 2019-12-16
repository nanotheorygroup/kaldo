"""
Ballistico
Anharmonic Lattice Dynamics
"""
from opt_einsum import contract
import numpy as np
import ballistico.controllers.harmonic as bha
import ballistico.controllers.anharmonic as ban
import ballistico.controllers.anharmonic_tf as bantf
import ballistico.controllers.statistic as bst
import tensorflow as tf
from ballistico.helpers.tools import is_calculated
from ballistico.helpers.tools import lazy_property
from ballistico.helpers.tools import apply_boundary_with_cell

FOLDER_NAME = 'ald-output'
FREQUENCY_THRESHOLD = 0.


class Phonons:
    def __init__(self, **kwargs):
        """The phonons object exposes all the phononic properties of a system,

        Parameters
        ----------
        finite_difference : FiniteDifference
            contains all the information about the system and the derivatives of the potential.
        is_classic : bool
            specifies if the system is classic, `True` or quantum, `False`
        temperature : float
            defines the temperature of the simulation.
        folder (optional) : string
            specifies where to store the data files. Default is `output`.
        kpts (optional) : (3) tuple
            defines the number of k points to use to create the k mesh. Default is [1, 1, 1].
        frequency_threshold (optional) : float
            ignores all phonons with frequency below `frequency_threshold` THz, Default is 0.001.
        sigma_in (optional) : float or None
            defines the width of the energy conservation smearing in the phonons scattering calculation.
            If `None` the width is calculated dynamically. Otherwise the input value corresponds to the
            width in THz. Default is None.
        broadening_shape (optional) : string
            defines the algorithm to use to calculate the broadening. Available broadenings are `gauss` and `triangle`.
            Default is `gauss`.
        is_tf_backend (optional) : bool
            defines if the third order phonons scattering calculations should be performed on tensorflow (True) or
            numpy (False). Default is True.
        Returns
        -------
        Phonons
            An instance of the `Phonons` class.

        """
        self.finite_difference = kwargs['finite_difference']
        self.is_classic = bool(kwargs['is_classic'])
        if 'temperature' in kwargs:
            self.temperature = float(kwargs['temperature'])
        if 'folder' in kwargs:
            self.folder = kwargs['folder']
        else:
            self.folder = FOLDER_NAME
        if 'kpts' in kwargs:
            self.kpts = np.array(kwargs['kpts'])
        else:
            self.kpts = np.array([1, 1, 1])
        if 'frequency_threshold' in kwargs:
            self.frequency_threshold = kwargs['frequency_threshold']
        else:
            self.frequency_threshold = FREQUENCY_THRESHOLD
        if 'sigma_in' in kwargs:
            self.sigma_in = kwargs['sigma_in']
        else:
            self.sigma_in = None
        if 'broadening_shape' in kwargs:
            self.broadening_shape = kwargs['broadening_shape']
        else:
            self.broadening_shape = 'gauss'
        if 'is_tf_backend' in kwargs:
            self.is_tf_backend = kwargs['is_tf_backend']
        else:
            self.is_tf_backend = False
        if 'is_nw' in kwargs:
            self.is_nw = kwargs['is_nw']
        else:
            self.is_nw = False
        self.atoms = self.finite_difference.atoms
        self.supercell = np.array(self.finite_difference.supercell)
        self.n_k_points = int(np.prod(self.kpts))
        self.n_modes = self.atoms.get_masses().shape[0] * 3
        self.n_phonons = self.n_k_points * self.n_modes
        self.is_able_to_calculate = True

        self.cell_inv = np.linalg.inv(self.atoms.cell)
        self.replicated_cell = self.finite_difference.replicated_atoms.cell
        self.replicated_cell_inv = np.linalg.inv(self.replicated_cell)



    @lazy_property(is_storing=True, is_reduced_path=True)
    def frequencies(self):
        frequencies = bha.calculate_second_order_observable(self, 'frequencies')
        return frequencies


    @lazy_property(is_storing=True, is_reduced_path=True)
    def velocities(self):
        velocities = bha.calculate_second_order_observable(self, 'velocities')
        return velocities


    @lazy_property(is_storing=True, is_reduced_path=True)
    def dos(self):
        dos = bha.calculate_density_of_states(self.frequencies, self.kpts)
        return dos


    @lazy_property(is_storing=True, is_reduced_path=False)
    def occupations(self):
        occupations =  bst.calculate_occupations(self)
        return occupations


    @lazy_property(is_storing=True, is_reduced_path=False)
    def heat_capacity(self):
        """Calculate the heat capacity for each k point in k_points and each mode.
        If classical, it returns the Boltzmann constant in W/m/K. If quantum it returns

        .. math::

            c_\\mu = k_B \\frac{\\nu_\\mu^2}{ \\tilde T^2} n_\\mu (n_\\mu + 1)

        where the frequency :math:`\\nu` and the temperature :math:`\\tilde T` are in THz.

        Returns
        -------
        c_v : np.array(n_k_points, n_modes)
            heat capacity in W/m/K for each k point and each mode
        """
        c_v = bst.calculate_c_v(self)
        return c_v


    @lazy_property(is_storing=False, is_reduced_path=False)
    def rescaled_eigenvectors(self):
        n_particles = self.atoms.positions.shape[0]
        n_modes = self.n_modes
        masses = self.atoms.get_masses()
        rescaled_eigenvectors = self.eigenvectors[:, :, :].reshape(
            (self.n_k_points, n_particles, 3, n_modes), order='C') / np.sqrt(
            masses[np.newaxis, :, np.newaxis, np.newaxis])
        rescaled_eigenvectors = rescaled_eigenvectors.reshape((self.n_k_points, n_modes, n_modes), order='C')
        return rescaled_eigenvectors


    @property
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


    @property
    def gamma(self):
        gamma = self._ps_and_gamma[:, 1]
        return gamma


    @property
    def ps(self):
        ps = self._ps_and_gamma[:, 0]
        return ps


#################
# Private methods
#################


    @lazy_property(is_storing=True, is_reduced_path=True)
    def _dynmat_derivatives(self):
        dynmat_derivatives = bha.calculate_second_order_observable(self, 'dynmat_derivatives')
        return dynmat_derivatives


    @lazy_property(is_storing=True, is_reduced_path=True)
    def _eigensystem(self):
        eigensystem = bha.calculate_second_order_observable(self, 'eigensystem', q_points=None)
        return eigensystem


    @lazy_property(is_storing=False, is_reduced_path=True)
    def _physical_modes(self):
        physical_modes = (self.frequencies.reshape(self.n_phonons) > self.frequency_threshold)
        if self.is_nw:
            physical_modes[:4] = False
        else:
            physical_modes[:3] = False
        return physical_modes


    @lazy_property(is_storing=False, is_reduced_path=True)
    def _chi_k(self):
        chi = np.zeros((self.n_k_points, self.finite_difference.n_replicas), dtype=np.complex)
        for index_q in range(self.n_k_points):
            k_point = self._q_vec_from_q_index(index_q)
            chi[index_q] = self._chi(k_point)
        return chi


    @lazy_property(is_storing=False, is_reduced_path=True)
    def _omegas(self):
        return self.frequencies * 2 * np.pi


    @lazy_property(is_storing=True, is_reduced_path=True)
    def _velocities_af(self):
        velocities_AF = bha.calculate_second_order_observable(self, 'velocities_AF')
        return velocities_AF


    @lazy_property(is_storing=True, is_reduced_path=False)
    def _ps_and_gamma(self):
        if is_calculated('ps_gamma_and_gamma_tensor', self):
            ps_and_gamma = self._ps_gamma_and_gamma_tensor[:, :2]
        else:
            ps_and_gamma = self._calculate_ps_and_gamma(is_gamma_tensor_enabled=False)
        return ps_and_gamma


    @lazy_property(is_storing=True, is_reduced_path=False)
    def _ps_gamma_and_gamma_tensor(self):
        ps_gamma_and_gamma_tensor = self._calculate_ps_and_gamma(is_gamma_tensor_enabled=True)
        return ps_gamma_and_gamma_tensor


    @lazy_property(is_storing=False, is_reduced_path=False)
    def _scattering_matrix_without_diagonal(self):
        frequencies = self._keep_only_physical(self.frequencies.reshape((self.n_phonons), order='C'))
        gamma_tensor = self._keep_only_physical(self._ps_gamma_and_gamma_tensor[:, 2:])
        scattering_matrix_without_diagonal = contract('a,ab,b->ab', 1 / frequencies, gamma_tensor, frequencies)
        return scattering_matrix_without_diagonal


    @lazy_property(is_storing=False, is_reduced_path=False)
    def _scattering_matrix(self):
        scattering_matrix = -1 * self._scattering_matrix_without_diagonal
        gamma = self._keep_only_physical(self.gamma.reshape((self.n_phonons), order='C'))
        scattering_matrix = scattering_matrix + np.diag(gamma)
        return scattering_matrix


    @lazy_property(is_storing=False, is_reduced_path=False)
    def _frequencies_tf(self):
        frequencies_tf = tf.convert_to_tensor(self.frequencies.astype(float))
        return frequencies_tf


    @lazy_property(is_storing=False, is_reduced_path=False)
    def _omega_tf(self):
        omega_tf = tf.convert_to_tensor(self._omegas.astype(float))
        return omega_tf


    @lazy_property(is_storing=False, is_reduced_path=False)
    def _occupations_tf(self):
        density_tf = tf.convert_to_tensor(self.occupations.astype(float))
        return density_tf


    @property
    def _is_amorphous(self):
        is_amorphous = (self.kpts == (1, 1, 1)).all()
        return is_amorphous

    def _q_index_from_q_vec(self, q_vec):
        # the input q_vec is in the unit sphere
        rescaled_qpp = np.round((q_vec * self.kpts).T, 0).astype(np.int)
        q_index = np.ravel_multi_index(rescaled_qpp, self.kpts, mode='wrap')
        return q_index

    def _q_vec_from_q_index(self, q_index):
        # the output q_vec is in the unit sphere
        q_vec = np.array(np.unravel_index(q_index, (self.kpts))).T / self.kpts
        apply_boundary_with_cell(q_vec)
        return q_vec

    def allowed_index_qpp(self, index_q, is_plus):
        index_qp_full = np.arange(self.n_k_points)
        q_vec = self._q_vec_from_q_index(index_q)
        qp_vec = self._q_vec_from_q_index(index_qp_full)
        qpp_vec = q_vec[np.newaxis, :] + (int(is_plus) * 2 - 1) * qp_vec[:, :]
        index_qpp_full = self._q_index_from_q_vec(qpp_vec)
        return index_qpp_full

    def _keep_only_physical(self, operator):
        physical_modes = self._physical_modes
        if operator.shape == (self.n_phonons, self.n_phonons):
            index = np.outer(physical_modes, physical_modes)
            return operator[index].reshape((physical_modes.sum(), physical_modes.sum()), order='C')
        else:
            return operator[physical_modes, ...]


    def _chi(self, qvec):
        dxij = self.finite_difference.list_of_replicas
        cell_inv = self.cell_inv
        chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
        return chi_k


    def _calculate_ps_and_gamma(self, is_gamma_tensor_enabled=True):
        print('Projection started')
        if self.is_tf_backend:
            print('Backend, tensorflow')
            controller = bantf
        else:
            print('Backend, numpy')
            controller = ban
        if self._is_amorphous:
            ps_and_gamma = controller.project_amorphous(self, is_gamma_tensor_enabled)
        else:
            ps_and_gamma = controller.project_crystal(self, is_gamma_tensor_enabled)
        return ps_and_gamma


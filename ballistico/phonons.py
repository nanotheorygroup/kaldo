from opt_einsum import contract
import numpy as np
import ballistico.harmonic as bha
import ballistico.anharmonic as ban
import ballistico.statistic as bst
import tensorflow as tf
from .tools import timeit, lazy_property, is_calculated

from .tools import lazy_property

FOLDER_NAME = 'output'
FREQUENCY_THRESHOLD = 0.001


class Phonons:
    def __init__(self, **kwargs):
        self.finite_difference = kwargs['finite_difference']
        self.is_classic = bool(kwargs['is_classic'])
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
        if 'conserve_momentum' in kwargs:
            self.conserve_momentum = kwargs['conserve_momentum']
        else:
            self.conserve_momentum = False
        if 'sigma_in' in kwargs:
            self.sigma_in = kwargs['sigma_in']
        else:
            self.sigma_in = None
        if 'broadening_shape' in kwargs:
            self.broadening_shape = kwargs['broadening_shape']
        else:
            self.broadening_shape = 'gauss'

        self.atoms = self.finite_difference.atoms
        self.supercell = np.array(self.finite_difference.supercell)
        self.n_k_points = int(np.prod(self.kpts))
        self.n_modes = self.atoms.get_masses().shape[0] * 3
        self.n_phonons = self.n_k_points * self.n_modes
        self.is_able_to_calculate = True

        self.cell_inv = np.linalg.inv(self.atoms.cell)
        self.replicated_cell = self.finite_difference.replicated_atoms.cell
        self.replicated_cell_inv = np.linalg.inv(self.replicated_cell)
        self.replicated_cell = self.finite_difference.replicated_atoms.cell
        self.list_of_replicas = self.finite_difference.list_of_replicas()
        if self.list_of_replicas.shape == (3,):
            self.n_replicas = 1
        else:
            self.n_replicas = self.list_of_replicas.shape[0]


    @lazy_property(is_storing=False, is_reduced_path=True)
    def k_points(self):
        k_points = bha.calculate_k_points(self)
        return k_points


    @lazy_property(is_storing=True, is_reduced_path=True)
    def dynmat(self):
        dynmat = bha.calculate_dynamical_matrix(self)
        return dynmat


    @lazy_property(is_storing=True, is_reduced_path=True)
    def frequencies(self):
        frequencies = bha.calculate_second_order_observable(self, 'frequencies')
        return frequencies


    @lazy_property(is_storing=True, is_reduced_path=True)
    def eigensystem(self):
        eigensystem = bha.calculate_eigensystem(self)
        return eigensystem


    @lazy_property(is_storing=True, is_reduced_path=True)
    def velocities(self):
        velocities = bha.calculate_second_order_observable(self, 'velocities')
        return velocities


    @lazy_property(is_storing=True, is_reduced_path=True)
    def velocities_AF(self):
        velocities_AF = bha.calculate_second_order_observable(self, 'velocities_AF')
        return velocities_AF


    @lazy_property(is_storing=True, is_reduced_path=True)
    def dos(self):
        dos = bha.calculate_density_of_states(self.frequencies, self.kpts)
        return dos


    @lazy_property(is_storing=True, is_reduced_path=False)
    def occupations(self):
        occupations =  bst.calculate_occupations(self)
        return occupations


    @lazy_property(is_storing=True, is_reduced_path=False)
    def c_v(self):
        c_v =  bst.calculate_c_v(self)
        return c_v


    @property
    def eigenvalues(self):
        eigenvalues = self.eigensystem[:, :, -1]
        return eigenvalues


    @property
    def eigenvectors(self):
        eigenvectors = self.eigensystem[:, :, :-1]
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


    @lazy_property(is_storing=False, is_reduced_path=True)
    def _physical_modes(self):
        physical_modes = (self.frequencies.reshape(self.n_phonons) > self.frequency_threshold)
        return physical_modes


    @lazy_property(is_storing=False, is_reduced_path=True)
    def _chi_k(self):
        chi = np.zeros((self.n_k_points, self.n_replicas), dtype=np.complex)
        for index_k in range(self.n_k_points):
            k_point = self.k_points[index_k]
            chi[index_k] = self._chi(k_point)
        return chi


    @lazy_property(is_storing=False, is_reduced_path=True)
    def _omegas(self):
        return self.frequencies * 2 * np.pi


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


    @property
    def _is_amorphous(self):
        is_amorphous = (self.kpts == (1, 1, 1)).all()
        return is_amorphous


    def _keep_only_physical(self, operator):
        physical_modes = self._physical_modes
        if operator.shape == (self.n_phonons, self.n_phonons):
            index = np.outer(physical_modes, physical_modes)
            return operator[index].reshape((physical_modes.sum(), physical_modes.sum()), order='C')
        else:
            return operator[physical_modes, ...]


    def _apply_boundary_with_cell(self, dxij):
        # exploit periodicity to calculate the shortest distance, which may not be the one we have
        sxij = dxij.dot(self.replicated_cell_inv)
        sxij = sxij - np.round(sxij)
        dxij = sxij.dot(self.replicated_cell)
        return dxij


    def _chi(self, qvec):
        dxij = self.list_of_replicas
        cell_inv = self.cell_inv
        chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
        return chi_k


    def _calculate_ps_and_gamma(self, is_gamma_tensor_enabled=True):
        print('Projection started')
        if self._is_amorphous:
            ps_and_gamma = ban.project_amorphous(self, is_gamma_tensor_enabled)
        else:
            ps_and_gamma = ban.project_crystal(self, is_gamma_tensor_enabled)
        return ps_and_gamma


    @lazy_property(is_storing=False, is_reduced_path=False)
    def frequencies_tf(self):
        frequencies_tf = tf.convert_to_tensor(self.frequencies.astype(float))
        return frequencies_tf

    @lazy_property(is_storing=False, is_reduced_path=False)
    def omega_tf(self):
        omega_tf = tf.convert_to_tensor(self._omegas.astype(float))
        return omega_tf

    @lazy_property(is_storing=False, is_reduced_path=False)
    def density_tf(self):
        density_tf = tf.convert_to_tensor(self.occupations.astype(float))
        return density_tf
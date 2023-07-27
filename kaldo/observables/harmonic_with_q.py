from kaldo.grid import wrap_coordinates
from kaldo.observables.forceconstant import chi
from kaldo.observables.observable import Observable
import numpy as np
from opt_einsum import contract
from kaldo.helpers.storage import lazy_property
import tensorflow as tf
from scipy.linalg.lapack import zheev
from kaldo.helpers.logger import get_logger, log_size

logging = get_logger()

MIN_N_MODES_TO_STORE = 1000


class HarmonicWithQ(Observable):

    def __init__(self, q_point, second,
                 distance_threshold=None,
                 storage='numpy',
                 is_nw=False,
                 is_unfolding=False,
                 is_amorphous=False,
                 *kargs,
                 **kwargs):
        super().__init__(*kargs, **kwargs)
        self.q_point = q_point
        self.atoms = second.atoms
        self.n_modes = self.atoms.positions.shape[0] * 3
        self.supercell = second.supercell
        self.second = second
        self.distance_threshold = distance_threshold
        self.physical_mode = np.ones((1, self.n_modes), dtype=bool)
        self.is_nw = is_nw
        self.is_unfolding = is_unfolding
        self.is_amorphous = is_amorphous
        if (q_point == [0, 0, 0]).all():
            if self.is_nw:
                self.physical_mode[0, :4] = False
            else:
                self.physical_mode[0, :3] = False
        if self.n_modes > MIN_N_MODES_TO_STORE:
            self.storage = storage
        else:
            self.storage = 'memory'

    @lazy_property(label='<q_point>')
    def frequency(self):
        frequency = self.calculate_frequency()[np.newaxis, :]
        return frequency

    @lazy_property(label='<q_point>')
    def velocity(self):
        velocity = self.calculate_velocity()
        return velocity

    @lazy_property(label='<q_point>')
    def participation_ratio(self):
        participation_ratio = self.calculate_participation_ratio()
        return participation_ratio

    @lazy_property(label='<q_point>')
    def _dynmat_derivatives_x(self):
        if self.is_unfolding:
            _dynmat_derivatives = self.calculate_dynmat_derivatives_unfolded(direction=0)
        else:
            _dynmat_derivatives = self.calculate_dynmat_derivatives(direction=0)
        return _dynmat_derivatives

    @lazy_property(label='<q_point>')
    def _dynmat_derivatives_y(self):
        if self.is_unfolding:
            _dynmat_derivatives = self.calculate_dynmat_derivatives_unfolded(direction=1)
        else:
            _dynmat_derivatives = self.calculate_dynmat_derivatives(direction=1)
        return _dynmat_derivatives

    @lazy_property(label='<q_point>')
    def _dynmat_derivatives_z(self):
        if self.is_unfolding:
            _dynmat_derivatives = self.calculate_dynmat_derivatives_unfolded(direction=2)
        else:
            _dynmat_derivatives = self.calculate_dynmat_derivatives(direction=2)
        return _dynmat_derivatives

    @lazy_property(label='<q_point>')
    def _dynmat_fourier(self):
        dynmat_fourier = self.calculate_dynmat_fourier()
        return dynmat_fourier

    @lazy_property(label='<q_point>')
    def _eigensystem(self):
        if self.is_unfolding:
            _eigensystem = self.calculate_eigensystem_unfolded(only_eigenvals=False)
        else:
            _eigensystem = self.calculate_eigensystem(only_eigenvals=False)
        return _eigensystem

    @lazy_property(label='<q_point>')
    def _sij_x(self):
        _sij = self.calculate_sij(direction=0)
        return _sij

    @lazy_property(label='<q_point>')
    def _sij_y(self):
        _sij = self.calculate_sij(direction=1)
        return _sij

    @lazy_property(label='<q_point>')
    def _sij_z(self):
        _sij = self.calculate_sij(direction=2)
        return _sij

    def calculate_frequency(self):
        # TODO: replace calculate_eigensystem() with eigensystem
        if self.is_unfolding:
            eigenvals = self.calculate_eigensystem_unfolded(only_eigenvals=True)
        else:
            eigenvals = self.calculate_eigensystem(only_eigenvals=True)
        frequency = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
        return frequency.real

    def calculate_dynmat_derivatives(self, direction):
        q_point = self.q_point
        distance_threshold = self.distance_threshold
        atoms = self.atoms
        list_of_replicas = self.second.list_of_replicas
        replicated_cell = self.second.replicated_atoms.cell
        replicated_cell_inv = self.second._replicated_cell_inv
        cell_inv = self.second.cell_inv
        dynmat = self.second.dynmat
        positions = self.atoms.positions
        n_unit_cell = atoms.positions.shape[0]
        n_modes = n_unit_cell * 3
        n_replicas = np.prod(self.supercell)
        shape = (1, n_unit_cell * 3, n_unit_cell * 3)
        dir = ['_x', '_y', '_z']
        type = complex if (not self.is_amorphous) else float
        log_size(shape, type, name='dynamical_matrix_derivative_' + dir[direction])
        if self.is_amorphous:
            distance = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distance = wrap_coordinates(distance, replicated_cell, replicated_cell_inv)
            dynmat_derivatives = contract('ij,ibjc->ibjc',
                                          tf.convert_to_tensor(distance[..., direction]),
                                          dynmat[0, :, :, 0, :, :],
                                          backend='tensorflow')
        else:
            distance = positions[:, np.newaxis, np.newaxis, :] - (
                    positions[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])

            if distance_threshold is not None:

                distance_to_wrap = positions[:, np.newaxis, np.newaxis, :] - (
                    self.second.replicated_atoms.positions.reshape(n_replicas, n_unit_cell, 3)[
                    np.newaxis, :, :, :])

                shape = (n_unit_cell, 3, n_unit_cell, 3)
                type = complex
                dynmat_derivatives = np.zeros(shape, dtype=type)
                for l in range(n_replicas):
                    wrapped_distance = wrap_coordinates(distance_to_wrap[:, l, :, :], replicated_cell,
                                                        replicated_cell_inv)
                    mask = (np.linalg.norm(wrapped_distance, axis=-1) < distance_threshold)
                    id_i, id_j = np.argwhere(mask).T
                    dynmat_derivatives[id_i, :, id_j, :] += contract('f,fbc->fbc', distance[id_i, l, id_j, direction], \
                                                                     dynmat.numpy()[0, id_i, :, 0, id_j, :] *
                                                                     chi(q_point, list_of_replicas, cell_inv)[l])
            else:
                dynmat_derivatives = contract('ilj,ibljc,l->ibjc',
                                              tf.convert_to_tensor(distance.astype(complex)[..., direction]),
                                              tf.cast(dynmat[0], tf.complex128),
                                              tf.convert_to_tensor(
                                                  chi(q_point, list_of_replicas, cell_inv).flatten().astype(
                                                      complex)),
                                              backend='tensorflow')
        dynmat_derivatives = tf.reshape(dynmat_derivatives, (n_modes, n_modes))
        return dynmat_derivatives

    def calculate_sij(self, direction):
        q_point = self.q_point
        shape = (3 * self.atoms.positions.shape[0], 3 * self.atoms.positions.shape[0])
        if self.is_amorphous and (self.q_point == np.array([0, 0, 0])).all():
            type = float
        else:
            type = complex
        eigenvects = self._eigensystem[1:, :]
        if direction == 0:
            dynmat_derivatives = self._dynmat_derivatives_x
        if direction == 1:
            dynmat_derivatives = self._dynmat_derivatives_y
        if direction == 2:
            dynmat_derivatives = self._dynmat_derivatives_z
        if self.atoms.positions.shape[0] > 500:
            # We want to print only for big systems
            logging.info('Flux operators for q = ' + str(q_point) + ', direction = ' + str(direction))
            dir = ['_x', '_y', '_z']
            log_size(shape, type, name='sij' + dir[direction])
        if self.is_amorphous and (self.q_point == np.array([0, 0, 0])).all():
            sij = tf.tensordot(eigenvects, dynmat_derivatives, (0, 1))
            sij = tf.tensordot(eigenvects, sij, (0, 1))
        else:
            eigenvects = tf.cast(eigenvects, tf.complex128)
            dynmat_derivatives = tf.cast(dynmat_derivatives, tf.complex128)
            sij = tf.tensordot(eigenvects, dynmat_derivatives, (0, 1))
            sij = tf.tensordot(tf.math.conj(eigenvects), sij, (0, 1))
        return sij

    def calculate_velocity(self):
        frequency = self.frequency[0]
        velocity = np.zeros((self.n_modes, 3))
        inverse_sqrt_freq = tf.cast(tf.convert_to_tensor(1 / np.sqrt(frequency)), tf.complex128)
        if self.is_amorphous:
            inverse_sqrt_freq = tf.cast(inverse_sqrt_freq, tf.float64)
        for alpha in range(3):
            if alpha == 0:
                sij = self._sij_x
            if alpha == 1:
                sij = self._sij_y
            if alpha == 2:
                sij = self._sij_z
            velocity_AF = 1 / (2 * np.pi) * contract('mn,m,n->mn', sij,
                                                     inverse_sqrt_freq, inverse_sqrt_freq, backend='tensorflow') / 2
            velocity_AF = tf.where(tf.math.is_nan(tf.math.real(velocity_AF)), 0., velocity_AF)
            velocity[..., alpha] = contract('mm->m', velocity_AF.numpy().imag)
        return velocity[np.newaxis, ...]

    def calculate_dynmat_fourier(self):
        q_point = self.q_point
        distance_threshold = self.distance_threshold
        atoms = self.atoms
        n_unit_cell = atoms.positions.shape[0]
        n_replicas = np.prod(self.supercell)
        dynmat = self.second.dynmat
        cell_inv = self.second.cell_inv
        replicated_cell_inv = self.second._replicated_cell_inv
        is_at_gamma = (q_point == (0, 0, 0)).all()
        list_of_replicas = self.second.list_of_replicas
        log_size((self.n_modes, self.n_modes), complex, name='dynmat_fourier')
        if distance_threshold is not None:
            shape = (n_unit_cell, 3, n_unit_cell, 3)
            type = complex
            dyn_s = np.zeros(shape, dtype=type)
            replicated_cell = self.second.replicated_atoms.cell

            for l in range(n_replicas):
                distance_to_wrap = atoms.positions[:, np.newaxis, :] - (
                    self.second.replicated_atoms.positions.reshape(n_replicas, n_unit_cell, 3)[np.newaxis, l, :, :])

                distance_to_wrap = wrap_coordinates(distance_to_wrap, replicated_cell, replicated_cell_inv)

                mask = np.linalg.norm(distance_to_wrap, axis=-1) < distance_threshold
                id_i, id_j = np.argwhere(mask).T
                dyn_s[id_i, :, id_j, :] += dynmat.numpy()[0, id_i, :, 0, id_j, :] * \
                                           chi(q_point, list_of_replicas, cell_inv)[l]
        else:
            if is_at_gamma:
                if self.is_amorphous:
                    dyn_s = dynmat[0]
                else:
                    dyn_s = contract('ialjb->iajb', dynmat[0], backend='tensorflow')
            else:
                dyn_s = contract('ialjb,l->iajb',
                                 tf.cast(dynmat[0], tf.complex128),
                                 tf.convert_to_tensor(chi(q_point, list_of_replicas, cell_inv).flatten()),
                                 backend='tensorflow')
        dyn_s = tf.reshape(dyn_s, (self.n_modes, self.n_modes))
        return dyn_s

    def calculate_eigensystem(self, only_eigenvals):
        dyn_s = self._dynmat_fourier
        if only_eigenvals:
            esystem = tf.linalg.eigvalsh(dyn_s)
        else:
            log_size(self._dynmat_fourier.shape, type=complex, name='eigensystem')
            esystem = tf.linalg.eigh(dyn_s)
            esystem = tf.concat(axis=0, values=(esystem[0][tf.newaxis, :], esystem[1]))
        return esystem

    def calculate_participation_ratio(self):
        n_atoms = self.n_modes // 3
        esystem = self._eigensystem[1:, :]
        eigenvects = tf.transpose(esystem)
        eigenvects = np.reshape(eigenvects, (self.n_modes, n_atoms, 3))
        participation_ratio = np.power(np.linalg.norm(eigenvects, axis=2), 4)
        participation_ratio = np.reciprocal(np.sum(participation_ratio, axis=1) * n_atoms)
        return participation_ratio

    def calculate_eigensystem_unfolded(self, only_eigenvals=False):
        # This algorithm should be the same as the ShengBTE version
        q_point = self.q_point
        supercell = self.supercell
        atoms = self.atoms
        cell = atoms.cell
        n_unit_cell = atoms.positions.shape[0]
        fc_s = self.second.dynmat.numpy()
        fc_s = fc_s.reshape((n_unit_cell, 3, supercell[0], supercell[1], supercell[2], n_unit_cell, 3))
        supercell_positions = self.second.supercell_positions
        supercell_norms = 1 / 2 * np.linalg.norm(supercell_positions, axis=1) ** 2
        dyn_s = np.zeros((n_unit_cell, 3, n_unit_cell, 3), dtype=complex)
        supercell_replicas = self.second.supercell_replicas
        for ind in range(supercell_replicas.shape[0]):
            supercell_replica = supercell_replicas[ind]
            replica_position = np.tensordot(supercell_replica, cell, (-1, 0))
            distance = replica_position[None, None, :] + (atoms.positions[:, None, :] - atoms.positions[None, :, :])
            projection = (contract('la,ija->ijl', supercell_positions, distance) - supercell_norms[None, None, :])
            mask = (projection <= 1e-6).all(axis=-1)
            neq = (np.abs(projection) <= 1e-6).sum(axis=-1)
            weight = 1.0 / neq
            coefficient = weight * mask
            if coefficient.any():
                supercell_index = supercell_replica%supercell
                qr = 2. * np.pi * np.dot(q_point[:], supercell_replica[:])
                dyn_s[:, :, :, :] += np.exp(-1j * qr) * contract('jbia,ij->iajb',
                                                                 fc_s[:, :, supercell_index[0], supercell_index[1],
                                                                 supercell_index[2], :, :], coefficient)
        dyn = dyn_s[...].reshape((n_unit_cell * 3, n_unit_cell * 3))

        # omega2 = 2 * pi * frequency^2
        # todo: performance check with zheev vs tf.linalg.eigh(dyn) (+ eigvalsh(dyn)
        if only_eigenvals:
            omega2, __, info = zheev(dyn, compute_v=0)
            return omega2
        else:
            omega2, eigenvect, info = zheev(dyn, compute_v=1)
            esystem = np.vstack((omega2, eigenvect))
            return esystem

    def calculate_dynmat_derivatives_unfolded(self, direction=None):
        # This algorithm should be the same as the ShengBTE version
        q_point = self.q_point
        supercell = self.supercell
        atoms = self.atoms
        cell = atoms.cell
        n_unit_cell = atoms.positions.shape[0]
        ddyn_s = np.zeros((n_unit_cell, 3, n_unit_cell, 3), dtype=complex)
        fc_s = self.second.dynmat.numpy()
        fc_s = fc_s.reshape((n_unit_cell, 3, supercell[0], supercell[1], supercell[2], n_unit_cell, 3))
        supercell_positions = self.second.supercell_positions
        supercell_norms = 1 / 2 * np.linalg.norm(supercell_positions, axis=1) ** 2
        supercell_replicas = self.second.supercell_replicas
        for ind in range(supercell_replicas.shape[0]):
            supercell_replica = supercell_replicas[ind]
            replica_position = np.tensordot(supercell_replica, cell, (-1, 0))
            distance = replica_position[None, None, :] + (atoms.positions[:, None, :] - atoms.positions[None, :, :])
            projection = (contract('la,ija->ijl', supercell_positions, distance) - supercell_norms[None, None, :])
            mask = (projection <= 1e-6).all(axis=-1)
            neq = (np.abs(projection) <= 1e-6).sum(axis=-1)
            weight = 1.0 / neq
            coefficient = weight * mask
            if coefficient.any():
                supercell_index = supercell_replica % supercell
                qr = 2. * np.pi * np.dot(q_point, supercell_replica)
                ddyn_s[:, :, :, :] += replica_position[direction] * np.exp(-1j * qr) *\
                                      contract('jbia,ij->iajb', fc_s[:, :, supercell_index[0],
                                          supercell_index[1], supercell_index[2], :, :], coefficient)
        return ddyn_s.reshape((n_unit_cell * 3, n_unit_cell * 3))

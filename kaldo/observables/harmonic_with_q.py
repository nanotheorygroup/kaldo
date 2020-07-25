
from kaldo.helpers.tools import timeit
from kaldo.grid import wrap_coordinates
from scipy.linalg.lapack import dsyev
from kaldo.observables.forceconstant import chi
from kaldo.observables.observable import Observable
import numpy as np
from opt_einsum import contract
import ase.units as units
from kaldo.helpers.storage import lazy_property
import tensorflow as tf
from kaldo.helpers.logger import get_logger, log_size
logging = get_logger()

class HarmonicWithQ(Observable):

    def __init__(self, q_point, second,
                 distance_threshold=None, storage='numpy', *kargs, **kwargs):

        super().__init__(*kargs, **kwargs)
        self.q_point = q_point
        self.atoms = second.atoms
        self.n_modes = self.atoms.positions.shape[0] * 3
        self.supercell = second.supercell
        self.is_amorphous = (self.supercell == (1, 1, 1))
        self.replicated_atoms = second.replicated_atoms
        self.list_of_replicas = second.list_of_replicas
        self.second = second.value
        self.distance_threshold = distance_threshold
        self.storage = storage


    @lazy_property(label='<q_point>')
    def frequency(self):
        frequency = self.calculate_frequency()
        return frequency


    @lazy_property(label='<q_point>')
    def velocity(self):
        velocity = self.calculate_velocity()
        return velocity


    @lazy_property(label='<q_point>')
    def _dynmat_derivatives(self):
        _dynmat_derivatives = self.calculate_dynmat_derivatives()
        return _dynmat_derivatives


    @lazy_property(label='<q_point>')
    def _eigensystem(self):
        _eigensystem = self.calculate_eigensystem()
        return _eigensystem


    @lazy_property(label='<q_point>')
    def _velocity_af(self):
        _velocity_af = self.calculate_velocity_af()
        return _velocity_af


    def calculate_frequency(self):
        #TODO: replace calculate_eigensystem() with eigensystem
        eigenvals = self.calculate_eigensystem(only_eigenvals=True)
        frequency = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
        return frequency.real


    def calculate_dynmat_derivatives(self):
        q_point = self.q_point
        is_amorphous = self.is_amorphous
        distance_threshold = self.distance_threshold
        atoms = self.atoms
        list_of_replicas = self.list_of_replicas
        replicated_cell = self.replicated_atoms.cell
        replicated_cell_inv = np.linalg.inv(self.replicated_atoms.cell)
        dynmat = self.calculate_dynmat()
        positions = self.atoms.positions
        n_unit_cell = atoms.positions.shape[0]
        n_modes = n_unit_cell * 3
        n_replicas = np.prod(self.supercell)

        if distance_threshold is not None:
            logging.info('Using folded flux operators')
        cell_inv = np.linalg.inv(self.atoms.cell)

        shape = (1, n_unit_cell * 3, n_unit_cell * 3, 3)
        if is_amorphous:
            type = np.float
        else:
            type = np.complex
        log_size(shape, type, name='dynamical_matrix_derivative')
        ddyn = np.zeros(shape).astype(type)
        if is_amorphous:
            distance = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distance = wrap_coordinates(distance, replicated_cell, replicated_cell_inv)
            dynmat_derivatives = contract('ija,ibjc->ibjca',
                                          tf.convert_to_tensor(distance),
                                          dynmat[0, :, :, 0, :, :],
                                          backend='tensorflow')
        else:
            distance = positions[:, np.newaxis, np.newaxis, :] - (
                    positions[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])

            distance_to_wrap = positions[:, np.newaxis, np.newaxis, :] - (
                self.replicated_atoms.positions.reshape(n_replicas, n_unit_cell, 3)[
                np.newaxis, :, :, :])

            list_of_replicas = self.list_of_replicas

            if distance_threshold is not None:
                shape = (n_unit_cell, 3, n_unit_cell, 3, 3)
                type = np.complex
                log_size(shape, type, name='dynmat_derivatives')
                dynmat_derivatives = np.zeros(shape, dtype=type)
                for l in range(n_replicas):
                    wrapped_distance = wrap_coordinates(distance_to_wrap[:, l, :, :], replicated_cell,
                                                        replicated_cell_inv)
                    mask = (np.linalg.norm(wrapped_distance, axis=-1) < distance_threshold)
                    id_i, id_j = np.argwhere(mask).T
                    dynmat_derivatives[id_i, :, id_j, :, :] += contract('fa,fbc->fbca', distance[id_i, l, id_j, :], \
                                                                         dynmat[0, id_i, :, 0, id_j, :] *
                                                                         chi(q_point, list_of_replicas, cell_inv)[l], backend='tensorflow')
            else:

                dynmat_derivatives = contract('ilja,ibljc,l->ibjca',
                                              tf.convert_to_tensor(distance.astype(np.complex)),
                                              tf.cast(dynmat[0], tf.complex128),
                                              tf.convert_to_tensor(chi(q_point, list_of_replicas, cell_inv).flatten().astype(np.complex)),
                                              backend='tensorflow')
        ddyn[0] = tf.reshape(dynmat_derivatives, (n_modes, n_modes, 3))
        return ddyn


    def calculate_sij(self):
        q_point = self.q_point
        is_amorphous = self.is_amorphous
        shape = (1, 3 * self.atoms.positions.shape[0], 3 * self.atoms.positions.shape[0], 3)
        if is_amorphous:
            type = np.float
        else:
            type = np.complex
        log_size(shape, type, name='sij')
        sij = np.zeros(shape, dtype=type)
        if self.atoms.positions.shape[0] > 100:
            # We want to print only for big systems
            logging.info('Flux operators for q = ' + str(q_point))
        dynmat_derivatives = self._dynmat_derivatives

        eigenvects = self._eigensystem[:, 1:, :]
        sij_single = tf.tensordot(eigenvects[0], dynmat_derivatives[0], (0, 1))
        if is_amorphous:
            sij_single = tf.tensordot(eigenvects[0], sij_single, (0, 1))
        else:
            sij_single = tf.tensordot(eigenvects[0].conj(), sij_single, (0, 1))

        sij[0] = sij_single
        return sij


    def calculate_velocity_af(self):
        n_modes = self.n_modes
        sij = self.calculate_sij()
        frequency = self.frequency[0]
        sij = tf.reshape(sij, (n_modes, n_modes, 3))
        inverse_sqrt_freq = tf.cast(tf.convert_to_tensor(1 / np.sqrt(frequency)), tf.complex128)
        velocity_AF = 1 / (2 * np.pi) * contract('mna,m,n->mna', sij,
                               inverse_sqrt_freq, inverse_sqrt_freq, backend='tensorflow') / 2
        return velocity_AF


    def calculate_velocity(self):
        velocity_AF = self._velocity_af
        # velocity = 1j * tf.reduce_sum(velocity_AF, axis=1)
        # velocity = 1j * tf.linalg.diag_part(velocity_AF, k=0)
        velocity_np = 1j * contract('kmma->kma', velocity_AF.numpy()[np.newaxis, ...])
        velocity =tf.convert_to_tensor(velocity_np)
        return tf.math.real(velocity)


    def calculate_eigensystem(self, only_eigenvals=False):
        q_point = self.q_point
        is_amorphous = self.is_amorphous
        distance_threshold = self.distance_threshold
        atoms = self.atoms
        n_unit_cell = atoms.positions.shape[0]
        n_replicas = np.prod(self.supercell)
        if distance_threshold is not None:
            logging.info('Using folded dynamical matrix.')
        if is_amorphous:
            dtype = np.float
        else:
            dtype = np.complex
        if only_eigenvals:
            esystem = np.zeros((1, n_unit_cell * 3), dtype=dtype)
        else:
            shape = (1, n_unit_cell * 3 + 1, n_unit_cell * 3)
            log_size(shape, dtype, name='eigensystem')
            esystem = np.zeros(shape, dtype=dtype)
        cell_inv = np.linalg.inv(self.atoms.cell)
        replicated_cell_inv = np.linalg.inv(self.replicated_atoms.cell)

        qvec = q_point
        dynmat = self.calculate_dynmat()
        is_at_gamma = (qvec == (0, 0, 0)).all()

        list_of_replicas = self.list_of_replicas
        if distance_threshold is not None:
            shape = (n_unit_cell, 3, n_unit_cell, 3)
            type = np.complex
            log_size(shape, type, name='dynamical_matrix')
            dyn_s = np.zeros(shape, dtype=type)
            replicated_cell = self.replicated_atoms.cell

            for l in range(n_replicas):
                distance_to_wrap = atoms.positions[:, np.newaxis, :] - (
                    self.replicated_atoms.positions.reshape(n_replicas, n_unit_cell ,3)[np.newaxis, l, :, :])

                distance_to_wrap = wrap_coordinates(distance_to_wrap, replicated_cell, replicated_cell_inv)

                mask = np.linalg.norm(distance_to_wrap, axis=-1) < distance_threshold
                id_i, id_j = np.argwhere(mask).T

                dyn_s[id_i, :, id_j, :] += dynmat[0, id_i, :, 0, id_j, :] * chi(qvec, list_of_replicas, cell_inv)[l]
        else:
            if is_at_gamma:
                dyn_s = contract('ialjb->iajb', dynmat[0], backend='tensorflow')
            else:
                dyn_s = contract('ialjb,l->iajb',
                                 tf.cast(dynmat[0], tf.complex128),
                                 tf.convert_to_tensor(chi(qvec, list_of_replicas, cell_inv).flatten()),
                                 backend='tensorflow')
        dyn_s = tf.reshape(dyn_s, (self.n_modes, self.n_modes))


        if only_eigenvals:
            evals = np.linalg.eigvalsh(dyn_s)
            esystem[0] = evals
        else:
            if is_at_gamma:
                evals, evects = dsyev(dyn_s)[:2]
            else:
                evals, evects = np.linalg.eigh(dyn_s)
                # evals, evects = zheev(dyn_s)[:2]
            esystem[0] = np.vstack((evals, evects))
        return esystem


    def calculate_dynmat(self):
        mass = self.atoms.get_masses()
        dynmat = contract('mialjb,i,j->mialjb', tf.convert_to_tensor(self.second), tf.convert_to_tensor(1 / np.sqrt(mass)), tf.convert_to_tensor(1 / np.sqrt(mass)), backend='tensorflow')
        evtotenjovermol = units.mol / (10 * units.J)
        return dynmat * evtotenjovermol


from kaldo.grid import wrap_coordinates
from kaldo.observables.forceconstant import chi
from kaldo.observables.observable import Observable
import numpy as np
from opt_einsum import contract
import ase.units as units
from kaldo.helpers.storage import lazy_property
import tensorflow as tf
from kaldo.helpers.logger import get_logger, log_size
logging = get_logger()

MIN_N_MODES_TO_STORE = 1000
EVTOTENJOVERMOL = units.mol / (10 * units.J)


class HarmonicWithQ(Observable):

    def __init__(self, q_point, second,
                 distance_threshold=None, storage='numpy', is_nw=False, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.q_point = q_point
        self.atoms = second.atoms
        self.n_modes = self.atoms.positions.shape[0] * 3
        self.supercell = second.supercell
        self.is_amorphous = (np.array(self.supercell) == [1, 1, 1]).all()
        self.replicated_atoms = second.replicated_atoms
        self.list_of_replicas = second.list_of_replicas
        self.second = second
        self.distance_threshold = distance_threshold
        self.physical_mode= np.ones((1, self.n_modes), dtype=bool)
        self.is_nw = is_nw
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
    def _dynmat_derivatives(self):
        _dynmat_derivatives = self.calculate_dynmat_derivatives()
        return _dynmat_derivatives


    @lazy_property(label='<q_point>')
    def _dynmat(self):
        _dynmat = self.calculate_dynmat()
        return _dynmat
    

    @lazy_property(label='<q_point>')
    def _dynmat_fourier(self):
        dynmat_fourier = self.calculate_dynmat_fourier()
        return dynmat_fourier


    @lazy_property(label='<q_point>')
    def _eigensystem(self):
        _eigensystem = self.calculate_eigensystem(only_eigenvals=False)
        return _eigensystem


    @lazy_property(label='<q_point>')
    def _velocity_af(self):
        _velocity_af = self.calculate_velocity_af()
        return _velocity_af


    @lazy_property(label='<q_point>')
    def _sij(self):
        _sij = self.calculate_sij()
        return _sij


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
        replicated_cell_inv = self.second._replicated_cell_inv
        cell_inv = self.second.cell_inv
        dynmat = self._dynmat
        positions = self.atoms.positions
        n_unit_cell = atoms.positions.shape[0]
        n_modes = n_unit_cell * 3
        n_replicas = np.prod(self.supercell)
        shape = (1, n_unit_cell * 3, n_unit_cell * 3, 3)
        if is_amorphous:
            type = np.float
        else:
            type = np.complex
        log_size(shape, type, name='dynamical_matrix_derivative')
        dynmat_derivatives = np.zeros(shape).astype(type)
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

            list_of_replicas = self.list_of_replicas

            if distance_threshold is not None:

                distance_to_wrap = positions[:, np.newaxis, np.newaxis, :] - (
                    self.replicated_atoms.positions.reshape(n_replicas, n_unit_cell, 3)[
                    np.newaxis, :, :, :])

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
                                                                         dynmat.numpy()[0, id_i, :, 0, id_j, :] *
                                                                         chi(q_point, list_of_replicas, cell_inv)[l])
            else:

                dynmat_derivatives = contract('ilja,ibljc,l->ibjca',
                                              tf.convert_to_tensor(distance.astype(np.complex)),
                                              tf.cast(dynmat[0], tf.complex128),
                                              tf.convert_to_tensor(chi(q_point, list_of_replicas, cell_inv).flatten().astype(np.complex)),
                                              backend='tensorflow')
        dynmat_derivatives = tf.reshape(dynmat_derivatives, (n_modes, n_modes, 3))
        return dynmat_derivatives


    def calculate_sij(self):
        q_point = self.q_point
        is_amorphous = self.is_amorphous
        shape = (3 * self.atoms.positions.shape[0], 3 * self.atoms.positions.shape[0], 3)
        if is_amorphous:
            type = np.float
        else:
            type = np.complex
        if self.atoms.positions.shape[0] > 100:
            # We want to print only for big systems
            logging.info('Flux operators for q = ' + str(q_point))
        dynmat_derivatives = self._dynmat_derivatives
        log_size(shape, type, name='sij')
        eigenvects = self._eigensystem[1:, :]
        if is_amorphous:
            sij = tf.tensordot(eigenvects, dynmat_derivatives, (0, 1))
            sij = tf.tensordot(eigenvects, sij, (0, 1))
        else:
            eigenvects = tf.cast(eigenvects, tf.complex128)
            sij = tf.tensordot(eigenvects, dynmat_derivatives, (0, 1))
            sij = tf.tensordot(tf.math.conj(eigenvects), sij, (0, 1))
        return sij


    def calculate_velocity_af(self):
        n_modes = self.n_modes
        sij = self._sij
        frequency = self.frequency[0]
        sij = tf.reshape(sij, (n_modes, n_modes, 3))
        inverse_sqrt_freq = tf.cast(tf.convert_to_tensor(1 / np.sqrt(frequency)), tf.complex128)
        velocity_AF = 1 / (2 * np.pi) * contract('mna,m,n->mna', sij,
                               inverse_sqrt_freq, inverse_sqrt_freq, backend='tensorflow') / 2
        return velocity_AF


    def calculate_velocity(self):
        velocity_AF = self._velocity_af
        velocity_AF = tf.where(tf.math.is_nan(tf.math.real(velocity_AF)), 0., velocity_AF)
        velocity = contract('kmma->kma', velocity_AF.numpy()[np.newaxis, ...])
        return velocity.imag


    def calculate_dynmat_fourier(self):
        q_point = self.q_point
        distance_threshold = self.distance_threshold
        atoms = self.atoms
        n_unit_cell = atoms.positions.shape[0]
        n_replicas = np.prod(self.supercell)
        dynmat = self._dynmat
        cell_inv = self.second.cell_inv
        replicated_cell_inv = self.second._replicated_cell_inv
        is_at_gamma = (q_point == (0, 0, 0)).all()
        is_amorphous = (n_replicas == 1)
        list_of_replicas = self.list_of_replicas
        if distance_threshold is not None:
            shape = (n_unit_cell, 3, n_unit_cell, 3)
            type = np.complex
            log_size(shape, type, name='dynmat_fourier')
            dyn_s = np.zeros(shape, dtype=type)
            replicated_cell = self.replicated_atoms.cell

            for l in range(n_replicas):
                distance_to_wrap = atoms.positions[:, np.newaxis, :] - (
                    self.replicated_atoms.positions.reshape(n_replicas, n_unit_cell ,3)[np.newaxis, l, :, :])

                distance_to_wrap = wrap_coordinates(distance_to_wrap, replicated_cell, replicated_cell_inv)

                mask = np.linalg.norm(distance_to_wrap, axis=-1) < distance_threshold
                id_i, id_j = np.argwhere(mask).T
                dyn_s[id_i, :, id_j, :] += dynmat.numpy()[0, id_i, :, 0, id_j, :] * chi(q_point, list_of_replicas, cell_inv)[l]
        else:
            if is_at_gamma:
                if is_amorphous:
                    dyn_s = dynmat[0]
                else:
                    dyn_s = contract('ialjb->iajb', dynmat[0], backend='tensorflow')
            else:
                log_size((self.n_modes, self.n_modes), np.complex, name='dynmat_fourier')
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
            esystem = tf.linalg.eigh(dyn_s)
            esystem = tf.concat(axis=0, values=(esystem[0][tf.newaxis, :], esystem[1]))
        return esystem


    def calculate_dynmat(self):
        mass = self.atoms.get_masses()
        dynmat = self.second.value * 1 / np.sqrt(mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        dynmat = dynmat * 1 / np.sqrt(mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])
        evtotenjovermol = units.mol / (10 * units.J)
        return tf.convert_to_tensor(dynmat * evtotenjovermol)


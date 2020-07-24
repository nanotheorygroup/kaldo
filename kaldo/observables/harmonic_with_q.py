
from kaldo.helpers.tools import timeit
from kaldo.grid import wrap_coordinates
from scipy.linalg.lapack import dsyev
from kaldo.observables.forceconstant import chi
from kaldo.observables.observable import Observable
import numpy as np
from opt_einsum import contract
import ase.units as units
from kaldo.helpers.storage import lazy_property
from kaldo.helpers.storage import DEFAULT_STORE_FORMATS

from kaldo.helpers.logger import get_logger, log_size
logging = get_logger()

class HarmonicWithQ(Observable):

    def __init__(self, q_point, atoms, supercell, replicated_atoms, list_of_replicas, second, is_amorphous=False,
                 distance_threshold=None, storage='numpy', *kargs, **kwargs):

        super().__init__(*kargs, **kwargs)
        self.q_point = q_point
        self.atoms = atoms
        self.n_modes = self.atoms.positions.shape[0] * 3
        self.supercell = supercell
        self.replicated_atoms = replicated_atoms
        self.list_of_replicas = list_of_replicas
        self.second = second.value
        self.is_amorphous = is_amorphous
        self.distance_threshold = distance_threshold
        self.storage = storage
        self.store_format = {}
        for observable in DEFAULT_STORE_FORMATS:
            self.store_format[observable] = DEFAULT_STORE_FORMATS[observable] \
                if self.storage == 'formatted' else self.storage


    @property
    def frequency(self):
        frequency = self.calculate_frequency()
        return frequency


    def calculate_frequency(self):
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
            dynmat_derivatives = contract('ija,ibjc->ibjca', distance, dynmat[0, :, :, 0, :, :])
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
                    dynmat_derivatives[id_i, :, id_j, :, :] += np.einsum('fa,fbc->fbca', distance[id_i, l, id_j, :], \
                                                                         dynmat[0, id_i, :, 0, id_j, :] *
                                                                         chi(q_point, list_of_replicas, cell_inv)[l])
            else:

                dynmat_derivatives = contract('ilja,ibljc,l->ibjca', distance, dynmat[0],
                                              chi(q_point, list_of_replicas, cell_inv).flatten())
        ddyn[0] = dynmat_derivatives.reshape((n_modes, n_modes, 3))
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
        dynmat_derivatives = self.calculate_dynmat_derivatives()

        eigenvects = self.calculate_eigensystem(only_eigenvals=False)[:, 1:, :]
        sij_single = np.tensordot(eigenvects[0], dynmat_derivatives[0], (0, 1))
        if is_amorphous:
            sij_single = np.tensordot(eigenvects[0], sij_single, (0, 1))
        else:
            sij_single = np.tensordot(eigenvects[0].conj(), sij_single, (0, 1))

        sij[0] = sij_single
        return sij


    def calculate_velocity_af(self):
        n_modes = self.n_modes
        sij = self.calculate_sij()
        frequency = self.frequency
        sij = sij.reshape((1, n_modes, n_modes, 3))
        velocity_AF = contract('kmna,kmn->kmna', sij,
                               1 / (2 * np.pi * np.sqrt(frequency[:, :, np.newaxis]) * np.sqrt(
                                   frequency[:, np.newaxis, :]))) / 2
        return velocity_AF


    def calculate_velocity(self):
        velocity_AF = self.calculate_velocity_af()
        velocity = 1j * contract('kmma->kma', velocity_AF)
        return velocity.real


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
                dyn_s = contract('ialjb->iajb', dynmat[0])
            else:
                dyn_s = contract('ialjb,l->iajb', dynmat[0], chi(qvec, list_of_replicas, cell_inv).flatten())
        dyn_s = dyn_s.reshape((self.n_modes, self.n_modes))


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
        dynmat = contract('mialjb,i,j->mialjb', self.second, 1 / np.sqrt(mass), 1 / np.sqrt(mass))
        evtotenjovermol = units.mol / (10 * units.J)
        return dynmat * evtotenjovermol

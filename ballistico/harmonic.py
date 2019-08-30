from opt_einsum import contract
import numpy as np
import ase.units as units
import os
from .helper import lazy_property


EVTOTENJOVERMOL = units.mol / (10 * units.J)
DELTA_DOS = 1
NUM_DOS = 100
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J
THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
FOLDER_NAME = 'output'
FREQUENCY_THRESHOLD = 0.001



def calculate_density_of_states(frequencies, k_mesh, delta=DELTA_DOS, num=NUM_DOS):
    n_modes = frequencies.shape[-1]
    frequencies = frequencies.reshape((k_mesh[0], k_mesh[1], k_mesh[2], n_modes), order='C')
    n_k_points = np.prod(k_mesh)
    # increase_factor = 3
    omega_kl = np.zeros((n_k_points, n_modes))
    for mode in range(n_modes):
        omega_kl[:, mode] = frequencies[..., mode].flatten()
    # Energy axis and dos
    omega_e = np.linspace(0., np.amax(omega_kl) + 5e-3, num=num)
    dos_e = np.zeros_like(omega_e)
    # Sum up contribution from all q-points and branches
    for omega_l in omega_kl:
        diff_el = (omega_e[:, np.newaxis] - omega_l[np.newaxis, :]) ** 2
        dos_el = 1. / (diff_el + (0.5 * delta) ** 2)
        dos_e += dos_el.sum(axis=1)
    dos_e *= 1. / (n_k_points * np.pi) * 0.5 * delta
    return omega_e, dos_e


class Harmonic:
    def __init__(self, *args, **kwargs):
        self.finite_difference = kwargs['finite_difference']
        if 'folder' in kwargs:
            self.folder_name = kwargs['folder']
        else:
            self.folder_name = FOLDER_NAME
        if 'kpts' in kwargs:
            self.kpts = np.array(kwargs['kpts'])
        else:
            self.kpts = np.array([1, 1, 1])
        if 'frequency_threshold' in kwargs:
            self.frequency_threshold = kwargs['frequency_threshold']
        else:
            self.frequency_threshold = FREQUENCY_THRESHOLD

        self.atoms = self.finite_difference.atoms
        self.supercell = np.array (self.finite_difference.supercell)
        self.n_k_points = int(np.prod (self.kpts))
        self.n_modes = self.atoms.get_masses ().shape[0] * 3
        self.n_phonons = self.n_k_points * self.n_modes

        # TODO: Move cell_inv and replicated_cell_inv to finitedifference
        self.cell_inv = np.linalg.inv(self.atoms.cell)
        self.replicated_cell = self.finite_difference.replicated_atoms.cell
        self.replicated_cell_inv = np.linalg.inv(self.replicated_cell)
        self.is_able_to_calculate = True
        self.replicated_cell = self.finite_difference.replicated_atoms.cell
        self.list_of_replicas = self.finite_difference.list_of_replicas()


    @lazy_property(is_storing=False, is_reduced_path=True)
    def k_points(self):
        k_points =  self.calculate_k_points()
        return k_points


    @lazy_property(is_storing=True, is_reduced_path=True)
    def dynmat(self):
        dynmat =  self.calculate_dynamical_matrix()
        return dynmat


    @lazy_property(is_storing=True, is_reduced_path=True)
    def frequencies(self):
        frequencies =  self.calculate_second_order_observable('frequencies')
        return frequencies


    @lazy_property(is_storing=True, is_reduced_path=True)
    def eigensystem(self):
        eigensystem =  self.calculate_eigensystem()
        return eigensystem


    @lazy_property(is_storing=True, is_reduced_path=True)
    def dynmat_derivatives(self):
        dynmat_derivatives =  self.calculate_second_order_observable('dynmat_derivatives')
        return dynmat_derivatives


    @lazy_property(is_storing=True, is_reduced_path=True)
    def velocities(self):
        velocities =  self.calculate_second_order_observable('velocities')
        return velocities


    @lazy_property(is_storing=True, is_reduced_path=True)
    def velocities_AF(self):
        velocities_AF =  self.calculate_second_order_observable('velocities_AF')
        return velocities_AF


    @lazy_property(is_storing=True, is_reduced_path=True)
    def dos(self):
        dos = calculate_density_of_states(self.frequencies, self.kpts)
        return dos


    @lazy_property(is_storing=False, is_reduced_path=True)
    def physical_modes(self):
        physical_modes = (self.frequencies.reshape(self.n_phonons) > self.frequency_threshold)
        return physical_modes


    @property
    def is_amorphous(self):
        is_amorphous = (self.kpts == (1, 1, 1)).all()
        return is_amorphous


    @property
    def eigenvalues(self):
        eigenvalues = self.eigensystem[:, :, -1]
        return eigenvalues


    @property
    def eigenvectors(self):
        eigenvectors = self.eigensystem[:, :, :-1]
        return eigenvectors



    def chi(self, qvec):
        dxij = self.list_of_replicas
        cell_inv = self.cell_inv
        chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
        return chi_k


    def calculate_k_points(self):
        k_size = self.kpts
        n_k_points = self.n_k_points
        k_points = np.zeros ((n_k_points, 3))
        for index_k in range (n_k_points):
            k_points[index_k] = np.unravel_index (index_k, k_size, order='C') / k_size
        return k_points


    def apply_boundary_with_cell(self, dxij):
        # exploit periodicity to calculate the shortest distance, which may not be the one we have
        sxij = dxij.dot(self.replicated_cell_inv)
        sxij = sxij - np.round(sxij)
        dxij = sxij.dot(self.replicated_cell)
        return dxij


    def calculate_dynamical_matrix(self):
        atoms = self.atoms
        second_order = self.finite_difference.second_order.copy()
        list_of_replicas = self.list_of_replicas
        geometry = atoms.positions
        n_particles = geometry.shape[0]
        n_replicas = list_of_replicas.shape[0]
        is_second_reduced = (second_order.size == n_particles * 3 * n_replicas * n_particles * 3)
        if is_second_reduced:
            dynmat = second_order.reshape((n_particles, 3, n_replicas, n_particles, 3), order='C')
        else:
            dynmat = second_order.reshape((n_replicas, n_particles, 3, n_replicas, n_particles, 3), order='C')[0]
        mass = np.sqrt(atoms.get_masses())
        dynmat /= mass[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        dynmat /= mass[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

        # TODO: probably we want to move this unit conversion somewhere more appropriate
        dynmat *= EVTOTENJOVERMOL
        return dynmat

    def calculate_eigensystem(self, k_list=None):
        if k_list is not None:
            k_points = k_list
        else:
            k_points = self.k_points
        atoms = self.atoms
        n_unit_cell = atoms.positions.shape[0]
        n_k_points = k_points.shape[0]

        # Here we store the eigenvalues in the last column
        eigensystem = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3 + 1)).astype(np.complex)
        for index_k in range(n_k_points):
            eigensystem[index_k, :, -1], eigensystem[index_k, :, :-1] = self.calculate_eigensystem_for_k(k_points[index_k])
        return eigensystem


    def calculate_second_order_observable(self, observable, k_list=None):
        if k_list is not None:
            k_points = k_list
        else:
            k_points = self.k_points
        atoms = self.atoms
        n_unit_cell = atoms.positions.shape[0]
        n_k_points = k_points.shape[0]
        if observable == 'frequencies':
            tensor = np.zeros((n_k_points, n_unit_cell * 3))
            function = self.calculate_frequencies_for_k
        elif observable == 'dynmat_derivatives':
            tensor = np.zeros((n_k_points, n_unit_cell * 3,  n_unit_cell * 3, 3)).astype(np.complex)
            function = self.calculate_dynmat_derivatives_for_k
        elif observable == 'velocities_AF':
            tensor = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3, 3)).astype(np.complex)
            function = self.calculate_velocities_AF_for_k
        elif observable == 'velocities':
            tensor = np.zeros((n_k_points, n_unit_cell * 3, 3))
            function = self.calculate_velocities_for_k
        else:
            raise TypeError('Operator not recognized')
        for index_k in range(n_k_points):
            tensor[index_k] = function(k_points[index_k])
        return tensor


    def calculate_eigensystem_for_k(self, qvec, only_eigenvals=False):
        dynmat = self.dynmat
        atoms = self.atoms
        geometry = atoms.positions
        n_particles = geometry.shape[0]
        n_phonons = n_particles * 3
        if self.is_amorphous:
            dyn_s = dynmat[:, :, 0, :, :]
        else:
            dyn_s = contract('ialjb,l->iajb', dynmat, self.chi(qvec))
        dyn_s = dyn_s.reshape((n_phonons, n_phonons), order='C')
        if only_eigenvals:
            evals = np.linalg.eigvalsh(dyn_s)
            return evals
        else:
            evals, evects = np.linalg.eigh(dyn_s)
            return evals, evects

    def calculate_dynmat_derivatives_for_k(self, qvec):
        dynmat = self.dynmat
        atoms = self.atoms
        geometry = atoms.positions
        n_particles = geometry.shape[0]
        n_phonons = n_particles * 3
        geometry = atoms.positions
        list_of_replicas = self.list_of_replicas
        if self.is_amorphous:
            dxij = self.apply_boundary_with_cell(geometry[:, np.newaxis, :] - geometry[np.newaxis, :, :])
            dynmat_derivatives = contract('ija,ibjc->ibjca', dxij, dynmat[:, :, 0, :, :])
        else:
            dxij = self.apply_boundary_with_cell(geometry[:, np.newaxis, np.newaxis, :] - (
                    geometry[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :]))
            dynmat_derivatives = contract('ilja,ibljc,l->ibjca', dxij, dynmat, self.chi(qvec))
        dynmat_derivatives = dynmat_derivatives.reshape((n_phonons, n_phonons, 3), order='C')
        return dynmat_derivatives

    def calculate_frequencies_for_k(self, qvec):
        rescaled_qvec = qvec * self.kpts
        if (np.round(rescaled_qvec) == qvec * self.kpts).all():
            k_index = np.ravel_multi_index(rescaled_qvec.astype(int), self.kpts, order='C')
            eigenvals = self.eigenvalues[k_index]
        else:
            eigenvals = self.calculate_eigensystem_for_k(qvec, only_eigenvals=True)
        frequencies = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
        return frequencies

    def calculate_velocities_AF_for_k(self, qvec):
        rescaled_qvec = qvec * self.kpts
        if (np.round(rescaled_qvec) == qvec * self.kpts).all():
            k_index = np.ravel_multi_index(rescaled_qvec.astype(int), self.kpts, order='C')
            dynmat_derivatives = self.dynmat_derivatives[k_index]
            frequencies = self.frequencies[k_index]
            eigenvects = self.eigenvectors[k_index]
        else:
            dynmat_derivatives = self.calculate_dynmat_derivatives_for_k(qvec)
            frequencies = self.calculate_frequencies_for_k(qvec)
            _, eigenvects = self.calculate_eigensystem_for_k(qvec)

        frequencies_threshold = self.frequency_threshold
        condition = frequencies > frequencies_threshold
        velocities_AF = contract('im,ija,jn->mna', eigenvects[:, :].conj(), dynmat_derivatives, eigenvects[:, :])
        velocities_AF = contract('mna,mn->mna', velocities_AF,
                                          1 / (2 * np.pi * np.sqrt(frequencies[:, np.newaxis]) * np.sqrt(frequencies[np.newaxis, :])))
        velocities_AF[np.invert(condition), :, :] = 0
        velocities_AF[:, np.invert(condition), :] = 0
        velocities_AF = velocities_AF / 2
        return velocities_AF

    def calculate_velocities_for_k(self, qvec):
        rescaled_qvec = qvec * self.kpts
        if (np.round(rescaled_qvec) == qvec * self.kpts).all():
            k_index = np.ravel_multi_index(rescaled_qvec.astype(int), self.kpts, order='C')
            velocities_AF = self.velocities_AF[k_index]
        else:
            velocities_AF = self.calculate_velocities_AF_for_k(qvec)

        velocities = 1j * np.diagonal(velocities_AF).T
        return velocities
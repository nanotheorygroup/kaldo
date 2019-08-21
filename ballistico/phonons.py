import os


import numpy as np
import ase.units as units
from opt_einsum import contract
from .conductivity_controller import ConductivityController
from .anharmonic_controller import AnharmonicController

EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J
THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15

FREQUENCY_THRESHOLD = 0.001


OCCUPATIONS_FILE = 'occupations.npy'
C_V_FILE = 'c_v.npy'
FOLDER_NAME = 'output'
GAMMA_FILE = 'gamma.npy'
PS_FILE = 'phase_space.npy'


DELTA_DOS = 1
NUM_DOS = 100




def lazy_property(fn):
    attr = '_lazy__' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr):
            filename = self.folder_name + '/' + fn.__name__ + '.npy'
            try:
                loaded_attr = np.load (filename)
            except FileNotFoundError:
                print(filename, 'not found, calculating', fn.__name__)
                loaded_attr = fn(self)
                np.save (filename, loaded_attr)
            else:
                print('loading', filename)
            setattr(self, attr, loaded_attr)
        return getattr(self, attr)
    return _lazy_property


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




class Phonons:
    def __init__(self, finite_difference, is_classic, temperature, folder=FOLDER_NAME, kpts = (1, 1, 1), sigma_in=None, frequency_threshold=FREQUENCY_THRESHOLD, broadening_shape='gauss'):
        self.finite_difference = finite_difference
        self.atoms = finite_difference.atoms
        self.supercell = np.array (finite_difference.supercell)
        self.kpts = np.array (kpts)
        self.is_classic = is_classic
        self.n_k_points = np.prod (self.kpts)
        self.n_modes = self.atoms.get_masses ().shape[0] * 3
        self.n_phonons = self.n_k_points * self.n_modes
        self.temperature = temperature

        self._dos = None
        self._occupations = None
        self._full_scattering_plus = None
        self._full_scattering_minus = None
        self._k_points = None
        self.folder_name = folder
        self.sigma_in = sigma_in
        self._c_v = None
        self.is_able_to_calculate = True
        self.broadening_shape = broadening_shape

        if self.is_classic:
            classic_string = 'classic'
        else:
            classic_string = 'quantum'
        folder = self.folder_name + '/' + str (self.temperature) + '/' + classic_string + '/'
        if self.sigma_in is not None:
            folder += 'sigma_in_' + str (self.sigma_in).replace ('.', '_') + '/'
        folders = [self.folder_name, folder]
        for folder in folders:
            if not os.path.exists (folder):
                os.makedirs (folder)
        if frequency_threshold is not None:
            self.frequency_threshold = frequency_threshold
        else:
            self.frequency_threshold = FREQUENCY_THRESHOLD
        self.replicated_cell = self.finite_difference.replicated_atoms.cell
        self.list_of_replicas = self.finite_difference.list_of_replicas()
        self._ps = None
        self._gamma = None
        self._gamma_tensor = None

    @staticmethod
    def apply_boundary_with_cell(cell, cellinv, dxij):
        # exploit periodicity to calculate the shortest distance, which may not be the one we have
        sxij = dxij.dot(cellinv)
        sxij = sxij - np.round(sxij)
        dxij = sxij.dot(cell)
        return dxij

    @lazy_property
    def k_points(self):
        k_points =  self.calculate_k_points()
        return k_points

    @lazy_property
    def dynmat(self):
        dynmat =  self.calculate_dynamical_matrix()
        return dynmat

    @lazy_property
    def frequencies(self):
        frequencies =  self.calculate_second_order_observable('frequencies')
        return frequencies

    @lazy_property
    def eigenvalues(self):
        eigenvalues =  self.calculate_second_order_observable('eigenvalues')
        return eigenvalues

    @lazy_property
    def eigenvectors(self):
        eigenvectors =  self.calculate_second_order_observable('eigenvectors')
        return eigenvectors

    @lazy_property
    def dynmat_derivatives(self):
        dynmat_derivatives =  self.calculate_second_order_observable('dynmat_derivatives')
        return dynmat_derivatives

    @lazy_property
    def velocities(self):
        velocities =  self.calculate_second_order_observable('velocities')
        return velocities

    @lazy_property
    def velocities_AF(self):
        velocities_AF =  self.calculate_second_order_observable('velocities_AF')
        return velocities_AF

    @lazy_property
    def dos(self):
        dos = calculate_density_of_states(self.frequencies, self.kpts)
        return dos

    @property
    def occupations(self):
        return self._occupations

    @occupations.getter
    def occupations(self):
        if self._occupations is None:
            try:
                folder = self.folder_name
                folder += '/' + str(self.temperature) + '/'
                if self.is_classic:
                    folder += 'classic/'
                else:
                    folder += 'quantum/'
                self._occupations = np.load (folder + OCCUPATIONS_FILE)
            except FileNotFoundError as e:
                print(e)
        if self._occupations is None:
            frequencies = self.frequencies

            temp = self.temperature * KELVINTOTHZ
            density = np.zeros_like(frequencies)
            physical_modes = frequencies > self.frequency_threshold

            if self.is_classic is False:
                density[physical_modes] = 1. / (np.exp(frequencies[physical_modes] / temp) - 1.)
            else:
                density[physical_modes] = temp / frequencies[physical_modes]
            self.occupations = density
        return self._occupations

    @occupations.setter
    def occupations(self, new_occupations):
        folder = self.folder_name
        folder += '/' + str(self.temperature) + '/'
        if self.is_classic:
            folder += 'classic/'
        else:
            folder += 'quantum/'
        np.save (folder + OCCUPATIONS_FILE, new_occupations)
        self._occupations = new_occupations


    @property
    def c_v(self):
        return self._c_v

    @c_v.getter
    def c_v(self):
        if self._c_v is None:
            try:
                folder = self.folder_name
                folder += '/' + str(self.temperature) + '/'
                if self.is_classic:
                    folder += 'classic/'
                else:
                    folder += 'quantum/'
                self._c_v = np.load (folder + C_V_FILE)
            except FileNotFoundError as e:
                print(e)
        if self._c_v is None:
            frequencies = self.frequencies
            c_v = np.zeros_like (frequencies)
            physical_modes = frequencies > self.frequency_threshold
            temperature = self.temperature * KELVINTOTHZ

            if (self.is_classic):
                c_v[physical_modes] = KELVINTOJOULE
            else:
                f_be = self.occupations
                c_v[physical_modes] = KELVINTOJOULE * f_be[physical_modes] * (f_be[physical_modes] + 1) * self.frequencies[physical_modes] ** 2 / \
                                      (temperature ** 2)
            self.c_v = c_v
        return self._c_v

    @c_v.setter
    def c_v(self, new_c_v):
        folder = self.folder_name
        folder += '/' + str(self.temperature) + '/'
        if self.is_classic:
            folder += 'classic/'
        else:
            folder += 'quantum/'
        np.save (folder + C_V_FILE, new_c_v)
        self._c_v = new_c_v


    @property
    def gamma(self):
        return self._gamma

    @gamma.getter
    def gamma(self):
        if self._gamma is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._gamma = np.load (folder + GAMMA_FILE)
            except FileNotFoundError as e:
                print(e)
                AnharmonicController(self).calculate_gamma(is_gamma_tensor_enabled=False)
        return self._gamma

    @gamma.setter
    def gamma(self, new_gamma):
        folder = self.folder_name
        folder += '/'
        np.save (folder + GAMMA_FILE, new_gamma)
        self._gamma = new_gamma

    @property
    def ps(self):
        return self._ps

    @ps.getter
    def ps(self):
        if self._ps is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._ps = np.load (folder + PS_FILE)
            except FileNotFoundError as e:
                print(e)
                AnharmonicController(self).calculate_gamma(is_gamma_tensor_enabled=False)
        return self._ps

    @ps.setter
    def ps(self, new_ps):
        folder = self.folder_name
        folder += '/'
        np.save (folder + GAMMA_FILE, new_ps)
        self._ps = new_ps

    @property
    def gamma_tensor(self):
        if self._gamma_tensor is None:
            AnharmonicController(self).calculate_gamma(is_gamma_tensor_enabled=True)
        return  self._gamma_tensor

    def calculate_k_points(self):
        k_size = self.kpts
        n_k_points = np.prod (k_size)
        k_points = np.zeros ((n_k_points, 3))
        for index_k in range (n_k_points):
            k_points[index_k] = np.unravel_index (index_k, k_size, order='C') / k_size
        return k_points

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
        elif observable == 'eigenvalues':
            tensor = np.zeros((n_k_points, n_unit_cell * 3))
            function = self.calculate_eigenvalues_for_k
        elif observable == 'eigenvectors':
            tensor = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3)).astype(np.complex)
            function = self.calculate_eigenvectors_for_k
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

    def calculate_eigenvalues_for_k(self, qvec):
        return self.calculate_eigensystem_for_k(qvec, only_eigvals=True)

    def calculate_eigenvectors_for_k(self, qvec):
        return self.calculate_eigensystem_for_k(qvec, only_eigvals=False)

    def calculate_eigensystem_for_k(self, qvec, only_eigvals=False):
        dynmat = self.dynmat
        atoms = self.atoms
        geometry = atoms.positions
        n_particles = geometry.shape[0]
        n_phonons = n_particles * 3
        replicated_cell = self.replicated_cell
        cell_inv = np.linalg.inv(self.atoms.cell)
        list_of_replicas = self.list_of_replicas
        replicated_cell_inv = np.linalg.inv(replicated_cell)
        is_amorphous = (self.kpts == (1, 1, 1)).all()
        dxij = self.apply_boundary_with_cell(replicated_cell, replicated_cell_inv, list_of_replicas[np.newaxis, :, np.newaxis, :])
        if is_amorphous:
            dyn_s = dynmat[:, :, 0, :, :]
        else:
            chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
            dyn_s = contract('ialjb,ilj->iajb', dynmat, chi_k)
        dyn_s = dyn_s.reshape((n_phonons, n_phonons), order='C')
        if only_eigvals:
            evals = np.linalg.eigvalsh(dyn_s)
            return evals
        else:
            # TODO: here we are diagonalizing twice to calculate the same quantity, we'll need to change this
            _, evects = np.linalg.eigh(dyn_s)
            return evects

    def calculate_dynmat_derivatives_for_k(self, qvec):
        dynmat = self.dynmat
        atoms = self.atoms
        geometry = atoms.positions
        n_particles = geometry.shape[0]
        n_phonons = n_particles * 3
        replicated_cell = self.replicated_cell
        geometry = atoms.positions
        cell_inv = np.linalg.inv(self.atoms.cell)
        list_of_replicas = self.list_of_replicas
        replicated_cell_inv = np.linalg.inv(replicated_cell)
        is_amorphous = (self.kpts == (1, 1, 1)).all()
        dxij = self.apply_boundary_with_cell(replicated_cell, replicated_cell_inv, list_of_replicas[np.newaxis, :, np.newaxis, :])
        if is_amorphous:
            dxij = self.apply_boundary_with_cell(replicated_cell, replicated_cell_inv, geometry[:, np.newaxis, :] - geometry[np.newaxis, :, :])
            dynmat_derivatives = contract('ija,ibjc->ibjca', dxij, dynmat[:, :, 0, :, :])
        else:
            chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
            dxij = self.apply_boundary_with_cell(replicated_cell, replicated_cell_inv, geometry[:, np.newaxis, np.newaxis, :] - (
                    geometry[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :]))
            dynmat_derivatives = contract('ilja,ibljc,ilj->ibjca', dxij, dynmat, chi_k)
        dynmat_derivatives = dynmat_derivatives.reshape((n_phonons, n_phonons, 3), order='C')
        return dynmat_derivatives

    def calculate_frequencies_for_k(self, qvec):
        rescaled_qvec = qvec * self.kpts
        if (np.round(rescaled_qvec) == qvec * self.kpts).all():
            k_index = np.ravel_multi_index(rescaled_qvec.astype(int), self.kpts, order='C')
            eigenvals = self.eigenvalues[k_index]
        else:
            eigenvals = self.calculate_eigenvalues_for_k(qvec)
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
            eigenvects = self.calculate_eigenvectors_for_k(qvec)

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


    def calculate_conductivity(self, method='rta', max_n_iterations=None):
        if max_n_iterations and method != 'sc':
            raise TypeError('Only self consistent method support n_iteration parameter')

        conductivity_controller = ConductivityController(self)
        if method == 'rta':
            conductivity = conductivity_controller.calculate_conductivity_rta()
        elif method == 'af':
            conductivity = conductivity_controller.calculate_conductivity_AF()
        elif method == 'inverse':
            conductivity = conductivity_controller.calculate_conductivity_inverse()
        elif method == 'sc':
            conductivity = conductivity_controller.calculate_conductivity_sc(max_n_iterations)
        else:
            raise TypeError('Conductivity method not recognized')
        return conductivity
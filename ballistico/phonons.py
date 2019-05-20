import numpy as np
import os
import ballistico.phonons_calculator
import ase.units as units
from scipy.sparse import save_npz

import numpy as np
import sparse

from scipy.optimize import minimize

MAX_ITERATIONS_SC = 500

import sparse

ENERGY_THRESHOLD = 0.001
GAMMA_CUTOFF = 0


FREQUENCIES_FILE = 'frequencies.npy'
EIGENVALUES_FILE = 'eigenvalues.npy'
EIGENVECTORS_FILE = 'eigenvectors.npy'
VELOCITIES_FILE = 'velocities.npy'
GAMMA_FILE = 'gamma.npy'
DOS_FILE = 'dos.npy'
OCCUPATIONS_FILE = 'occupations.npy'
K_POINTS_FILE = 'k_points.npy'
C_V_FILE = 'c_v.npy'
SCATTERING_MATRIX_FILE = 'scattering_matrix'
FOLDER_NAME = 'phonons_calculated'


class Phonons (object):
    def __init__(self, finite_difference, folder=FOLDER_NAME, kpts = (1, 1, 1), is_classic = False, temperature
    = 300, sigma_in=None, energy_threshold=ENERGY_THRESHOLD, gamma_cutoff=GAMMA_CUTOFF, broadening_shape='gauss'):
        self.finite_difference = finite_difference
        self.atoms = finite_difference.atoms
        self.supercell = np.array (finite_difference.supercell)
        self.kpts = np.array (kpts)
        self.is_classic = is_classic
        self.n_k_points = np.prod (self.kpts)
        self.n_modes = self.atoms.get_masses ().shape[0] * 3
        self.n_phonons = self.n_k_points * self.n_modes
        self.temperature = temperature

        self._frequencies = None
        self._velocities = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._dos = None
        self._occupations = None
        self._full_scattering_plus = None
        self._full_scattering_minus = None
        self._n_k_points = None
        self._n_modes = None
        self._n_phonons = None
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
        if energy_threshold is not None:
            self.energy_threshold = energy_threshold
        else:
            self.energy_threshold = ENERGY_THRESHOLD
        
        if gamma_cutoff is not None:
            self.gamma_cutoff = gamma_cutoff
        else:
            self.gamma_cutoff = GAMMA_CUTOFF
            
        self.replicated_cell = self.finite_difference.replicated_atoms.cell
        replicated_cell_inv = np.linalg.inv(self.replicated_cell)
        replicated_atoms_positions = (self.__apply_boundary_with_cell(self.replicated_cell, replicated_cell_inv, self.finite_difference.replicated_atoms.positions))

        n_replicas = np.prod (self.finite_difference.supercell)
        atoms = self.finite_difference.atoms
        n_unit_atoms = self.finite_difference.atoms.positions.shape[0]
        list_of_replicas = (
                replicated_atoms_positions.reshape ((n_replicas, n_unit_atoms, 3)) -
                atoms.positions[np.newaxis, :, :])
        self.list_of_index = list_of_replicas[:, 0, :]



    @property
    def frequencies(self):
        return self._frequencies

    @frequencies.getter
    def frequencies(self):
        if self._frequencies is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._frequencies = np.load (folder + FREQUENCIES_FILE)
            except FileNotFoundError as e:
                print(e)

                frequencies, eigenvalues, eigenvectors, velocities = ballistico.phonons_calculator.calculate_second_k_list(
                    self.k_points,
                    self.atoms,
                    self.finite_difference.second_order,
                    self.list_of_index,
                    self.replicated_cell,
                    self.energy_threshold)
                self.frequencies = frequencies
                self.eigenvalues = eigenvalues
                self.velocities = velocities
                self.eigenvectors = eigenvectors
        return self._frequencies

    @frequencies.setter
    def frequencies(self, new_frequencies):
        folder = self.folder_name
        folder += '/'
        np.save (folder + FREQUENCIES_FILE, new_frequencies)
        self._frequencies = new_frequencies

    @property
    def velocities(self):
        return self._velocities

    @velocities.getter
    def velocities(self):
        if self._velocities is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._velocities = np.load (folder + VELOCITIES_FILE)
            except FileNotFoundError as e:
                print(e)

                frequencies, eigenvalues, eigenvectors, velocities = ballistico.phonons_calculator.calculate_second_k_list(
                    self.k_points,
                    self.atoms,
                    self.finite_difference.second_order,
                    self.list_of_index,
                    self.replicated_cell,
                    self.energy_threshold)
                self.frequencies = frequencies
                self.eigenvalues = eigenvalues
                self.velocities = velocities
                self.eigenvectors = eigenvectors
        return self._velocities

    @velocities.setter
    def velocities(self, new_velocities):
        folder = self.folder_name
        folder += '/'
        np.save (folder + VELOCITIES_FILE, new_velocities)
        self._velocities = new_velocities

    @property
    def eigenvectors(self):
        return self._eigenvectors

    @eigenvectors.getter
    def eigenvectors(self):
        if self._eigenvectors is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._eigenvectors = np.load (folder + EIGENVECTORS_FILE)
            except FileNotFoundError as e:
                print(e)
        return self._eigenvectors

    @eigenvectors.setter
    def eigenvectors(self, new_eigenvectors):
        folder = self.folder_name
        folder += '/'
        np.save (folder + EIGENVECTORS_FILE, new_eigenvectors)
        self._eigenvectors = new_eigenvectors

    @property
    def eigenvalues(self):
        return self._eigenvalues

    @eigenvalues.getter
    def eigenvalues(self):
        if self._eigenvalues is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._eigenvalues = np.load (folder + EIGENVALUES_FILE)
            except FileNotFoundError as e:
                print(e)
                frequencies, eigenvalues, eigenvectors, velocities = ballistico.phonons_calculator.calculate_second_k_list(
                    self.k_points,
                    self.atoms,
                    self.finite_difference.second_order,
                    self.list_of_index,
                    self.replicated_cell,
                    self.energy_threshold)
                self.frequencies = frequencies
                self.eigenvalues = eigenvalues
                self.velocities = velocities
                self.eigenvectors = eigenvectors
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, new_eigenvalues):
        folder = self.folder_name
        folder += '/'
        np.save (folder + EIGENVALUES_FILE, new_eigenvalues)
        self._eigenvalues = new_eigenvalues

    @property
    def full_scattering_plus(self):
        return self._full_scattering_plus

    @full_scattering_plus.getter
    def full_scattering_plus(self):
        if self._full_scattering_plus is None:
            folder = self.folder_name
            folder += '/' + str(self.temperature) + '/'
            if self.is_classic:
                folder += 'classic/'
            else:
                folder += 'quantum/'
            if self.sigma_in is not None:
                folder += 'sigma_in_' + str(self.sigma_in).replace('.', '_') + '/'
            try:
                n_phonons = self.n_phonons
                nu, nu_p, nu_pp, pot_times_delta = np.loadtxt(folder + SCATTERING_MATRIX_FILE + '_1').T

                self._full_scattering_plus = sparse.COO((nu.astype(int), nu_p.astype(int), nu_pp.astype(int)),
                                                        pot_times_delta, (n_phonons, n_phonons, n_phonons))
                nu, nu_p, nu_pp, pot_times_delta = np.loadtxt(folder + SCATTERING_MATRIX_FILE + '_0').T
                self._full_scattering_minus = sparse.COO((nu.astype(int), nu_p.astype(int), nu_pp.astype(int)),
                                                        pot_times_delta, (n_phonons, n_phonons, n_phonons))
            except OSError as e:
                print(e)
                self._full_scattering_plus, self._full_scattering_minus = ballistico.phonons_calculator.calculate_gamma(
                    self.atoms,
                    self.frequencies,
                    self.velocities,
                    self.occupations,
                    self.kpts,
                    self.eigenvectors,
                    self.list_of_index,
                    self.finite_difference.third_order,
                    self.sigma_in,
                    self.broadening_shape,
                    self.energy_threshold,
                    folder + '/' + SCATTERING_MATRIX_FILE
                )
        return self._full_scattering_plus


    @property
    def full_scattering_minus(self):
        return self._full_scattering_minus

    @full_scattering_minus.getter
    def full_scattering_minus(self):
        if self._full_scattering_minus is None:
            folder = self.folder_name
            folder += '/' + str(self.temperature) + '/'
            if self.is_classic:
                folder += 'classic/'
            else:
                folder += 'quantum/'
            if self.sigma_in is not None:
                folder += 'sigma_in_' + str(self.sigma_in).replace('.', '_') + '/'
            try:
                n_phonons = self.n_phonons
                nu, nu_p, nu_pp, pot_times_delta = np.loadtxt(folder + SCATTERING_MATRIX_FILE + '_1').T

                self._full_scattering_plus = sparse.COO((nu.astype(int), nu_p.astype(int), nu_pp.astype(int)),
                                                        pot_times_delta, (n_phonons, n_phonons, n_phonons))
                nu, nu_p, nu_pp, pot_times_delta = np.loadtxt(folder + SCATTERING_MATRIX_FILE + '_0').T
                self._full_scattering_minus = sparse.COO((nu.astype(int), nu_p.astype(int), nu_pp.astype(int)),
                                                        pot_times_delta, (n_phonons, n_phonons, n_phonons))
            except OSError as e:
                print(e)
                self._full_scattering_plus, self._full_scattering_minus = ballistico.phonons_calculator.calculate_gamma(
                    self.atoms,
                    self.frequencies,
                    self.velocities,
                    self.occupations,
                    self.kpts,
                    self.eigenvectors,
                    self.list_of_index,
                    self.finite_difference.third_order,
                    self.sigma_in,
                    self.broadening_shape,
                    self.energy_threshold,
                    folder + '/' + SCATTERING_MATRIX_FILE
                )

        return self._full_scattering_minus

    @property
    def gamma(self):
        n_kpoints = np.prod(self.kpts)
        gamma = (self.full_scattering_minus.sum(axis=2).sum(axis=1).reshape((n_kpoints, self.n_modes)) + \
                 self.full_scattering_plus.sum(axis=2).sum(axis=1).reshape((n_kpoints, self.n_modes))).todense()
        return gamma

    @property
    def gamma_tensor_plus(self):
        return (self.full_scattering_plus.sum(axis=1) - self.full_scattering_plus.sum(axis=2)).todense()

    @property
    def gamma_tensor_minus(self):
        return (self.full_scattering_minus.sum(axis=1) + self.full_scattering_minus.sum(axis=2)).todense()

    @property
    def dos(self):
        return self._dos

    @dos.getter
    def dos(self):
        if self._dos is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._dos = np.load (folder + DOS_FILE)
            except FileNotFoundError as e:
                print(e)
                dos = ballistico.phonons_calculator.calculate_density_of_states(
                    self.frequencies,
                    self.kpts
                )
                self.dos = dos
        return self._dos

    @dos.setter
    def dos(self, new_dos):
        folder = self.folder_name
        folder += '/'
        np.save (folder + DOS_FILE, new_dos)
        self._dos = new_dos

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
            
            kelvinoverthz = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
            temp = self.temperature * kelvinoverthz
            density = np.zeros_like(frequencies)
            physical_modes = frequencies > self.energy_threshold

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
    def k_points(self):
        return self._k_points

    @k_points.getter
    def k_points(self):
        if self._k_points is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._k_points = np.load (folder + K_POINTS_FILE)
            except FileNotFoundError as e:
                print(e)
        if self._k_points is None:
            k_size = self.kpts
            n_k_points = np.prod (k_size)
            k_points = np.zeros ((n_k_points, 3))
            for index_k in range (n_k_points):
                k_points[index_k] = np.unravel_index (index_k, k_size, order='C') / k_size
            self.k_points = k_points
        return self._k_points

    @k_points.setter
    def k_points(self, new_k_points):
        folder = self.folder_name
        folder += '/'
        np.save (folder + K_POINTS_FILE, new_k_points)
        self._k_points = new_k_points

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
            physical_modes = frequencies > self.energy_threshold
            kelvinoverjoule = units.kB / units.J
            kelvinoverthz = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
            temperature = self.temperature * kelvinoverthz

            if (self.is_classic):
                c_v[physical_modes] = kelvinoverjoule
            else:
                f_be = self.occupations
                c_v[physical_modes] = kelvinoverjoule * f_be[physical_modes] * (f_be[physical_modes] + 1) * self.frequencies[physical_modes] ** 2 / \
                         (temperature ** 2)
            self.c_v = c_v * 1e21
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
        
    def __apply_boundary_with_cell(self, cell, cellinv, dxij):
        # exploit periodicity to calculate the shortest distance, which may not be the one we have
        sxij = dxij.dot(cellinv)
        sxij = sxij - np.round(sxij)
        dxij = sxij.dot(cell)
        return dxij

    def second_quantities_k_list(self, klist):
        return ballistico.phonons_calculator.calculate_second_k_list(
            klist,
            self.atoms,
            self.finite_difference.second_order,
            self.list_of_index,
            self.replicated_cell,
            self.energy_threshold)

    def conductivity(self, mfp):
        volume = np.linalg.det(self.atoms.cell) / 1000
        frequencies = self.frequencies.reshape((self.n_phonons), order='C')
        physical_modes = (frequencies > self.energy_threshold)
        c_v = self.c_v.reshape((self.n_phonons), order='C')
        velocities = self.velocities.real.reshape((self.n_phonons, 3), order='C') / 10
        conductivity_per_mode = np.zeros((self.n_phonons, 3, 3))
        conductivity_per_mode[physical_modes, :, :] = 1 / (volume * self.n_k_points) * c_v[physical_modes, np.newaxis, np.newaxis] * \
                                                      velocities[physical_modes, :, np.newaxis] * mfp[physical_modes,np.newaxis, :]

        return conductivity_per_mode

    def calculate_conductivity_inverse(self):
        
        scattering_matrix = (self.gamma_tensor_minus + self.gamma_tensor_plus)

        velocities = self.velocities.real.reshape((self.n_phonons, 3), order='C') / 10
        frequencies = self.frequencies.reshape((self.n_k_points * self.n_modes), order='C')
        physical_modes = (frequencies > self.energy_threshold)  # & (velocities > 0)[:, 2]
        gamma = self.gamma.reshape((self.n_phonons), order='C')
        a_in = - 1 * scattering_matrix.reshape((self.n_phonons, self.n_phonons), order='C')
        a_in = np.einsum('a,ab,b->ab', 1 / frequencies, a_in, frequencies)
        a_out = np.zeros_like(gamma)
        a_out[physical_modes] = gamma[physical_modes]
        a_out_inverse = np.zeros_like(gamma)
        a_out_inverse[physical_modes] = 1 / a_out[physical_modes]
        a = np.diag(a_out) + a_in

        # Let's remove the unphysical modes from the matrix
        index = np.outer(physical_modes, physical_modes)
        a = a[index].reshape((physical_modes.sum(), physical_modes.sum()), order='C')
        a_inverse = np.linalg.inv(a)
        lambd = np.zeros((self.n_phonons, 3))
        lambd[physical_modes, :] = a_inverse.dot(velocities[physical_modes, :])
        conductivity_per_mode = self.conductivity(lambd)
        evals = np.linalg.eigvalsh(a)
        print('negative eigenvals : ', (evals < 0).sum())
        return conductivity_per_mode

    def transmission_caltech(self, gamma, velocity, length):
        kn = abs(velocity / (length * gamma))
        transmission = (1 - kn * (1 - np.exp(- 1. / kn))) * kn
        return length / abs(velocity) * transmission

    def transmission_matthiesen(self, gamma, velocity, length):
        #        gamma + np.abs(velocity) / length
        transmission = (gamma * length / abs(velocity) + 1.) ** (-1)
        return length / abs(velocity) * transmission

    def calculate_conductivity_variational(self, n_iterations=MAX_ITERATIONS_SC):
        
        frequencies = self.frequencies.reshape((self.n_phonons), order='C')
        physical_modes = (frequencies > self.energy_threshold)

        velocities = self.velocities.real.reshape((self.n_phonons, 3), order='C') / 10
        physical_modes = physical_modes  # & (velocities > 0)[:, 2]

        gamma = self.gamma.reshape((self.n_phonons), order='C')
        scattering_matrix = (self.gamma_tensor_minus + self.gamma_tensor_plus)

        a_in = - 1 * scattering_matrix.reshape((self.n_phonons, self.n_phonons), order='C')
        a_in = np.einsum('a,ab,b->ab', 1 / frequencies, a_in, frequencies)
        a_out = np.zeros_like(gamma)
        a_out[physical_modes] = gamma[physical_modes]
        a_out_inverse = np.zeros_like(gamma)
        a_out_inverse[physical_modes] = 1 / a_out[physical_modes]
        a = np.diag(a_out) + a_in
        b = a_out_inverse[:, np.newaxis] * velocities[:, :]
        a_out_inverse_a_in_to_n_times_b = np.copy(b)
        f_n = np.copy(b)
        conductivity_value = np.zeros((3, 3, n_iterations))

        for n_iteration in range(n_iterations):

            a_out_inverse_a_in_to_n_times_b[:, :] = -1 * (a_out_inverse[:, np.newaxis] * a_in[:, physical_modes]).dot(
                a_out_inverse_a_in_to_n_times_b[physical_modes, :])
            f_n += a_out_inverse_a_in_to_n_times_b

            conductivity_value[:, :, n_iteration] = self.conductivity(f_n).sum(0)

        conductivity_per_mode = self.conductivity(f_n)
        if n_iteration == (MAX_ITERATIONS_SC - 1):
            print('Max iterations reached')
        return conductivity_per_mode, conductivity_value

    def calculate_conductivity_sc(self, tolerance=0.01, length_thresholds=None, is_rta=False,
                                  n_iterations=MAX_ITERATIONS_SC, finite_size_method='matthiesen'):
        
        volume = np.linalg.det(self.atoms.cell) / 1000
        velocities = self.velocities.real.reshape((self.n_k_points, self.n_modes, 3), order='C') / 10
        lambd_0 = np.zeros((self.n_k_points * self.n_modes, 3))
        velocities = velocities.reshape((self.n_phonons, 3), order='C')
        frequencies = self.frequencies.reshape((self.n_phonons), order='C')
        gamma = self.gamma.reshape((self.n_phonons), order='C').copy()
        physical_modes = (frequencies > self.energy_threshold)  # & (velocities > 0)[:, 2]
        if not is_rta:

            scattering_matrix = (self.gamma_tensor_minus + self.gamma_tensor_plus)

            scattering_matrix = scattering_matrix.reshape((self.n_phonons,
                                                           self.n_phonons), order='C')
            scattering_matrix = np.einsum('a,ab,b->ab', 1 / frequencies, scattering_matrix, frequencies)

        for alpha in range(3):
            if length_thresholds:
                if length_thresholds[alpha]:
                    if finite_size_method == 'matthiesen':
                        gamma[physical_modes] += abs(velocities[physical_modes, alpha]) / (
                                    1 / 2 * length_thresholds[alpha])

        tau_0 = np.zeros_like(gamma)
        tau_0[physical_modes] = 1 / gamma[physical_modes]

        lambd_0[physical_modes, alpha] = tau_0[physical_modes] * velocities[physical_modes, alpha]
        c_v = self.c_v.reshape((self.n_phonons), order='C')
        lambd_n = lambd_0.copy()
        conductivity_per_mode = np.zeros((self.n_phonons, 3, 3))

        for n_iteration in range(n_iterations):
            for alpha in range(3):
                for beta in range(3):
                    conductivity_per_mode[physical_modes, alpha, beta] = 1 / (volume * self.n_k_points) * \
                                                                         c_v[physical_modes] * velocities[
                                                                             physical_modes, alpha] * lambd_n[
                                                                             physical_modes, beta]
            if is_rta:
                return conductivity_per_mode, lambd_0

            tau_0 = tau_0.reshape((self.n_phonons), order='C')

            # calculate the shift in mft
            delta_lambd = tau_0[:, np.newaxis] * scattering_matrix.dot(lambd_n)
            lambd_n = lambd_0 + delta_lambd

        for alpha in range(3):

            for mu in np.argwhere(physical_modes):
                if length_thresholds:
                    if length_thresholds[alpha]:

                        if finite_size_method == 'caltech':
                            transmission = (1 - np.abs(lambd_n[mu, alpha]) / length_thresholds[alpha] * (
                                        1 - np.exp(-length_thresholds[alpha] / np.abs(lambd_n[mu, alpha]))))
                            lambd_n[mu] = lambd_n[mu] * transmission

        for alpha in range(3):
            for beta in range(3):
                conductivity_per_mode[physical_modes, alpha, beta] = 1 / (volume * self.n_k_points) * c_v[
                    physical_modes] * velocities[physical_modes, alpha] * lambd_n[physical_modes, beta]

        if n_iteration == (MAX_ITERATIONS_SC - 1):
            print('Convergence not reached')

        return conductivity_per_mode, lambd_n

    def calculate_conductivity_sheng(self, n_iterations=20):
        
        velocities = self.velocities.real.reshape((self.n_phonons, 3), order='C') / 10
        frequencies = self.frequencies.reshape((self.n_k_points * self.n_modes), order='C')
        physical_modes = (frequencies > self.energy_threshold)  # & (velocities > 0)[:, 2]
        tau = np.zeros(frequencies.shape)
        gamma = self.gamma
        tau[physical_modes] = 1 / gamma.reshape((self.n_phonons), order='C')[physical_modes]

        # F_0 = tau * velocities[:, :] * frequencies
        F_0 = tau[:, np.newaxis] * velocities * frequencies[:, np.newaxis]
        F_n = F_0.copy()
        for iteration in range(n_iterations):
            DeltaF = 0
            for is_plus in (1, 0):
                if is_plus:
                    DeltaF -= sparse.tensordot(self.full_scattering_plus, F_n, (1, 0))
                    DeltaF += sparse.tensordot(self.full_scattering_plus, F_n, (2, 0))

                else:
                    DeltaF += sparse.tensordot(self.full_scattering_minus, F_n, (1, 0))
                    DeltaF += sparse.tensordot(self.full_scattering_minus, F_n, (2, 0))

            F_n = F_0 + tau[:, np.newaxis] * DeltaF.sum(axis=1)

            conductivity_per_mode = self.conductivity(F_n / frequencies[:, np.newaxis])


        return conductivity_per_mode
    
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
        self._gamma = None
        self._gamma_tensor = None

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
    def gamma(self):
        if self._gamma is None:
            self.calculate_gamma(is_gamma_tensor_enabled=False)
        return self._gamma

    @property
    def gamma_tensor(self):
        if self._gamma_tensor is None:
            self.calculate_gamma(is_gamma_tensor_enabled=True)
        return  self._gamma_tensor

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

    def calculate_gamma(self, is_gamma_tensor_enabled=False):
        folder = self.folder_name
        folder += '/' + str(self.temperature) + '/'
        if self.is_classic:
            folder += 'classic/'
        else:
            folder += 'quantum/'
        if self.sigma_in is not None:
            folder += 'sigma_in_' + str(self.sigma_in).replace('.', '_') + '/'
        n_phonons = self.n_phonons
        is_plus_label = ['_0', '_1']
        file = None
        self._gamma = np.zeros(n_phonons)
        if is_gamma_tensor_enabled:
            self._gamma_tensor = np.zeros((n_phonons, n_phonons))
        for is_plus in [1, 0]:
            read_nu = -1
            file = None
            progress_filename = folder + '/' + SCATTERING_MATRIX_FILE + is_plus_label[is_plus]
            try:
                file = open(progress_filename, 'r+')
            except FileNotFoundError as err:
                print(err)
            else:
                for line in file:
                    read_nu, read_nup, read_nupp, value = np.fromstring(line, dtype=np.float, sep=' ')
                    read_nu = int(read_nu)
                    read_nup = int(read_nup)
                    read_nupp = int(read_nupp)
                    self._gamma[read_nu] += value
                    if is_gamma_tensor_enabled:
                        if is_plus:
                            self._gamma_tensor[read_nu, read_nup] -= value
                            self._gamma_tensor[read_nu, read_nupp] += value
                        else:
                            self._gamma_tensor[read_nu, read_nup] += value
                            self._gamma_tensor[read_nu, read_nupp] += value

            print('starting third order ')
            atoms = self.atoms
            frequencies = self.frequencies
            velocities = self.velocities
            density = self.occupations
            k_size = self.kpts
            eigenvectors = self.eigenvectors
            list_of_replicas = self.list_of_index
            third_order = self.finite_difference.third_order
            sigma_in = self.sigma_in
            broadening = self.broadening_shape
            frequencies_threshold = self.energy_threshold

            density = density.flatten(order='C')
            nptk = np.prod(k_size)
            n_particles = atoms.positions.shape[0]

            print('Lifetime calculation')

            # TODO: We should write this in a better way
            if list_of_replicas.shape == (3,):
                n_replicas = 1
            else:
                n_replicas = list_of_replicas.shape[0]

            cell_inv = np.linalg.inv(atoms.cell)

            is_amorphous = (k_size == (1, 1, 1)).all()

            if is_amorphous:
                chi = 1
            else:
                rlattvec = cell_inv * 2 * np.pi
                chi = np.zeros((nptk, n_replicas), dtype=np.complex)
                for index_k in range(np.prod(k_size)):
                    i_k = np.array(np.unravel_index(index_k, k_size, order='C'))
                    k_point = i_k / k_size
                    realq = np.matmul(rlattvec, k_point)
                    for l in range(n_replicas):
                        chi[index_k, l] = np.exp(1j * list_of_replicas[l].dot(realq))
            print('Projection started')
            n_modes = n_particles * 3
            nptk = np.prod(k_size)

            # print('n_irreducible_q_points = ' + str(int(len(unique_points))) + ' : ' + str(unique_points))
            process_string = ['Minus processes: ', 'Plus processes: ']
            masses = atoms.get_masses()
            rescaled_eigenvectors = eigenvectors[:, :, :].reshape((nptk, n_particles, 3, n_modes), order='C') / np.sqrt(
                masses[np.newaxis, :, np.newaxis, np.newaxis])
            rescaled_eigenvectors = rescaled_eigenvectors.reshape((nptk, n_particles * 3, n_modes), order='C')
            rescaled_eigenvectors = rescaled_eigenvectors.swapaxes(1, 2).reshape(nptk * n_modes, n_modes, order='C')

            index_kp_vec = np.arange(np.prod(k_size))
            i_kp_vec = np.array(np.unravel_index(index_kp_vec, k_size, order='C'))

            is_amorphous = (nptk == 1)
            if broadening == 'gauss':
                broadening_function = ballistico.phonons_calculator.gaussian_delta
            elif broadening == 'lorentz':
                broadening_function = ballistico.phonons_calculator.lorentzian_delta
            elif broadening == 'triangle':
                broadening_function = ballistico.phonons_calculator.triangular_delta
            read_nu = read_nu + 1

            for nu_single in range(read_nu, self.n_phonons):
                index_k, mu = np.unravel_index(nu_single, [nptk, n_modes], order='C')

                if not file:
                    file = open(progress_filename, 'a+')
                if frequencies[index_k, mu] > frequencies_threshold:


                    gamma_out = ballistico.phonons_calculator.calculate_single_gamma(is_plus, index_k, mu, i_kp_vec, index_kp_vec,
                                                       frequencies,
                                                       velocities, density,
                                                       cell_inv, k_size, n_modes, nptk, n_replicas,
                                                       rescaled_eigenvectors, chi, third_order, sigma_in,
                                                       frequencies_threshold, is_amorphous, broadening_function)

                    if gamma_out:
                        nup_vec, nupp_vec, pot_times_dirac = gamma_out
                        self._gamma[nu_single] += pot_times_dirac.sum()
                        for nup_index in range(nup_vec.shape[0]):
                            nup = nup_vec[nup_index]
                            nupp = nupp_vec[nup_index]
                            if is_gamma_tensor_enabled:
                                if is_plus:
                                    self.gamma_tensor[nu_single, nup] -= pot_times_dirac[nup_index]
                                    self.gamma_tensor[nu_single, nupp] += pot_times_dirac[nup_index]
                                else:
                                    self.gamma_tensor[nu_single, nup] += pot_times_dirac[nup_index]
                                    self.gamma_tensor[nu_single, nupp] += pot_times_dirac[nup_index]

                        nu_vec = np.ones(nup_vec.shape[0]).astype(int) * nu_single
                        # try:
                        np.savetxt(file, np.vstack([nu_vec, gamma_out]).T, fmt='%i %i %i %.8e')
                        # except ValueError as err:
                        #     print(err)
                print(process_string[is_plus] + 'q-point = ' + str(index_k))
            file.close()


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
        
        scattering_matrix = self.gamma_tensor

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
        scattering_matrix = self.gamma_tensor

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

    def calculate_conductivity_rta(self, length_thresholds=None, finite_size_method='matthiesen'):
        
        volume = np.linalg.det(self.atoms.cell) / 1000
        velocities = self.velocities.real.reshape((self.n_k_points, self.n_modes, 3), order='C') / 10
        lambd_0 = np.zeros((self.n_k_points * self.n_modes, 3))
        velocities = velocities.reshape((self.n_phonons, 3), order='C')
        frequencies = self.frequencies.reshape((self.n_phonons), order='C')
        gamma = self.gamma.reshape((self.n_phonons), order='C').copy()
        physical_modes = (frequencies > self.energy_threshold)

        tau_0 = np.zeros_like(gamma)
        tau_0[physical_modes] = 1 / gamma[physical_modes]

        for alpha in range(3):
            lambd_0[physical_modes, alpha] = tau_0[physical_modes] * velocities[physical_modes, alpha]
            if length_thresholds:
                if length_thresholds[alpha]:
                    if finite_size_method == 'matthiesen':
                        gamma[physical_modes] += abs(velocities[physical_modes, alpha]) / (
                                    1 / 2 * length_thresholds[alpha])


        c_v = self.c_v.reshape((self.n_phonons), order='C')
        lambd_n = lambd_0.copy()
        conductivity_per_mode = np.zeros((self.n_phonons, 3, 3))


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


        return conductivity_per_mode, lambd_n

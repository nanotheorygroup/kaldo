import numpy as np
import os
import ballistico.phonons_calculator
import ase.units as units
import sparse
from scipy.sparse import load_npz, save_npz
from sparse import COO


ENERGY_THRESHOLD = 0.001
GAMMA_CUTOFF = 0


FREQUENCIES_FILE = 'frequencies.npy'
EIGENVALUES_FILE = 'eigenvalues.npy'
EIGENVECTORS_FILE = 'eigenvectors.npy'
VELOCITIES_FILE = 'velocities.npy'
GAMMA_FILE = 'gamma.npy'
SCATTERING_MATRIX_FILE = 'scattering_matrix.npy'
FULL_SCATTERING_FILE_PLUS = 'full_scattering_plus.npz'
FULL_SCATTERING_FILE_MINUS = 'full_scattering_minus.npz'
DOS_FILE = 'dos.npy'
OCCUPATIONS_FILE = 'occupations.npy'
K_POINTS_FILE = 'k_points.npy'
C_V_FILE = 'c_v.npy'
THIRD_ORDER_PROJECTION_WITH_PROGRESS_FILE = 'third'

FOLDER_NAME = 'ballistico'


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
                self._full_scattering_plus = COO.from_scipy_sparse(load_npz(folder + FULL_SCATTERING_FILE_PLUS)) \
                    .reshape((self.n_phonons, self.n_phonons, self.n_phonons))
                self._full_scattering_minus = COO.from_scipy_sparse(load_npz(folder + FULL_SCATTERING_FILE_MINUS)) \
                    .reshape((self.n_phonons, self.n_phonons, self.n_phonons))
            except FileNotFoundError as e:
                print(e)
                self.full_scattering_plus, self.full_scattering_minus = ballistico.phonons_calculator.calculate_gamma(
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
                    folder + '/' + THIRD_ORDER_PROJECTION_WITH_PROGRESS_FILE
                )
        return self._full_scattering_plus

    @full_scattering_plus.setter
    def full_scattering_plus(self, new_full_scattering_plus):
        folder = self.folder_name
        folder += '/' + str(self.temperature) + '/'
        if self.is_classic:
            folder += 'classic/'
        else:
            folder += 'quantum/'
        if self.sigma_in is not None:
            folder += 'sigma_in_' + str(self.sigma_in).replace('.', '_') + '/'

        save_npz(folder + FULL_SCATTERING_FILE_PLUS, new_full_scattering_plus.reshape(
            (self.n_phonons * self.n_phonons, self.n_phonons)).to_scipy_sparse())
        self._full_scattering_plus = new_full_scattering_plus

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
                self._full_scattering_plus = COO.from_scipy_sparse(load_npz(folder + FULL_SCATTERING_FILE_PLUS)) \
                    .reshape((self.n_phonons, self.n_phonons, self.n_phonons))
                self._full_scattering_minus = COO.from_scipy_sparse(load_npz(folder + FULL_SCATTERING_FILE_MINUS)) \
                    .reshape((self.n_phonons, self.n_phonons, self.n_phonons))
            except FileNotFoundError as e:
                print(e)
                self.full_scattering_plus, self.full_scattering_minus = ballistico.phonons_calculator.calculate_gamma(
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
                    folder + '/' + THIRD_ORDER_PROJECTION_WITH_PROGRESS_FILE
                )
        return self._full_scattering_minus

    @full_scattering_minus.setter
    def full_scattering_minus(self, new_full_scattering_minus):
        folder = self.folder_name
        folder += '/' + str(self.temperature) + '/'
        if self.is_classic:
            folder += 'classic/'
        else:
            folder += 'quantum/'
        if self.sigma_in is not None:
            folder += 'sigma_in_' + str(self.sigma_in).replace('.', '_') + '/'

        save_npz(folder + FULL_SCATTERING_FILE_MINUS, new_full_scattering_minus.reshape(
            (self.n_phonons * self.n_phonons, self.n_phonons)).to_scipy_sparse())
        self._full_scattering_minus = new_full_scattering_minus

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

    def save_csv_data(self):
        frequencies = self.frequencies
        lifetime = 1. / self.gamma
        n_modes = frequencies.shape[1]
        if self.is_classic:
            filename = "data_classic"
        else:
            filename = "data_quantum"
        filename = filename + '_' + str (self.temperature)
        filename = filename + ".csv"

        filename = self.folder_name + filename
        print('saving ' + filename)
        with open (filename, "w") as csv:
            str_to_write = 'k_x,k_y,k_z,'
            for i in range (n_modes):
                str_to_write += 'frequencies_' + str (i) + ' (THz),'
            for i in range (n_modes):
                str_to_write += 'tau_' + str (i) + ' (ps),'
            for alpha in range (3):
                coord = 'x'
                if alpha == 1:
                    coord = 'y'
                if alpha == 2:
                    coord = 'z'

                for i in range (n_modes):
                    str_to_write += 'v^' + coord + '_' + str (i) + ' (km/s),'
            str_to_write += '\n'
            csv.write (str_to_write)
            velocities = self.velocities.reshape((np.prod(self.kpts), n_modes, 3), order='C')
            for k in range (self.q_points ().shape[0]):
                str_to_write = str (self.q_points ()[k, 0]) + ',' + str (self.q_points ()[k, 1]) + ',' + str (
                    self.q_points ()[k, 2]) + ','
                for i in range (n_modes):
                    str_to_write += str(self.energies[k, i] / (2 * np.pi)) + ','
                for i in range (n_modes):
                    str_to_write += str(lifetime[k, i]) + ','

                for alpha in range(3):
                    for i in range(n_modes):
                        str_to_write += str(velocities[k, i, alpha]) + ','
                str_to_write += '\n'
                csv.write(str_to_write)

    def second_quantities_k_list(self, klist):
        return ballistico.phonons_calculator.calculate_second_k_list(
            klist,
            self.atoms,
            self.finite_difference.second_order,
            self.list_of_index,
            self.replicated_cell,
            self.energy_threshold)




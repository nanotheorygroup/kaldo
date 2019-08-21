import os


import numpy as np
import scipy.special
import ase.units as units
import sparse
from opt_einsum import contract_expression, contract
import time

EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J
THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15

MAX_ITERATIONS_SC = 500
FREQUENCY_THRESHOLD = 0.001


GAMMA_FILE = 'gamma.npy'
PS_FILE = 'phase_space.npy'
OCCUPATIONS_FILE = 'occupations.npy'
C_V_FILE = 'c_v.npy'
SCATTERING_MATRIX_FILE = 'scattering_matrix'
FOLDER_NAME = 'output'


IS_SCATTERING_MATRIX_ENABLED = True
IS_DELTA_CORRECTION_ENABLED = False
DELTA_THRESHOLD = 2
DELTA_DOS = 1
NUM_DOS = 100


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


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





def calculate_broadening(velocity, cellinv, k_size):
    # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
    delta_k = cellinv / k_size
    base_sigma = ((np.tensordot(velocity, delta_k, [-1, 1])) ** 2).sum(axis=-1)
    base_sigma = np.sqrt(base_sigma / 6.)
    return base_sigma


def gaussian_delta(params):
    # alpha is a factor that tells whats the ration between the width of the gaussian
    # and the width of allowed phase space
    delta_energy = params[0]
    # allowing processes with width sigma and creating a gaussian with width sigma/2
    # we include 95% (erf(2/sqrt(2)) of the probability of scattering. The erf makes the total area 1
    sigma = params[1]
    if IS_DELTA_CORRECTION_ENABLED:
        correction = scipy.special.erf(DELTA_THRESHOLD / np.sqrt(2))
    else:
        correction = 1
    gaussian = 1 / np.sqrt(np.pi * sigma ** 2) * np.exp(- delta_energy ** 2 / (sigma ** 2))
    return gaussian / correction


def triangular_delta(params):
    delta_energy = np.abs(params[0])
    deltaa = np.abs(params[1])
    out = np.zeros_like(delta_energy)
    out[delta_energy < deltaa] = 1. / deltaa * (1 - delta_energy[delta_energy < deltaa] / deltaa)
    return out


def lorentzian_delta(params):
    delta_energy = params[0]
    gamma = params[1]
    if IS_DELTA_CORRECTION_ENABLED:
        # TODO: replace these hardcoded values
        # numerical value of the integral of a lorentzian over +- DELTA_TRESHOLD * gamma
        corrections = {
            1: 0.704833,
            2: 0.844042,
            3: 0.894863,
            4: 0.920833,
            5: 0.936549,
            6: 0.947071,
            7: 0.954604,
            8: 0.960263,
            9: 0.964669,
            10: 0.968195}
        correction = corrections[DELTA_THRESHOLD]
    else:
        correction = 1
    lorentzian = 1 / np.pi * 1 / 2 * gamma / (delta_energy ** 2 + (gamma / 2) ** 2)
    return lorentzian / correction


def calculate_single_gamma(is_plus, index_k, mu, index_kp_full, frequencies, density, nptk, first_evect, second_evect, first_chi, second_chi, scaled_potential, sigma_in,
                           frequencies_threshold, omegas, kpp_mapping, sigma_small, broadening_function):
    second_sign = (int(is_plus) * 2 - 1)

    omegas_difference = np.abs(omegas[index_k, mu] + second_sign * omegas[index_kp_full, :, np.newaxis] -
                               omegas[kpp_mapping, np.newaxis, :])

    condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_small) & \
                (frequencies[index_kp_full, :, np.newaxis] > frequencies_threshold) & \
                (frequencies[kpp_mapping, np.newaxis, :] > frequencies_threshold)
    interactions = np.array(np.where(condition)).T

    # TODO: Benchmark something fast like
    # interactions = np.array(np.unravel_index (np.flatnonzero (condition), condition.shape)).T
    if interactions.size != 0:
        # Create sparse index
        index_kp_vec = interactions[:, 0]
        index_kpp_vec = kpp_mapping[index_kp_vec]
        mup_vec = interactions[:, 1]
        mupp_vec = interactions[:, 2]


        if is_plus:
            dirac_delta = density[index_kp_vec, mup_vec] - density[index_kpp_vec, mupp_vec]

        else:
            dirac_delta = .5 * (1 + density[index_kp_vec, mup_vec] + density[index_kpp_vec, mupp_vec])

        dirac_delta /= (omegas[index_kp_vec, mup_vec] * omegas[index_kpp_vec, mupp_vec])
        if np.array(sigma_small).size == 1:

            dirac_delta *= broadening_function(
                [omegas_difference[index_kp_vec, mup_vec, mupp_vec], 2 * np.pi * sigma_small])

        else:
            dirac_delta *= broadening_function(
                [omegas_difference[index_kp_vec, mup_vec, mupp_vec], 2 * np.pi * sigma_small[
                    index_kp_vec, mup_vec, mupp_vec]])

        shapes = []
        for tens in scaled_potential, first_evect, first_chi, second_evect, second_chi:
            shapes.append(tens.shape)
        expr = contract_expression('litj,kni,kl,kmj,kt->knm', *shapes)
        scaled_potential = expr(scaled_potential,
                                first_evect,
                                first_chi,
                                second_evect,
                                second_chi
                                )

        scaled_potential = scaled_potential[index_kp_vec, mup_vec, mupp_vec]
        pot_times_dirac = np.abs(scaled_potential) ** 2 * dirac_delta

        #TODO: move units conversion somewhere else
        gammatothz = 1e11 * units.mol * EVTOTENJOVERMOL ** 2
        pot_times_dirac = units._hbar * np.pi / 4. * pot_times_dirac / omegas[index_k, mu] / nptk * gammatothz

        return index_kp_vec, mup_vec, index_kpp_vec, mupp_vec, pot_times_dirac, dirac_delta


def apply_boundary_with_cell(cell, cellinv, dxij):
    # exploit periodicity to calculate the shortest distance, which may not be the one we have
    sxij = dxij.dot(cellinv)
    sxij = sxij - np.round(sxij)
    dxij = sxij.dot(cell)
    return dxij


class Phonons (object):
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
                self.calculate_gamma(is_gamma_tensor_enabled=False)
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
                self.calculate_gamma(is_gamma_tensor_enabled=False)
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
            self.calculate_gamma(is_gamma_tensor_enabled=True)
        return  self._gamma_tensor


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
        dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, list_of_replicas[np.newaxis, :, np.newaxis, :])
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
        dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, list_of_replicas[np.newaxis, :, np.newaxis, :])
        if is_amorphous:
            dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, geometry[:, np.newaxis, :] - geometry[np.newaxis, :, :])
            dynmat_derivatives = contract('ija,ibjc->ibjca', dxij, dynmat[:, :, 0, :, :])
        else:
            chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
            dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, geometry[:, np.newaxis, np.newaxis, :] - (
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





    @timeit
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
        self._ps = np.zeros(n_phonons)
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
                    read_nu, read_nup, read_nupp, value, value_ps = np.fromstring(line, dtype=np.float, sep=' ')
                    read_nu = int(read_nu)
                    read_nup = int(read_nup)
                    read_nupp = int(read_nupp)
                    self._gamma[read_nu] += value
                    self._ps[read_nu] += value_ps
                    if is_gamma_tensor_enabled:
                        if is_plus:
                            self.gamma_tensor[read_nu, read_nup] -= value
                            self.gamma_tensor[read_nu, read_nupp] += value
                        else:
                            self.gamma_tensor[read_nu, read_nup] += value
                            self.gamma_tensor[read_nu, read_nupp] += value

            atoms = self.atoms
            frequencies = self.frequencies
            velocities = self.velocities
            density = self.occupations
            k_size = self.kpts
            eigenvectors = self.eigenvectors
            list_of_replicas = self.list_of_replicas
            third_order = self.finite_difference.third_order
            sigma_in = self.sigma_in
            broadening = self.broadening_shape
            frequencies_threshold = self.frequency_threshold
            omegas = 2 * np.pi * frequencies

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
                cell_inv = np.linalg.inv(atoms.cell)
                replicated_cell = self.finite_difference.replicated_atoms.cell
                replicated_cell_inv = np.linalg.inv(replicated_cell)
                chi = np.zeros((nptk, n_replicas), dtype=np.complex)
                dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, list_of_replicas)

                for index_k in range(np.prod(k_size)):
                    i_k = np.array(np.unravel_index(index_k, k_size, order='C'))
                    k_point = i_k / k_size
                    realq = np.matmul(rlattvec, k_point)
                    chi[index_k] = np.exp(1j * dxij.dot(realq))

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

            # TODO: find a way to use initial_mu correctly, when restarting
            read_nu += 1
            if (read_nu < nptk * n_modes):
                initial_k, initial_mu = np.unravel_index(read_nu, (nptk, n_modes))
                # initial_k = initial_k + 1

                for index_k in range(initial_k, nptk):

                    i_k = np.array(np.unravel_index(index_k, k_size, order='C'))


                    i_kpp_vec = i_k[:, np.newaxis] + (int(is_plus) * 2 - 1) * i_kp_vec[:, :]
                    index_kpp_vec = np.ravel_multi_index(i_kpp_vec, k_size, order='C', mode='wrap')

                    if is_plus:
                        first_evect = rescaled_eigenvectors.reshape((nptk, n_modes, n_modes))
                    else:
                        first_evect = rescaled_eigenvectors.conj().reshape((nptk, n_modes, n_modes))
                    second_evect = rescaled_eigenvectors.conj().reshape((nptk, n_modes, n_modes))[index_kpp_vec]

                    if is_plus:
                        first_chi = chi
                    else:
                        first_chi = chi.conj()
                    second_chi = chi.conj()[index_kpp_vec]

                    if broadening == 'gauss':
                        broadening_function = gaussian_delta
                    elif broadening == 'lorentz':
                        broadening_function = lorentzian_delta
                    elif broadening == 'triangle':
                        broadening_function = triangular_delta

                    # +1 if is_plus, -1 if not is_plus
                    if sigma_in is None:
                        sigma_tensor_np = calculate_broadening(velocities[index_kp_vec, :, np.newaxis, :] -
                                                               velocities[index_kpp_vec, np.newaxis, :, :], cell_inv,
                                                               k_size)
                        sigma_small = sigma_tensor_np
                    else:
                        sigma_small = sigma_in


                    for mu in range(n_modes):
                        if index_k == initial_k and mu < initial_mu:
                            break

                        nu_single = np.ravel_multi_index([index_k, mu], [nptk, n_modes], order='C')
                        if not file:
                            file = open(progress_filename, 'a+')
                        if frequencies[index_k, mu] > frequencies_threshold:

                            scaled_potential = sparse.tensordot(third_order, rescaled_eigenvectors[nu_single, :], (0, 0))
                            scaled_potential = scaled_potential.reshape((n_replicas, n_modes, n_replicas, n_modes),
                                                                        order='C')

                            gamma_out = calculate_single_gamma(is_plus, index_k, mu, index_kp_vec, frequencies, density, nptk,
                                                               first_evect, second_evect, first_chi, second_chi,
                                                               scaled_potential, sigma_small, frequencies_threshold, omegas, index_kpp_vec, sigma_small, broadening_function)

                            if gamma_out:
                                index_kp_out, mup_out, index_kpp_out, mupp_out, pot_times_dirac, dirac = gamma_out
                                nup_vec = np.ravel_multi_index(np.array([index_kp_out, mup_out]),
                                                               np.array([nptk, n_modes]), order='C')
                                nupp_vec = np.ravel_multi_index(np.array([index_kpp_out, mupp_out]),
                                                                np.array([nptk, n_modes]), order='C')

                                self._gamma[nu_single] += pot_times_dirac.sum()
                                self._ps[nu_single] += dirac.sum()

                                for nup_index in range(nup_vec.shape[0]):
                                    nup = nup_vec[nup_index]
                                    nupp = nupp_vec[nup_index]
                                    if is_gamma_tensor_enabled:
                                        if is_plus:
                                            self._gamma_tensor[nu_single, nup] -= pot_times_dirac[nup_index]
                                            self._gamma_tensor[nu_single, nupp] += pot_times_dirac[nup_index]
                                        else:
                                            self._gamma_tensor[nu_single, nup] += pot_times_dirac[nup_index]
                                            self._gamma_tensor[nu_single, nupp] += pot_times_dirac[nup_index]

                                nu_vec = np.ones(nup_vec.shape[0]).astype(int) * nu_single
                                # try:
                                np.savetxt(file, np.vstack([nu_vec, nup_vec, nupp_vec, pot_times_dirac, dirac]).T, fmt='%i %i %i %.8e %.8e')
                                # except ValueError as err:
                                #     print(err)
                        # print(process_string[is_plus] + 'q-point = ' + str(index_k))
                file.close()


    def conductivity(self, mfp):
        volume = np.linalg.det(self.atoms.cell) / 1000
        frequencies = self.frequencies.reshape((self.n_phonons), order='C')
        physical_modes = (frequencies > self.frequency_threshold)
        c_v = self.c_v.reshape((self.n_phonons), order='C') * 1e21
        velocities = self.velocities.real.reshape((self.n_phonons, 3), order='C') / 10
        conductivity_per_mode = np.zeros((self.n_phonons, 3, 3))
        conductivity_per_mode[physical_modes, :, :] = 1 / (volume * self.n_k_points) * c_v[physical_modes, np.newaxis, np.newaxis] * \
                                                      velocities[physical_modes, :, np.newaxis] * mfp[physical_modes,np.newaxis, :]

        return conductivity_per_mode



    def calculate_conductivity_inverse(self):

        scattering_matrix = self.gamma_tensor

        velocities = self.velocities.real.reshape((self.n_phonons, 3), order='C') / 10
        frequencies = self.frequencies.reshape((self.n_k_points * self.n_modes), order='C')
        physical_modes = (frequencies > self.frequency_threshold)  # & (velocities > 0)[:, 2]
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
        physical_modes = (frequencies > self.frequency_threshold)

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

    def calculate_conductivity_rta(self):
        volume = np.linalg.det(self.atoms.cell)
        gamma = self.gamma.reshape((self.n_k_points, self.n_modes)).copy()
        physical_modes = (self.frequencies > self.frequency_threshold)
        tau = 1 / gamma
        tau[np.invert(physical_modes)] = 0
        self.velocities[np.isnan(self.velocities)] = 0
        conductivity_per_mode = np.zeros((self.n_k_points, self.n_modes, 3, 3))
        conductivity_per_mode[:, :, :, :] = contract('kn,kna,kn,knb->knab', self.c_v[:, :], self.velocities[:, :, :], tau[:, :], self.velocities[:, :, :])
        conductivity_per_mode = 1e22 / (volume * self.n_k_points) * conductivity_per_mode
        conductivity_per_mode = conductivity_per_mode.reshape((self.n_phonons, 3, 3))
        return conductivity_per_mode

    def calculate_conductivity_AF(self, gamma_in=None):
        volume = np.linalg.det(self.atoms.cell)
        if gamma_in is not None:
            gamma = gamma_in * np.ones((self.n_k_points, self.n_modes))
        else:
            gamma = self.gamma.reshape((self.n_k_points, self.n_modes)).copy()
        omega = 2 * np.pi * self.frequencies
        physical_modes = (self.frequencies[:, :, np.newaxis] > self.frequency_threshold) * (self.frequencies[:, np.newaxis, :] > self.frequency_threshold)

        lorentz = (gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2 / (((gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2) ** 2 +
                                                                           (omega[:, :, np.newaxis] - omega[:, np.newaxis, :]) ** 2)
        lorentz[np.invert(physical_modes)] = 0
        conductivity_per_mode = np.zeros((self.n_k_points, self.n_modes, self.n_modes, 3, 3))
        conductivity_per_mode[:, :, :, :, :] = contract('kn,knma,knm,knmb->knmab', self.c_v[:, :], self.velocities_AF[:, :, :, :], lorentz[:, :, :], self.velocities_AF[:, :, :, :])
        conductivity_per_mode = 1e22 / (volume * self.n_k_points) * conductivity_per_mode

        kappa = contract('knmab->knab', conductivity_per_mode)
        kappa = kappa.reshape((self.n_phonons, 3, 3))

        return kappa


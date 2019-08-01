import os


import numpy as np
import scipy.special
import ase.units as units
import sparse
from opt_einsum import contract_expression, contract

# import tensorflow as tf
# tf.enable_eager_execution()

EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J
THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15

MAX_ITERATIONS_SC = 500
ENERGY_THRESHOLD = 0.001
GAMMA_CUTOFF = 0


FREQUENCIES_FILE = 'frequencies.npy'
EIGENVALUES_FILE = 'eigenvalues.npy'
EIGENVECTORS_FILE = 'eigenvectors.npy'
VELOCITIES_AF_FILE = 'velocities_af.npy'
VELOCITIES_FILE = 'velocities.npy'
GAMMA_FILE = 'gamma.npy'
PS_FILE = 'phase_space.npy'
DOS_FILE = 'dos.npy'
OCCUPATIONS_FILE = 'occupations.npy'
K_POINTS_FILE = 'k_points.npy'
C_V_FILE = 'c_v.npy'
SCATTERING_MATRIX_FILE = 'scattering_matrix'
FOLDER_NAME = 'phonons_calculated'


IS_SCATTERING_MATRIX_ENABLED = True
IS_SORTING_EIGENVALUES = False
DIAGONALIZATION_ALGORITHM = np.linalg.eigh
# DIAGONALIZATION_ALGORITHM = scipy.linalg.lapack.zheev

IS_DELTA_CORRECTION_ENABLED = False
DELTA_THRESHOLD = 2
DELTA_DOS = 1
NUM_DOS = 100


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


def calculate_single_gamma(is_plus, index_k, mu, i_kp_full, index_kp_full, frequencies, velocities, density, cell_inv,
                           k_size,
                           n_modes, nptk, n_replicas, evect, chi, third_order, sigma_in,
                           frequencies_threshold, is_amorphous, broadening_function):
    i_k = np.array(np.unravel_index(index_k, k_size, order='C'))

    omegas = 2 * np.pi * frequencies
    nu = np.ravel_multi_index([index_k, mu], [nptk, n_modes], order='C')

    i_kpp_vec = i_k[:, np.newaxis] + (int(is_plus) * 2 - 1) * i_kp_full[:, :]
    index_kpp_vec = np.ravel_multi_index(i_kpp_vec, k_size, order='C', mode='wrap')

    # +1 if is_plus, -1 if not is_plus
    second_sign = (int(is_plus) * 2 - 1)
    if sigma_in is None:
        sigma_tensor_np = calculate_broadening(velocities[index_kp_full, :, np.newaxis, :] -
                                               velocities[index_kpp_vec, np.newaxis, :, :], cell_inv, k_size)
        sigma_small = sigma_tensor_np
    else:
        sigma_small = sigma_in

    omegas_difference = np.abs(omegas[index_k, mu] + second_sign * omegas[index_kp_full, :, np.newaxis] -
                               omegas[index_kpp_vec, np.newaxis, :])

    condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_small) & \
                (frequencies[index_kp_full, :, np.newaxis] > frequencies_threshold) & \
                (frequencies[index_kpp_vec, np.newaxis, :] > frequencies_threshold)
    interactions = np.array(np.where(condition)).T

    # TODO: Benchmark something fast like
    # interactions = np.array(np.unravel_index (np.flatnonzero (condition), condition.shape)).T
    if interactions.size != 0:
        # Create sparse index
        index_kp_vec = interactions[:, 0]
        index_kpp_vec = index_kpp_vec[index_kp_vec]
        mup_vec = interactions[:, 1]
        mupp_vec = interactions[:, 2]
        nup_vec = np.ravel_multi_index(np.array([index_kp_vec, mup_vec]),
                                       np.array([nptk, n_modes]), order='C')
        nupp_vec = np.ravel_multi_index(np.array([index_kpp_vec, mupp_vec]),
                                        np.array([nptk, n_modes]), order='C')

        # prepare evect
        # scaled_potential = sparse.tensordot(third_order, evect[nu, :], [0, 0])
        # scaled_potential = np.zeros((n_replicas * n_modes, n_replicas * n_modes), dtype=np.complex128)
        # for evect_index in range(n_modes):
        #     scaled_potential += third_order[evect_index, :, :].todense() * evect[nu, evect_index]
        scaled_potential = sparse.tensordot(third_order, evect[nu, :], (0, 0))
        scaled_potential = scaled_potential.reshape((n_replicas, n_modes, n_replicas, n_modes), order='C')

        if is_plus:
            dirac_delta = density[nup_vec] - density[nupp_vec]

        else:
            dirac_delta = .5 * (1 + density[nup_vec] + density[nupp_vec])

        dirac_delta /= (omegas[index_kp_vec, mup_vec] * omegas[index_kpp_vec, mupp_vec])
        if sigma_in is None:
            dirac_delta *= broadening_function(
                [omegas_difference[index_kp_vec, mup_vec, mupp_vec], 2 * np.pi * sigma_tensor_np[
                    index_kp_vec, mup_vec, mupp_vec]])

        else:
            dirac_delta *= broadening_function(
                [omegas_difference[index_kp_vec, mup_vec, mupp_vec], 2 * np.pi * sigma_in])

        if is_plus:
            first_evect = evect[nup_vec]
        else:
            first_evect = evect.conj()[nup_vec]
        second_evect = evect.conj()[nupp_vec]
        if is_amorphous:
            scaled_potential = contract('litj,aj,ai->a', scaled_potential, second_evect, first_evect)
        else:
            if is_plus:
                first_chi = chi[index_kp_vec]
            else:
                first_chi = chi.conj()[index_kp_vec]
            second_chi = chi.conj()[index_kpp_vec]
            shapes = []
            for tens in scaled_potential, first_chi, second_chi, second_evect, first_evect:
                shapes.append(tens.shape)

            expr = contract_expression('litj,al,at,aj,ai->a', *shapes)
            scaled_potential = expr(scaled_potential,
                                    first_chi,
                                    second_chi,
                                    second_evect,
                                    first_evect
                                    # backend='tensorflow',
                                    )

        # gamma contracted on one index
        pot_times_dirac = np.abs(scaled_potential) ** 2 * dirac_delta
        gammatothz = 1e11 * units.mol * EVTOTENJOVERMOL ** 2
        pot_times_dirac = units._hbar * np.pi / 4. * pot_times_dirac / omegas[index_k, mu] / nptk * gammatothz

        pot_times_dirac_davide = pot_times_dirac.sum() * THZTOMEV / (2 * np.pi)
        print(frequencies[index_k, mu], pot_times_dirac_davide)

        return nup_vec.astype(int), nupp_vec.astype(int), pot_times_dirac, dirac_delta





def apply_boundary_with_cell(cell, cellinv, dxij):
    # exploit periodicity to calculate the shortest distance, which may not be the one we have
    sxij = dxij.dot(cellinv)
    sxij = sxij - np.round(sxij)
    dxij = sxij.dot(cell)
    return dxij


class Phonons (object):
    def __init__(self, finite_difference, folder=FOLDER_NAME, kpts = (1, 1, 1), is_classic = False, temperature
    = 300, sigma_in=None, energy_threshold=ENERGY_THRESHOLD, gamma_cutoff=GAMMA_CUTOFF, broadening_shape='gauss', velocity_dq=None):
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
        self._velocities_AF = None
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
        self.velocity_dq = velocity_dq
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
        self.list_of_replicas = self.finite_difference.list_of_replicas()
        self._ps = None
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
                self.calculate_second_k_list()
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
                self.calculate_second_k_list()

        return self._velocities

    @velocities.setter
    def velocities(self, new_velocities):
        folder = self.folder_name
        folder += '/'
        np.save (folder + VELOCITIES_FILE, new_velocities)
        self._velocities = new_velocities

    @property
    def velocities_AF(self):
        return self._velocities_AF

    @velocities_AF.getter
    def velocities_AF(self):
        if self._velocities_AF is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._velocities_AF = np.load (folder + VELOCITIES_AF_FILE)
            except FileNotFoundError as e:
                print(e)
                self.calculate_second_k_list()

        return self._velocities_AF

    @velocities_AF.setter
    def velocities_AF(self, new_velocities_AF):
        folder = self.folder_name
        folder += '/'
        np.save (folder + VELOCITIES_AF_FILE, new_velocities_AF)
        self._velocities_AF = new_velocities_AF

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
                self.calculate_second_k_list()

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
                self.calculate_second_k_list()
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, new_eigenvalues):
        folder = self.folder_name
        folder += '/'
        np.save (folder + EIGENVALUES_FILE, new_eigenvalues)
        self._eigenvalues = new_eigenvalues


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
        np.save (folder + GAMMA_FILE, ps)
        self._ps = new_ps

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
                dos = calculate_density_of_states(
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

            temp = self.temperature * KELVINTOTHZ
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

    def diagonalize_second_order_single_k(self, qvec, dynmat, frequencies_threshold, dq=None):
        # TODO: remove duplicate arguments from this method
        atoms = self.atoms
        geometry = atoms.positions
        n_particles = geometry.shape[0]
        n_phonons = n_particles * 3
        atoms = self.atoms
        replicated_cell = self.replicated_cell
        pos = self.finite_difference.atoms.positions
        geometry = atoms.positions
        n_particles = geometry.shape[0]
        cell_inv = np.linalg.inv(self.atoms.cell)
        list_of_replicas = self.list_of_replicas
        n_replicas = list_of_replicas.shape[0]
        replicated_cell_inv = np.linalg.inv(replicated_cell)

        if dq is not None:
            # dynmat = dynmat.reshape((n_particles, 3, n_replicas, n_particles, 3))
            dynbase = np.zeros((n_particles, 3, n_particles, 3), dtype=np.complex)
            # pos = apply_boundary_with_cell(cell, cell_inv, pos)
            q_vec = 2 * np.pi * cell_inv.dot(qvec)
            for iat in range(n_particles):
                for alpha in range(3):
                    for id_replica in range(n_replicas):
                        for jat in range(n_particles):
                            for beta in range(3):
                                dxij = - (list_of_replicas[id_replica, :])
                                # dxij = pos[iat, :] - (list_of_replicas[id_replica, :] + pos[jat, :])
                                dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, dxij)

                                # phase = 2 * np.pi * qvec.dot(dxij)
                                phase = -1 * dxij.dot(q_vec)
                                # chi_k = np.exp(1j * 2 * np.pi * (dxij.dot(cell_inv)).dot(qvec))

                                dynbase[iat, alpha, jat, beta] += dynmat[iat, alpha, id_replica, jat, beta] * np.exp(
                                    1j * phase)
            perturbx = np.zeros((n_particles, 3, n_particles, 3, 3), dtype=np.complex)
            for gamma in range(3):

                dqx = np.zeros(3)
                dqx[gamma] = dq
                # dqmod = np.linalg.norm(dq) * 2 * np.pi
                # dqx = dqx.dot(atoms.cell)
                q_prime_vec = 2 * np.pi * (cell_inv.dot(qvec + atoms.cell.dot(dqx)))
                dqmod = np.linalg.norm(2 * np.pi * (dqx))

                perturb = np.zeros((n_particles, 3, n_particles, 3), dtype=np.complex)
                for iat in range(n_particles):
                    for alpha in range(3):
                        for id_replica in range(n_replicas):
                            for jat in range(n_particles):
                                for beta in range(3):
                                    dxij = - (list_of_replicas[id_replica, :])
                                    # dxij = pos[iat, :] - (list_of_replicas[id_replica, :] + pos[jat, :])
                                    dxij = self.__apply_boundary_with_cell(replicated_cell, replicated_cell_inv, dxij)

                                    # phase = 2 * np.pi * (qvec + dqx).dot(dxij)
                                    phase = -1 * dxij.dot(q_prime_vec)
                                    perturb[iat, alpha, jat, beta] += dynmat[iat, alpha, id_replica, jat, beta] * np.exp(
                                        1j * phase)

                perturbx[..., gamma] = (perturb - dynbase) / dqmod
            dyn_s, ddyn_s = dynbase, perturbx

        else:
            # dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, list_of_replicas)
            # kpoint = 2 * np.pi * (cell_inv).dot(qvec)
            # chi_k = np.exp(1j * dxij.dot(kpoint))

            # dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, geometry[:, np.newaxis, np.newaxis, :] - (
            #         geometry[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :]))
            is_amorphous = (self.kpts == (1, 1, 1)).all()
            dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, list_of_replicas[np.newaxis, :, np.newaxis, :])
            if is_amorphous:

                # dx_chi = contract('la,l->la', dxij, chi_k)
                # ddyn_s = 1j * contract('la,ibljc->ibjca', dx_chi, dynmat)

                # dyn_s = contract('ialjb,l->iajb', dynmat, chi_k)
                dyn_s = dynmat[:, :, 0, :, :]
                dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, geometry[:, np.newaxis, :] - geometry[np.newaxis, :, :])

                ddyn_s = contract('ija,ibjc->ibjca', dxij, dyn_s)
                ddyn_s = (ddyn_s + contract('ija,jcib->ibjca', dxij, dyn_s)) / 2.
                ddyn_s = (ddyn_s - ddyn_s.swapaxes(0, 2).swapaxes(1, 3)) / 2.

            else:
                chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
                # dx_chi = contract('la,l->la', dxij, chi_k)
                # ddyn_s = 1j * contract('la,ibljc->ibjca', dx_chi, dynmat)

                # dyn_s = contract('ialjb,l->iajb', dynmat, chi_k)
                dyn_s = contract('ialjb,ilj->iajb', dynmat, chi_k)
                dyn_s = (dyn_s + contract('jblia,ilj->iajb', dynmat, chi_k.conj())) / 2.
                dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, geometry[:, np.newaxis, np.newaxis, :] - (
                        geometry[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :]))

                ddyn_s = contract('ilja,ibljc,ilj->ibjca', dxij, dynmat, chi_k)
                ddyn_s = (ddyn_s - contract('ilja,jclib,ilj->ibjca', dxij, dynmat, chi_k.conj())) / 2.
                # ddyn_s = (ddyn_s  + contract('ilja,jclib,ilj->ibjca', dxij, dynmat, chi_k)) / 2.


        # dxij = apply_boundary_with_cell(replicated_cell, replicated_cell_inv, list_of_replicas)
        # kpoint = 2 * np.pi * (cell_inv).dot(qvec)
        # chi_k = np.exp(1j * dxij.dot(kpoint))
        # dyn_s_new = contract('ialjb,l->iajb', dynmat, chi_k)
        # dx_chi = contract('la,l->la', dxij, chi_k)
        # ddyn_s_new = 1j * contract('la,ibljc->ibjca', dx_chi, dynmat)
        # print(np.abs(ddyn_s_new - ddyn_s).sum())

        ddyn_s = ddyn_s.reshape((n_phonons, n_phonons, 3), order='C')
        dyn_s = dyn_s.reshape((n_phonons, n_phonons), order='C')
        out = DIAGONALIZATION_ALGORITHM(dyn_s)
        eigenvals, eigenvects = out[0], out[1]
        if IS_SORTING_EIGENVALUES:
            idx = eigenvals.argsort()
            eigenvals = eigenvals[idx]
            eigenvects = eigenvects[:, idx]

        frequencies = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
        # velocities = np.zeros((frequencies.shape[0], 3), dtype=np.complex)
        condition = frequencies > frequencies_threshold
        # for mu in range(n_particles * 3):
        #     if frequencies[mu] > frequencies_threshold:
        #         velocities[mu, :] = contract('i,ija,j->a', eigenvects[:, mu].conj(), ddyn_s, eigenvects[:, mu]) / (
        #                 2 * (2 * np.pi) * frequencies[mu])
        #


        velocities_AF = contract('im,ija,jn->mna', eigenvects[:, :].conj(), ddyn_s, eigenvects[:, :])
        velocities_AF = contract('mna,mn->mna', velocities_AF,
                                          1 / (2 * np.pi * np.sqrt(frequencies[:, np.newaxis]) * np.sqrt(frequencies[np.newaxis, :])))
        velocities_AF[np.invert(condition), :, :] = 0
        velocities_AF[:, np.invert(condition), :] = 0
        velocities_AF = velocities_AF / 2
        velocities = 1j * np.diagonal(velocities_AF).T
        # eigenvals_2 = np.zeros_like(eigenvals).astype(np.complex)
        # eigenvals_2[condition] = contract('im,ij,jm->m', eigenvects[:, condition].conj(), dyn_s, eigenvects[:, condition])
        # frequencies_2 = np.abs(eigenvals_2) ** .5 * np.sign(eigenvals_2) / (np.pi * 2.)

        # velocities2 = contract('ik,ija,jk,j->ja', eigenvects.conj(), ddyn_s, eigenvects,
        #                                1 / (2 * (2 * np.pi) * frequencies[:]))
                # eigenvals[mu] = np.real(contract('i,ij,j->', eigenvects[:, mu].conj(), dyn, eigenvects[:, mu]))

                # if (qvec == [0,0,0]).all():
        #     frequencies[:3] = 0.
        #     velocities[:3,:] = 0.
        # frequencies = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)

        return frequencies, eigenvals, eigenvects, velocities, velocities_AF



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
                            self._gamma_tensor[read_nu, read_nup] -= value
                            self._gamma_tensor[read_nu, read_nupp] += value
                        else:
                            self._gamma_tensor[read_nu, read_nup] += value
                            self._gamma_tensor[read_nu, read_nupp] += value

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
            if broadening == 'gauss':
                broadening_function = gaussian_delta
            elif broadening == 'lorentz':
                broadening_function = lorentzian_delta
            elif broadening == 'triangle':
                broadening_function = triangular_delta
            read_nu = read_nu + 1

            for nu_single in range(read_nu, self.n_phonons):
                index_k, mu = np.unravel_index(nu_single, [nptk, n_modes], order='C')

                if not file:
                    file = open(progress_filename, 'a+')
                if frequencies[index_k, mu] > frequencies_threshold:
                    gamma_out = calculate_single_gamma(is_plus, index_k, mu, i_kp_vec, index_kp_vec,
                                                       frequencies,
                                                       velocities, density,
                                                       cell_inv, k_size, n_modes, nptk, n_replicas,
                                                       rescaled_eigenvectors, chi, third_order, sigma_in,
                                                       frequencies_threshold, is_amorphous, broadening_function)

                    if gamma_out:
                        nup_vec, nupp_vec, pot_times_dirac, dirac = gamma_out
                        self._gamma[nu_single] += pot_times_dirac.sum()
                        self._ps[nu_single] += dirac.sum()

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
                        np.savetxt(file, np.vstack([nu_vec, gamma_out]).T, fmt='%i %i %i %.8e %.8e')
                        # except ValueError as err:
                        #     print(err)
                print(process_string[is_plus] + 'q-point = ' + str(index_k))
            file.close()


    def conductivity(self, mfp):
        volume = np.linalg.det(self.atoms.cell) / 1000
        frequencies = self.frequencies.reshape((self.n_phonons), order='C')
        physical_modes = (frequencies > self.energy_threshold)
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

    def calculate_conductivity_rta(self):
        volume = np.linalg.det(self.atoms.cell)
        gamma = self.gamma.reshape((self.n_k_points, self.n_modes)).copy()
        physical_modes = (self.frequencies > self.energy_threshold)
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
        physical_modes = (self.frequencies[:, :, np.newaxis] > self.energy_threshold) * (self.frequencies[:, np.newaxis, :] > self.energy_threshold)
        # lorentz = 2 / (gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :])
        lorentz = ((gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2) / (((gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2) ** 2 +
                                                                           ((omega[:, :, np.newaxis] - omega[:, np.newaxis, :]) / 2) ** 2)
        lorentz[np.invert(physical_modes)] = 0
        conductivity_per_mode = np.zeros((self.n_k_points, self.n_modes, self.n_modes, 3, 3))
        conductivity_per_mode[:, :, :, :, :] = contract('kn,knma,knm,knmb->knmab', self.c_v[:, :], self.velocities_AF[:, :, :, :], lorentz[:, :, :], self.velocities_AF[:, :, :, :])
        conductivity_per_mode = 1e22 / (volume * self.n_k_points) * conductivity_per_mode
        reduced_conductivity_per_mode = np.diagonal(conductivity_per_mode, axis1=1, axis2=2)
        # reduced_conductivity_per_mode = conductivity_per_mode.sum(axis=2)
        reduced_conductivity_per_mode = reduced_conductivity_per_mode.swapaxes(3, 2).swapaxes(2, 1)
        is_amorphous = (self.kpts == (1, 1, 1)).all()

        # if is_amorphous:
        temp = self.temperature
        delta = gamma
        omega = omega

        cnv = 47.992374
        x = omega / 2. / np.pi * cnv / temp
        expx = np.exp(x)
        cvx = x * x * expx / (expx - 1.0) ** 2
        cvqm = cvx.sum()/volume
        lorentz = 1 / np.pi * (gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2 / (((gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2) ** 2 +
                                                                           (omega[:, :, np.newaxis] - omega[:, np.newaxis, :]) ** 2)
        lorentz[np.invert(physical_modes)] = 0
        kappa = contract('ki,kija,kij,kija->a', cvx[:, :], self.velocities_AF[:, :, :, :], lorentz[:, :, :], self.velocities_AF[:, :, :, :])

        kboltz = 0.13806504
        kappa = kboltz / volume * np.pi * kappa

        kappa = np.mean(kappa)
        print(kappa, cvqm)
        return reduced_conductivity_per_mode.reshape((self.n_phonons, 3, 3))


    def calculate_second_k_list(self, k_list=None):
        if k_list is not None:
            k_points = k_list
        else:
            k_points = self.k_points
        atoms = self.atoms
        second_order = self.finite_difference.second_order.copy()
        list_of_replicas = self.list_of_replicas
        replicated_cell = self.replicated_cell
        frequencies_threshold = self.energy_threshold

        n_unit_cell = atoms.positions.shape[0]
        n_k_points = k_points.shape[0]

        frequencies = np.zeros((n_k_points, n_unit_cell * 3))
        eigenvalues = np.zeros((n_k_points, n_unit_cell * 3))
        eigenvectors = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3)).astype(np.complex)
        velocities = np.zeros((n_k_points, n_unit_cell * 3, 3))
        velocities_AF = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3, 3))

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

        for index_k in range(n_k_points):
            # freq, eval, evect, vels = self.diagonalize_second_order_single_k(k_points[index_k], dynmat,
            #                                                                  frequencies_threshold, dq=0.0001)
            freq, eval, evect, vels, vels_AF = self.diagonalize_second_order_single_k(k_points[index_k], dynmat,
                                                                             frequencies_threshold, dq=self.velocity_dq)
            frequencies[index_k, :] = freq
            eigenvalues[index_k, :] = eval
            eigenvectors[index_k, :, :] = evect
            velocities[index_k, :, :] = vels.real
            velocities_AF[index_k, : , :, :] = vels_AF.real

        # TODO: change the way we deal with two different outputs
        if k_list is not None:
            return frequencies, eigenvalues, eigenvectors, velocities, velocities_AF
        else:
            self.frequencies = frequencies
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            self.velocities = velocities
            self.velocities_AF = velocities_AF
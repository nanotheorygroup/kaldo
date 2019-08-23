import sparse
import scipy.special
import numpy as np
from opt_einsum import contract_expression
import ase.units as units
from .helper import timeit
from .helper import lazy_property
import os

DELTA_THRESHOLD = 2
IS_DELTA_CORRECTION_ENABLED = False
EVTOTENJOVERMOL = units.mol / (10 * units.J)
SCATTERING_MATRIX_FILE = 'scattering_matrix'
GAMMA_FILE = 'gamma.npy'
GAMMA_TENSOR_FILE = 'gamma_tensor.npy'
PS_FILE = 'phase_space.npy'

KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12



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




class AnharmonicController:
    def __init__(self, phonons):
        self.phonons = phonons
        folder_name = self.phonons.folder_name
        folder_name += '/' + str(self.phonons.temperature) + '/'
        if self.phonons.is_classic:
            folder_name += 'classic'
        else:
            folder_name += 'quantum'
        self.folder_name = folder_name
        self._ps = None
        self._gamma = None
        self._gamma_tensor = None



    @lazy_property(is_storing=True)
    def occupations(self):
        occupations =  self.calculate_occupations()
        return occupations


    @lazy_property(is_storing=True)
    def c_v(self):
        c_v =  self.calculate_c_v()
        return c_v

    @lazy_property(is_storing=True)
    def gamma_sparse_plus(self):
        gamma_data =  self.calculate_gamma_sparse(is_plus=True)
        return gamma_data

    @lazy_property(is_storing=True)
    def gamma_sparse_minus(self):
        gamma_data =  self.calculate_gamma_sparse(is_plus=False)
        return gamma_data

    @lazy_property(is_storing=False)
    def rescaled_eigenvectors(self):
        # TODO: this doesn't need to be saved
        rescaled_eigenvectors = self.calculate_rescaled_eigenvectors()
        return rescaled_eigenvectors

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
        np.save (folder + PS_FILE, new_ps)
        self._ps = new_ps

    @property
    def gamma_tensor(self):
        return self._gamma_tensor

    @gamma_tensor.getter
    def gamma_tensor(self):
        if self._gamma_tensor is None:

            try:
                folder = self.folder_name
                folder += '/'
                self._gamma_tensor = np.load (folder + GAMMA_TENSOR_FILE)
            except FileNotFoundError as e:
                print(e)
                self.calculate_gamma(is_gamma_tensor_enabled=True)

        return  self._gamma_tensor

    @gamma_tensor.setter
    def gamma_tensor(self, new_gamma_tensor):
        folder = self.folder_name
        folder += '/'
        np.save (folder + GAMMA_TENSOR_FILE, new_gamma_tensor)
        self._gamma_tensor = new_gamma_tensor

    def calculate_rescaled_eigenvectors(self):
        n_particles = self.phonons.atoms.positions.shape[0]
        n_modes = self.phonons.n_modes
        masses = self.phonons.atoms.get_masses()
        rescaled_eigenvectors = self.phonons.eigenvectors[:, :, :].reshape(
            (self.phonons.n_k_points, n_particles, 3, n_modes), order='C') / np.sqrt(
            masses[np.newaxis, :, np.newaxis, np.newaxis])
        rescaled_eigenvectors = rescaled_eigenvectors.reshape((self.phonons.n_k_points, n_particles * 3, n_modes),
                                                              order='C')
        rescaled_eigenvectors = rescaled_eigenvectors.swapaxes(1, 2)
        rescaled_eigenvectors = rescaled_eigenvectors.reshape((self.phonons.n_k_points, n_modes, n_modes), order='C')
        return rescaled_eigenvectors

    def calculate_occupations(self):
        frequencies = self.phonons.frequencies
        temp = self.phonons.temperature * KELVINTOTHZ
        density = np.zeros_like(frequencies)
        physical_modes = frequencies > self.phonons.frequency_threshold
        if self.phonons.is_classic is False:
            density[physical_modes] = 1. / (np.exp(frequencies[physical_modes] / temp) - 1.)
        else:
            density[physical_modes] = temp / frequencies[physical_modes]
        return density


    def calculate_c_v(self):
        frequencies = self.phonons.frequencies
        c_v = np.zeros_like (frequencies)
        physical_modes = frequencies > self.phonons.frequency_threshold
        temperature = self.phonons.temperature * KELVINTOTHZ

        if (self.phonons.is_classic):
            c_v[physical_modes] = KELVINTOJOULE
        else:
            f_be = self.occupations
            c_v[physical_modes] = KELVINTOJOULE * f_be[physical_modes] * (f_be[physical_modes] + 1) * self.phonons.frequencies[physical_modes] ** 2 / \
                                  (temperature ** 2)
        return c_v


    def calculate_gamma(self, is_gamma_tensor_enabled=False):
        n_particles = self.phonons.atoms.positions.shape[0]
        n_modes = n_particles * 3
        k_size = self.phonons.kpts
        nptk = np.prod(k_size)

        n_phonons = self.phonons.n_phonons
        self._gamma = np.zeros(n_phonons)
        self._ps = np.zeros(n_phonons)
        if is_gamma_tensor_enabled:
            self.gamma_tensor = np.zeros((n_phonons, n_phonons))

        for is_plus in (1, 0):
            if is_plus:
                gamma_data = self.gamma_sparse_plus
            else:
                gamma_data = self.gamma_sparse_minus
            index_k, mu, index_kp, mup, index_kpp, mupp, pot_times_dirac, dirac = gamma_data

            nu_vec = np.ravel_multi_index(np.array([index_k, mu], dtype=int),
                                           np.array([nptk, n_modes]), order='C')
            nup_vec = np.ravel_multi_index(np.array([index_kp, mup], dtype=int),
                                           np.array([nptk, n_modes]), order='C')
            nupp_vec = np.ravel_multi_index(np.array([index_kpp, mupp], dtype=int),
                                            np.array([nptk, n_modes]), order='C')
            gamma_sparse = sparse.COO([nu_vec, nup_vec, nupp_vec], pot_times_dirac, (n_phonons, n_phonons, n_phonons))
            ps_sparse = sparse.COO([nu_vec, nup_vec, nupp_vec], dirac, (n_phonons, n_phonons, n_phonons))
            self._gamma += gamma_sparse.sum(axis=2).sum(axis=1).todense()
            self._ps += ps_sparse.sum(axis=2).sum(axis=1).todense()
            if is_gamma_tensor_enabled:
                if is_plus:
                    self.gamma_tensor -= gamma_sparse.sum(axis=2).todense()
                    self.gamma_tensor += gamma_sparse.sum(axis=1).todense()
                else:
                    self.gamma_tensor += gamma_sparse.sum(axis=2).todense()
                    self.gamma_tensor += gamma_sparse.sum(axis=1).todense()


    @timeit
    def calculate_gamma_sparse(self, is_plus):

        folder = self.phonons.folder_name
        if self.phonons.sigma_in is not None:
            folder += 'sigma_in_' + str(self.phonons.sigma_in).replace('.', '_') + '/'
        is_plus_label = ['_0', '_1']
        progress_filename = folder + '/' + SCATTERING_MATRIX_FILE + is_plus_label[is_plus]
        gamma_data = None
        try:
            gamma_data = np.loadtxt(progress_filename)
        except IOError:
            print('No gamma partial file found. Calculating')

        n_particles = self.phonons.atoms.positions.shape[0]

        print('Lifetime calculation')

        # TODO: We should write this in a better way
        if self.phonons.list_of_replicas.shape == (3,):
            n_replicas = 1
        else:
            n_replicas = self.phonons.list_of_replicas.shape[0]


        is_amorphous = (self.phonons.kpts == (1, 1, 1)).all()

        if is_amorphous:
            chi = 1
        else:
            chi = np.zeros((self.phonons.n_k_points, n_replicas), dtype=np.complex)
            dxij = self.phonons.apply_boundary_with_cell(self.phonons.list_of_replicas)

            for index_k in range(self.phonons.n_k_points):
                i_k = np.array(np.unravel_index(index_k, self.phonons.kpts, order='C'))

                #TODO: Is the following division correct? Should we unravel instead
                k_point = i_k / self.phonons.kpts
                realq = np.matmul(self.phonons.cell_inv * 2 * np.pi, k_point)
                chi[index_k] = np.exp(1j * dxij.dot(realq))

        print('Projection started')
        n_modes = n_particles * 3
        index_kp_vec = np.arange(self.phonons.n_k_points)
        i_kp_vec = np.array(np.unravel_index(index_kp_vec, self.phonons.kpts, order='C'))

        if gamma_data is not None:
            initial_k = int(gamma_data[-1, 0])
            initial_mu = int(gamma_data[-1, 1])
            if initial_mu == n_modes - 1:
                initial_k += 1
            else:
                initial_mu += 1
        else:
            initial_k = 0
            initial_mu = 0

        for index_k in range(initial_k, self.phonons.n_k_points):
            i_k = np.array(np.unravel_index(index_k, self.phonons.kpts, order='C'))
            i_kpp_vec = i_k[:, np.newaxis] + (int(is_plus) * 2 - 1) * i_kp_vec[:, :]
            index_kpp_vec = np.ravel_multi_index(i_kpp_vec, self.phonons.kpts, order='C', mode='wrap')
            if is_plus:
                first_evect = self.rescaled_eigenvectors
                first_chi = chi
            else:
                first_evect = self.rescaled_eigenvectors.conj()
                first_chi = chi.conj()
            second_evect = self.rescaled_eigenvectors.conj()[index_kpp_vec]
            second_chi = chi.conj()[index_kpp_vec]
            if self.phonons.sigma_in is None:
                sigma_small = self.calculate_broadening(index_kp_vec, index_kpp_vec)
            else:
                sigma_small = self.phonons.sigma_in
            for mu in range(n_modes):
                if index_k == initial_k and mu < initial_mu:
                    break
                nu_single = np.ravel_multi_index([index_k, mu], [self.phonons.n_k_points, n_modes], order='C')
                if self.phonons.frequencies[index_k, mu] > self.phonons.frequency_threshold:
                    scaled_potential = sparse.tensordot(self.phonons.finite_difference.third_order,
                                                        self.rescaled_eigenvectors.reshape((self.phonons.n_k_points * n_modes, n_modes),order='C')[nu_single, :], (0, 0))
                    scaled_potential = scaled_potential.reshape((n_replicas, n_modes, n_replicas, n_modes),
                                                                order='C')
                    gamma_out = self.calculate_single_gamma(is_plus, index_k, mu, index_kp_vec, index_kpp_vec,
                                                            first_evect, second_evect, first_chi, second_chi,
                                                            scaled_potential, sigma_small)
                    if gamma_out is not None:
                        k_vec = np.ones(gamma_out.shape[0]).astype(int) * index_k
                        mu_vec = np.ones(gamma_out.shape[0]).astype(int) * mu
                        gamma_data_to_append = np.vstack([k_vec, mu_vec, gamma_out.T]).T
                        if gamma_data is None:
                            gamma_data = gamma_data_to_append
                        else:
                            gamma_data = np.append(gamma_data, gamma_data_to_append, axis=0)
                        np.savetxt(progress_filename, gamma_data_to_append, fmt='%i %i %i %i %i %i %.8e %.8e')
        os.remove(progress_filename)
        return gamma_data.T


    def calculate_single_gamma(self, is_plus, index_k, mu, index_kp_full, kpp_mapping, first_evect, second_evect, first_chi, second_chi, scaled_potential, sigma_small):
        frequencies = self.phonons.frequencies
        density = self.phonons.occupations
        frequencies_threshold = self.phonons.frequency_threshold
        if self.phonons.broadening_shape == 'gauss':
            broadening_function = gaussian_delta
        elif self.phonons.broadening_shape == 'lorentz':
            broadening_function = lorentzian_delta
        elif self.phonons.broadening_shape == 'triangle':
            broadening_function = triangular_delta
        else:
            raise TypeError('Broadening shape not supported')

        second_sign = (int(is_plus) * 2 - 1)
        omegas = 2 * np.pi * self.phonons.frequencies
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
            pot_times_dirac = units._hbar * np.pi / 4. * pot_times_dirac / omegas[index_k, mu] / self.phonons.n_k_points * gammatothz

            return np.vstack([index_kp_vec, mup_vec, index_kpp_vec, mupp_vec, pot_times_dirac, dirac_delta]).T

    def calculate_broadening(self, index_kp_vec, index_kpp_vec):
        cellinv = self.phonons.cell_inv
        k_size = self.phonons.kpts
        velocity = self.phonons.velocities[index_kp_vec, :, np.newaxis, :] - self.phonons.velocities[index_kpp_vec, np.newaxis, :, :]
        # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
        delta_k = cellinv / k_size
        base_sigma = ((np.tensordot(velocity, delta_k, [-1, 1])) ** 2).sum(axis=-1)
        base_sigma = np.sqrt(base_sigma / 6.)
        return base_sigma

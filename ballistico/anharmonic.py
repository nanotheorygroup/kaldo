import sparse
import scipy.special
import numpy as np
from opt_einsum import contract
import ase.units as units
from .helper import timeit, lazy_property, is_calculated
from .harmonic import Harmonic
import tensorflow as tf


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




class Anharmonic(Harmonic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_classic = bool(kwargs['is_classic'])
        self.temperature = float(kwargs['temperature'])
        if 'sigma_in' in kwargs:
            self.sigma_in = kwargs['sigma_in']
        else:
            self.sigma_in = None
        if 'broadening_shape' in kwargs:
            self.broadening_shape = kwargs['broadening_shape']
        else:
            self.broadening_shape = 'gauss'


    @lazy_property(is_storing=True, is_reduced_path=False)
    def occupations(self):
        occupations =  self.calculate_occupations()
        return occupations


    @lazy_property(is_storing=True, is_reduced_path=False)
    def c_v(self):
        c_v =  self.calculate_c_v()
        return c_v


    @lazy_property(is_storing=True, is_reduced_path=False)
    def ps_and_gamma(self):
        if is_calculated('ps_gamma_and_gamma_tensor', self):
            ps_and_gamma = self.ps_gamma_and_gamma_tensor[:, :2]
        else:
            ps_and_gamma = self.calculate_gamma_sparse(is_gamma_tensor_enabled=False)
        return ps_and_gamma


    @lazy_property(is_storing=True, is_reduced_path=False)
    def ps_gamma_and_gamma_tensor(self):
        ps_gamma_and_gamma_tensor = self.calculate_gamma_sparse(is_gamma_tensor_enabled=True)
        return ps_gamma_and_gamma_tensor


    @lazy_property(is_storing=False, is_reduced_path=False)
    def rescaled_eigenvectors(self):
        rescaled_eigenvectors = self.calculate_rescaled_eigenvectors()
        return rescaled_eigenvectors


    @property
    def gamma_tensor(self):
        gamma_tensor = self.ps_gamma_and_gamma_tensor[:, 2:]
        return gamma_tensor


    @property
    def gamma(self):
        gamma = self.ps_and_gamma[:, 1]
        return gamma


    @property
    def ps(self):
        ps = self.ps_and_gamma[:, 0]
        return ps


    def calculate_rescaled_eigenvectors(self):
        n_particles = self.atoms.positions.shape[0]
        n_modes = self.n_modes
        masses = self.atoms.get_masses()
        rescaled_eigenvectors = self.eigenvectors[:, :, :].reshape(
            (self.n_k_points, n_particles, 3, n_modes), order='C') / np.sqrt(
            masses[np.newaxis, :, np.newaxis, np.newaxis])
        rescaled_eigenvectors = rescaled_eigenvectors.reshape((self.n_k_points, n_particles * 3, n_modes),
                                                              order='C')
        rescaled_eigenvectors = rescaled_eigenvectors.swapaxes(1, 2)
        rescaled_eigenvectors = rescaled_eigenvectors.reshape((self.n_k_points, n_modes, n_modes), order='C')
        return rescaled_eigenvectors


    def calculate_occupations(self):
        frequencies = self.frequencies
        temp = self.temperature * KELVINTOTHZ
        density = np.zeros_like(frequencies)
        physical_modes = frequencies > self.frequency_threshold
        if self.is_classic is False:
            density[physical_modes] = 1. / (np.exp(frequencies[physical_modes] / temp) - 1.)
        else:
            density[physical_modes] = temp / frequencies[physical_modes]
        return density


    def calculate_c_v(self):
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
        return c_v


    def calculate_c_v_2d(self):
        frequencies = self.frequencies
        c_v = np.zeros((self.n_k_points, self.n_modes, self.n_modes))
        temperature = self.temperature * KELVINTOTHZ
        physical_modes = frequencies > self.frequency_threshold

        if (self.is_classic):
            c_v[:, :, :] = KELVINTOJOULE
        else:
            f_be = self.occupations
            c_v_omega = KELVINTOJOULE * f_be * (f_be + 1) * frequencies / (temperature ** 2)
            c_v_omega[np.invert(physical_modes)] = 0
            freq_sq = (frequencies[:, :, np.newaxis] + frequencies[:, np.newaxis, :]) / 2 * (c_v_omega[:, :, np.newaxis] + c_v_omega[:, np.newaxis, :]) / 2
            c_v[:, :, :] = freq_sq
        return c_v


    @timeit
    def calculate_gamma_sparse(self, is_gamma_tensor_enabled=False):
        # The ps and gamma matrix stores ps, gamma and then the scattering matrix
        if is_gamma_tensor_enabled:
            ps_and_gamma = np.zeros((self.n_phonons, 2 + self.n_phonons))
        else:
            ps_and_gamma = np.zeros((self.n_phonons, 2))
        for is_plus in (1, 0):
            n_particles = self.atoms.positions.shape[0]
            print('Lifetime calculation')
            n_replicas = self.n_replicas
            if self.is_amorphous:
                # TODO: change this definition
                chi = np.array([1j])
            else:
                chi = np.zeros((self.n_k_points, n_replicas), dtype=np.complex)
                for index_k in range(self.n_k_points):
                    k_point = self.k_points[index_k]
                    chi[index_k] = self.chi(k_point)

            print('Projection started')
            n_modes = n_particles * 3
            index_kp_vec = np.arange(self.n_k_points)
            i_kp_vec = np.array(np.unravel_index(index_kp_vec, self.kpts, order='C'))
            for index_k in range(self.n_k_points):
                i_k = np.array(np.unravel_index(index_k, self.kpts, order='C'))
                i_kpp_vec = i_k[:, np.newaxis] + (int(is_plus) * 2 - 1) * i_kp_vec[:, :]
                index_kpp_vec = np.ravel_multi_index(i_kpp_vec, self.kpts, order='C', mode='wrap')
                if is_plus:
                    first_evect = self.rescaled_eigenvectors
                    first_chi = chi
                else:
                    first_evect = self.rescaled_eigenvectors.conj()
                    first_chi = chi.conj()
                second_evect = self.rescaled_eigenvectors.conj()[index_kpp_vec]
                second_chi = chi.conj()[index_kpp_vec]
                if self.sigma_in is None:
                    sigma_small = self.calculate_broadening(index_kp_vec, index_kpp_vec)
                else:
                    sigma_small = self.sigma_in
                for mu in range(n_modes):
                    nu_single = np.ravel_multi_index([index_k, mu], [self.n_k_points, n_modes], order='C')
                    if nu_single % 200 == 0 or self.is_amorphous:
                        print('calculating third', nu_single, np.round(nu_single/self.n_phonons, 2) * 100, '%')
                    if self.frequencies[index_k, mu] > self.frequency_threshold:

                        out = self.calculate_single_gamma(is_plus, index_k, mu, index_kp_vec, index_kpp_vec,
                                                          first_evect, second_evect, first_chi, second_chi,
                                                          sigma_small, is_gamma_tensor_enabled)
                        if out is not None:
                            ps_and_gamma[nu_single] += out
        return ps_and_gamma


    def calculate_single_gamma(self, is_plus, index_k, mu, index_kp_full, kpp_mapping, first_evect, second_evect, first_chi, second_chi, sigma_small, is_gamma_tensor_enabled):
        nu_single = np.ravel_multi_index([index_k, mu], [self.n_k_points, self.n_modes], order='C')
        frequencies = self.frequencies
        density = self.occupations
        frequencies_threshold = self.frequency_threshold
        if self.broadening_shape == 'gauss':
            broadening_function = gaussian_delta
        elif self.broadening_shape == 'lorentz':
            broadening_function = lorentzian_delta
        elif self.broadening_shape == 'triangle':
            broadening_function = triangular_delta
        else:
            raise TypeError('Broadening shape not supported')

        second_sign = (int(is_plus) * 2 - 1)
        omegas = 2 * np.pi * self.frequencies
        omegas_difference = np.abs(omegas[index_k, mu] + second_sign * omegas[index_kp_full, :, np.newaxis] -
                                   omegas[kpp_mapping, np.newaxis, :])

        condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_small) & \
                    (frequencies[index_kp_full, :, np.newaxis] > frequencies_threshold) & \
                    (frequencies[kpp_mapping, np.newaxis, :] > frequencies_threshold)
        interactions = np.array(np.where(condition)).T

        # TODO: Benchmark something fast like
        # interactions = np.array(np.unravel_index (np.flatnonzero (condition), condition.shape)).T
        if interactions.size != 0:
            zeroth_evect = self.rescaled_eigenvectors.reshape((self.n_k_points * self.n_modes, self.n_modes), order='C')[
                           nu_single, :]
            scaled_potential = sparse.tensordot(self.finite_difference.third_order, zeroth_evect, (0, 0))
            # Create sparse index
            index_kp_vec = interactions[:, 0]
            index_kpp_vec = kpp_mapping[index_kp_vec]
            mup_vec = interactions[:, 1]
            mupp_vec = interactions[:, 2]
            if is_plus:
                dirac_delta = density[index_kp_vec, mup_vec] - density[index_kpp_vec, mupp_vec]
                # dirac_delta = density[index_k, mu] * density[index_kp_vec, mup_vec] * (
                #             density[index_kpp_vec, mupp_vec] + 1)

            else:
                dirac_delta = .5 * (1 + density[index_kp_vec, mup_vec] + density[index_kpp_vec, mupp_vec])
                # dirac_delta = .5 * density[index_k, mu] * (density[index_kp_vec, mup_vec] + 1) * (
                #             density[index_kpp_vec, mupp_vec] + 1)

            dirac_delta /= (omegas[index_kp_vec, mup_vec] * omegas[index_kpp_vec, mupp_vec])
            if np.array(sigma_small).size == 1:
                dirac_delta *= broadening_function(
                    [omegas_difference[index_kp_vec, mup_vec, mupp_vec], 2 * np.pi * sigma_small])
            else:
                dirac_delta *= broadening_function(
                    [omegas_difference[index_kp_vec, mup_vec, mupp_vec], 2 * np.pi * sigma_small[
                        index_kp_vec, mup_vec, mupp_vec]])
            if self.is_amorphous:
                scaled_potential = contract('ij,ni,mj->nm', scaled_potential.real, first_evect[0].real,
                                            second_evect[0].real, optimize='optimal')
                scaled_potential = scaled_potential[np.newaxis, ...]

            else:
                scaled_potential = scaled_potential.reshape(
                    (self.n_replicas, self.n_modes, self.n_replicas, self.n_modes),
                    order='C')
                scaled_potential = contract('litj,kni,kl,kmj,kt->knm', scaled_potential, first_evect, first_chi,
                                            second_evect, second_chi)
            scaled_potential = scaled_potential[index_kp_vec, mup_vec, mupp_vec]
            pot_times_dirac = np.abs(scaled_potential) ** 2 * dirac_delta

            #TODO: move units conversion somewhere else
            gammatothz = 1e11 * units.mol * EVTOTENJOVERMOL ** 2
            pot_times_dirac = units._hbar * np.pi / 4. * pot_times_dirac / omegas[index_k, mu] / self.n_k_points * gammatothz

            nup_vec = np.ravel_multi_index(np.array([index_kp_vec, mup_vec], dtype=int),
                                           np.array([self.n_k_points, self.n_modes]), order='C')
            nupp_vec = np.ravel_multi_index(np.array([index_kpp_vec, mupp_vec], dtype=int),
                                            np.array([self.n_k_points, self.n_modes]), order='C')

            # The ps and gamma array stores first ps then gamma then the scattering array
            if is_gamma_tensor_enabled:
                ps_and_gamma_sparse = np.zeros(2 + self.n_phonons)
                # We need to use bincount together with fancy indexing here. See:
                # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
                if is_plus:
                    result = np.bincount(nup_vec, pot_times_dirac, self.n_phonons)
                    ps_and_gamma_sparse[2:] -= result
                else:
                    result = np.bincount(nup_vec, pot_times_dirac, self.n_phonons)
                    ps_and_gamma_sparse[2:] += result

                result = np.bincount(nupp_vec, pot_times_dirac, self.n_phonons)
                ps_and_gamma_sparse[2:] += result
            else:
                ps_and_gamma_sparse = np.zeros(2)
            ps_and_gamma_sparse[0] = dirac_delta.sum()
            ps_and_gamma_sparse[1] = pot_times_dirac.sum()
            return ps_and_gamma_sparse

    def calculate_broadening(self, index_kp_vec, index_kpp_vec):
        cellinv = self.cell_inv
        k_size = self.kpts
        velocity = self.velocities[index_kp_vec, :, np.newaxis, :] - self.velocities[index_kpp_vec, np.newaxis, :, :]
        # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
        delta_k = cellinv / k_size
        base_sigma = ((np.tensordot(velocity, delta_k, [-1, 1])) ** 2).sum(axis=-1)
        base_sigma = np.sqrt(base_sigma / 6.)
        return base_sigma

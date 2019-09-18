import sparse
import scipy.special
import numpy as np
from opt_einsum import contract
import ase.units as units
from .helper import timeit, lazy_property, is_calculated

DELTA_THRESHOLD = 2
IS_DELTA_CORRECTION_ENABLED = False
EVTOTENJOVERMOL = units.mol / (10 * units.J)
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





def calculate_rescaled_eigenvectors(phonons):
    n_particles = phonons.atoms.positions.shape[0]
    n_modes = phonons.n_modes
    masses = phonons.atoms.get_masses()
    rescaled_eigenvectors = phonons.eigenvectors[:, :, :].reshape(
        (phonons.n_k_points, n_particles, 3, n_modes), order='C') / np.sqrt(
        masses[np.newaxis, :, np.newaxis, np.newaxis])
    rescaled_eigenvectors = rescaled_eigenvectors.reshape((phonons.n_k_points, n_particles * 3, n_modes),
                                                          order='C')
    rescaled_eigenvectors = rescaled_eigenvectors.swapaxes(1, 2)
    rescaled_eigenvectors = rescaled_eigenvectors.reshape((phonons.n_k_points, n_modes, n_modes), order='C')
    return rescaled_eigenvectors


def calculate_occupations(phonons):
    frequencies = phonons.frequencies
    temp = phonons.temperature * KELVINTOTHZ
    density = np.zeros_like(frequencies)
    physical_modes = frequencies > phonons.frequency_threshold
    if phonons.is_classic is False:
        density[physical_modes] = 1. / (np.exp(frequencies[physical_modes] / temp) - 1.)
    else:
        density[physical_modes] = temp / frequencies[physical_modes]
    return density


def calculate_c_v(phonons):
    frequencies = phonons.frequencies
    c_v = np.zeros_like (frequencies)
    physical_modes = frequencies > phonons.frequency_threshold
    temperature = phonons.temperature * KELVINTOTHZ

    if (phonons.is_classic):
        c_v[physical_modes] = KELVINTOJOULE
    else:
        f_be = phonons.occupations
        c_v[physical_modes] = KELVINTOJOULE * f_be[physical_modes] * (f_be[physical_modes] + 1) * phonons.frequencies[physical_modes] ** 2 / \
                              (temperature ** 2)
    return c_v


def calculate_dirac_delta(phonons, index_k, mu, is_plus):
    if phonons.frequencies[index_k, mu] <= phonons.frequency_threshold:
        return None

    frequencies = phonons.frequencies
    density = phonons.occupations
    frequencies_threshold = phonons.frequency_threshold
    if phonons.broadening_shape == 'gauss':
        broadening_function = gaussian_delta
    elif phonons.broadening_shape == 'lorentz':
        broadening_function = lorentzian_delta
    elif phonons.broadening_shape == 'triangle':
        broadening_function = triangular_delta
    else:
        raise TypeError('Broadening shape not supported')
    index_kp_full = np.arange(phonons.n_k_points)
    i_kp_vec = np.array(np.unravel_index(index_kp_full, phonons.kpts, order='C'))
    i_k = np.array(np.unravel_index(index_k, phonons.kpts, order='C'))
    i_kpp_vec = i_k[:, np.newaxis] + (int(is_plus) * 2 - 1) * i_kp_vec[:, :]
    index_kpp_full = np.ravel_multi_index(i_kpp_vec, phonons.kpts, order='C', mode='wrap')
    if phonons.sigma_in is None:
        sigma_small = calculate_broadening(phonons, index_kp_full, index_kpp_full)
    else:
        sigma_small = phonons.sigma_in

    second_sign = (int(is_plus) * 2 - 1)
    omegas = phonons.omegas
    omegas_difference = np.abs(
        omegas[index_k, mu] + second_sign * omegas[index_kp_full, :, np.newaxis] -
        omegas[index_kpp_full, np.newaxis, :])

    condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_small) & \
                (frequencies[index_kp_full, :, np.newaxis] > frequencies_threshold) & \
                (frequencies[index_kpp_full, np.newaxis, :] > frequencies_threshold)
    interactions = np.array(np.where(condition)).T

    # TODO: Benchmark something fast like
    # interactions = np.array(np.unravel_index (np.flatnonzero (condition), condition.shape)).T
    if interactions.size == 0:
        return None
    # Create sparse index
    index_kp_vec = interactions[:, 0]
    index_kpp_vec = index_kpp_full[index_kp_vec]
    mup_vec = interactions[:, 1]
    mupp_vec = interactions[:, 2]
    if is_plus:
        dirac_delta = density[index_kp_vec, mup_vec] - density[index_kpp_vec, mupp_vec]
        # dirac_delta = density[index_k, mu] * density[index_kp_vec, mup_vec] * (
        #             density[index_kpp_vec, mupp_vec] + 1)

    else:
        dirac_delta = .5 * (
                1 + density[index_kp_vec, mup_vec] + density[index_kpp_vec, mupp_vec])
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

    index_kp = index_kp_vec
    mup = mup_vec
    index_kpp = index_kpp_vec
    mupp = mupp_vec
    current_delta = dirac_delta
    return current_delta, index_kp, mup, index_kpp, mupp

def calculate_dirac_delta_amorphous(phonons, mu):

    frequencies = phonons.frequencies
    density = phonons.occupations
    frequencies_threshold = phonons.frequency_threshold
    if phonons.broadening_shape == 'gauss':
        broadening_function = gaussian_delta
    elif phonons.broadening_shape == 'lorentz':
        broadening_function = lorentzian_delta
    elif phonons.broadening_shape == 'triangle':
        broadening_function = triangular_delta
    else:
        raise TypeError('Broadening shape not supported')

    for is_plus in [1, 0]:
        sigma_small = phonons.sigma_in

        if phonons.frequencies[0, mu] > phonons.frequency_threshold:

            second_sign = (int(is_plus) * 2 - 1)
            omegas = phonons.omegas
            omegas_difference = np.abs(
                omegas[0, mu] + second_sign * omegas[0, :, np.newaxis] -
                omegas[0, np.newaxis, :])

            condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_small) & \
                        (frequencies[0, :, np.newaxis] > frequencies_threshold) & \
                        (frequencies[0, np.newaxis, :] > frequencies_threshold)
            interactions = np.array(np.where(condition)).T

            # TODO: Benchmark something fast like
            # interactions = np.array(np.unravel_index (np.flatnonzero (condition), condition.shape)).T
            if interactions.size != 0:
                # Create sparse index

                mup_vec = interactions[:, 0]
                mupp_vec = interactions[:, 1]
                if is_plus:
                    dirac_delta = density[0, mup_vec] - density[0, mupp_vec]
                    # dirac_delta = density[0, mu] * density[0, mup_vec] * (
                    #             density[0, mupp_vec] + 1)

                else:
                    dirac_delta = .5 * (
                            1 + density[0, mup_vec] + density[0, mupp_vec])
                    # dirac_delta = .5 * density[0, mu] * (density[0, mup_vec] + 1) * (
                    #             density[0, mupp_vec] + 1)

                dirac_delta /= (omegas[0, mup_vec] * omegas[0, mupp_vec])
                dirac_delta *= broadening_function(
                    [omegas_difference[mup_vec, mupp_vec], 2 * np.pi * sigma_small])

                try:
                    mup = np.concatenate([mup, mup_vec])
                    mupp = np.concatenate([mupp, mupp_vec])
                    current_delta = np.concatenate([current_delta, dirac_delta])
                except NameError:
                    mup = mup_vec
                    mupp = mupp_vec
                    current_delta = dirac_delta
    try:
        return current_delta, mup, mupp
    except:
        return None

@timeit
def calculate_gamma_sparse(phonons, is_gamma_tensor_enabled=False):
    # The ps and gamma matrix stores ps, gamma and then the scattering matrix
    if is_gamma_tensor_enabled:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2 + phonons.n_phonons))
    else:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2))
    print('Projection started')
    if phonons.is_amorphous:
        print_every = 1
        project = project_amorphous
    else:
        print_every = 200
        project = project_crystal
    for nu_single in range(phonons.n_phonons):
        if nu_single % print_every == 0:
            print('calculating third', nu_single, np.round(nu_single / phonons.n_phonons, 2) * 100,
                  '%')
        potential_times_evect = sparse.tensordot(phonons.finite_difference.third_order,
                                            phonons.rescaled_eigenvectors.reshape(
                                                (phonons.n_k_points * phonons.n_modes, phonons.n_modes),
                                                order='C')[nu_single, :], (0, 0))

        ps_and_gamma[nu_single] = project(phonons, potential_times_evect, nu_single, is_gamma_tensor_enabled)
        ps_and_gamma[nu_single, 1:] /= phonons.frequencies.flatten()[nu_single]
    return ps_and_gamma


def project_amorphous(phonons, potential_times_evect, mu, is_gamma_tensor_enabled=False):
    if is_gamma_tensor_enabled==True:
        raise ValueError('is_gamma_tensor_enabled=True not supported')
    ps_and_gamma_sparse = np.zeros(2)
    out = calculate_dirac_delta_amorphous(phonons, mu)
    if not out:
        return ps_and_gamma_sparse
    dirac_delta, mup_vec, mupp_vec = out

    scaled_potential = contract('ij,ni,mj->nm', potential_times_evect.real, phonons.rescaled_eigenvectors[0].real,
                                phonons.rescaled_eigenvectors[0].real,
                                optimize='optimal')
    scaled_potential = scaled_potential[np.newaxis, ...]
    scaled_potential = scaled_potential[0, mup_vec, mupp_vec]
    pot_times_dirac = np.abs(scaled_potential) ** 2 * dirac_delta

    # TODO: move units conversion somewhere else
    gammatothz = 1e11 * units.mol * EVTOTENJOVERMOL ** 2
    pot_times_dirac = units._hbar / 8. * pot_times_dirac / phonons.n_k_points * gammatothz
    ps_and_gamma_sparse[0] = dirac_delta.sum()
    ps_and_gamma_sparse[1] = pot_times_dirac.sum()
    return ps_and_gamma_sparse

def project_crystal(phonons, potential_times_evect, nu_single, is_gamma_tensor_enabled=False):
    index_k, mu = np.unravel_index(nu_single, (phonons.n_k_points, phonons.n_modes), order='C')
    if is_gamma_tensor_enabled:
        ps_and_gamma_sparse = np.zeros(2 + phonons.n_phonons)
    else:
        ps_and_gamma_sparse = np.zeros(2)
    for is_plus in (1, 0):
        out = calculate_dirac_delta(phonons, index_k, mu, is_plus)
        if not out:
            continue
        potential_times_evect = potential_times_evect.reshape(
            (phonons.n_replicas, phonons.n_modes, phonons.n_replicas, phonons.n_modes),
            order='C')

        dirac_delta, index_kp_vec, mup_vec, index_kpp_vec, mupp_vec = out

        # The ps and gamma array stores first ps then gamma then the scattering array

        if is_plus:

            scaled_potential = contract('litj,ai,al,aj,at->a', potential_times_evect,
                                        phonons.rescaled_eigenvectors[index_kp_vec, mup_vec],
                                        phonons.chi_k[index_kp_vec],
                                        phonons.rescaled_eigenvectors.conj()[index_kpp_vec, mupp_vec],
                                        phonons.chi_k[index_kpp_vec].conj())
        else:

            scaled_potential = contract('litj,ai,al,aj,at->a', potential_times_evect,
                                        phonons.rescaled_eigenvectors.conj()[index_kp_vec, mup_vec],
                                        phonons.chi_k[index_kp_vec].conj(),
                                        phonons.rescaled_eigenvectors.conj()[index_kpp_vec, mupp_vec],
                                        phonons.chi_k[index_kpp_vec].conj())
        pot_times_dirac = np.abs(scaled_potential) ** 2 * dirac_delta

        # TODO: move units conversion somewhere else
        gammatothz = 1e11 * units.mol * EVTOTENJOVERMOL ** 2
        pot_times_dirac = units._hbar / 8. * pot_times_dirac / phonons.n_k_points * gammatothz

        if is_gamma_tensor_enabled:
            # We need to use bincount together with fancy indexing here. See:
            # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
            nup_vec = np.ravel_multi_index(np.array([index_kp_vec, mup_vec], dtype=int),
                                           np.array([phonons.n_k_points, phonons.n_modes]), order='C')
            nupp_vec = np.ravel_multi_index(np.array([index_kpp_vec, mupp_vec], dtype=int),
                                            np.array([phonons.n_k_points, phonons.n_modes]), order='C')

            result = np.bincount(nup_vec, pot_times_dirac, phonons.n_phonons)
            if is_plus:
                ps_and_gamma_sparse[2:] -= result
            else:
                ps_and_gamma_sparse[2:] += result

            result = np.bincount(nupp_vec, pot_times_dirac, phonons.n_phonons)
            ps_and_gamma_sparse[2:] += result
        ps_and_gamma_sparse[0] += dirac_delta.sum()
        ps_and_gamma_sparse[1] += pot_times_dirac.sum()

    return ps_and_gamma_sparse


def calculate_broadening(phonons, index_kp_vec, index_kpp_vec):
    cellinv = phonons.cell_inv
    k_size = phonons.kpts
    velocity = phonons.velocities[index_kp_vec, :, np.newaxis, :] - phonons.velocities[index_kpp_vec, np.newaxis, :, :]
    # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
    delta_k = cellinv / k_size
    base_sigma = ((np.tensordot(velocity, delta_k, [-1, 1])) ** 2).sum(axis=-1)
    base_sigma = np.sqrt(base_sigma / 6.)
    return base_sigma

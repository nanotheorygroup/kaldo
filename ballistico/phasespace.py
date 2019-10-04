import scipy.special
import numpy as np


DELTA_THRESHOLD = 2


def gaussian_delta(params):
    # alpha is a factor that tells whats the ration between the width of the gaussian
    # and the width of allowed phase space
    delta_energy = params[0]
    # allowing processes with width sigma and creating a gaussian with width sigma/2
    # we include 95% (erf(2/sqrt(2)) of the probability of scattering. The erf makes the total area 1
    sigma = params[1]
    gaussian = 1 / np.sqrt(np.pi * sigma ** 2) * np.exp(- delta_energy ** 2 / (sigma ** 2))
    return gaussian


def triangular_delta(params):
    delta_energy = np.abs(params[0])
    deltaa = np.abs(params[1])
    out = np.zeros_like(delta_energy)
    out[delta_energy < deltaa] = 1. / deltaa * (1 - delta_energy[delta_energy < deltaa] / deltaa)
    return out


def lorentzian_delta(params):
    delta_energy = params[0]
    gamma = params[1]
    lorentzian = 1 / np.pi * 1 / 2 * gamma / (delta_energy ** 2 + (gamma / 2) ** 2)
    return lorentzian



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
        sigma_small = calculate_broadening(phonons, index_kpp_full)
    else:
        try:
            phonons.sigma_in.size
        except AttributeError:
            sigma_small = phonons.sigma_in
        else:
            sigma_small = phonons.sigma_in.reshape((phonons.n_k_points, phonons.n_modes))[index_k, mu]

    second_sign = (int(is_plus) * 2 - 1)
    omegas = phonons._omegas
    omegas_difference = np.abs(
        omegas[index_k, mu] + second_sign * omegas[:, :, np.newaxis] -
        omegas[index_kpp_full, np.newaxis, :])

    condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_small) & \
                (frequencies[:, :, np.newaxis] > frequencies_threshold) & \
                (frequencies[index_kpp_full, np.newaxis, :] > frequencies_threshold)
    interactions = np.array(np.where(condition)).T

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
            omegas = phonons._omegas
            omegas_difference = np.abs(
                omegas[0, mu] + second_sign * omegas[0, :, np.newaxis] -
                omegas[0, np.newaxis, :])

            condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_small) & \
                        (frequencies[0, :, np.newaxis] > frequencies_threshold) & \
                        (frequencies[0, np.newaxis, :] > frequencies_threshold)
            interactions = np.array(np.where(condition)).T

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


def calculate_broadening(phonons, index_kpp_vec):
    cellinv = phonons.cell_inv
    k_size = phonons.kpts
    velocity = phonons.velocities[:, :, np.newaxis, :] - phonons.velocities[index_kpp_vec, np.newaxis, :, :]
    # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
    delta_k = cellinv / k_size
    base_sigma = ((np.tensordot(velocity, delta_k, [-1, 1])) ** 2).sum(axis=-1)
    base_sigma = np.sqrt(base_sigma / 6.)
    return base_sigma


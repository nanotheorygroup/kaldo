"""
Ballistico
Anharmonic Lattice Dynamics
"""
import sparse
from opt_einsum import contract
import ase.units as units
from ballistico.tools.tools import timeit
import numpy as np


DELTA_THRESHOLD = 2
EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
GAMMATOTHZ = 1e11 * units.mol * EVTOTENJOVERMOL ** 2


@timeit
def project_amorphous(phonons, is_gamma_tensor_enabled=False):
    if is_gamma_tensor_enabled == True:
        raise ValueError('is_gamma_tensor_enabled=True not supported')

    n_particles = phonons.atoms.positions.shape[0]
    n_modes = phonons.n_modes
    masses = phonons.atoms.get_masses()
    rescaled_eigenvectors = phonons.eigenvectors.reshape(
        (n_particles, 3, n_modes), order='C') / np.sqrt(
        masses[:, np.newaxis, np.newaxis])
    rescaled_eigenvectors = rescaled_eigenvectors.reshape((n_modes, n_modes),
                                                          order='C')

    # The ps and gamma matrix stores ps, gamma and then the scattering matrix
    ps_and_gamma = np.zeros((phonons.n_phonons, 2))
    for nu_single in range(phonons.n_phonons):

        # ps_and_gamma_sparse = np.zeros(2)
        out = calculate_dirac_delta_amorphous(phonons, nu_single)
        if not out:
            continue
        dirac_delta, mup_vec, mupp_vec = out

        potential_times_evect = sparse.tensordot(phonons.finite_difference.third_order,
                                                 rescaled_eigenvectors[:, nu_single], (0, 0))
        scaled_potential = contract('ij,in,jm->nm', potential_times_evect.real,
                                    rescaled_eigenvectors.real,
                                    rescaled_eigenvectors.real,
                                    optimize='optimal')
        scaled_potential = scaled_potential[np.newaxis, ...]
        scaled_potential = scaled_potential[0, mup_vec, mupp_vec]
        pot_times_dirac = np.abs(scaled_potential) ** 2 * dirac_delta

        pot_times_dirac = units._hbar / 8. * pot_times_dirac / phonons.n_k_points * GAMMATOTHZ
        ps_and_gamma[nu_single, 0] = dirac_delta.sum()
        ps_and_gamma[nu_single, 1] = pot_times_dirac.sum()
        ps_and_gamma[nu_single, 1:] /= phonons.frequencies.flatten()[nu_single]

        THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15

        print(phonons.frequencies[0, nu_single], ps_and_gamma[nu_single, 1] * THZTOMEV / (2 * np.pi))
        print('calculating third', nu_single, np.round(nu_single / phonons.n_phonons, 2) * 100,
              '%')
    return ps_and_gamma


def calculate_index_kpp(phonons, index_k, is_plus):
    index_kp_full = np.arange(phonons.n_k_points)
    kp_vec = np.array(np.unravel_index(index_kp_full, phonons.kpts, order='C'))
    k = np.array(np.unravel_index(index_k, phonons.kpts, order='C'))
    kpp_vec = k[:, np.newaxis] + (int(is_plus) * 2 - 1) * kp_vec[:, :]
    index_kpp_full = np.ravel_multi_index(kpp_vec, phonons.kpts, order='C', mode='wrap')
    return index_kpp_full


@timeit
def project_crystal(phonons, is_gamma_tensor_enabled=False):

    # The ps and gamma matrix stores ps, gamma and then the scattering matrix
    if is_gamma_tensor_enabled:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2 + phonons.n_phonons))
    else:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2))

    n_particles = phonons.atoms.positions.shape[0]
    n_modes = phonons.n_modes
    masses = phonons.atoms.get_masses()
    rescaled_eigenvectors = phonons.eigenvectors[:, :, :].reshape(
        (phonons.n_k_points, n_particles, 3, n_modes), order='C') / np.sqrt(
        masses[np.newaxis, :, np.newaxis, np.newaxis])
    rescaled_eigenvectors = rescaled_eigenvectors.reshape((phonons.n_k_points, n_modes, n_modes), order='C')

    for index_k in range(phonons.n_k_points):
        for mu in range(phonons.n_modes):
            nu_single = np.ravel_multi_index([index_k, mu], (phonons.n_k_points, phonons.n_modes), order='C')

            if nu_single % 200 == 0:
                print('calculating third', nu_single, np.round(nu_single / phonons.n_phonons, 2) * 100,
                      '%')
            potential_times_evect = np.zeros((phonons.n_replicas * phonons.n_modes, phonons.n_replicas * phonons.n_modes), dtype=np.complex)
            for i in range(phonons.n_modes):
                mask = phonons.finite_difference.third_order.coords[0] == i
                potential_times_evect[phonons.finite_difference.third_order.coords[1][mask], phonons.finite_difference.third_order.coords[2][mask]] += phonons.finite_difference.third_order.data[mask] * rescaled_eigenvectors[index_k, i, mu]

            potential_times_evect = potential_times_evect.reshape(
                (phonons.n_replicas, phonons.n_modes, phonons.n_replicas, phonons.n_modes),
                order='C').transpose(0, 2, 1, 3).reshape(
                (phonons.n_replicas * phonons.n_replicas, phonons.n_modes, phonons.n_modes))

            for is_plus in (1, 0):


                out = calculate_dirac_delta_crystal(phonons, index_k, mu, is_plus)
                if not out:
                    continue
                dirac_delta, index_kp_vec, mup_vec, index_kpp_vec, mupp_vec = out
                index_kpp_full = calculate_index_kpp(phonons, index_k, is_plus)
                if is_plus:
                    scaled_potential = contract('litj,kim,kl,kjn,kt->kmn', potential_times_evect,
                                                rescaled_eigenvectors,
                                                phonons._chi_k,
                                                rescaled_eigenvectors[index_kpp_full].conj(),
                                                phonons._chi_k[index_kpp_full].conj()
                                                )

                    # chi_prod = np.einsum('kt,kl->ktl', phonons._chi_k, phonons._chi_k[index_kpp_full].conj())
                    # chi_prod = chi_prod.reshape((phonons.n_k_points, phonons.n_replicas ** 2))
                    # scaled_potential = np.tensordot(chi_prod, potential_times_evect, (1, 0))
                    # scaled_potential = np.einsum('kij,kim->kjm', scaled_potential, rescaled_eigenvectors)
                    # scaled_potential = np.einsum('kjm,kjn->kmn', scaled_potential, rescaled_eigenvectors[index_kpp_full].conj())
                else:
                    scaled_potential = contract('litj,kim,kl,kjn,kt->kmn', potential_times_evect,
                                                rescaled_eigenvectors.conj(),
                                                phonons._chi_k.conj(),
                                                rescaled_eigenvectors[index_kpp_full].conj(),
                                                phonons._chi_k[index_kpp_full].conj())

                    # chi_prod = np.einsum('kt,kl->ktl', phonons._chi_k.conj(), phonons._chi_k[index_kpp_full].conj())
                    # chi_prod = chi_prod.reshape((phonons.n_k_points, phonons.n_replicas ** 2))
                    # scaled_potential = np.tensordot(chi_prod, potential_times_evect, (1, 0))
                    # scaled_potential = np.einsum('kij,kim->kjm', scaled_potential, rescaled_eigenvectors.conj())
                    # scaled_potential = np.einsum('kjm,kjn->kmn', scaled_potential, rescaled_eigenvectors[index_kpp_full].conj())

                pot_times_dirac = np.abs(scaled_potential[index_kp_vec, mup_vec, mupp_vec]) ** 2 * dirac_delta

                pot_times_dirac = units._hbar / 8. * pot_times_dirac / phonons.n_k_points * GAMMATOTHZ

                if is_gamma_tensor_enabled:
                    # We need to use bincount together with fancy indexing here. See:
                    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
                    nup_vec = np.ravel_multi_index(np.array([index_kp_vec, mup_vec], dtype=int),
                                                   np.array([phonons.n_k_points, phonons.n_modes]), order='C')
                    nupp_vec = np.ravel_multi_index(np.array([index_kpp_vec, mupp_vec], dtype=int),
                                                    np.array([phonons.n_k_points, phonons.n_modes]), order='C')

                    result = np.bincount(nup_vec, pot_times_dirac, phonons.n_phonons)

                    # The ps and gamma array stores first ps then gamma then the scattering array
                    if is_plus:
                        ps_and_gamma[nu_single, 2:] -= result
                    else:
                        ps_and_gamma[nu_single, 2:] += result

                    result = np.bincount(nupp_vec, pot_times_dirac, phonons.n_phonons)
                    ps_and_gamma[nu_single, 2:] += result
                ps_and_gamma[nu_single, 0] += dirac_delta.sum()
                ps_and_gamma[nu_single, 1] += pot_times_dirac.sum()

            ps_and_gamma[nu_single, 1:] /= phonons.frequencies.flatten()[nu_single]

    return ps_and_gamma


def calculate_dirac_delta_crystal(phonons, index_k, mu, is_plus):
    if phonons.frequencies[index_k, mu] <= phonons.frequency_threshold:
        return None

    frequencies = phonons.frequencies
    density = phonons.occupations
    frequencies_threshold = phonons.frequency_threshold
    if phonons.broadening_shape == 'gauss':
        broadening_function = gaussian_delta
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
    physical_modes = phonons._physical_modes.reshape((phonons.n_k_points, phonons.n_modes))
    condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_small) & \
                (physical_modes[:, :, np.newaxis]) & \
                (physical_modes[index_kpp_full, np.newaxis, :])

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

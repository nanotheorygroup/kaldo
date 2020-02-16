"""
Ballistico
Anharmonic Lattice Dynamics
"""
import sparse
import ase.units as units
from ballistico.helpers.tools import timeit, allowed_index_qpp
import numpy as np
from ballistico.controllers.dirac_kernel import gaussian_delta, triangular_delta, lorentz_delta
from ballistico.helpers.logger import get_logger
logging = get_logger()


DELTA_THRESHOLD = 2
EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
GAMMATOTHZ = 1e11 * units.mol * EVTOTENJOVERMOL ** 2


@timeit
def project_amorphous(phonons):
    n_modes = phonons.n_modes
    rescaled_eigenvectors = phonons._rescaled_eigenvectors.reshape((n_modes, n_modes),
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

        scaled_potential = np.einsum('ij,in,jm->nm', potential_times_evect.real,
                                    rescaled_eigenvectors.real,
                                    rescaled_eigenvectors.real,
                                    optimize='optimal')
        scaled_potential = scaled_potential[np.newaxis, ...]
        scaled_potential = scaled_potential[0, mup_vec, mupp_vec]

        # pot_small = phonons.calculate_third_k0m0_k1m1_k2m2(False, 0, nu_single, 0, mup_vec[0],
        #                                            0, mupp_vec[0])

        pot_times_dirac = np.abs(scaled_potential) ** 2 * dirac_delta

        ps_and_gamma[nu_single, 0] = dirac_delta.sum()
        ps_and_gamma[nu_single, 1] = pot_times_dirac.sum()
        ps_and_gamma[nu_single, 1:] = ps_and_gamma[nu_single, 1:] / 8. / phonons.n_k_points * units._hbar * GAMMATOTHZ
        ps_and_gamma[nu_single, 1:] = ps_and_gamma[nu_single, 1:] / phonons.frequency.flatten()[nu_single]

        THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
        logging.info('calculating third ' + str(nu_single) + ', ' + str(np.round(nu_single / phonons.n_phonons, 2) * 100) + '%')
        logging.info(str(phonons.frequency[0, nu_single]) + ': ' + str(ps_and_gamma[nu_single, 1] * THZTOMEV / (2 * np.pi)))

    return ps_and_gamma


def calculate_third_k0m0_k1m1_k2m2(phonons, is_plus, k0, m0, k1, m1, k2, m2):
    evect = phonons._rescaled_eigenvectors
    phonons.finite_difference.third_order_delta = 0.001
    third_k0m0_k2m2 = phonons.finite_difference.calculate_single_third_on_phonons(k0, m0, k2, m2, evect,
                                                                               phonons._chi_k)
    third_k0m0_k2m2 = third_k0m0_k2m2.reshape((phonons.finite_difference.n_replicas, phonons.n_modes))
    if is_plus:
        third_k0m0_k1m1_k2m2 = np.einsum('li,l,i->', third_k0m0_k2m2, phonons._chi_k[k1, :],
                                         evect[k1, :, m1])
    else:
        third_k0m0_k1m1_k2m2 = np.einsum('li,l,i->', third_k0m0_k2m2, phonons._chi_k[k1, :].conj(),
                                         evect[k1, :, m1].conj())
    return third_k0m0_k1m1_k2m2

@timeit
def project_crystal(phonons, ):
    is_gamma_tensor_enabled = phonons.is_gamma_tensor_enabled
    # The ps and gamma matrix stores ps, gamma and then the scattering matrix
    if is_gamma_tensor_enabled:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2 + phonons.n_phonons))
    else:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2))
    n_replicas = phonons.finite_difference.n_replicas
    rescaled_eigenvectors = phonons._rescaled_eigenvectors

    for index_k in range(phonons.n_k_points):
        for mu in range(phonons.n_modes):
            nu_single = np.ravel_multi_index([index_k, mu], (phonons.n_k_points, phonons.n_modes))

            if nu_single % 200 == 0:
                logging.info('calculating third ' + str(nu_single) + ': ' + str(np.round(nu_single / phonons.n_phonons, 2) * 100) + '%')
            potential_times_evect = np.zeros((n_replicas * phonons.n_modes, n_replicas * phonons.n_modes), dtype=np.complex)
            for i in range(phonons.n_modes):
                mask = phonons.finite_difference.third_order.coords[0] == i
                potential_times_evect[phonons.finite_difference.third_order.coords[1][mask], phonons.finite_difference.third_order.coords[2][mask]] += phonons.finite_difference.third_order.data[mask] * rescaled_eigenvectors[index_k, i, mu]

            potential_times_evect = potential_times_evect.reshape(
                (n_replicas, phonons.n_modes, n_replicas, phonons.n_modes),
                order='C').transpose(0, 2, 1, 3).reshape(
                (n_replicas * n_replicas, phonons.n_modes, phonons.n_modes))

            for is_plus in (1, 0):


                out = calculate_dirac_delta_crystal(phonons, index_k, mu, is_plus)
                if not out:
                    continue
                dirac_delta, index_kp_vec, mup_vec, index_kpp_vec, mupp_vec = out
                index_kpp_full = allowed_index_qpp(index_k, is_plus, phonons.kpts)
                if is_plus:
                    #TODO: This can be faster using the contract opt_einsum
                    chi_prod = np.einsum('kt,kl->ktl', phonons._chi_k, phonons._chi_k[index_kpp_full].conj())
                    chi_prod = chi_prod.reshape((phonons.n_k_points, n_replicas ** 2))
                    scaled_potential = np.tensordot(chi_prod, potential_times_evect, (1, 0))
                    scaled_potential = np.einsum('kij,kim->kjm', scaled_potential, rescaled_eigenvectors)
                    scaled_potential = np.einsum('kjm,kjn->kmn', scaled_potential, rescaled_eigenvectors[index_kpp_full].conj())
                else:
                    chi_prod = np.einsum('kt,kl->ktl', phonons._chi_k.conj(), phonons._chi_k[index_kpp_full].conj())
                    chi_prod = chi_prod.reshape((phonons.n_k_points, n_replicas ** 2))
                    scaled_potential = np.tensordot(chi_prod, potential_times_evect, (1, 0))
                    scaled_potential = np.einsum('kij,kim->kjm', scaled_potential, rescaled_eigenvectors.conj())
                    scaled_potential = np.einsum('kjm,kjn->kmn', scaled_potential, rescaled_eigenvectors[index_kpp_full].conj())

                # pot_small = calculate_third_k0m0_k1m1_k2m2(phonons, is_plus, index_k, mu, index_kp_vec[0], mup_vec[0], index_kpp_vec[0], mupp_vec[0])

                pot_times_dirac = np.abs(scaled_potential[index_kp_vec, mup_vec, mupp_vec]) ** 2 * dirac_delta

                pot_times_dirac = units._hbar / 8. * pot_times_dirac / phonons.n_k_points * GAMMATOTHZ

                if is_gamma_tensor_enabled:
                    # We need to use bincount together with fancy indexing here. See:
                    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
                    nup_vec = np.ravel_multi_index(np.array([index_kp_vec, mup_vec], dtype=int),
                                                   np.array([phonons.n_k_points, phonons.n_modes]))
                    nupp_vec = np.ravel_multi_index(np.array([index_kpp_vec, mupp_vec], dtype=int),
                                                    np.array([phonons.n_k_points, phonons.n_modes]))

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

            ps_and_gamma[nu_single, 1:] /= phonons.frequency.flatten()[nu_single]

    return ps_and_gamma


def calculate_dirac_delta_crystal(phonons, index_q, mu, is_plus):
    physical_modes = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    if not physical_modes[index_q, mu]:
        return None

    density = phonons.population
    if phonons.broadening_shape == 'gauss':
        broadening_function = gaussian_delta
    elif phonons.broadening_shape == 'triangle':
        broadening_function = triangular_delta
    elif phonons.broadening_shape == 'lorentz':
        broadening_function = lorentz_delta

    else:
        raise TypeError('Broadening shape not supported')

    index_qpp_full = allowed_index_qpp(index_q, is_plus, phonons.kpts)
    if phonons.third_bandwidth is None:
        sigma_small = calculate_broadening(phonons, index_qpp_full)
    else:
        try:
            phonons.third_bandwidth.size
        except AttributeError:
            sigma_small = phonons.third_bandwidth
        else:
            sigma_small = phonons.third_bandwidth.reshape((phonons.n_k_points, phonons.n_modes))[index_q, mu]

    second_sign = (int(is_plus) * 2 - 1)
    omegas = phonons._omegas
    omegas_difference = np.abs(
        omegas[index_q, mu] + second_sign * omegas[:, :, np.newaxis] -
        omegas[index_qpp_full, np.newaxis, :])
    physical_modes = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_small) & \
                (physical_modes[:, :, np.newaxis]) & \
                (physical_modes[index_qpp_full, np.newaxis, :])

    interactions = np.array(np.where(condition)).T

    if interactions.size == 0:
        return None
    # Create sparse index
    index_qp = interactions[:, 0]
    index_qpp = index_qpp_full[index_qp]
    mup_vec = interactions[:, 1]
    mupp_vec = interactions[:, 2]
    if is_plus:
        dirac_delta = density[index_qp, mup_vec] - density[index_qpp, mupp_vec]
        # dirac_delta = density[index_k, mu] * density[index_kp_vec, mup_vec] * (
        #             density[index_kpp_vec, mupp_vec] + 1)

    else:
        dirac_delta = .5 * (
                1 + density[index_qp, mup_vec] + density[index_qpp, mupp_vec])
        # dirac_delta = .5 * density[index_k, mu] * (density[index_kp_vec, mup_vec] + 1) * (
        #             density[index_kpp_vec, mupp_vec] + 1)

    dirac_delta /= (omegas[index_qp, mup_vec] * omegas[index_qpp, mupp_vec])
    if np.array(sigma_small).size == 1:
        dirac_delta *= broadening_function(
            omegas_difference[index_qp, mup_vec, mupp_vec], 2 * np.pi * sigma_small)
    else:
        dirac_delta *= broadening_function(
            omegas_difference[index_qp, mup_vec, mupp_vec], 2 * np.pi * sigma_small[
                index_qp, mup_vec, mupp_vec])

    return dirac_delta, index_qp, mup_vec, index_qpp, mupp_vec

def calculate_dirac_delta_amorphous(phonons, mu):
    density = phonons.population
    if phonons.broadening_shape == 'gauss':
        broadening_function = gaussian_delta
    elif phonons.broadening_shape == 'triangle':
        broadening_function = triangular_delta
    else:
        raise TypeError('Broadening shape not supported')
    physical_modes = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    for is_plus in [1, 0]:
        sigma_small = phonons.third_bandwidth

        if physical_modes[0, mu]:

            second_sign = (int(is_plus) * 2 - 1)
            omegas = phonons._omegas
            omegas_difference = np.abs(
                omegas[0, mu] + second_sign * omegas[0, :, np.newaxis] -
                omegas[0, np.newaxis, :])

            condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_small) & \
                            (physical_modes[0, :, np.newaxis]) & \
                            (physical_modes[0, np.newaxis, :])
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
                    omegas_difference[mup_vec, mupp_vec], 2 * np.pi * sigma_small)

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
    cellinv = phonons.finite_difference.cell_inv
    k_size = phonons.kpts
    velocity = phonons.velocity[:, :, np.newaxis, :] - phonons.velocity[index_kpp_vec, np.newaxis, :, :]
    # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
    delta_k = cellinv / k_size
    base_sigma = ((np.tensordot(velocity, delta_k, [-1, 1])) ** 2).sum(axis=-1)
    base_sigma = np.sqrt(base_sigma / 6.)
    return base_sigma



"""
Ballistico
Anharmonic Lattice Dynamics
"""
import sparse
import ase.units as units
from ballistico.helpers.tools import timeit
import numpy as np
from ballistico.controllers.dirac_kernel import gaussian_delta, triangular_delta, lorentz_delta
from ballistico.helpers.logger import get_logger
from opt_einsum import contract
from ballistico.grid import wrap_coordinates

logging = get_logger()


DELTA_THRESHOLD = 2
GAMMATOTHZ = 1e11 * units.mol * (units.mol / (10 * units.J)) ** 2


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
        pot_times_dirac /= (phonons._omegas[0, mup_vec] * phonons._omegas[0, mupp_vec])

        ps_and_gamma[nu_single, 0] = dirac_delta.sum()
        ps_and_gamma[nu_single, 1] = pot_times_dirac.sum()
        ps_and_gamma[nu_single, 1:] = ps_and_gamma[nu_single, 1:] * np.pi / 4. / phonons.n_k_points * units._hbar * GAMMATOTHZ
        ps_and_gamma[nu_single, 1:] = ps_and_gamma[nu_single, 1:] / phonons._omegas.flatten()[nu_single]

        THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
        logging.info('calculating third ' + str(nu_single) + ', ' + str(np.round(nu_single / phonons.n_phonons, 2) * 100) + '%')
        logging.info(str(phonons.frequency[0, nu_single]) + ': ' + str(ps_and_gamma[nu_single, 1] * THZTOMEV / (2 * np.pi)))

    return ps_and_gamma


@timeit
def project_crystal(phonons):
    is_gamma_tensor_enabled = phonons.is_gamma_tensor_enabled
    # The ps and gamma matrix stores ps, gamma and then the scattering matrix
    if is_gamma_tensor_enabled:
        scattering_tensor = np.zeros((phonons.n_phonons, phonons.n_phonons))
    ps_and_gamma = np.zeros((phonons.n_phonons, 2))
    n_replicas = phonons.finite_difference.n_replicas
    rescaled_eigenvectors = phonons._rescaled_eigenvectors
    n_atoms = phonons.finite_difference.n_atoms
    third_order = phonons.finite_difference.third_order.reshape((n_atoms * 3, n_replicas, n_atoms * 3, n_replicas, n_atoms * 3))

    for index_k in range(phonons.n_k_points):
        for mu in range(phonons.n_modes):
            nu_single = np.ravel_multi_index([index_k, mu], (phonons.n_k_points, phonons.n_modes))

            if nu_single % 200 == 0:
                logging.info('calculating third ' + str(nu_single) + ': ' + str(np.round(nu_single / phonons.n_phonons, 2) * 100) + '%')
            potential_times_evect = contract('iljtn,i->ljtn', third_order, rescaled_eigenvectors[index_k, :, mu])
            for is_plus in (1, 0):
                out = calculate_dirac_delta_crystal(phonons, index_k, mu, is_plus)
                if not out:
                    continue
                dirac_delta, index_kp_vec, mup_vec, index_kpp_vec, mupp_vec = out

                index_kpp_full = phonons._allowed_third_phonons_index(index_k, is_plus)
                if is_plus:
                    chi_1 = phonons._chi_k
                    evect_1 = rescaled_eigenvectors
                else:
                    chi_1 = phonons._chi_k.conj()
                    evect_1 = rescaled_eigenvectors.conj()
                chi_2 = phonons._chi_k[index_kpp_full].conj()
                evect_2 = rescaled_eigenvectors[index_kpp_full].conj()
                scaled_potential = contract('tilj,kt,kl,kim,kjn->kmn',
                                            potential_times_evect,
                                            chi_1,
                                            chi_2,
                                            evect_1,
                                            evect_2)

                # pot_small = calculate_third_k0m0_k1m1_k2m2(phonons, is_plus, index_k, mu, index_kp_vec[0], mup_vec[0], index_kpp_vec[0], mupp_vec[0])

                pot_times_dirac = np.abs(scaled_potential[index_kp_vec, mup_vec, mupp_vec]) ** 2 * dirac_delta

                pot_times_dirac /= (phonons._omegas[index_kp_vec, mup_vec] * phonons._omegas[index_kpp_vec, mupp_vec])
                pot_times_dirac = np.pi * units._hbar / 4. * pot_times_dirac / phonons.n_k_points * GAMMATOTHZ

                if is_gamma_tensor_enabled:
                    # We need to use bincount together with fancy indexing here. See:
                    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
                    nup_vec = np.ravel_multi_index(np.array([index_kp_vec, mup_vec], dtype=int),
                                                   np.array([phonons.n_k_points, phonons.n_modes]))
                    nupp_vec = np.ravel_multi_index(np.array([index_kpp_vec, mupp_vec], dtype=int),
                                                    np.array([phonons.n_k_points, phonons.n_modes]))

                    # The ps and gamma array stores first ps then gamma then the scattering array
                    sign = -1 if is_plus else 1
                    scattering_tensor[nu_single] += sign * np.bincount(nup_vec, pot_times_dirac, phonons.n_phonons)
                    scattering_tensor[nu_single] += np.bincount(nupp_vec, pot_times_dirac, phonons.n_phonons)
                ps_and_gamma[nu_single, 0] += dirac_delta.sum()
                ps_and_gamma[nu_single, 1] += pot_times_dirac.sum()

            ps_and_gamma[nu_single, 1:] /= phonons._omegas.flatten()[nu_single]
            if is_gamma_tensor_enabled:
                scattering_tensor[nu_single] /= phonons._omegas.flatten()[nu_single]
    if is_gamma_tensor_enabled:
        return np.hstack([ps_and_gamma, scattering_tensor])
    else:
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

    index_qpp_full = phonons._allowed_third_phonons_index(index_q, is_plus)
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
        # dirac_delta = density[index_q, mu] * density[index_qp, mup_vec] * (
        #             density[index_qpp, mupp_vec] + 1)

    else:
        dirac_delta = .5 * (
                1 + density[index_qp, mup_vec] + density[index_qpp, mupp_vec])
        # dirac_delta = .5 * density[index_q, mu] * (density[index_qp, mup_vec] + 1) * (
        #             density[index_qpp, mupp_vec] + 1)

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
    elif phonons.broadening_shape == 'lorentz':
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



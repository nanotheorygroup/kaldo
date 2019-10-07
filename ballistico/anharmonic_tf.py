import scipy.special
import numpy as np
from opt_einsum import contract
import ase.units as units
from .tools import timeit, lazy_property, is_calculated
import tensorflow as tf

DELTA_THRESHOLD = 2
IS_DELTA_CORRECTION_ENABLED = False
EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
GAMMATOTHZ = 1e11 * units.mol * EVTOTENJOVERMOL ** 2



def calculate_dirac_delta_crystal(phonons, index_kpp_full, index_k, mu, is_plus):
    if phonons.frequencies[index_k, mu] <= phonons.frequency_threshold:
        return None
    if phonons.sigma_in:
        sigma_tf = tf.constant(phonons.sigma_in, dtype=tf.float64)
    else:
        sigma_tf = calculate_broadening(phonons, index_kpp_full)
    second_sign = (int(is_plus) * 2 - 1)
    omegas_difference = tf.abs(phonons.omega_tf[index_k, mu] + second_sign * phonons.omega_tf[:, :, tf.newaxis] -
        tf.gather(phonons.omega_tf, index_kpp_full)[:, tf.newaxis, :])
    condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_tf) & \
                (phonons.frequencies_tf[:, :, tf.newaxis] > phonons.frequency_threshold) & \
                (tf.gather(phonons.frequencies_tf, index_kpp_full)[:, tf.newaxis, :] > phonons.frequency_threshold)
    interactions = tf.where(condition)
    if interactions.shape[0] > 0:
        index_kp_vec = tf.cast(interactions[:, 0], dtype=tf.int32)
        # kp_vec = tf.unravel_index(index_kp_vec, phonons.kpts)
        # kpp_vec = k[:, tf.newaxis] + (int(is_plus) * 2 - 1) * tf.cast(kp_vec[:, :], dtype=tf.int32)
        # kpp_vec = tf.math.floormod(kpp_vec, phonons.kpts[:, tf.newaxis])
        # index_kpp_vec = tf.cast(phonons.kpts[2] * phonons.kpts[1] * kpp_vec[0, :] + phonons.kpts[2] * kpp_vec[1, :] + kpp_vec[2, :], dtype=tf.int32)
        index_kpp_vec = tf.gather(index_kpp_full, index_kp_vec)
        mup_vec = tf.cast(interactions[:, 1], dtype=tf.int32)
        mupp_vec = tf.cast(interactions[:, 2], dtype=tf.int32)
        coords_1 = tf.stack((index_kp_vec, mup_vec), axis=-1)
        coords_2 = tf.stack((index_kpp_vec, mupp_vec), axis=-1)
        if sigma_tf.shape != []:
            coords_3 = tf.stack((index_kp_vec, mup_vec, mupp_vec), axis=-1)
            sigma_tf = tf.gather_nd(sigma_tf, coords_3)
        if is_plus:
            dirac_delta_tf = tf.gather_nd(phonons.density_tf, coords_1) - tf.gather_nd(phonons.density_tf, coords_2)

        else:
            dirac_delta_tf = 0.5 * (1 + tf.gather_nd(phonons.density_tf, coords_1) + tf.gather_nd(phonons.density_tf, coords_2))
        dirac_delta_tf /= (tf.gather_nd(phonons.omega_tf, coords_1) * tf.gather_nd(phonons.omega_tf, coords_2))
        dirac_delta_tf = dirac_delta_tf * 1 / tf.sqrt(np.pi * (2 * np.pi * sigma_tf) ** 2) * tf.exp(
            - (phonons.omega_tf[index_k, mu] + second_sign * tf.gather_nd(phonons.omega_tf, coords_1) - tf.gather_nd(phonons.omega_tf, coords_2)) ** 2 / ((2 * np.pi * sigma_tf) ** 2))
        index_kp = index_kp_vec
        mup = mup_vec
        index_kpp = index_kpp_vec
        mupp = mupp_vec
        return dirac_delta_tf, index_kp, mup, index_kpp, mupp

def calculate_dirac_delta_amorphous(phonons, mu):
    if phonons.frequencies[0, mu] < phonons.frequency_threshold:
        return None
    if phonons.broadening_shape == 'triangle':
        delta_threshold = 1
    else:
        delta_threshold = DELTA_THRESHOLD
    density_tf = phonons.density_tf
    omega_tf = phonons.omega_tf
    sigma_tf = tf.constant(phonons.sigma_in, dtype=tf.float64)
    for is_plus in (1, 0):
        second_sign = (int(is_plus) * 2 - 1)
        omegas_difference = np.abs(phonons._omegas[0, mu] + second_sign * phonons._omegas[0, :, np.newaxis] -
                                   phonons._omegas[0, np.newaxis, :])
        condition = (omegas_difference < delta_threshold * 2 * np.pi * sigma_tf) & \
                    (phonons.frequencies[0, :, np.newaxis] > phonons.frequency_threshold) & \
                    (phonons.frequencies[0, np.newaxis, :] > phonons.frequency_threshold)
        interactions = tf.where(condition)
        if interactions.shape[0] > 0:
            # Create sparse index
            mup_vec = interactions[:, 0]
            mupp_vec = interactions[:, 1]
            if is_plus:
                dirac_delta_tf = tf.gather(density_tf[0], mup_vec) - tf.gather(density_tf[0], mupp_vec)
            else:
                dirac_delta_tf = 1 + tf.gather(density_tf[0], mup_vec) + tf.gather(density_tf[0], mupp_vec)
            dirac_delta_tf = dirac_delta_tf / tf.gather(omega_tf[0], mup_vec) / tf.gather(omega_tf[0], mupp_vec)
            omegas_difference_tf = tf.abs(phonons._omegas[0, mu] + second_sign * tf.gather(omega_tf[0], mup_vec) - tf.gather(omega_tf[0],
                                                                                                                mupp_vec))

            if phonons.broadening_shape == 'gauss':
                dirac_delta_tf = dirac_delta_tf * 1 / tf.sqrt(np.pi * (2 * np.pi * sigma_tf) ** 2) * np.exp(
                    - omegas_difference_tf ** 2 / ((2 * np.pi * sigma_tf) ** 2))
            elif phonons.broadening_shape == 'triangle':
                dirac_delta_tf = dirac_delta_tf * 1. / (2 * np.pi * sigma_tf) * (1 - omegas_difference_tf / (2 * np.pi * sigma_tf))
            else:
                raise TypeError('Broadening function not implemented.')

            try:
                mup = tf.concat([mup, mup_vec], 0)
                mupp = tf.concat([mupp, mupp_vec], 0)
                current_delta = tf.concat([current_delta, dirac_delta_tf], 0)
            except NameError:
                mup = mup_vec
                mupp = mupp_vec
                current_delta = dirac_delta_tf
    try:
        return current_delta, mup, mupp
    except:
        return None


def project_amorphous(phonons, is_gamma_tensor_enabled=False):
    if is_gamma_tensor_enabled == True:
        raise ValueError('is_gamma_tensor_enabled=True not supported')
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
    # The ps and gamma matrix stores ps, gamma and then the scattering matrix
    ps_and_gamma = np.zeros((phonons.n_phonons, 2))
    print('Projection started')
    evect_tf = tf.convert_to_tensor(
        rescaled_eigenvectors.reshape((phonons.n_phonons, phonons.n_modes)).astype(float))

    coords = phonons.finite_difference.third_order.coords
    data = phonons.finite_difference.third_order.data
    coords = np.vstack([coords[1], coords[2], coords[0]])
    third_tf = tf.SparseTensor(coords.T, data, (
    phonons.n_modes * phonons.n_replicas, phonons.n_modes * phonons.n_replicas, phonons.n_modes))

    third_tf = tf.sparse.reshape(third_tf, ((phonons.n_modes * phonons.n_replicas) ** 2, phonons.n_modes))
    for nu_single in range(phonons.n_phonons):
        out = calculate_dirac_delta_amorphous(phonons, nu_single)
        if not out:
            continue
        third_nu_tf = tf.sparse.sparse_dense_matmul(third_tf,
                                                    tf.reshape(evect_tf[nu_single, :], ((phonons.n_modes, 1))))
        third_nu_tf = tf.reshape(third_nu_tf,
                                 (phonons.n_modes * phonons.n_replicas, phonons.n_modes * phonons.n_replicas))

        dirac_delta_tf, mup_vec, mupp_vec = out
        scaled_potential_tf = tf.einsum('ij,ni,mj->nm', third_nu_tf, evect_tf, evect_tf)
        coords = tf.stack((mup_vec, mupp_vec), axis=-1)
        # pot_times_dirac_tf = tf.SparseTensor(coords, tf.abs(tf.gather_nd(scaled_potential_tf, coords)) ** 2 * dirac_delta_tf, (n_phonons, n_phonons))

        pot_times_dirac = tf.reduce_sum(tf.abs(tf.gather_nd(scaled_potential_tf,
                                                            coords)) ** 2 * dirac_delta_tf).numpy() * units._hbar / 8. / phonons.n_k_points * GAMMATOTHZ
        dirac_delta = tf.reduce_sum(dirac_delta_tf).numpy()

        ps_and_gamma[nu_single, 0] = dirac_delta
        ps_and_gamma[nu_single, 1] = pot_times_dirac
        ps_and_gamma[nu_single, 1:] /= phonons.frequencies.flatten()[nu_single]

        THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
        print('calculating third', nu_single, np.round(nu_single / phonons.n_phonons, 2) * 100,
                  '%')
        print(phonons.frequencies[0, nu_single], ps_and_gamma[nu_single, 1] * THZTOMEV / (2 * np.pi))

    return ps_and_gamma


def project_crystal(phonons, is_gamma_tensor_enabled=False):

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
    evect_tf = tf.convert_to_tensor(rescaled_eigenvectors.reshape((phonons.n_phonons, phonons.n_modes)))

    coords = phonons.finite_difference.third_order.coords
    data = phonons.finite_difference.third_order.data
    coords = np.vstack([coords[1], coords[2], coords[0]])
    third_tf = tf.SparseTensor(coords.T, data, (
    phonons.n_modes * phonons.n_replicas, phonons.n_modes * phonons.n_replicas, phonons.n_modes))

    third_tf = tf.sparse.reshape(third_tf, ((phonons.n_modes * phonons.n_replicas) ** 2, phonons.n_modes))

    third_tf = tf.cast(third_tf, dtype=tf.complex128)

    chi_k = tf.convert_to_tensor(phonons._chi_k)
    evect_tf = tf.reshape(evect_tf, (phonons.n_k_points, phonons.n_modes, phonons.n_modes))

    # The ps and gamma matrix stores ps, gamma and then the scattering matrix
    if is_gamma_tensor_enabled:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2 + phonons.n_phonons))
    else:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2))
    print('Projection started')
    second_minus = tf.math.conj(evect_tf)
    second_minus_chi = tf.math.conj(chi_k)

    index_kp_full = tf.range(phonons.n_k_points)
    kp_vec = tf.unravel_index(index_kp_full, phonons.kpts)
    for nu_single in range(phonons.n_phonons):
        if nu_single % 200 == 0:
            print('calculating third', nu_single, np.round(nu_single / phonons.n_phonons, 2) * 100,
                  '%')
        index_k, mu = np.unravel_index(nu_single, (phonons.n_k_points, phonons.n_modes), order='C')

        third_nu_tf = tf.sparse.sparse_dense_matmul(third_tf,
                                                    tf.reshape(evect_tf[index_k, mu, :], ((phonons.n_modes, 1))))

        third_nu_tf = tf.cast(
            tf.reshape(third_nu_tf, (phonons.n_replicas, phonons.n_modes, phonons.n_replicas, phonons.n_modes)),
            dtype=tf.complex128)


        for is_plus in (0, 1):
            k = tf.unravel_index(index_k, phonons.kpts)
            kpp_vec = k[:, tf.newaxis] + (int(is_plus) * 2 - 1) * kp_vec[:, :]
            kpp_vec = tf.math.floormod(kpp_vec, phonons.kpts[:, tf.newaxis])
            index_kpp_full = phonons.kpts[2] * phonons.kpts[1] * kpp_vec[0, :] + phonons.kpts[2] * kpp_vec[1,
                                                                                                   :] + kpp_vec[2, :]

            out = calculate_dirac_delta_crystal(phonons, index_kpp_full, index_k, mu, is_plus)
            if not out:
                continue


            if is_plus:

                second = evect_tf
                second_chi = chi_k
            else:

                second = second_minus
                second_chi = second_minus_chi

            third = tf.math.conj(tf.gather(evect_tf, index_kpp_full))
            third_chi = tf.math.conj(tf.gather(chi_k, index_kpp_full))
            dirac_delta, index_kp_vec, mup_vec, index_kpp_vec, mupp_vec = out

            # The ps and gamma array stores first ps then gamma then the scattering array


            scaled_potential = contract('litj,kl,kni,kt,kmj->knm', third_nu_tf, second_chi, second, third_chi, third, backend='tensorflow')
            scaled_potential = tf.gather_nd(scaled_potential, tf.stack([index_kp_vec, mup_vec, mupp_vec], axis=-1))
            pot_times_dirac = tf.abs(
                scaled_potential) ** 2 * dirac_delta * units._hbar / 8. / phonons.n_k_points * GAMMATOTHZ
            if is_gamma_tensor_enabled:
                # We need to use bincount together with fancy indexing here. See:
                # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments

                nup_vec = index_kp_vec * phonons.n_modes + mup_vec
                nupp_vec = index_kpp_vec * phonons.n_modes + mupp_vec

                result = tf.math.bincount(nup_vec, pot_times_dirac, phonons.n_phonons)
                if is_plus:
                    ps_and_gamma[nu_single, 2:] -= result
                else:
                    ps_and_gamma[nu_single, 2:] += result

                result = tf.math.bincount(nupp_vec, pot_times_dirac, phonons.n_phonons)
                ps_and_gamma[nu_single, 2:] += result
            ps_and_gamma[nu_single, 0] += tf.reduce_sum(dirac_delta)
            ps_and_gamma[nu_single, 1] += tf.reduce_sum(pot_times_dirac)
        ps_and_gamma[nu_single, 1:] /= phonons.frequencies.flatten()[nu_single]
    return ps_and_gamma


def calculate_broadening(phonons, index_kpp_vec):
    cellinv = phonons.cell_inv
    k_size = phonons.kpts
    velocities_tf = tf.convert_to_tensor(phonons.velocities)
    velocity_difference = velocities_tf[:, :, tf.newaxis, :] - tf.gather(velocities_tf, index_kpp_vec)[:,tf.newaxis, :, :]
    # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
    delta_k = cellinv / k_size
    base_sigma = tf.reduce_sum((tf.tensordot(velocity_difference, delta_k, [-1, 1])) ** 2, axis=-1)
    base_sigma = tf.sqrt(base_sigma / 6.)
    return base_sigma

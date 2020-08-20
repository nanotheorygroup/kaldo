"""
kaldo
Anharmonic Lattice Dynamics
"""
import numpy as np
import ase.units as units
from kaldo.helpers.tools import timeit
import tensorflow as tf
from opt_einsum import contract
from kaldo.helpers.logger import get_logger, log_size
from kaldo.controllers.dirac_kernel import gaussian_delta, triangular_delta, lorentz_delta

logging = get_logger()

DELTA_THRESHOLD = 2
EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
GAMMATOTHZ = 1e11 * units.mol * EVTOTENJOVERMOL ** 2


@timeit
def project_amorphous(phonons):
    frequency = phonons.frequency
    omega = 2 * np.pi * frequency
    population = phonons.population
    n_replicas = phonons.forceconstants.n_replicas
    rescaled_eigenvectors = phonons._rescaled_eigenvectors.astype(float)
    # The ps and gamma matrix stores ps, gamma and then the scattering matrix
    ps_and_gamma = np.zeros((phonons.n_phonons, 2))
    evect_tf = tf.convert_to_tensor(rescaled_eigenvectors[0])

    coords = phonons.forceconstants.third.value.coords
    data = phonons.forceconstants.third.value.data
    coords = np.vstack([coords[1], coords[2], coords[0]])
    third_tf = tf.SparseTensor(coords.T, data, (
        phonons.n_modes * n_replicas, phonons.n_modes * n_replicas, phonons.n_modes))

    third_tf = tf.sparse.reshape(third_tf, ((phonons.n_modes * n_replicas) ** 2, phonons.n_modes))
    physical_mode = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    logging.info('Projection started')
    for nu_single in range(phonons.n_phonons):
        sigma_tf = tf.constant(phonons.third_bandwidth, dtype=tf.float64)

        out = calculate_dirac_delta_amorphous(omega,
                                              population,
                                              physical_mode,
                                              sigma_tf,
                                              phonons.broadening_shape,
                                              nu_single)
        if not out:
            continue
        third_nu_tf = tf.sparse.sparse_dense_matmul(third_tf,
                                                    tf.reshape(evect_tf[:, nu_single], ((phonons.n_modes, 1))))
        third_nu_tf = tf.reshape(third_nu_tf,
                                 (phonons.n_modes * n_replicas, phonons.n_modes * n_replicas))

        dirac_delta_tf, mup_vec, mupp_vec = out
        scaled_potential_tf = tf.einsum('ij,in,jm->nm', third_nu_tf, evect_tf, evect_tf)
        coords = tf.stack((mup_vec, mupp_vec), axis=-1)
        # pot_times_dirac_tf = tf.SparseTensor(coords, tf.abs(tf.gather_nd(scaled_potential_tf, coords)) ** 2 \
        # * dirac_delta_tf, (n_phonons, n_phonons))

        pot_times_dirac = tf.gather_nd(scaled_potential_tf,coords) **  2
        pot_times_dirac = pot_times_dirac / tf.gather(omega[0], mup_vec) / tf.gather(omega[0], mupp_vec)
        pot_times_dirac = tf.reduce_sum(tf.abs(pot_times_dirac) * dirac_delta_tf)
        pot_times_dirac = np.pi * units._hbar / 4. * pot_times_dirac / phonons.n_k_points * GAMMATOTHZ

        dirac_delta = tf.reduce_sum(dirac_delta_tf)

        ps_and_gamma[nu_single, 0] = dirac_delta.numpy()
        ps_and_gamma[nu_single, 1] = pot_times_dirac.numpy()
        ps_and_gamma[nu_single, 1:] /= omega.flatten()[nu_single]

        THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
        logging.info('calculating third ' + str(nu_single) + ': ' + str(np.round(nu_single / \
                                                                                 phonons.n_phonons, 2) * 100) + '%')
        logging.info(str(frequency.reshape(phonons.n_phonons)[nu_single]) + ': ' + \
                     str(ps_and_gamma[nu_single, 1] * THZTOMEV / (2 * np.pi)))

    return ps_and_gamma


@timeit
def project_crystal(phonons):
    is_gamma_tensor_enabled = phonons.is_gamma_tensor_enabled
    n_replicas = phonons.forceconstants.third.n_replicas

    try:
        sparse_third = phonons.forceconstants.third.value.reshape((phonons.n_modes, -1))
        # transpose
        sparse_coords = tf.stack([sparse_third.coords[1], sparse_third.coords[0]], -1)
        third_tf = tf.SparseTensor(sparse_coords,
                                   sparse_third.data,
                                   ((phonons.n_modes * n_replicas) ** 2, phonons.n_modes))
        is_sparse = True
    except AttributeError:
        third_tf = tf.convert_to_tensor(phonons.forceconstants.third.value)
        is_sparse = False
    third_tf = tf.cast(third_tf, dtype=tf.complex64)
    k_mesh = phonons._reciprocal_grid.unitary_grid(is_wrapping=False)
    n_k_points = k_mesh.shape[0]
    _chi_k = tf.convert_to_tensor(phonons.forceconstants.third._chi_k(k_mesh))
    _chi_k = tf.cast(_chi_k, dtype=tf.complex64)
    evect_tf = tf.convert_to_tensor(phonons._rescaled_eigenvectors)
    evect_tf = tf.cast(evect_tf, dtype=tf.complex64)
    # The ps and gamma matrix stores ps, gamma and then the scattering matrix
    if is_gamma_tensor_enabled:
        shape = (phonons.n_phonons, 2 + phonons.n_phonons)
        log_size(shape, name='scattering_tensor')
        ps_and_gamma = np.zeros(shape)
    else:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2))
    second_minus = tf.math.conj(evect_tf)
    second_minus_chi = tf.math.conj(_chi_k)
    logging.info('Projection started')
    population = phonons.population
    broadening_shape = phonons.broadening_shape
    physical_mode = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    omega = phonons.omega
    if not phonons.third_bandwidth:
        velocity_tf = tf.convert_to_tensor(phonons.velocity)
    for nu_single in range(phonons.n_phonons):
        if nu_single % 200 == 0:
            logging.info('calculating third ' + str(nu_single) +  ', ' + \
                         str(np.round(nu_single / phonons.n_phonons, 2) * 100) + '%')
        index_k, mu = np.unravel_index(nu_single, (n_k_points, phonons.n_modes))
        if is_sparse:
            third_nu_tf = tf.sparse.sparse_dense_matmul(third_tf, evect_tf[index_k, :, mu, tf.newaxis])
        else:
            third_nu_tf = contract('ijk,i->jk', third_tf, evect_tf[index_k, :, mu], backend='tensorflow')
            third_nu_tf = tf.reshape(third_nu_tf, (n_replicas * n_replicas, phonons.n_modes, phonons.n_modes))

        third_nu_tf = tf.cast(
            tf.reshape(third_nu_tf, (n_replicas, phonons.n_modes, n_replicas, phonons.n_modes)),
            dtype=tf.complex64)
        third_nu_tf = tf.transpose(third_nu_tf, (0, 2, 1, 3))
        third_nu_tf = tf.reshape(third_nu_tf, (n_replicas * n_replicas, phonons.n_modes, phonons.n_modes))
        for is_plus in (0, 1):
            index_kpp_full = phonons._allowed_third_phonons_index(index_k, is_plus)
            index_kpp_full = tf.cast(index_kpp_full, dtype=tf.int32)

            if phonons.third_bandwidth:
                sigma_tf = tf.constant(phonons.third_bandwidth, dtype=tf.float64)
            else:
                cellinv = phonons.forceconstants.cell_inv
                k_size = phonons.kpts
                sigma_tf = calculate_broadening(velocity_tf, cellinv, k_size, index_kpp_full)

            out = calculate_dirac_delta_crystal(omega,
                                                population,
                                                physical_mode,
                                                sigma_tf,
                                                broadening_shape,
                                                index_kpp_full,
                                                index_k,
                                                mu,
                                                is_plus)
            if not out:
                continue

            if is_plus:

                second = evect_tf
                second_chi = _chi_k
            else:

                second = second_minus
                second_chi = second_minus_chi

            third = tf.math.conj(tf.gather(evect_tf, index_kpp_full))
            third_chi = tf.math.conj(tf.gather(_chi_k, index_kpp_full))
            dirac_delta, index_kp_vec, mup_vec, index_kpp_vec, mupp_vec = out

            # The ps and gamma array stores first ps then gamma then the scattering array
            chi_prod = tf.einsum('kt,kl->ktl', second_chi, third_chi)
            chi_prod = tf.reshape(chi_prod, (n_k_points, n_replicas ** 2))
            scaled_potential = tf.tensordot(chi_prod, third_nu_tf, (1, 0))
            scaled_potential = tf.einsum('kij,kim->kjm', scaled_potential, second)
            scaled_potential = tf.einsum('kjm,kjn->kmn', scaled_potential, third)

            scaled_potential = tf.gather_nd(scaled_potential, tf.stack([index_kp_vec, mup_vec, mupp_vec], axis=-1))
            pot_times_dirac = tf.abs(
                scaled_potential) ** 2 * dirac_delta

            nup_vec = index_kp_vec * phonons.n_modes + mup_vec
            nupp_vec = index_kpp_vec * phonons.n_modes + mupp_vec
            pot_times_dirac = tf.cast(pot_times_dirac, dtype=tf.float64)
            pot_times_dirac = pot_times_dirac / tf.gather(omega.flatten(), nup_vec) / tf.gather(omega.flatten(), nupp_vec)

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
            ps_and_gamma[nu_single, 0] += tf.reduce_sum(dirac_delta) / phonons.n_k_points
            ps_and_gamma[nu_single, 1] += tf.reduce_sum(pot_times_dirac)
        ps_and_gamma[nu_single, 1:] /= omega.flatten()[nu_single]
        ps_and_gamma[nu_single, 1:] *= np.pi * units._hbar / 4 / n_k_points * GAMMATOTHZ
    return ps_and_gamma


def calculate_dirac_delta_crystal(omega, population, physical_mode, sigma_tf, broadening_shape,
                                  index_kpp_full, index_k, mu, is_plus, is_balanced=False):
    if not physical_mode[index_k, mu]:
        return None
    if broadening_shape == 'gauss':
        broadening_function = gaussian_delta
    elif broadening_shape == 'triangle':
        broadening_function = triangular_delta
    elif broadening_shape == 'lorentz':
        broadening_function = lorentz_delta
    else:
        raise ('Broadening function not implemented')
    second_sign = (int(is_plus) * 2 - 1)
    omegas_difference = tf.abs(omega[index_k, mu] + second_sign * omega[:, :, tf.newaxis] -
                               tf.gather(omega, index_kpp_full)[:, tf.newaxis, :])


    condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_tf) & \
                (physical_mode[:, :, np.newaxis]) & \
                (physical_mode[index_kpp_full, np.newaxis, :])
    interactions = tf.where(condition)
    if interactions.shape[0] > 0:
        index_kp_vec = tf.cast(interactions[:, 0], dtype=tf.int32)
        index_kpp_vec = tf.gather(index_kpp_full, index_kp_vec)
        mup_vec = tf.cast(interactions[:, 1], dtype=tf.int32)
        mupp_vec = tf.cast(interactions[:, 2], dtype=tf.int32)
        coords_1 = tf.stack((index_kp_vec, mup_vec), axis=-1)
        coords_2 = tf.stack((index_kpp_vec, mupp_vec), axis=-1)
        if sigma_tf.shape != []:
            coords_3 = tf.stack((index_kp_vec, mup_vec, mupp_vec), axis=-1)
            sigma_tf = tf.gather_nd(sigma_tf, coords_3)
        if is_plus:
            dirac_delta_tf = tf.gather_nd(population, coords_1) - tf.gather_nd(population, coords_2)
            if is_balanced:
                # Detail balance
                # (n0) * (n1) * (n2 + 2) - (n0 + 1) * (n1 + 1) * (n2) = 0
                dirac_delta_tf = 0.5 * (tf.gather_nd(population, coords_1) + 1) * (tf.gather_nd(population, coords_2)) / (population[index_k, mu])
                dirac_delta_tf += 0.5 * (tf.gather_nd(population, coords_1)) * (tf.gather_nd(population, coords_2) + 1) / (1 + population[index_k, mu])
        else:
            dirac_delta_tf = 0.5 * (1 + tf.gather_nd(population, coords_1) + tf.gather_nd(population, coords_2))
            if is_balanced:
                # Detail balance
                # (n0) * (n1 + 1) * (n2 + 2) - (n0 + 1) * (n1) * (n2) = 0
                dirac_delta_tf = 0.25 * (tf.gather_nd(population, coords_1)) * (tf.gather_nd(population, coords_2)) / (population[index_k, mu])
                dirac_delta_tf += 0.25 * (tf.gather_nd(population, coords_1) + 1) * (tf.gather_nd(population, coords_2) + 1) / (1 + population[index_k, mu])
        omegas_difference_tf = (omega[index_k, mu] + second_sign * tf.gather_nd(omega, coords_1) - tf.gather_nd(
                omega, coords_2))

        dirac_delta_tf = dirac_delta_tf * broadening_function(omegas_difference_tf, 2 * np.pi * sigma_tf)

        index_kp = index_kp_vec
        mup = mup_vec
        index_kpp = index_kpp_vec
        mupp = mupp_vec
        return tf.cast(dirac_delta_tf, dtype=tf.float32), index_kp, mup, index_kpp, mupp


def calculate_dirac_delta_amorphous(omega, population, physical_mode, sigma_tf, broadening_shape, mu):
    if not physical_mode[0, mu]:
        return None
    if broadening_shape == 'triangle':
        delta_threshold = 1
    else:
        delta_threshold = DELTA_THRESHOLD
    for is_plus in (1, 0):
        second_sign = (int(is_plus) * 2 - 1)
        omegas_difference = np.abs(omega[0, mu] + second_sign * omega[0, :, np.newaxis] -
                                   omega[0, np.newaxis, :])
        condition = (omegas_difference < delta_threshold * 2 * np.pi * sigma_tf) & \
                (physical_mode[0, :, np.newaxis]) & \
                (physical_mode[0, np.newaxis, :])
        interactions = tf.where(condition)
        if interactions.shape[0] > 0:
            # Create sparse index
            mup_vec = interactions[:, 0]
            mupp_vec = interactions[:, 1]
            if is_plus:
                dirac_delta_tf = tf.gather(population[0], mup_vec) - tf.gather(population[0], mupp_vec)
            else:
                dirac_delta_tf = 0.5 * (1 + tf.gather(population[0], mup_vec) + tf.gather(population[0], mupp_vec))
            omegas_difference_tf = tf.abs(omega[0, mu] + second_sign * omega[0, mup_vec] - omega[0, mupp_vec])

            if broadening_shape == 'gauss':
                broadening_function = gaussian_delta
            elif broadening_shape == 'triangle':
                broadening_function = triangular_delta
            elif broadening_shape == 'lorentz':
                broadening_function = lorentz_delta
            else:
                raise('Broadening function not implemented')

            dirac_delta_tf = dirac_delta_tf * broadening_function(omegas_difference_tf,  2 * np.pi * sigma_tf)

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


def calculate_broadening(velocity_tf, cellinv, k_size, index_kpp_vec):
    velocity_difference = velocity_tf[:, :, tf.newaxis, :] - tf.gather(velocity_tf, index_kpp_vec)[:,tf.newaxis, :, :]
    # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
    delta_k = cellinv / k_size
    base_sigma = tf.reduce_sum((tf.tensordot(velocity_difference, delta_k, [-1, 1])) ** 2, axis=-1)
    base_sigma = tf.sqrt(base_sigma / 6.)
    return base_sigma
"""
kaldo
Anharmonic Lattice Dynamics

This module contains functions for calculating anharmonic properties of phonons
in both amorphous and crystalline materials.
"""

import numpy as np
import ase.units as units
import tensorflow as tf
from opt_einsum import contract
from kaldo.helpers.logger import get_logger, log_size
from kaldo.helpers.tools import timeit
from kaldo.controllers.dirac_kernel import gaussian_delta, triangular_delta, lorentz_delta

logging = get_logger()

# Constants
GAMMA_TO_THZ = 1e11 * units.mol * (units.mol / (10 * units.J)) ** 2
HBAR = units._hbar
THZ_TO_MEV = units.J * HBAR * 2 * np.pi * 1e15


@timeit
def project_amorphous(phonons):
    """
    Project anharmonic properties for amorphous materials.

    Args:
        phonons: Phonon object containing material properties

    Returns:
        np.ndarray: Array containing projected properties
    """
    is_balanced = phonons.is_balanced
    frequency = phonons.frequency
    omega = 2 * np.pi * frequency
    population = phonons.population
    n_replicas = phonons.forceconstants.n_replicas
    rescaled_eigenvectors = phonons._rescaled_eigenvectors.astype(float)

    ps_and_gamma = np.zeros((phonons.n_phonons, 2))
    evect_tf = tf.convert_to_tensor(rescaled_eigenvectors[0])

    coords = phonons.forceconstants.third.value.coords
    coords = np.vstack([coords[1], coords[2], coords[0]])
    coords = tf.cast(coords.T, dtype=tf.int64)
    data = phonons.forceconstants.third.value.data
    third_tf = tf.SparseTensor(
        coords, data, (phonons.n_modes * n_replicas, phonons.n_modes * n_replicas, phonons.n_modes)
    )

    third_tf = tf.sparse.reshape(third_tf, ((phonons.n_modes * n_replicas) ** 2, phonons.n_modes))
    physical_mode = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    logging.info("Projection started")

    hbar = HBAR * (1e-6 if phonons.is_classic else 1)

    for nu_single in range(phonons.n_phonons):
        sigma_tf = tf.constant(phonons.third_bandwidth, dtype=tf.float64)

        ps_and_gamma[nu_single] = project_amorphous_mu(
            omega,
            population,
            physical_mode,
            sigma_tf,
            phonons.broadening_shape,
            nu_single,
            is_balanced,
            third_tf,
            evect_tf,
            phonons.n_modes,
            n_replicas,
            hbar,
        )

        if nu_single % 200 == 0:
            logging.info("calculating third " + f"{nu_single}" + ": " + \
                                                        f"{100*nu_single/phonons.n_phonons:.2f}%")
            logging.info(f"{frequency.reshape(phonons.n_phonons)[nu_single]}: " + \
                                        f"{ps_and_gamma[nu_single, 1] * THZ_TO_MEV / (2 * np.pi)}")

    return ps_and_gamma


def project_amorphous_mu(
    omega,
    population,
    physical_mode,
    sigma_tf,
    broadening_shape,
    nu_single,
    is_balanced,
    third_tf,
    evect_tf,
    n_modes,
    n_replicas,
    hbar,
):
    """
    Project anharmonic properties for a single mode in amorphous materials.

    Args:
        (various): Parameters describing the phonon properties and material

    Returns:
        np.ndarray: Projected properties for the single mode
    """
    if not physical_mode[0, nu_single]:
        return np.zeros(2)

    dirac_delta_result = calculate_dirac_delta_amorphous(
        omega, population, physical_mode, sigma_tf, broadening_shape, nu_single, is_balanced
    )

    if not dirac_delta_result:
        return np.zeros(2)

    dirac_delta_tf, mup_vec, mupp_vec = dirac_delta_result
    dirac_delta = tf.reduce_sum(dirac_delta_tf)

    third_nu_tf = tf.sparse.sparse_dense_matmul(third_tf, tf.reshape(evect_tf[:, nu_single], ((n_modes, 1))))
    third_nu_tf = tf.reshape(third_nu_tf, (n_modes * n_replicas, n_modes * n_replicas))

    scaled_potential_tf = tf.einsum("ij,in,jm->nm", third_nu_tf, evect_tf, evect_tf)
    coords = tf.stack((mup_vec, mupp_vec), axis=-1)
    pot_times_dirac = tf.gather_nd(scaled_potential_tf, coords) ** 2
    pot_times_dirac = pot_times_dirac / tf.gather(omega[0], mup_vec) / tf.gather(omega[0], mupp_vec)
    pot_times_dirac = tf.reduce_sum(tf.abs(pot_times_dirac) * dirac_delta_tf)
    pot_times_dirac = np.pi * hbar / 4.0 * pot_times_dirac * GAMMA_TO_THZ

    ps_and_gamma = np.zeros(2)
    ps_and_gamma[0] = dirac_delta.numpy()
    ps_and_gamma[1] = pot_times_dirac.numpy()
    ps_and_gamma[1] /= omega.flatten()[nu_single]

    return ps_and_gamma

@timeit
def project_crystal(phonons):

    n_replicas = phonons.forceconstants.third.n_replicas

    try:
        sparse_third = phonons.forceconstants.third.value.reshape((phonons.n_modes, -1))
        sparse_coords = tf.stack([sparse_third.coords[1], sparse_third.coords[0]], -1)
        sparse_coords = tf.cast(sparse_coords, dtype=tf.int64)
        third_tf = tf.SparseTensor(
            sparse_coords, sparse_third.data, ((phonons.n_modes * n_replicas) ** 2, phonons.n_modes)
        )
        is_sparse = True
    except AttributeError:
        third_tf = tf.convert_to_tensor(phonons.forceconstants.third.value)
        is_sparse = False
    third_tf = tf.cast(third_tf, dtype=tf.complex128)

    k_mesh = phonons._reciprocal_grid.unitary_grid(is_wrapping=False)
    n_k_points = k_mesh.shape[0]
    _chi_k = tf.convert_to_tensor(phonons.forceconstants.third._chi_k(k_mesh))
    _chi_k = tf.cast(_chi_k, dtype=tf.complex128)
    evect_tf = tf.convert_to_tensor(phonons._rescaled_eigenvectors)
    evect_tf = tf.cast(evect_tf, dtype=tf.complex128)


    second_minus = tf.math.conj(evect_tf)
    second_minus_chi = tf.math.conj(_chi_k)
    logging.info("Projection started")

    population = phonons.population
    broadening_shape = phonons.broadening_shape
    physical_mode = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    omega = phonons.omega
    velocity_tf = tf.convert_to_tensor(phonons.velocity)
    cell_inv = phonons.forceconstants.cell_inv
    kpts = phonons.kpts

    hbar = HBAR * (1e-6 if phonons.is_classic else 1)
    n_modes = phonons.n_modes
    n_phonons = phonons.n_phonons

    sparse_phase = []
    sparse_potential = []

    for nu_single in range(n_phonons):
        if nu_single % 200 == 0:
            logging.info(f"calculating third {nu_single}: {100 * nu_single / phonons.n_phonons:.2f}%")

        sparse_phase.append([])
        sparse_potential.append([])

        index_k, mu = np.unravel_index(nu_single, (n_k_points, phonons.n_modes))
        if not physical_mode[index_k, mu]:
            sparse_phase[nu_single].extend([None, None])
            sparse_potential[nu_single].extend([None, None])
            continue

        for is_plus in (0, 1):
            index_kpp_full = tf.cast(phonons._allowed_third_phonons_index(index_k, is_plus), dtype=tf.int32)

            if phonons.third_bandwidth:
                sigma_tf = tf.constant(phonons.third_bandwidth, dtype=tf.float64)
            else:
                sigma_tf = calculate_broadening(velocity_tf, cell_inv, kpts, index_kpp_full)

            out = calculate_dirac_delta_crystal(
                omega,
                physical_mode,
                sigma_tf,
                broadening_shape,
                index_kpp_full,
                index_k,
                mu,
                is_plus,
                n_k_points,
                n_modes,
            )

            if out is None:
                sparse_phase[nu_single].append(None)
                sparse_potential[nu_single].append(None)
                continue

            sparse_phase[nu_single].append(out)

            sparse_potential[nu_single].append(
                sparse_potential_mu(
                    nu_single,
                    evect_tf,
                    out,
                    index_k,
                    mu,
                    n_k_points,
                    n_modes,
                    is_plus,
                    is_sparse,
                    index_kpp_full,
                    _chi_k,
                    second_minus,
                    second_minus_chi,
                    third_tf,
                    n_replicas,
                    omega,
                    hbar,
                )
            )
    return calculate_ps_and_gamma(
        sparse_phase,
        sparse_potential,
        population,
        phonons.is_balanced,
        n_k_points,
        n_modes,
        n_phonons,
        phonons.is_gamma_tensor_enabled
    )

def calculate_ps_and_gamma(sparse_phase, sparse_potential, population, is_balanced, n_k_points, n_modes, n_phonons, is_gamma_tensor_enabled):
    if is_gamma_tensor_enabled:
        shape = (n_phonons, 2 + n_phonons)
        log_size(shape, name="scattering_tensor")
        ps_and_gamma = np.zeros((n_phonons, 2 + n_phonons))
    else:
        ps_and_gamma = np.zeros((n_phonons, 2))

    for nu_single in range(n_phonons):
        index_k, mu = np.unravel_index(nu_single, (n_k_points, n_modes))
        for is_plus in (0, 1):
            if sparse_phase[nu_single][is_plus] is None:
                continue

            pop_vec = sparse_population_mu(
                    is_plus,
                    mu,
                    index_k,
                    sparse_phase[nu_single][is_plus].indices,
                    population,
                    is_balanced,
                )

            flat_indices = tf.sparse.reshape(
                sparse_phase[nu_single][is_plus], tf.cast([n_k_points * n_modes, n_k_points * n_modes], tf.int64)
            ).indices

            ps_and_gamma[nu_single, 0] += tf.reduce_sum(sparse_phase[nu_single][is_plus].values * pop_vec) / n_k_points

            contrib = sparse_potential[nu_single][is_plus] * pop_vec
            ps_and_gamma[nu_single, 1] += tf.reduce_sum(contrib)

            if is_gamma_tensor_enabled:
                nup = tf.cast(flat_indices[:, 0], tf.int32)
                nupp = tf.cast(flat_indices[:, 1], tf.int32)
                result_nup = tf.math.bincount(nup, contrib, n_k_points * n_modes)
                result_nupp = tf.math.bincount(nupp, contrib, n_k_points * n_modes)
                if is_plus:
                    ps_and_gamma[nu_single, 2:] -= result_nup
                else:
                    ps_and_gamma[nu_single, 2:] += result_nup
                ps_and_gamma[nu_single, 2:] += result_nupp
    return ps_and_gamma

def sparse_potential_mu(
    nu_single, evect_tf, sparse_phase, index_k, mu, n_k_points, n_modes, is_plus, is_sparse,
    index_kpp_full, _chi_k, second_minus, second_minus_chi, third_tf, n_replicas, omega, hbar
):
    index_kp_vec, mup_vec, index_kpp_vec, mupp_vec = tf.unstack(sparse_phase.indices, axis=1)
    nup_vec = index_kp_vec * n_modes + mup_vec
    nupp_vec = index_kpp_vec * n_modes + mupp_vec

    second, second_chi = (evect_tf, _chi_k) if is_plus else (second_minus, second_minus_chi)
    third = tf.math.conj(tf.gather(evect_tf, index_kpp_full))
    third_chi = tf.math.conj(tf.gather(_chi_k, index_kpp_full))

    chi_prod = tf.einsum("kt,kl->ktl", second_chi, third_chi)
    chi_prod = tf.reshape(chi_prod, (n_k_points, n_replicas**2))

    if is_sparse:
        third_nu_tf = tf.sparse.sparse_dense_matmul(third_tf, evect_tf[index_k, :, mu, tf.newaxis])
    else:
        third_nu_tf = contract("ijk,i->jk", third_tf, evect_tf[index_k, :, mu], backend="tensorflow")
        third_nu_tf = tf.reshape(third_nu_tf, (n_replicas * n_replicas, n_modes, n_modes))

    third_nu_tf = tf.cast(tf.reshape(third_nu_tf, (n_replicas, n_modes, n_replicas, n_modes)), dtype=tf.complex128)
    third_nu_tf = tf.transpose(third_nu_tf, (0, 2, 1, 3))
    third_nu_tf = tf.reshape(third_nu_tf, (n_replicas * n_replicas, n_modes, n_modes))

    scaled_potential = tf.tensordot(chi_prod, third_nu_tf, (1, 0))
    scaled_potential = tf.einsum("kij,kim->kjm", scaled_potential, second)
    scaled_potential = tf.einsum("kjm,kjn->kmn", scaled_potential, third)
    scaled_potential = tf.gather_nd(scaled_potential, tf.stack([index_kp_vec, mup_vec, mupp_vec], axis=-1))

    pot_times_dirac = tf.abs(scaled_potential) ** 2 * sparse_phase.values
    pot_times_dirac = (tf.cast(pot_times_dirac, dtype=tf.float64) * np.pi * hbar/ 4 * GAMMA_TO_THZ / omega.flatten()[nu_single]
                       / n_k_points)
    pot_times_dirac = pot_times_dirac / tf.gather(omega.flatten(), nup_vec) / tf.gather(omega.flatten(), nupp_vec)

    return pot_times_dirac



def sparse_population_mu(is_plus, mu, index_k, indices, population, is_balanced):
    index_kp_vec, mup_vec, index_kpp_vec, mupp_vec = tf.unstack(indices, axis=1)

    population_1 = population[index_kp_vec, mup_vec]
    population_2 = population[index_kpp_vec, mupp_vec]
    population_0 = population[index_k, mu]

    if is_plus:
        population_delta = population_1 - population_2
        if is_balanced:
            population_delta = 0.5 * (population_1 + 1) * population_2 / population_0
            population_delta += 0.5 * population_1 * (population_2 + 1) / (1 + population_0)
    else:
        population_delta = 0.5 * (1 + population_1 + population_2)
        if is_balanced:
            population_delta = 0.25 * population_1 * population_2 / population_0
            population_delta += 0.25 * (population_1 + 1) * (population_2 + 1) / (1 + population_0)

    return population_delta

def calculate_dirac_delta_crystal(
    omega,
    physical_mode,
    sigma_tf,
    broadening_shape,
    index_kpp_full,
    index_k,
    mu,
    is_plus,
    n_k_points,
    n_modes,
    default_delta_threshold=2
):
    """
    Calculate the Dirac delta function for crystalline materials.

    Args:
        (various): Parameters describing the phonon properties and material

    Returns:
        tuple: Dirac delta function and related indices
    """
    if not physical_mode[index_k, mu]:
        return None

    if broadening_shape == "gauss":
        broadening_function = gaussian_delta
    elif broadening_shape == "triangle":
        broadening_function = triangular_delta
    elif broadening_shape == "lorentz":
        broadening_function = lorentz_delta
    else:
        raise ValueError("Broadening function not implemented")

    second_sign = int(is_plus) * 2 - 1
    omegas_difference = tf.abs(
        omega[index_k, mu] + second_sign * omega[:, :, tf.newaxis] - tf.gather(omega, index_kpp_full)[:, tf.newaxis, :]
    )

    condition = (
        (omegas_difference < default_delta_threshold * 2 * np.pi * sigma_tf)
        & (physical_mode[:, :, np.newaxis])
        & (physical_mode[index_kpp_full, np.newaxis, :])
    )
    interactions = tf.where(condition)

    if interactions.shape[0] <= 0:
        return None

    index_kp_vec = tf.cast(interactions[:, 0], dtype=tf.int32)
    index_kpp_vec = tf.gather(index_kpp_full, index_kp_vec)
    mup_vec = tf.cast(interactions[:, 1], dtype=tf.int32)
    mupp_vec = tf.cast(interactions[:, 2], dtype=tf.int32)
    coords_1 = tf.stack((index_kp_vec, mup_vec), axis=-1)
    coords_2 = tf.stack((index_kpp_vec, mupp_vec), axis=-1)
    omegas_difference_tf = (
        omega[index_k, mu] + second_sign * tf.gather_nd(omega, coords_1) - tf.gather_nd(omega, coords_2)
    )

    if sigma_tf.shape != []:
        coords_3 = tf.stack((index_kp_vec, mup_vec, mupp_vec), axis=-1)
        sigma_tf = tf.gather_nd(sigma_tf, coords_3)
    dirac_delta_tf = broadening_function(omegas_difference_tf, 2 * np.pi * sigma_tf)
    sparse_phase = tf.sparse.SparseTensor(tf.stack([
        tf.cast(index_kp_vec, tf.int64),
        tf.cast(mup_vec, tf.int64),
        tf.cast(index_kpp_vec, tf.int64),
        tf.cast(mupp_vec, tf.int64),
    ], axis=-1), dirac_delta_tf,
        [n_k_points, n_modes, n_k_points, n_modes])

    return sparse_phase


def calculate_dirac_delta_amorphous(
    omega,
    population,
    physical_mode,
    sigma_tf,
    broadening_shape,
    mu,
    is_balanced,
    default_delta_threshold=2,
):
    """
    Calculate the Dirac delta function for amorphous materials.

    Args:
        (various): Parameters describing the phonon properties and material

    Returns:
        tuple: Dirac delta function and related indices
    """
    if not physical_mode[0, mu]:
        return None

    if broadening_shape == "triangle":
        delta_threshold = 1
    else:
        delta_threshold = default_delta_threshold

    if broadening_shape == "gauss":
        broadening_function = gaussian_delta
    elif broadening_shape == "triangle":
        broadening_function = triangular_delta
    elif broadening_shape == "lorentz":
        broadening_function = lorentz_delta
    else:
        raise ValueError("Broadening function not implemented")

    for is_plus in (1, 0):

        second_sign = int(is_plus) * 2 - 1
        omegas_difference = np.abs(omega[0, mu] + second_sign * omega[0, :, np.newaxis] - omega[0, np.newaxis, :])
        condition = (
            (omegas_difference < delta_threshold * 2 * np.pi * sigma_tf)
            & (physical_mode[0, :, np.newaxis])
            & (physical_mode[0, np.newaxis, :])
        )
        interactions = tf.where(condition)

        if interactions.shape[0] <= 0:
            continue

        # Create sparse index

        mup_vec = interactions[:, 0]
        mupp_vec = interactions[:, 1]
        mup_mapped = tf.gather(population[0], mup_vec)
        mupp_mapped = tf.gather(population[0], mupp_vec)

        if is_plus:
            dirac_delta_tf = mup_mapped - mupp_mapped

            if is_balanced:
                # Detailed balance
                # n0 * n1 * (n2 + 2) = (n0 + 1) * (n1 + 1) * n2
                dirac_delta_tf = 0.5 * (mup_mapped + 1) * (mupp_mapped) / (population[0, mu])
                dirac_delta_tf += 0.5 * (mup_mapped) * (mupp_mapped + 1) / (1 + population[0, mu])
        else:
            dirac_delta_tf = 0.5 * (1 + mup_mapped + mupp_mapped)

            if is_balanced:
                # Detailed balance
                # n0 * (n1 + 1) * (n2 + 2) = (n0 + 1) * n1 * n2
                dirac_delta_tf = 0.25 * (mup_mapped) * (mupp_mapped) / (population[0, mu])
                dirac_delta_tf += 0.25 * (mup_mapped + 1) * (mupp_mapped + 1) / (1 + population[0, mu])

        omegas_difference_tf = tf.abs(omega[0, mu] + second_sign * omega[0, mup_vec] - omega[0, mupp_vec])
        dirac_delta_tf *= broadening_function(omegas_difference_tf, 2 * np.pi * sigma_tf)

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


def calculate_broadening(velocity_tf, cell_inv, k_size, index_kpp_vec):
    """
    Calculate the broadening for the Dirac delta function.

    Args:
        velocity_tf (tf.Tensor): Velocity tensor
        cell_inv (tf.Tensor): Inverse of the unit cell
        k_size (int): Size of the k-point mesh
        index_kpp_vec (tf.Tensor): Indices of the k-points

    Returns:
        tf.Tensor: Calculated broadening
    """
    velocity_difference = velocity_tf[:, :, tf.newaxis, :] - tf.gather(velocity_tf, index_kpp_vec)[:, tf.newaxis, :, :]
    delta_k = cell_inv / k_size
    base_sigma = tf.reduce_sum((tf.tensordot(velocity_difference, delta_k, [-1, 1])) ** 2, axis=-1)
    base_sigma = tf.sqrt(base_sigma / 6.0)
    return base_sigma

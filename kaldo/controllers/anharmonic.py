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


def calculate_ps_and_gamma(sparse_phase, sparse_potential, population, is_balanced, n_phonons, is_amorphous,
                           is_gamma_tensor_enabled=False, hbar_factor=1):
    # Set up the output array
    if is_gamma_tensor_enabled:
        shape = (n_phonons, 2 + n_phonons)
        log_size(shape, name="scattering_tensor")
        ps_and_gamma = np.zeros((n_phonons, 2 + n_phonons))
    else:
        ps_and_gamma = np.zeros((n_phonons, 2))

    for nu_single in range(n_phonons):
        population_0 = population[nu_single]

        for is_plus in (0, 1):
            if sparse_phase[nu_single][is_plus] is None:
                continue

            # Extract indices - both cases use the same 2D format now
            nup_vec, nupp_vec = tf.unstack(sparse_phase[nu_single][is_plus].indices, axis=1)
            sparse_phase_nu = sparse_phase[nu_single][is_plus].values
            # Apply hbar_factor to potential values for classical/quantum
            sparse_pot_nu = sparse_potential[nu_single][is_plus].values * hbar_factor
            
            # Use direct indexing for both cases
            population_1 = tf.gather(population, nup_vec)
            population_2 = tf.gather(population, nupp_vec)
                
            single_pop_delta = population_delta(is_plus, population_1, population_2, population_0, is_balanced)

            # Accumulate phase space and gamma
            ps_and_gamma[nu_single, 0] += tf.reduce_sum(sparse_phase_nu * single_pop_delta)
            contrib = sparse_pot_nu * sparse_phase_nu * single_pop_delta
            ps_and_gamma[nu_single, 1] += tf.reduce_sum(contrib)

            # Handle gamma tensor for crystal case
            if is_gamma_tensor_enabled:
                nup = tf.cast(nup_vec, tf.int32)
                nupp = tf.cast(nupp_vec, tf.int32)
                result_nup = tf.math.bincount(nup, contrib, n_phonons)
                result_nupp = tf.math.bincount(nupp, contrib, n_phonons)
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
    nup_vec, nupp_vec = tf.unstack(sparse_phase.indices, axis=1)

    index_kp_vec, mup_vec = tf.unravel_index(nup_vec, (n_k_points, n_modes))
    index_kpp_vec, mupp_vec = tf.unravel_index(nupp_vec, (n_k_points, n_modes))

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

    pot_times_dirac = tf.abs(scaled_potential) ** 2
    pot_times_dirac = (tf.cast(pot_times_dirac, dtype=tf.float64) * np.pi * hbar/ 4 * GAMMA_TO_THZ / omega.flatten()[nu_single]
                       / n_k_points)
    pot_times_dirac = pot_times_dirac / tf.gather(omega.flatten(), nup_vec) / tf.gather(omega.flatten(), nupp_vec)

    # Return as sparse tensor using the same indices as sparse_phase
    sparse_potential_tensor = tf.SparseTensor(
        indices=sparse_phase.indices,
        values=pot_times_dirac,
        dense_shape=sparse_phase.dense_shape
    )
    return sparse_potential_tensor



def population_delta(is_plus, population_1, population_2, population_0, is_balanced):
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
    
    # Convert to flattened indices to match amorphous format (n_phonons, n_phonons)
    nup_indices = tf.cast(index_kp_vec * n_modes + mup_vec, tf.int64)
    nupp_indices = tf.cast(index_kpp_vec * n_modes + mupp_vec, tf.int64)
    
    sparse_phase = tf.sparse.SparseTensor(tf.stack([
        nup_indices,
        nupp_indices,
    ], axis=-1), dirac_delta_tf,
        [n_k_points * n_modes, n_k_points * n_modes])

    return sparse_phase


def calculate_dirac_delta_amorphous(
    is_plus,
    mu,
    omega,
    physical_mode,
    sigma_tf,
    broadening_shape,
    n_phonons,
    default_delta_threshold=2):
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


    second_sign = int(is_plus) * 2 - 1
    omegas_difference = np.abs(omega[0, mu] + second_sign * omega[0, :, np.newaxis] - omega[0, np.newaxis, :])
    condition = (
        (omegas_difference < delta_threshold * 2 * np.pi * sigma_tf)
        & (physical_mode[0, :, np.newaxis])
        & (physical_mode[0, np.newaxis, :])
    )
    interactions = tf.where(condition)

    if interactions.shape[0] <= 0:
        return None

    mup_vec = interactions[:, 0]
    mupp_vec = interactions[:, 1]

    omegas_difference_tf = tf.abs(omega[0, mu] + second_sign * omega[0, mup_vec] - omega[0, mupp_vec])
    sparse_phase = broadening_function(omegas_difference_tf, 2 * np.pi * sigma_tf)

    sparse_phase = tf.sparse.SparseTensor(tf.stack([
        tf.cast(mup_vec, tf.int64),
        tf.cast(mupp_vec, tf.int64),
    ], axis=-1), sparse_phase,
        [n_phonons, n_phonons])

    return sparse_phase


# ---------------------------------------------------------------------------
# ShengBTE adaptive broadening (pair-based)
# ---------------------------------------------------------------------------
#
# Reference: Li, W., Carrete, J., Katcho, N. A., & Mingo, N.
# ShengBTE: A solver of the Boltzmann transport equation for phonons.
# Comput. Phys. Commun. 185, 1747-1758 (2014).
# doi:10.1016/j.cpc.2014.02.015

def calculate_broadening_shengbte(velocity_tf, cell_inv, k_size, index_kpp_vec):
    """
    ShengBTE-style adaptive broadening σ for the Dirac δ function in 3-phonon
    scattering.

    σ(q', μ', q'', μ'') = sqrt( (1/6) Σ_α [(v(q',μ') − v(q'',μ'')) · b_α/N_α]² )

    Ported from ShengBTE (bitbucket.org/sousaw/shengbte/src), specifically
      * `base_sigma(v)` in `Src/config.f90`
      * call sites in `Src/processes.f90` (e.g. line 122):
          sigma = scalebroad * base_sigma(vel_j - velocity(:, k, ss_p))

    Args:
        velocity_tf (tf.Tensor):   shape (n_k, n_m, 3), mode velocities
        cell_inv    (np.ndarray):  shape (3, 3), inverse unit cell; rows are the
                                   reciprocal basis vectors (no 2π)
        k_size      (array-like):  shape (3,), mesh dimensions
        index_kpp_vec (tf.Tensor): shape (n_k,) int — for each q' index,
                                   the corresponding q'' = q ± q' index

    Returns:
        tf.Tensor, shape (n_k, n_m, n_m): per-pair σ

    Notes
    -----
    This kernel is NOT invariant under the crystal point group when velocities
    at degenerate modes are specified in an arbitrary basis within the
    degenerate subspace (as numerical diagonalizers do by default). This shows
    up as a per-mode asymmetry of phase_space, bandwidth, and gamma tensor
    across symmetry-equivalent k-points. Use `calculate_broadening_tdep` for
    a gauge-invariant alternative when mode-resolved, symmetry-consistent
    results are needed (e.g. when combining with `use_q_symmetry=True`).

    Reference
    ---------
    Li, W., Carrete, J., Katcho, N. A., & Mingo, N. *ShengBTE: A solver of
    the Boltzmann transport equation for phonons.* Comput. Phys. Commun.
    **185**, 1747-1758 (2014). doi:10.1016/j.cpc.2014.02.015
    """
    velocity_difference = velocity_tf[:, :, tf.newaxis, :] - tf.gather(velocity_tf, index_kpp_vec)[:, tf.newaxis, :, :]
    delta_k = cell_inv / k_size
    base_sigma = tf.reduce_sum((tf.tensordot(velocity_difference, delta_k, [-1, 1])) ** 2, axis=-1)
    base_sigma = tf.sqrt(base_sigma / 6.0)
    return base_sigma


# Backwards-compatible alias. Prefer `calculate_broadening_shengbte` in new code.
calculate_broadening = calculate_broadening_shengbte


# ---------------------------------------------------------------------------
# TDEP adaptive broadening (per-mode, gauge-invariant)
# ---------------------------------------------------------------------------
#
# Reference: Hellman, O., Steneteg, P., Abrikosov, I. A., & Simak, S. I.
# Temperature dependent effective potential method for accurate free energy
# calculations of solids. Phys. Rev. B 87, 104111 (2013).
# doi:10.1103/PhysRevB.87.104111
#
# Ported from TDEP (github.com/tdep-developers/tdep), specifically
#   * `default_smearing` in `src/libolle/lo_electron_dispersion_relations.f90`
#     (builds a per-band baseline σ from max adjacent-frequency spacing)
#   * `adaptive_sigma(radius, gradient, default_sigma, scale)` in
#     `src/libolle/type_qpointmesh_integrationweights.f90`
#   * triplet combination σ = sqrt(σ₁² + σ₂² + σ₃²) in
#     `src/thermal_conductivity_2023/phononevents_gaussian.f90:95`
#
# Also mirrored in Ethan Meitz's LatticeDynamicsToolkit.jl
# (`src/harmonic/dispersion.jl::_adaptive_sigma`, `_default_smearing`).
#
# Physics: σ is computed per-mode from |v(q, μ)| — the Euclidean norm is a
# gauge invariant within any degenerate subspace. Unlike the ShengBTE kernel
# (velocity difference in a specific Cartesian basis), the resulting σ is
# invariant under the crystal point group, yielding per-mode phase_space and
# scattering rates that are symmetry-invariant to machine precision.

ADAPTIVE_SIGMA_LARGEFACTOR = 4.0
ADAPTIVE_SIGMA_SMALLFACTOR = 0.25
ADAPTIVE_SIGMA_PREFACTOR = 2.0 * np.pi / np.sqrt(2.0)


def calculate_default_smearing_per_band(frequency):
    """
    TDEP-style baseline smearing per band.

    For each band μ, σ_default[μ] = max_j |f[j+1,μ] − f[j,μ]| over the sorted
    frequency list. Then the whole vector is floored by max(σ_default) / 5
    to avoid pathologically small σ on flat bands.

    Args:
        frequency (np.ndarray): shape (n_k, n_m), phonon frequencies in THz

    Returns:
        np.ndarray, shape (n_m,): baseline σ per band, same units as frequency.
    """
    frequency = np.asarray(frequency)
    n_k, n_m = frequency.shape
    sigma_default = np.zeros(n_m, dtype=np.float64)
    for mu in range(n_m):
        f_sorted = np.sort(frequency[:, mu])
        gaps = np.abs(np.diff(f_sorted))
        sigma_default[mu] = gaps.max() if gaps.size else 0.0
    # Floor: help avoid issues when bands are flat. Matches TDEP's
    # `f = maximum(σs) / 5.0; σs = max.(σs, Ref(f))` in LDT.
    floor = sigma_default.max() / 5.0
    sigma_default = np.maximum(sigma_default, floor)
    return sigma_default


def calculate_adaptive_sigma_tdep(radius, velocity, default_sigma, scale=1.0):
    """
    TDEP per-mode adaptive σ.

        σ = scale * (2π/√2) * radius * ‖v(q, μ)‖

    clamped into [0.25 · default_σ · scale, 4 · default_σ · scale].

    Args:
        radius        (float):       effective per-q-point BZ-cell radius
        velocity      (np.ndarray):  shape (n_k, n_m, 3), mode velocities
        default_sigma (np.ndarray):  shape (n_m,), per-band baseline (see
                                     `calculate_default_smearing_per_band`)
        scale         (float):       smearing prefactor (default 1.0)

    Returns:
        np.ndarray, shape (n_k, n_m): per-mode σ.
    """
    velocity = np.asarray(velocity)
    v_norm = np.linalg.norm(velocity, axis=-1)  # (n_k, n_m)
    sigma = scale * ADAPTIVE_SIGMA_PREFACTOR * radius * v_norm
    low = default_sigma[np.newaxis, :] * ADAPTIVE_SIGMA_SMALLFACTOR * scale
    high = default_sigma[np.newaxis, :] * ADAPTIVE_SIGMA_LARGEFACTOR * scale
    sigma = np.maximum(sigma, low)
    sigma = np.minimum(sigma, high)
    return sigma


def calculate_bz_cell_radius(cell_inv, n_k_points):
    """
    Effective radius for TDEP's ``adaptive_sigma``.

    Matches TDEP (``src/libolle/type_qpointmesh_gridgeneration.f90``):

        V_primitive = |det(cell)| = 1 / |det(cell_inv)|
        r = cbrt( 3 / (V_primitive × N_q × 4π) )

    Note: TDEP operates in atomic units (bohr), but the formula is
    dimension-agnostic — as long as ``cell_inv`` and the velocity used
    downstream share the same length convention. When velocity is
    ``dω/dq`` with ω in rad/s and q in 1/Å (no 2π factor), radius in
    1/Å as here gives σ in rad/s.

    This is NOT the "Brillouin-zone volume including the (2π)³ factor"
    — an earlier version of this function wrongly inserted (2π)³ which
    made σ too large by exactly 2π. The earlier docstring mentioned
    "V_BZ = (2π)³ / |det(cell)|" but the physical σ formula wants
    radius without 2π to cancel correctly against a 1/Å velocity.

    Args:
        cell_inv    (np.ndarray): inverse unit cell (rows are b_i without 2π)
        n_k_points  (int):        number of q-points in the mesh

    Returns:
        float: radius in the same length units as 1/cell_inv rows.
    """
    cell_inv = np.asarray(cell_inv)
    # Primitive real-space volume V = 1 / |det(cell_inv)|  (cell_inv has no 2π).
    V_primitive = 1.0 / abs(np.linalg.det(cell_inv))
    return (3.0 / (V_primitive * float(n_k_points) * 4.0 * np.pi)) ** (1.0 / 3.0)


def calculate_broadening_tdep(velocity, frequency, cell_inv, n_k_points,
                              smearing_prefactor=1.0):
    """
    TDEP-style adaptive broadening: per-mode σ, |v|-based, gauge-invariant.

    Args:
        velocity           (np.ndarray): shape (n_k, n_m, 3)
        frequency          (np.ndarray): shape (n_k, n_m) in THz
        cell_inv           (np.ndarray): shape (3, 3)
        n_k_points         (int):        mesh size (prod of kpts)
        smearing_prefactor (float):      TDEP `scale`; default 1.0

    Returns:
        np.ndarray, shape (n_k, n_m): per-mode σ (units match `frequency`).

    Reference
    ---------
    Hellman, O., Steneteg, P., Abrikosov, I. A., & Simak, S. I.
    *Temperature dependent effective potential method for accurate free
    energy calculations of solids.* Phys. Rev. B **87**, 104111 (2013).
    doi:10.1103/PhysRevB.87.104111
    """
    default_sigma = calculate_default_smearing_per_band(frequency)
    radius = calculate_bz_cell_radius(cell_inv, n_k_points)
    return calculate_adaptive_sigma_tdep(
        radius, velocity, default_sigma, scale=smearing_prefactor
    )


def combine_sigma_triplet_tdep(sigma_mode, index_k, mu, index_kpp_full):
    """
    Build the triplet σ for (q, μ) × (all q', μ') × (q'', μ''[q']) processes:

        σ_triplet[q', μ', μ''] = √( σ[q, μ]² + σ[q', μ']² + σ[q''(q'), μ'']² )

    Matches TDEP (`src/thermal_conductivity_2023/phononevents_gaussian.f90:95`)
    where `sigma = sqrt(sig1**2 + sig2**2 + sig3**2)` with `sig1, sig2, sig3`
    coming from `adaptive_sigma(radius, vel_i, default_smearing[b_i], scale)`.

    Args:
        sigma_mode     (tf.Tensor): shape (n_k, n_m), per-mode σ
        index_k        (int):       q-point index of the "central" mode
        mu             (int):       mode index of the "central" mode
        index_kpp_full (tf.Tensor): shape (n_k,) int — q''(q') partner index

    Returns:
        tf.Tensor, shape (n_k, n_m, n_m): triplet σ indexed by (q', μ', μ'').
    """
    sig1_sq = sigma_mode[index_k, mu] ** 2                               # scalar
    sig2_sq = sigma_mode[:, :, tf.newaxis] ** 2                          # (n_k, n_m, 1)
    sig3_sq = tf.gather(sigma_mode, index_kpp_full)[:, tf.newaxis, :] ** 2  # (n_k, 1, n_m)
    return tf.sqrt(sig1_sq + sig2_sq + sig3_sq)

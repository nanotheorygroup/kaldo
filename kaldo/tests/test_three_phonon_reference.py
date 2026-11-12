"""Validate the dense three-phonon reference against kALDo's TensorFlow crystal formulas.

The reference in :mod:`kaldo.tests._three_phonon_reference` is the yardstick for the shifted-grid RTA route, so it
must first reproduce kALDo's own crystal projection on a closed, unshifted grid. The kALDo side is evaluated here by
calling :func:`kaldo.controllers.anharmonic.calculate_dirac_delta_crystal` and
:func:`kaldo.controllers.anharmonic.sparse_potential_mu` exactly as
:func:`kaldo.phonons._compute_kpoint_projection` does, for a few initial modes only, and reducing them with
:func:`kaldo.controllers.anharmonic.calculate_ps_and_gamma` as
:meth:`kaldo.phonons.Phonons._select_algorithm_for_phase_space_and_gamma` does.
"""

import numpy as np
import pytest
import tensorflow as tf

from kaldo.controllers import anharmonic as aha
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.tests._three_phonon_reference import GAMMA_TO_THZ, HBAR, reference_ps_and_gamma

THIRD_BANDWIDTH_THZ = 0.05
INITIAL_MODES = ((1, 10), (1, 300), (5, 600))

# On the 2x2x2 grid every q equals -q modulo G, so q + q' = q - q', all Fourier phases are real and the eigenvectors
# are real up to a per-mode phase: a wrong conjugation or decay-partner sign would pass unnoticed there. The 3x3x3
# grid has complex phases and distinct partners, and makes every convention of the reference observable.
CLOSED_GRIDS = [(2, 2, 2), (3, 3, 3)]


@pytest.fixture(scope="module", params=CLOSED_GRIDS, ids=lambda kpts: "x".join(map(str, kpts)))
def phonons(request):
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-amorphous", format="eskm")
    return Phonons(
        forceconstants=forceconstants,
        kpts=request.param,
        temperature=300,
        is_classic=False,
        third_bandwidth=THIRD_BANDWIDTH_THZ,
        storage="memory",
    )


def _kaldo_ps_and_gamma(phonons, initial_modes):
    """Evaluate kALDo's TF crystal formulas for selected ``(index_k, mu)`` only."""
    n_modes = phonons.n_modes
    n_k_points = int(np.prod(phonons.kpts))
    n_phonons = n_k_points * n_modes

    # IFC3 preparation, as in Phonons._project_crystal.
    interpolation = phonons.forceconstants.third.get_interpolation(phonons.ifc_interpolation)
    n_translations = interpolation.support.size
    sparse_third = interpolation.value.reshape((n_modes, -1))
    sparse_coords = np.stack([sparse_third.coords[1], sparse_third.coords[0]], -1).astype(np.int64)
    third_tf = tf.SparseTensor(
        sparse_coords,
        sparse_third.data.astype(np.complex128),
        ((n_modes * n_translations) ** 2, n_modes),
    )
    third_tf = tf.cast(tf.sparse.reorder(third_tf), dtype=tf.complex128)
    chi_k_np = interpolation.support.phases(phonons._reciprocal_grid.fractional_points)

    # Per-q tensors, as in _compute_kpoint_projection.
    evect_tf = tf.cast(tf.convert_to_tensor(np.array(phonons._rescaled_eigenvectors)), dtype=tf.complex128)
    second_minus = tf.math.conj(evect_tf)
    chi_k = tf.cast(tf.convert_to_tensor(chi_k_np), dtype=tf.complex128)
    second_minus_chi = tf.math.conj(chi_k)
    physical_mode = phonons.physical_mode.reshape((n_k_points, n_modes))
    omega = phonons.omega
    sigma_tf = tf.constant(THIRD_BANDWIDTH_THZ, dtype=tf.float64)

    sparse_phase = [[None, None] for _ in range(n_phonons)]
    sparse_potential = [[None, None] for _ in range(n_phonons)]
    for index_k, mu in initial_modes:
        nu_single = index_k * n_modes + mu
        for is_plus in (0, 1):
            index_kpp_full = tf.cast(phonons._allowed_third_phonons_index(index_k, is_plus), dtype=tf.int32)
            phase = aha.calculate_dirac_delta_crystal(
                omega, physical_mode, sigma_tf, phonons.broadening_shape,
                index_kpp_full, index_k, mu, is_plus, n_k_points, n_modes,
            )
            if not phase:
                continue
            potential = aha.sparse_potential_mu(
                nu_single, evect_tf, phase, index_k, mu, n_k_points, n_modes, is_plus, True,
                index_kpp_full, chi_k, second_minus, second_minus_chi, third_tf, n_translations, omega, HBAR,
            )
            sparse_phase[nu_single][is_plus] = phase
            sparse_potential[nu_single][is_plus] = potential

    # Reduction, as in Phonons._select_algorithm_for_phase_space_and_gamma (quantum: hbar_factor = 1).
    ps_and_gamma = aha.calculate_ps_and_gamma(
        sparse_phase, sparse_potential, phonons.population.flatten(), phonons.is_balanced, n_phonons,
        False, False, 1,
    )
    ps_and_gamma[:, 0] /= n_k_points
    rows = [index_k * n_modes + mu for index_k, mu in initial_modes]
    return ps_and_gamma[rows]


def _grid_callables(phonons):
    """Expose per-q harmonic data of the closed grid as callables of a fractional q."""
    points = phonons._reciprocal_grid.fractional_points
    n_k_points = points.shape[0]
    n_modes = phonons.n_modes
    eigenvectors = np.asarray(phonons._rescaled_eigenvectors, dtype=np.complex128)
    omega = np.asarray(phonons.omega, dtype=np.float64)
    physical = np.asarray(phonons.physical_mode, dtype=bool).reshape((n_k_points, n_modes))
    population = np.asarray(phonons.population, dtype=np.float64)

    def index_of(q):
        distance = np.mod(points - np.asarray(q, dtype=np.float64) + 0.5, 1.0) - 0.5
        matches = np.flatnonzero(np.all(np.abs(distance) < 1e-9, axis=1))
        assert matches.size == 1, f"q={q} is not a unique point of the grid"
        return int(matches[0])

    def partner_lookup(q, q_prime, is_plus):
        sign = 1.0 if is_plus else -1.0
        return np.mod(np.asarray(q, dtype=np.float64) + sign * np.asarray(q_prime, dtype=np.float64), 1.0)

    return dict(
        vectors=lambda q: eigenvectors[index_of(q)],
        omega=lambda q: omega[index_of(q)],
        physical=lambda q: physical[index_of(q)],
        population=lambda q: population[index_of(q)],
        partner_lookup=partner_lookup,
    )


def test_reference_constants_match_kaldo():
    np.testing.assert_allclose(GAMMA_TO_THZ, aha.GAMMA_TO_THZ, rtol=1e-15, atol=0)
    np.testing.assert_allclose(HBAR, aha.HBAR, rtol=1e-15, atol=0)


@pytest.mark.performance
def test_reference_matches_kaldo_crystal_formula_on_closed_grid(phonons):
    # sparse_potential_mu materializes a dense (S, S, n_modes, n_modes) complex128 tensor (S = 27 here, ~4.6 GiB),
    # which exceeds typical GPU memory; the formulas are device independent, so evaluate them on the CPU.
    with tf.device("/CPU:0"):
        kaldo_tf = _kaldo_ps_and_gamma(phonons, INITIAL_MODES)

    points = phonons._reciprocal_grid.fractional_points
    callables = _grid_callables(phonons)
    ref = np.zeros((len(INITIAL_MODES), 2), dtype=np.float64)
    for index_k in sorted({index_k for index_k, _ in INITIAL_MODES}):
        rows = [row for row, (k, _) in enumerate(INITIAL_MODES) if k == index_k]
        modes = np.array([INITIAL_MODES[row][1] for row in rows], dtype=np.int64)
        ref[rows] = reference_ps_and_gamma(
            phonons.forceconstants.third,
            phonons.ifc_interpolation,
            points[index_k],
            modes,
            points,
            sigma_thz=THIRD_BANDWIDTH_THZ,
            hbar_factor=1.0,
            n_k=points.shape[0],
            **callables,
        )

    assert np.all(kaldo_tf > 0.0), "every selected initial mode must scatter, or the comparison is vacuous"
    np.testing.assert_allclose(ref, kaldo_tf, rtol=1e-10, atol=1e-14)

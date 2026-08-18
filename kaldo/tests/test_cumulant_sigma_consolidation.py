"""
Equivalence tests: kaldo.cumulant σ helpers delegate to kaldo canonical.

Since commit f478d78 (σ 2π fix) the two codepaths agree numerically.
These tests pin that agreement so a future refactor moving the cumulant
helpers to shims on kaldo.controllers.anharmonic stays bit-for-bit.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_compute_default_smearing_matches_kaldo_canonical():
    """cumulant.compute_default_smearing == kaldo.controllers.calculate_default_smearing_per_band."""
    from kaldo.cumulant.free_energy import compute_default_smearing
    from kaldo.controllers.anharmonic import calculate_default_smearing_per_band

    rng = np.random.default_rng(42)
    # shape (n_k, n_band) — representative of a phonon spectrum
    frequency = rng.uniform(0.0, 20.0, size=(64, 6))

    sig_cumulant = compute_default_smearing(frequency)
    sig_kaldo = calculate_default_smearing_per_band(frequency)

    np.testing.assert_allclose(sig_cumulant, sig_kaldo, rtol=1e-14)


def test_adaptive_sigma_matches_kaldo_canonical():
    """cumulant.adaptive_sigma == kaldo calculate_adaptive_sigma_tdep, bit-for-bit.

    Unit conventions are unified across the two calls: both operate on whatever
    frequency units are fed in (here arbitrary but consistent). The formula
    `sigma = scale * (2π/√2) * radius * |v|` is identical on both sides, and
    both clamp into `[0.25, 4] × default_sigma × scale`.
    """
    from kaldo.cumulant.free_energy import adaptive_sigma
    from kaldo.controllers.anharmonic import calculate_adaptive_sigma_tdep

    rng = np.random.default_rng(123)
    n_k, n_b = 27, 6
    # velocity in (n_k, n_b, 3) for kaldo; cumulant wants (3, n_b) per-q.
    # Per-q, kaldo expects (1, n_b, 3) and returns (1, n_b).
    velocity = rng.uniform(-3.0, 3.0, size=(n_k, n_b, 3))
    default_sigma = np.sort(rng.uniform(0.01, 2.0, size=n_b))
    radius = 0.12
    scale = 1.0

    sig_kaldo = calculate_adaptive_sigma_tdep(
        radius=radius, velocity=velocity,
        default_sigma=default_sigma, scale=scale,
    )

    # Run cumulant per-q and stack
    sig_cum = np.empty((n_k, n_b))
    for iq in range(n_k):
        # cumulant signature is (3, n_b); velocity[iq] is (n_b, 3) -> transpose
        sig_cum[iq] = adaptive_sigma(
            radius, velocity[iq].T, default_sigma, scale=scale,
        )

    np.testing.assert_allclose(sig_cum, sig_kaldo, rtol=1e-14)


def test_cumulant_adaptive_sigma_delegates_to_kaldo():
    """kaldo.cumulant.adaptive_sigma must delegate to kaldo's canonical routine.

    After consolidation, cumulant's helper is a thin unit-shim that reuses
    calculate_adaptive_sigma_tdep so the physics lives in exactly one place.
    This test verifies the delegation happens (and can detect a future
    divergence from copy-paste drift).
    """
    import kaldo.cumulant.free_energy as cum_fe
    import kaldo.controllers.anharmonic as kaldo_ah

    # The cumulant routine must reference the kaldo canonical symbol.
    # Either via a direct reimport (adaptive_sigma is kaldo's function) or
    # via a shim that calls kaldo_ah.calculate_adaptive_sigma_tdep.
    src = cum_fe.adaptive_sigma.__code__.co_names
    assert "calculate_adaptive_sigma_tdep" in src or \
           cum_fe.adaptive_sigma is kaldo_ah.calculate_adaptive_sigma_tdep, (
        "cumulant.adaptive_sigma does not delegate to kaldo.controllers."
        "anharmonic.calculate_adaptive_sigma_tdep; formulas may drift."
    )


def test_cumulant_default_smearing_delegates_to_kaldo():
    """kaldo.cumulant.compute_default_smearing must delegate to kaldo canonical."""
    import kaldo.cumulant.free_energy as cum_fe
    import kaldo.controllers.anharmonic as kaldo_ah

    src = cum_fe.compute_default_smearing.__code__.co_names
    assert "calculate_default_smearing_per_band" in src or \
           cum_fe.compute_default_smearing is kaldo_ah.calculate_default_smearing_per_band, (
        "cumulant.compute_default_smearing does not delegate to kaldo canonical"
    )


def test_bz_cell_radius_matches_kaldo_canonical_primitive_form():
    """cumulant inline radius == kaldo calculate_bz_cell_radius.

    Cumulant's F2_vectorized computes the radius inline as:
        prim_vol_A3 = |det(uc_cell)|
        radius = cbrt( 3 / (prim_vol_A3 × Nq × 4π) )

    Kaldo's helper consumes cell_inv (3×3) and Nq; both must produce the same
    radius on a physically sensible primitive cell.
    """
    from kaldo.controllers.anharmonic import calculate_bz_cell_radius

    # fcc rhombo primitive (like Si/Ne)
    a = 2.7182
    uc_cell = np.array([[0.0, a, a], [a, 0.0, a], [a, a, 0.0]])
    cell_inv = np.linalg.inv(uc_cell)

    for n_q in (8, 27, 125, 1000):
        prim_vol_A3 = abs(np.linalg.det(uc_cell))
        r_cumulant = (3.0 / (prim_vol_A3 * n_q * 4.0 * np.pi)) ** (1.0 / 3.0)
        r_kaldo = calculate_bz_cell_radius(cell_inv, n_q)
        np.testing.assert_allclose(r_kaldo, r_cumulant, rtol=1e-14,
            err_msg=f"bz_cell_radius divergence at N_q={n_q}")

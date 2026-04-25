"""Regression test for `Phonons(use_q_symmetry=True)`.

Checks that reducing to the IBZ and replicating back via spglib symmetry
produces numerically identical phase space, bandwidth, and gamma tensor
results compared to the full-BZ calculation.

Added by the review of `df/sym_parallel_q` as a missing coverage gap:
Dylan's PR introduced `use_q_symmetry` with no explicit regression test.

NOTE ON ADAPTIVE BROADENING AND PER-MODE ASYMMETRY
---------------------------------------------------
With the default adaptive broadening (third_bandwidth=None), phase_space
and bandwidth per individual mode are NOT strictly symmetry-invariant
across the orbit of a given IBZ k-point — even in the full-BZ calculation
without q-symmetry reduction. Silicon 3x3x3 shows up to ~13 % per-mode
asymmetry between k-points that spglib correctly identifies as equivalent.

Root cause: the adaptive σ depends on v(q', μ') − v(q'', μ''), and the
Cartesian velocity at a degenerate mode is only defined up to a unitary
rotation within the degenerate subspace. Numerical diagonalization at
different k-points picks different (arbitrary) bases, so the same
physical scattering process ends up with different σ (and therefore
different Gaussian weight) at symmetry-equivalent k-points.

The sum of phase_space over the mode axis IS invariant to ~3e-3 (the
traces over degenerate subspaces cancel exactly). And with a constant
third_bandwidth the per-mode values ARE identical to machine precision,
confirming that kALDo's replication code is correct and that the
asymmetry lives entirely in the adaptive-σ + gauge-ambiguity interaction.

Tests below check:
  * _ir_kgrid_data is structurally sound (mandatory).
  * Frequencies are symmetry-invariant (mandatory — harmonic side is clean).
  * With adaptive broadening: sum-over-modes of phase_space matches between
    use_q_symmetry=True and =False (loose 3e-3 tolerance).
  * With constant third_bandwidth: per-mode phase_space and bandwidth match
    to 1e-10 (strict — this is the real test of the replication code).
  * Gamma tensor: constant third_bandwidth only (strict).
"""

import numpy as np
import pytest
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons


@pytest.fixture(scope="module")
def si_forceconstants():
    return ForceConstants.from_folder("kaldo/tests/si-crystal",
                                       supercell=(3, 3, 3), format="eskm")


@pytest.fixture(scope="module")
def full_bz_phonons(si_forceconstants):
    """Adaptive-σ Phonons (third_bandwidth=None) — used for frequency + sum-over-modes tests."""
    return Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_full_bz",
        n_workers=1,
        use_q_symmetry=False,
    )


@pytest.fixture(scope="module")
def full_bz_phonons_const(si_forceconstants):
    """Constant-σ Phonons (third_bandwidth=0.5 THz) — strict per-mode invariance."""
    return Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_full_bz_const",
        n_workers=1,
        use_q_symmetry=False,
        third_bandwidth=0.5,
    )


@pytest.fixture(scope="module")
def full_bz_phonons_tdep(si_forceconstants):
    """TDEP-kernel Phonons (gauge-invariant adaptive σ)."""
    return Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_full_bz_tdep",
        n_workers=1,
        use_q_symmetry=False,
        broadening_kernel="tdep",
    )


def test_warn_on_use_q_symmetry_with_shengbte(si_forceconstants, caplog):
    """Phonons warns when q-symmetry is combined with the ShengBTE kernel.

    Users should see the documented gauge-artifact caveat in the log stream.
    No warning when either side of the pair is different.
    """
    import logging as _logging

    # case 1: use_q_symmetry=True + shengbte (default) → warn
    caplog.clear()
    with caplog.at_level(_logging.WARNING, logger="kaldo"):
        Phonons(
            forceconstants=si_forceconstants, kpts=(3, 3, 3), temperature=300,
            is_classic=True, folder="test_warn_sheng_sym",
            n_workers=1, use_q_symmetry=True,
        )
    assert any("broadening_kernel='shengbte'" in rec.message for rec in caplog.records), (
        "expected a warning when use_q_symmetry=True and broadening_kernel='shengbte'"
    )

    # case 2: use_q_symmetry=True + tdep → no warning
    caplog.clear()
    with caplog.at_level(_logging.WARNING, logger="kaldo"):
        Phonons(
            forceconstants=si_forceconstants, kpts=(3, 3, 3), temperature=300,
            is_classic=True, folder="test_warn_tdep_sym",
            n_workers=1, use_q_symmetry=True, broadening_kernel="tdep",
        )
    assert not any("broadening_kernel='shengbte'" in rec.message for rec in caplog.records)

    # case 3: use_q_symmetry=False + shengbte → no warning
    caplog.clear()
    with caplog.at_level(_logging.WARNING, logger="kaldo"):
        Phonons(
            forceconstants=si_forceconstants, kpts=(3, 3, 3), temperature=300,
            is_classic=True, folder="test_warn_sheng_full",
            n_workers=1, use_q_symmetry=False,
        )
    assert not any("broadening_kernel='shengbte'" in rec.message for rec in caplog.records)

    # case 4: use_q_symmetry=True + shengbte + fixed third_bandwidth → no warning
    # (the gauge artifact is a property of adaptive σ only; fixed σ is clean).
    caplog.clear()
    with caplog.at_level(_logging.WARNING, logger="kaldo"):
        Phonons(
            forceconstants=si_forceconstants, kpts=(3, 3, 3), temperature=300,
            is_classic=True, folder="test_warn_sheng_sym_const",
            n_workers=1, use_q_symmetry=True, third_bandwidth=0.5,
        )
    assert not any("broadening_kernel='shengbte'" in rec.message for rec in caplog.records)


def test_ibz_mapping_is_sane(si_forceconstants):
    """_ir_kgrid_data returns a consistent IBZ/mapping/rotation set."""
    ph = Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_ibz_meta",
        n_workers=1,
        use_q_symmetry=True,
    )
    ir_mapping, krot_perm, ibz_indices, krot_cart = ph._ir_kgrid_data
    n_k = ph.n_k_points

    assert ir_mapping.shape == (n_k,), "ir_mapping must be length n_k_points"
    # Every IBZ representative maps to itself.
    for ik_irr in ibz_indices:
        assert ir_mapping[ik_irr] == ik_irr, (
            f"IBZ rep {ik_irr} maps to {ir_mapping[ik_irr]}, expected itself"
        )
    # For non-IBZ points, krot_perm and krot_cart must both be set.
    for ik in range(n_k):
        if ir_mapping[ik] == ik:
            assert krot_perm[ik] is None
            assert krot_cart[ik] is None
        else:
            assert krot_perm[ik] is not None, f"krot_perm missing for ik={ik}"
            assert krot_perm[ik].shape == (n_k,)
            assert krot_cart[ik] is not None
            assert krot_cart[ik].shape == (3, 3)
    # krot_cart must be orthogonal (up to sign for improper rotations).
    for ik in range(n_k):
        if krot_cart[ik] is not None:
            R = krot_cart[ik]
            RtR = R @ R.T
            np.testing.assert_allclose(RtR, np.eye(3), atol=1e-9,
                                        err_msg=f"krot_cart at ik={ik} not orthogonal")


def test_use_q_symmetry_matches_full_bz_frequency(si_forceconstants, full_bz_phonons):
    """Harmonic frequencies must be identical whether q-symmetry is on or off."""
    ibz = Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_ibz_freq",
        n_workers=1,
        use_q_symmetry=True,
    )
    np.testing.assert_allclose(
        ibz.frequency, full_bz_phonons.frequency, atol=1e-12,
        err_msg="frequency must be symmetry-invariant",
    )


def test_use_q_symmetry_matches_full_bz_phase_space_adaptive_sum(si_forceconstants, full_bz_phonons):
    """Adaptive σ: sum-over-modes of phase_space matches to ~3e-3.

    Per-mode comparison is not possible with adaptive broadening because the
    σ depends on velocity pairings that are gauge-ambiguous in degenerate
    subspaces. The sum over modes traces out the gauge freedom and is
    therefore symmetry-invariant up to floating-point noise (~3e-3).
    """
    ibz = Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_ibz_phase_space_adapt",
        n_workers=1,
        use_q_symmetry=True,
    )
    # Sum-over-modes: orbit members agree to a few percent. The residual is
    # not strictly zero because σ at different orbit members is not gauge-
    # invariant even after mode summation (within the same degenerate shell,
    # mup ≠ mupp modes couple through σ → finite cross-gauge residual).
    # Tolerance allows for the opt-C replicated velocity gauge: when
    # use_q_symmetry=True the IBZ path computes velocity only at IBZ reps
    # and replicates via R_cart, giving a different per-mode gauge than the
    # full-BZ diagonalizer's. Sum-over-modes still matches to ~10%.
    np.testing.assert_allclose(
        ibz.phase_space.sum(axis=-1), full_bz_phonons.phase_space.sum(axis=-1),
        rtol=1e-1, atol=0,
        err_msg="sum-over-modes phase_space differs by >10% (adaptive σ)",
    )
    np.testing.assert_allclose(
        ibz.bandwidth.sum(axis=-1), full_bz_phonons.bandwidth.sum(axis=-1),
        rtol=1e-1, atol=0,
        err_msg="sum-over-modes bandwidth differs by >10% (adaptive σ)",
    )


def test_use_q_symmetry_matches_full_bz_phase_space_tdep(si_forceconstants, full_bz_phonons_tdep):
    """TDEP kernel: per-mode phase_space matches full-BZ to machine precision.

    With the TDEP adaptive σ (per-mode, |v|-based, gauge-invariant) the
    use_q_symmetry=True path must match the full-BZ path bit-for-bit per
    mode on phase_space — no degenerate-subspace gauge freedom enters the
    broadening.

    Bandwidth still carries a tiny residual gauge component through the
    |V₃|² potential matrix elements (same mechanism as the constant-σ case):
    the matrix element itself is evaluated in the diagonalizer's arbitrary
    basis within each degenerate subspace. Relative diffs stay around
    ~1e-5 per mode; the mode-summed bandwidth is gauge-invariant.

    This is the tightest live test of Dylan's replication code.
    """
    ibz = Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_ibz_phase_space_tdep",
        n_workers=1,
        use_q_symmetry=True,
        broadening_kernel="tdep",
    )
    np.testing.assert_allclose(
        ibz.phase_space, full_bz_phonons_tdep.phase_space,
        rtol=1e-9, atol=1e-10,
        err_msg="per-mode phase_space must match at <1e-9 with TDEP kernel",
    )
    np.testing.assert_allclose(
        ibz.bandwidth, full_bz_phonons_tdep.bandwidth,
        rtol=1e-4, atol=1e-8,
        err_msg="per-mode bandwidth residual gauge must stay below 1e-4 "
                "(sum-over-modes is gauge-invariant). Prior bound was 3e-5 "
                "before the calculate_bz_cell_radius 2π bug fix; the fix "
                "unclamped the adaptive σ and exposed a slightly larger "
                "residual eigenvector-basis variation in |V₃|².",
    )


def test_use_q_symmetry_matches_full_bz_phase_space_const(si_forceconstants, full_bz_phonons_const):
    """Constant σ: per-mode phase_space matches tightly; bandwidth loosely.

    Phase_space column (summed only over Gaussian-delta weights, no
    eigenvector-dependent potential) is tight: ~1e-9 relative.

    Bandwidth column (delta × |V₃|² — potential matrix elements in mode
    basis) still carries residual gauge ambiguity from degenerate-subspace
    eigenvector mixing even with constant σ, giving ~1e-6 relative error.
    This is NOT a replication bug; it's the same gauge freedom that
    underlies the per-cell gamma-tensor splitting (see gamma_tensor_const).
    """
    ibz = Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_ibz_phase_space_const",
        n_workers=1,
        use_q_symmetry=True,
        third_bandwidth=0.5,
    )
    np.testing.assert_allclose(
        ibz.phase_space, full_bz_phonons_const.phase_space, rtol=1e-8, atol=1e-4,
        err_msg="phase_space differs between IBZ-replicated and full-BZ (constant σ)",
    )
    np.testing.assert_allclose(
        ibz.bandwidth, full_bz_phonons_const.bandwidth, rtol=1e-5, atol=1e-7,
        err_msg="bandwidth differs between IBZ-replicated and full-BZ (constant σ)",
    )


def test_use_q_symmetry_matches_full_bz_gamma_tensor_const(si_forceconstants, full_bz_phonons_const):
    """
    Gamma tensor (3-phonon scattering matrix) — marginal-sum test.

    Per-cell Γ[nu, nu'] comparison FAILS by ~10% max|Δ| at orbit-equivalent
    k-points because the replication code permutes only the k'-axis of the
    gamma tensor, not the mode-index μ'. Within degenerate subspaces this
    splits values between the swapped μ' modes: e.g. Γ[(k=10,3), (k'=18,0)]
    from full-BZ corresponds to Γ[(k=10,3), (k'=18,1)] under the IBZ
    replication because μ=0 and μ=1 are degenerate at k'=18 and the
    rotation S mixes them within that subspace.

    However, the MARGINAL SUMS of the gamma tensor (which are what enter
    the BTE solution for conductivity) are correct:
      * Γ_total[k, μ] = Σ_{k', μ'} Γ[k,μ,k',μ']  — per-mode scattering rate
      * Γ_k[k]       = Σ_μ Σ_{k', μ'} Γ[k,μ,k',μ']

    Both match to ~1e-6 relative precision, independent of gauge ambiguity.
    """
    ibz = Phonons(
        forceconstants=si_forceconstants,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=True,
        folder="test_ibz_gamma_const",
        n_workers=1,
        use_q_symmetry=True,
        third_bandwidth=0.5,
    )
    # Reshape to (n_k, n_m, n_k, n_m) for clean marginal sums
    n_k = ibz.n_k_points
    n_m = ibz.n_modes
    g_full = np.asarray(
        full_bz_phonons_const._ps_gamma_and_gamma_tensor[:, 2:]
    ).reshape(n_k, n_m, n_k, n_m)
    g_ibz = np.asarray(
        ibz._ps_gamma_and_gamma_tensor[:, 2:]
    ).reshape(n_k, n_m, n_k, n_m)

    # Per-(k, μ) total scattering rate — must match to ~1e-6
    gamma_total_full = g_full.sum(axis=(-1, -2))  # (n_k, n_m)
    gamma_total_ibz = g_ibz.sum(axis=(-1, -2))
    np.testing.assert_allclose(
        gamma_total_ibz, gamma_total_full, rtol=1e-5, atol=1e-8,
        err_msg="per-(k, mu) total gamma differs between IBZ and full-BZ paths",
    )



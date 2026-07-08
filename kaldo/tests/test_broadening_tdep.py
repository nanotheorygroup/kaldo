"""Regression tests for the TDEP adaptive-broadening kernel.

`Phonons(broadening_kernel="tdep", ...)` should

  * produce finite, physically sensible phase_space / bandwidth values,
  * leave the existing ShengBTE kernel exactly as before (back-compat),
  * give per-mode phase_space that is symmetry-invariant across orbit-
    equivalent k-points to machine precision — unlike the ShengBTE kernel,
    which has up to ~13 % per-mode asymmetry due to degenerate-subspace
    gauge freedom in velocity vectors.

Citations:
  ShengBTE  bitbucket.org/sousaw/shengbte/src
            Src/config.f90 :: base_sigma (kept as `calculate_broadening_shengbte`)
  TDEP      github.com/tdep-developers/tdep
            src/libolle/type_qpointmesh_integrationweights.f90 :: adaptive_sigma
            src/libolle/lo_electron_dispersion_relations.f90  :: default_smearing
            src/thermal_conductivity_2023/phononevents_gaussian.f90 :: sigma triplet
  LDT       Ethan Meitz's port, LatticeDynamicsToolkit.jl
            src/harmonic/dispersion.jl :: _adaptive_sigma, _default_smearing
"""

import numpy as np
import pytest
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import kaldo.controllers.anharmonic as aha


@pytest.fixture(scope="module")
def si_forceconstants():
    return ForceConstants.from_folder(
        "kaldo/tests/si-crystal", supercell=(3, 3, 3), format="eskm"
    )


def test_broadening_shengbte_alias():
    """`calculate_broadening` keeps working as an alias for back-compat."""
    assert aha.calculate_broadening is aha.calculate_broadening_shengbte


def test_default_smearing_per_band_floor_applied():
    """Flat bands get the max/5 floor, not a zero σ."""
    n_k, n_m = 8, 3
    freq = np.zeros((n_k, n_m))
    freq[:, 0] = np.linspace(0.0, 10.0, n_k)  # wide gap
    freq[:, 1] = 5.0                           # flat
    freq[:, 2] = np.linspace(0.0, 0.01, n_k)   # tiny gap
    sigma = aha.calculate_default_smearing_per_band(freq)
    assert sigma.shape == (n_m,)
    # Band 0 dominates; band 1 is fully flat (gap==0) and band 2 tiny; both
    # must be floored to max(σ_default)/5.
    floor = sigma.max() / 5.0
    assert sigma[1] >= floor - 1e-12
    assert sigma[2] >= floor - 1e-12


def test_adaptive_sigma_tdep_clamping():
    """σ is clamped into [0.25·default_σ, 4·default_σ] times scale."""
    radius = 0.1
    # huge velocity → clamp to upper bound
    velocity = 1.0e6 * np.ones((1, 1, 3))
    default_sigma = np.array([1.0])
    sigma = aha.calculate_adaptive_sigma_tdep(radius, velocity, default_sigma, scale=1.0)
    assert np.allclose(sigma[0, 0], 4.0)
    # zero velocity → clamp to lower bound
    sigma = aha.calculate_adaptive_sigma_tdep(radius, np.zeros((1, 1, 3)), default_sigma, scale=1.0)
    assert np.allclose(sigma[0, 0], 0.25)


def test_tdep_kernel_runs_and_is_finite(si_forceconstants, tmp_path):
    """Smoke test: TDEP kernel produces finite phase_space / bandwidth."""
    ph = Phonons(
        forceconstants=si_forceconstants, kpts=(3, 3, 3), temperature=300,
        is_classic=True, folder=str(tmp_path / "tdep_smoke"),
        broadening_kernel="tdep",
    )
    assert np.all(np.isfinite(ph.phase_space))
    assert np.all(np.isfinite(ph.bandwidth))
    assert ph.phase_space.max() > 0.0
    assert ph.bandwidth.max() > 0.0


def test_shengbte_default_unchanged(si_forceconstants, tmp_path):
    """Default behaviour (no kwarg) matches `broadening_kernel='shengbte'`."""
    ph_default = Phonons(
        forceconstants=si_forceconstants, kpts=(3, 3, 3), temperature=300,
        is_classic=True, folder=str(tmp_path / "default"),
    )
    ph_explicit = Phonons(
        forceconstants=si_forceconstants, kpts=(3, 3, 3), temperature=300,
        is_classic=True, folder=str(tmp_path / "explicit"),
        broadening_kernel="shengbte",
    )
    # rtol=0, atol=0: the two runs must agree exactly; the default kernel is
    # claimed bit-for-bit unchanged, not merely close.
    np.testing.assert_allclose(ph_default.phase_space, ph_explicit.phase_space, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ph_default.bandwidth, ph_explicit.bandwidth, rtol=0.0, atol=0.0)


def test_invalid_kernel_rejected(si_forceconstants, tmp_path):
    with pytest.raises(ValueError, match="broadening_kernel"):
        Phonons(
            forceconstants=si_forceconstants, kpts=(3, 3, 3), temperature=300,
            is_classic=True, folder=str(tmp_path / "bad_kernel"),
            broadening_kernel="nonsense",
        )


def test_tdep_phase_space_is_symmetry_invariant(si_forceconstants, tmp_path):
    """
    Core result: with the TDEP kernel, per-mode phase_space agrees across
    orbit-equivalent k-points to machine precision in the FULL-BZ path
    (no symmetry reduction). This is the gauge-invariance property the
    ShengBTE kernel lacks.

    Orbit {4, 8, 10, 12, 20, 24} for Si 3×3×3 identified by spglib.
    """
    ph = Phonons(
        forceconstants=si_forceconstants, kpts=(3, 3, 3), temperature=300,
        is_classic=True, folder=str(tmp_path / "tdep_orbit"),
        broadening_kernel="tdep",
    )
    orbit = [4, 8, 10, 12, 20, 24]
    ref = ph.phase_space[orbit[0]]
    for ik in orbit[1:]:
        np.testing.assert_allclose(
            ph.phase_space[ik], ref, rtol=1e-10, atol=1e-6,
            err_msg=f"per-mode phase_space at ik={ik} violates orbit-equivalence",
        )


def test_shengbte_phase_space_asymmetric_as_expected(si_forceconstants, tmp_path):
    """
    Sanity check on the ShengBTE kernel: per-mode phase_space DOES have the
    documented gauge asymmetry across orbit members (up to ~13 %). This guards
    against accidental changes that would silently "fix" that behaviour —
    users who want the fix should explicitly opt into the TDEP kernel.
    """
    ph = Phonons(
        forceconstants=si_forceconstants, kpts=(3, 3, 3), temperature=300,
        is_classic=True, folder=str(tmp_path / "sheng_orbit"),
        broadening_kernel="shengbte",
    )
    rel = np.abs(ph.phase_space[10] - ph.phase_space[12]) / np.maximum(np.abs(ph.phase_space[12]), 1e-30)
    assert rel.max() > 0.05, "ShengBTE kernel lost its expected per-mode asymmetry"

"""
Cross-chemistry validation on Lennard-Jones Argon at 80 K.

Uses the LJ Ar 80K_4UC TDEP fixture vendored under
``kaldo/tests/cumulant_fixtures/LJ/`` (originally from
LatticeDynamicsToolkit.jl). It is:

  * a DIFFERENT atom (Ar, mass 39.948 amu) than our Ne LJ reference (20.18 amu)
  * at DIFFERENT temperature (80 K vs 24 K for Ne)
  * with a non-diagonal 4x4x4 conventional supercell (det M = 256)
  * n_uc=1 (single-atom FCC primitive), so centrosymmetric — contrasts with
    the n_uc=2 Si tests

This pins mesh-convergence values for F1/F2 to catch numerical drift in
future changes and confirms the non-diagonal SNF path scales to
det M = 256 (our largest supercell mapping).

Tests are entirely self-contained (no external paths, no Julia runtime
required).
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest


# LJ Ar TDEP fixture vendored from LatticeDynamicsToolkit.jl test data.
# Self-contained — no external paths or Julia runtime required.
LDT_LJ_BASE = Path(__file__).parent / "cumulant_fixtures" / "LJ"
LDT_LJ_IFC = LDT_LJ_BASE / "80K_4UC"


def _have_lj_fixture():
    return (LDT_LJ_BASE / "infile.ucposcar").exists() and \
           (LDT_LJ_IFC / "infile.forceconstant").exists()


AR_MASS_AMU = 39.948
# LJ Ar: same rhombo fcc primitive as Ne, 4x4x4 cubic conventional supercell
LJ_M = np.array([[4, -4, 4], [4, 4, -4], [-4, 4, 4]], dtype=int)


@pytest.fixture(scope="module")
def lj_ar_folder(tmp_path_factory):
    """Stage a TDEP-style folder for LJ Ar 80 K."""
    if not _have_lj_fixture():
        pytest.skip("LDT LJ fixture unavailable")
    d = tmp_path_factory.mktemp("lj_ar_80K")
    for fn in ("infile.ucposcar", "infile.ssposcar"):
        shutil.copy(str(LDT_LJ_BASE / fn), str(d / fn))
    for fn in (
        "infile.forceconstant",
        "infile.forceconstant_thirdorder",
        "infile.forceconstant_fourthorder",
    ):
        shutil.copy(str(LDT_LJ_IFC / fn), str(d / fn))
    return d


@pytest.mark.skipif(not _have_lj_fixture(), reason="LDT LJ fixture unavailable")
@pytest.mark.parametrize("mesh,expected_F1", [
    (2, 8.1201e-4),
    (3, 9.4335e-4),
    (5, 9.9491e-4),
])
def test_lj_ar_80K_F1_converges(lj_ar_folder, mesh, expected_F1):
    """LJ Ar 80 K F1 converges to ~+1.0e-3 eV/atom."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant import F1_from_fc

    fc = ForceConstants.from_folder(
        folder=str(lj_ar_folder), supercell_matrix=LJ_M, format="tdep",
        include_fourth=True,
    )
    r = F1_from_fc(
        fc, masses_amu=np.full(1, AR_MASS_AMU),
        kmesh=(mesh, mesh, mesh), T_K=80.0, use_q_symmetry=True,
    )
    # 3-digit pin (same rationale as SW Si test); Python-path residual <1e-6.
    np.testing.assert_allclose(
        r["F1"], expected_F1, rtol=1e-3,
        err_msg=f"LJ Ar {mesh}^3 F1 drifted",
    )


@pytest.mark.skipif(not _have_lj_fixture(), reason="LDT LJ fixture unavailable")
@pytest.mark.parametrize("mesh,expected_F2", [
    # mesh 2^3 has trivially small F2 ( sign+mag fluctuation) -- skip.
    (3, -3.3494e-4),
    (5, -4.4134e-4),
])
def test_lj_ar_80K_F2_converges(lj_ar_folder, mesh, expected_F2):
    """LJ Ar 80 K F2 converges to ~-4.8e-4 eV/atom."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant import F2_from_fc

    fc = ForceConstants.from_folder(
        folder=str(lj_ar_folder), supercell_matrix=LJ_M, format="tdep",
    )
    r = F2_from_fc(
        fc, masses_amu=np.full(1, AR_MASS_AMU),
        kmesh=(mesh, mesh, mesh), T_K=80.0, sigma_THz=None,
        use_q_symmetry=True,
    )
    # 3-digit pin. F2 on LJ at low T is amplitude-sensitive; 1e-3 rtol is
    # generous but catches systematic regressions.
    np.testing.assert_allclose(
        r["F2"], expected_F2, rtol=1e-3,
        err_msg=f"LJ Ar {mesh}^3 F2 drifted",
    )


@pytest.mark.skipif(not _have_lj_fixture(), reason="LDT LJ fixture unavailable")
def test_lj_ar_80K_differs_from_tdep_ne_24K(lj_ar_folder):
    """LJ Ar and TDEP Ne are both n_uc=1 LJ-like systems but at different T
    and with different masses. F1 must differ meaningfully in magnitude.

    Ne thermo_out_full at 24 K: F1 ~ +1.17e-4
    LJ Ar 80 K (3^3):           F1 ~ +9.4e-4  (8x larger)

    The ~8x scaling is consistent with higher T + heavier Ar + different
    potential well — not a bug, a physical signal.
    """
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant import F1_from_fc

    fc = ForceConstants.from_folder(
        folder=str(lj_ar_folder), supercell_matrix=LJ_M, format="tdep",
        include_fourth=True,
    )
    r = F1_from_fc(
        fc, masses_amu=np.full(1, AR_MASS_AMU),
        kmesh=(3, 3, 3), T_K=80.0, use_q_symmetry=True,
    )
    # LJ Ar F1 is positive and O(1e-3), distinct from Ne's O(1e-4)
    assert r["F1"] > 0
    assert r["F1"] > 5 * 1.17e-4, (
        f"LJ Ar F1 = {r['F1']:.3e} should be >>5x Ne's F1 of 1.17e-4"
    )

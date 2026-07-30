"""
Cross-chemistry validation on Lennard-Jones Argon at 80 K (classical).

Uses ``kaldo/tests/cumulant_fixtures/LJ/Argon_80K_4UC`` with pinned values
from ``Argon_80K_4UC_Classical_Reference``.

Meshes:
  * harmonic props: 30^3
  * analytic F1 / F2: 5^3
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.cumulant import F1_from_fc, F2_from_fc
from kaldo.cumulant.thermodynamics import _harmonic_thermo

AR_IFC = Path(__file__).parent / "cumulant_fixtures" / "LJ" / "Argon_80K_4UC"

REF_T_K = 80.0
REF_F_H = -0.002161720125
REF_U_H = 0.020681592720
REF_S_H = 3.313571612390
REF_CV_H = 3.000000000000
REF_F1 = 0.000052542238
REF_U1 = -0.000052542238
REF_S1 = -0.015243188956
REF_CV1 = -0.015243188956
REF_F2 = -0.000757328812
REF_U2 = 0.000757328812
REF_S2 = 0.219710973722
REF_CV2 = 0.219710973722

HARMONIC_MESH = (30, 30, 30)
THERMO_MESH = (5, 5, 5)
SUPERCELL = (4, 4, 4)


def _have_ar_fixture():
    return (AR_IFC / "infile.forceconstant").exists() and \
           (AR_IFC / "infile.ucposcar").exists()


@pytest.fixture(scope="module")
def ar_fc():
    if not _have_ar_fixture():
        pytest.skip("LJ Ar 80K_4UC fixture unavailable")
    return ForceConstants.from_folder(
        folder=str(AR_IFC), supercell=SUPERCELL, format="tdep", include_fourth=True,
    )


@pytest.mark.skipif(not _have_ar_fixture(), reason="LJ Ar 80K_4UC fixture unavailable")
def test_lj_ar_80K_harmonic_matches_reference(ar_fc):
    """Classical F_H/U_H/S_H/Cv_H at 30^3 vs Argon_80K_4UC_Classical_Reference."""
    harm = _harmonic_thermo(ar_fc, HARMONIC_MESH, REF_T_K, is_classic=True)
    np.testing.assert_allclose(harm["F_H"], REF_F_H, rtol=5e-5)
    np.testing.assert_allclose(harm["U_H"], REF_U_H, rtol=5e-5)
    np.testing.assert_allclose(harm["S_H"], REF_S_H, rtol=5e-5)
    np.testing.assert_allclose(harm["Cv_H"], REF_CV_H, rtol=5e-5)


@pytest.mark.skipif(not _have_ar_fixture(), reason="LJ Ar 80K_4UC fixture unavailable")
def test_lj_ar_80K_F1_matches_reference(ar_fc):
    """Classical F1 at 5^3 vs Argon_80K_4UC_Classical_Reference."""
    masses = np.asarray(ar_fc.atoms.get_masses())
    r = F1_from_fc(
        ar_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        use_q_symmetry=True, is_classic=True,
    )
    np.testing.assert_allclose(r["F1"], REF_F1, rtol=1e-4)
    np.testing.assert_allclose(r["U1"], REF_U1, rtol=1e-4)
    np.testing.assert_allclose(r["S1"], REF_S1, rtol=1e-4)
    np.testing.assert_allclose(r["Cv1"], REF_CV1, rtol=1e-4)


@pytest.mark.skipif(not _have_ar_fixture(), reason="LJ Ar 80K_4UC fixture unavailable")
def test_lj_ar_80K_F2_matches_reference(ar_fc):
    """Classical F2 at 5^3 vs Argon_80K_4UC_Classical_Reference."""
    masses = np.asarray(ar_fc.atoms.get_masses())
    r = F2_from_fc(
        ar_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        sigma_THz=None, use_q_symmetry=True, is_classic=True,
    )
    np.testing.assert_allclose(r["F2"], REF_F2, rtol=1e-4)
    np.testing.assert_allclose(r["U2"], REF_U2, rtol=1e-4)
    np.testing.assert_allclose(r["S2"], REF_S2, rtol=1e-4)
    np.testing.assert_allclose(r["Cv2"], REF_CV2, rtol=1e-4)

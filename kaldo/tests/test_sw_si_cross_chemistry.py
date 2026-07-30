"""
Cross-chemistry validation on Stillinger-Weber Si at 1600 K.

Uses the SW Si ``1600K_3UC`` TDEP fixture under
``kaldo/tests/cumulant_fixtures/SW/`` with pinned classical reference
values in ``Silicon_1600K_3UC_Classical_Reference``.

Meshes (this suite):
  * harmonic props (F_H / U_H / S_H / Cv_H): 30^3
  * analytic F1 / F2: 5^3

Tests are self-contained (no external paths).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.cumulant import F1_from_fc, F2_from_fc
from kaldo.cumulant.thermodynamics import _harmonic_thermo

SW_IFC = Path(__file__).parent / "cumulant_fixtures" / "SW" / "1600K_3UC"

# Classical reference (Julia / LDT layout) at the meshes above.
REF_T_K = 1600.0
REF_F_H = -0.515229485446
REF_U_H = 0.413631854400
REF_S_H = 6.736869972408
REF_CV_H = 3.000000000000
REF_F1 = -0.000543896863
REF_U1 = 0.000543896863
REF_S1 = 0.007889578969
REF_CV1 = 0.007889578969
REF_F2 = -0.002515013780
REF_U2 = 0.002515013780
REF_S2 = 0.036481916272
REF_CV2 = 0.036481916272

HARMONIC_MESH = (30, 30, 30)
THERMO_MESH = (5, 5, 5)
SUPERCELL = (3, 3, 3)


def _have_sw_fixture():
    return (SW_IFC / "infile.forceconstant").exists() and \
           (SW_IFC / "infile.ucposcar").exists()


@pytest.fixture(scope="module")
def sw_fc():
    if not _have_sw_fixture():
        pytest.skip("SW Si 1600K_3UC fixture unavailable")
    return ForceConstants.from_folder(
        folder=str(SW_IFC), supercell=SUPERCELL, format="tdep", include_fourth=True,
    )


@pytest.mark.skipif(not _have_sw_fixture(), reason="SW Si 1600K_3UC fixture unavailable")
def test_sw_si_1600K_harmonic_matches_reference(sw_fc):
    """Classical F_H/U_H/S_H/Cv_H at 30^3 vs Silicon_1600K_3UC_Classical_Reference."""
    masses = np.asarray(sw_fc.atoms.get_masses())
    harm = _harmonic_thermo(sw_fc, HARMONIC_MESH, REF_T_K, is_classic=True)
    np.testing.assert_allclose(harm["F_H"], REF_F_H, rtol=5e-5)
    np.testing.assert_allclose(harm["U_H"], REF_U_H, rtol=5e-5)
    np.testing.assert_allclose(harm["S_H"], REF_S_H, rtol=5e-5)
    np.testing.assert_allclose(harm["Cv_H"], REF_CV_H, rtol=5e-5)
    assert masses.shape[0] == 2


@pytest.mark.skipif(not _have_sw_fixture(), reason="SW Si 1600K_3UC fixture unavailable")
def test_sw_si_1600K_F1_matches_reference(sw_fc):
    """Classical F1 at 5^3 vs Silicon_1600K_3UC_Classical_Reference."""
    masses = np.asarray(sw_fc.atoms.get_masses())
    r = F1_from_fc(
        sw_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        use_q_symmetry=True, is_classic=True,
    )
    np.testing.assert_allclose(r["F1"], REF_F1, rtol=1e-4)
    np.testing.assert_allclose(r["U1"], REF_U1, rtol=1e-4)
    np.testing.assert_allclose(r["S1"], REF_S1, rtol=1e-4)
    np.testing.assert_allclose(r["Cv1"], REF_CV1, rtol=1e-4)


@pytest.mark.skipif(not _have_sw_fixture(), reason="SW Si 1600K_3UC fixture unavailable")
def test_sw_si_1600K_F2_matches_reference(sw_fc):
    """Classical F2 at 5^3 vs Silicon_1600K_3UC_Classical_Reference."""
    masses = np.asarray(sw_fc.atoms.get_masses())
    r = F2_from_fc(
        sw_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        sigma_THz=None, use_q_symmetry=True, is_classic=True,
    )
    np.testing.assert_allclose(r["F2"], REF_F2, rtol=1e-4)
    np.testing.assert_allclose(r["U2"], REF_U2, rtol=1e-4)
    np.testing.assert_allclose(r["S2"], REF_S2, rtol=1e-4)
    np.testing.assert_allclose(r["Cv2"], REF_CV2, rtol=1e-4)

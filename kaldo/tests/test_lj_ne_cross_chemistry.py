"""
Cross-chemistry validation on Lennard-Jones Neon at 14 K (quantum).

Uses ``kaldo/tests/cumulant_fixtures/LJ/Neon_14K_4UC`` with pinned values
from ``Neon_14K_4UC_Quantum_Reference``.

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

NE_IFC = Path(__file__).parent / "cumulant_fixtures" / "LJ" / "Neon_14K_4UC"

REF_T_K = 14.0
REF_F_H = 0.007257533994
REF_U_H = 0.007761992603
REF_S_H = 0.418142934371
REF_CV_H = 0.987045346630
REF_F1 = 0.000070677388
REF_U1 = 0.000063045856
REF_S1 = -0.006325734499
REF_CV1 = -0.023459532844
REF_F2 = -0.000193988656
REF_U2 = -0.000042940939
REF_S2 = 0.125202569227
REF_CV2 = 0.236148282432

HARMONIC_MESH = (30, 30, 30)
THERMO_MESH = (5, 5, 5)
SUPERCELL = (4, 4, 4)


def _have_ne_fixture():
    return (NE_IFC / "infile.forceconstant").exists() and \
           (NE_IFC / "infile.ucposcar").exists()


@pytest.fixture(scope="module")
def ne_fc():
    if not _have_ne_fixture():
        pytest.skip("LJ Ne 14K_4UC fixture unavailable")
    return ForceConstants.from_folder(
        folder=str(NE_IFC), supercell=SUPERCELL, format="tdep", include_fourth=True,
    )


@pytest.mark.skipif(not _have_ne_fixture(), reason="LJ Ne 14K_4UC fixture unavailable")
def test_lj_ne_14K_harmonic_matches_reference(ne_fc):
    """Quantum F_H/U_H/S_H/Cv_H at 30^3 vs Neon_14K_4UC_Quantum_Reference."""
    harm = _harmonic_thermo(ne_fc, HARMONIC_MESH, REF_T_K, is_classic=False)
    np.testing.assert_allclose(harm["F_H"], REF_F_H, rtol=5e-5)
    np.testing.assert_allclose(harm["U_H"], REF_U_H, rtol=5e-5)
    np.testing.assert_allclose(harm["S_H"], REF_S_H, rtol=5e-5)
    np.testing.assert_allclose(harm["Cv_H"], REF_CV_H, rtol=5e-5)


@pytest.mark.skipif(not _have_ne_fixture(), reason="LJ Ne 14K_4UC fixture unavailable")
def test_lj_ne_14K_F1_matches_reference(ne_fc):
    """Quantum F1 at 5^3 vs Neon_14K_4UC_Quantum_Reference."""
    masses = np.asarray(ne_fc.atoms.get_masses())
    r = F1_from_fc(
        ne_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        use_q_symmetry=True, is_classic=False,
    )
    np.testing.assert_allclose(r["F1"], REF_F1, rtol=5e-5)
    np.testing.assert_allclose(r["U1"], REF_U1, rtol=5e-5)
    np.testing.assert_allclose(r["S1"], REF_S1, rtol=5e-5)
    np.testing.assert_allclose(r["Cv1"], REF_CV1, rtol=5e-5)


@pytest.mark.skipif(not _have_ne_fixture(), reason="LJ Ne 14K_4UC fixture unavailable")
def test_lj_ne_14K_F2_matches_reference(ne_fc):
    """Quantum F2 at 5^3 vs Neon_14K_4UC_Quantum_Reference."""
    masses = np.asarray(ne_fc.atoms.get_masses())
    r = F2_from_fc(
        ne_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        sigma_THz=None, use_q_symmetry=True, is_classic=False,
    )
    np.testing.assert_allclose(r["F2"], REF_F2, rtol=5e-5)
    np.testing.assert_allclose(r["U2"], REF_U2, rtol=5e-4)
    np.testing.assert_allclose(r["S2"], REF_S2, rtol=5e-5)
    np.testing.assert_allclose(r["Cv2"], REF_CV2, rtol=5e-5)

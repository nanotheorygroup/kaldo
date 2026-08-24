"""Parallel F1/F2 over IBZ q1 vs the serial path.

n_workers=2 must match n_workers=1. Flattened IFC tables are attached
via SharedMemory (not pickled nested quartet/triplet lists).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.cumulant import F1_from_fc, F2_from_fc
from kaldo.cumulant.thermodynamics import _harmonic_thermo

NE_IFC = Path(__file__).parent / "cumulant_fixtures" / "LJ" / "Neon_14K_4UC"
THERMO_MESH = (5, 5, 5)
SUPERCELL = (4, 4, 4)
REF_T_K = 14.0


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
def test_parallel_F1_matches_serial(ne_fc):
    masses = np.asarray(ne_fc.atoms.get_masses())
    serial = F1_from_fc(
        ne_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        use_q_symmetry=True, is_classic=False, n_workers=1,
    )
    parallel = F1_from_fc(
        ne_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        use_q_symmetry=True, is_classic=False, n_workers=2,
    )
    for key in ("F1", "S1", "Cv1", "U1"):
        np.testing.assert_allclose(
            parallel[key], serial[key], rtol=1e-7,
            err_msg=f"parallel F1 {key} drifted from serial",
        )


@pytest.mark.skipif(not _have_ne_fixture(), reason="LJ Ne 14K_4UC fixture unavailable")
def test_parallel_F2_matches_serial(ne_fc):
    masses = np.asarray(ne_fc.atoms.get_masses())
    serial = F2_from_fc(
        ne_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        sigma_THz=None, use_q_symmetry=True, is_classic=False, n_workers=1,
    )
    parallel = F2_from_fc(
        ne_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        sigma_THz=None, use_q_symmetry=True, is_classic=False, n_workers=2,
    )
    for key in ("F2", "S2", "Cv2", "U2"):
        np.testing.assert_allclose(
            parallel[key], serial[key], rtol=1e-7,
            err_msg=f"parallel F2 {key} drifted from serial",
        )


def test_parallel_harmonic_matches_serial(ne_fc):
    """Phase-1 F_H/S_H on two processes matches the in-process Phonons loop."""
    serial = _harmonic_thermo(
        ne_fc, THERMO_MESH, T_K=REF_T_K, is_classic=False, n_workers=1,
    )
    parallel = _harmonic_thermo(
        ne_fc, THERMO_MESH, T_K=REF_T_K, is_classic=False, n_workers=2,
    )
    for key in ("F_H", "U_H", "S_H", "Cv_H"):
        np.testing.assert_allclose(
            parallel[key], serial[key], rtol=1e-7,
            err_msg=f"parallel harmonic {key} drifted from serial",
        )


def test_n_workers_zero_raises_F1(ne_fc):
    masses = np.asarray(ne_fc.atoms.get_masses())
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        F1_from_fc(
            ne_fc, masses_amu=masses, kmesh=(2, 2, 2), T_K=REF_T_K,
            use_q_symmetry=True, n_workers=0,
        )

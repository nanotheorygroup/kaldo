"""
Prototype: SCContractors.from_tdep_folder -> V_4 on a TDEP run.

Demonstrates that the per-atom-quartet representation already used by
``kaldo.cumulant.contractors.SCContractors`` (Julia HDF5 path) can be built
directly from a TDEP run, no Julia involved. The V_4 evaluator is
unchanged.

Bounds we verify:
  * shape of the flat tables matches what V_4's einsum expects
  * V_4 returns a finite real number on a deterministic displacement
  * V_4 scales as u^4 (doubling u multiplies V_4 by 16 exactly)
  * V_2 is positive on a random displacement
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

LJ_TDEP = Path(__file__).parent / "cumulant_fixtures" / "LJ" / "80K_4UC"
SW_TDEP = Path(__file__).parent / "cumulant_fixtures" / "SW" / "100K_3UC"


def _stage_tdep_folder(tmp_path, ifc_dir: Path, struct_dir: Path) -> Path:
    """Lay out IFCs + ucposcar/ssposcar in a single folder, TDEP-style."""
    import shutil
    out = tmp_path / "tdep_stage"
    out.mkdir()
    for fn in ("infile.ucposcar", "infile.ssposcar"):
        shutil.copy(str(struct_dir / fn), str(out / fn))
    for fn in (
        "infile.forceconstant",
        "infile.forceconstant_thirdorder",
        "infile.forceconstant_fourthorder",
    ):
        shutil.copy(str(ifc_dir / fn), str(out / fn))
    return out


@pytest.mark.skipif(not LJ_TDEP.exists(), reason="LJ fixture missing")
def test_v4_from_tdep_lj_diagonal(tmp_path):
    """LJ Ar 4^3: diagonal supercell, n_uc=1, det M=64."""
    from kaldo.cumulant.contractors import SCContractors

    folder = _stage_tdep_folder(tmp_path, LJ_TDEP, LJ_TDEP.parent)
    sc = SCContractors.from_tdep_folder(folder, include_fourth=True)

    # 256 sc atoms, flat tables
    assert sc.n_atoms_sc == 256
    assert sc.a1_4.shape == sc.a2_4.shape == sc.a3_4.shape == sc.a4_4.shape
    assert sc.phi4.shape[1:] == (3, 3, 3, 3)
    assert sc.a1_4.shape[0] == sc.phi4.shape[0]

    rng = np.random.default_rng(42)
    u = 0.01 * rng.standard_normal((sc.n_atoms_sc, 3))
    v4 = sc.V4(u)
    assert np.isfinite(v4)

    # V_2 should be positive on a random displacement (positive-definite
    # under ASR + harmonic stability).
    v2 = sc.V2(u)
    assert v2 > 0
    # V_4 magnitudes for small u should scale as u^4: doubling u multiplies
    # V_4 by 16, within numerical tolerance.
    v4_double = sc.V4(2.0 * u)
    np.testing.assert_allclose(v4_double, 16.0 * v4, rtol=1e-10)


@pytest.mark.skipif(not SW_TDEP.exists(), reason="SW Si fixture missing")
def test_v4_from_tdep_sw_si_nondiagonal(tmp_path):
    """SW Si 100K_3UC: non-diagonal rhombo->cubic supercell, n_uc=2, det M=108."""
    from kaldo.cumulant.contractors import SCContractors

    folder = _stage_tdep_folder(tmp_path, SW_TDEP, SW_TDEP.parent)
    sc = SCContractors.from_tdep_folder(folder, include_fourth=True)

    assert sc.n_atoms_sc == 216
    assert sc.phi4.shape[1:] == (3, 3, 3, 3)

    rng = np.random.default_rng(0)
    u = 0.005 * rng.standard_normal((sc.n_atoms_sc, 3))
    v4 = sc.V4(u)
    v3 = sc.V3(u)
    v2 = sc.V2(u)
    assert np.isfinite(v4)
    assert np.isfinite(v3)
    assert v2 > 0

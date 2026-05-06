"""
Tests for non-diagonal TDEP supercell handling.

Two pieces:
  1. kaldo.interfaces.tdep_io.build_supercell_replica_mapping +
     wrap_lattice_vector_to_replica — low-level primitives for mapping
     TDEP per-atom quartets/triplets onto a non-diagonal ssposcar tiling.
  2. kaldo.cumulant.load_tdep_folder — high-level one-shot reader that
     returns the cumulant-format (neighbors, triplets, [quartets]) lists
     from any TDEP folder, diagonal or non-diagonal.

The si-tdep fixture is diagonal 5^3 (our reference cross-check). The
reference_si/T300_0 and ethan/thermo_out_full fixtures are non-diagonal
(rhombo primitive + cubic conventional ssposcar).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SI_TDEP_DIR = Path(__file__).parent / "si-tdep"

# Production-only fixtures: large DFT-quality Si IFCs and Ne TDEP run output.
# Set KALDO_TEST_SI_PROD and KALDO_TEST_NE_REF to enable.
# See kaldo/tests/_paths.py for details on env-var-gated test fixtures.
from kaldo.tests._paths import SI_PROD, NE_REF as NE_PROD


# ---------------------------------------------------------------------------
# SNF primitives: build_supercell_replica_mapping / wrap_lattice_vector_to_replica
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_build_supercell_replica_mapping_si_nondiagonal():
    """Si production (rhombo primitive + cubic 3x3x3 conv ssposcar, det M = 108).

    The mapping must produce 108 unique replicas and reproduce every sc
    atom position mod the supercell lattice.
    """
    import ase.io
    from kaldo.interfaces.tdep_io import build_supercell_replica_mapping

    uc = ase.io.read(str(SI_PROD / "infile.ucposcar"), format="vasp")
    sc = ase.io.read(str(SI_PROD / "infile.ssposcar"), format="vasp")
    m = build_supercell_replica_mapping(uc, sc)

    assert m["replica_table"].shape == (108, 3)
    assert m["atom_of_sc"].shape == (216,)
    assert set(np.unique(m["atom_of_sc"]).tolist()) == {0, 1}
    # round-trip check
    uc_pos = np.asarray(uc.positions); uc_cell = np.asarray(uc.cell)
    sc_cell = np.asarray(sc.cell)
    inv_sc = np.linalg.inv(sc_cell)
    for i in range(len(sc)):
        j = m["atom_of_sc"][i]
        R = m["replica_vector_of_sc"][i]
        r_expected = uc_pos[j] + R @ uc_cell
        diff_frac = (sc.positions[i] - r_expected) @ inv_sc
        diff_frac -= np.round(diff_frac)
        assert np.max(np.abs(diff_frac)) < 1e-4


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_wrap_lattice_vector_to_replica_si():
    """Arbitrary TDEP lattice vectors map to a unique replica id."""
    import ase.io
    from kaldo.interfaces.tdep_io import (
        build_supercell_replica_mapping, wrap_lattice_vector_to_replica,
    )
    uc = ase.io.read(str(SI_PROD / "infile.ucposcar"), format="vasp")
    sc = ase.io.read(str(SI_PROD / "infile.ssposcar"), format="vasp")
    m = build_supercell_replica_mapping(uc, sc)

    # Origin
    assert wrap_lattice_vector_to_replica(
        [0, 0, 0], m["replica_table"], m["M"]) >= 0
    # Arbitrary R in primitive basis
    for R in [[1, 0, 0], [-1, 0, 0], [2, -1, 3], [5, 5, 5]]:
        idx = wrap_lattice_vector_to_replica(R, m["replica_table"], m["M"])
        assert idx >= 0, f"R={R} did not map to a replica"
        # And it's in range
        assert 0 <= idx < 108


# ---------------------------------------------------------------------------
# load_tdep_folder: diagonal si-tdep
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SI_TDEP_DIR.exists(), reason="si-tdep fixture missing")
def test_load_tdep_folder_diagonal_sitdep():
    """si-tdep is diagonal 5^3; load_tdep_folder returns the right shapes."""
    from kaldo.cumulant import load_tdep_folder
    r = load_tdep_folder(SI_TDEP_DIR)
    assert r["is_diagonal"]
    assert np.allclose(r["M"], np.diag([5, 5, 5]))
    # 2 atoms, some neighbors / triplets
    assert len(r["uc_positions"]) == 2
    assert len(r["neighbors"]) == 2
    assert len(r["triplets"]) == 2
    # no IFC4 requested
    assert r["quartets"] is None


# ---------------------------------------------------------------------------
# load_tdep_folder: non-diagonal production Si
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_load_tdep_folder_nondiagonal_si_production():
    """reference_si/T300_0 is non-diagonal; load_tdep_folder still loads.

    This is the SNF-enabler: bypasses the diagonal guard in
    ForceConstants.from_folder(format='tdep') to let cumulant F1/F2
    consume non-diagonal production TDEP runs directly.
    """
    from kaldo.cumulant import load_tdep_folder, F2_vectorized, AMU
    r = load_tdep_folder(SI_PROD, include_fourth=False)
    assert not r["is_diagonal"]
    assert len(r["neighbors"]) == 2
    assert len(r["triplets"]) == 2

    # Actually run F2 at Si 2^3 and compare to pinned -6.43481e-5
    masses = np.full(2, 28.0855 * AMU)
    res = F2_vectorized(
        r["neighbors"], r["triplets"], masses,
        r["uc_positions"], r["uc_cell"],
        (2, 2, 2), 300.0, sigma_THz=None,
        use_q_symmetry=True, atoms=r["atoms"],
    )
    np.testing.assert_allclose(res["F2"], -6.43481e-5, rtol=5e-4)


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_load_tdep_folder_with_fourth_si_production():
    """include_fourth=True reads IFC4 too; F1 at Si 2^3 must match pinned value."""
    from kaldo.cumulant import load_tdep_folder, F1_vectorized, AMU
    r = load_tdep_folder(SI_PROD, include_fourth=True)
    assert r["quartets"] is not None
    assert len(r["quartets"]) == 2
    masses = np.full(2, 28.0855 * AMU)
    res = F1_vectorized(
        r["neighbors"], r["quartets"], masses,
        r["uc_positions"], r["uc_cell"],
        (2, 2, 2), 300.0,
        use_q_symmetry=True, atoms=r["atoms"],
    )
    np.testing.assert_allclose(res["F1"], -1.16592e-5, rtol=2e-4)


@pytest.mark.skipif(not NE_PROD.exists(), reason="non-diagonal Ne fixture unavailable")
def test_load_tdep_folder_nondiagonal_ne_production():
    """Ne thermo_out_full is non-diagonal (256 replicas); load_tdep_folder
    must also work there."""
    from kaldo.cumulant import load_tdep_folder
    r = load_tdep_folder(NE_PROD)
    assert not r["is_diagonal"]
    assert len(r["neighbors"]) == 1  # n_uc = 1 for Ne
    assert len(r["triplets"]) == 1

"""
Tests for non-diagonal TDEP supercell handling primitives.

kaldo.interfaces.tdep_io.build_supercell_replica_mapping +
wrap_lattice_vector_to_replica are the low-level primitives for mapping
TDEP per-atom pairs/triplets/quartets onto a non-diagonal ssposcar tiling.

The reference_si production fixture is non-diagonal (rhombohedral primitive
+ cubic conventional ssposcar, det M = 108); it is env-var-gated because it
is too large to vendor. Diagonal-supercell coverage of the same primitives
lives in test_parse_tdep_unified.py on the vendored si-tdep fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

# Production-only fixture: large DFT-quality non-diagonal Si IFCs.
# Set KALDO_TEST_SI_PROD to enable.
# See kaldo/tests/_paths.py for details on env-var-gated test fixtures.
from kaldo.tests._paths import SI_PROD


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


def _skewed_primitive():
    """Two-atom primitive with a deliberately skewed (triclinic) cell.

    The cell must NOT commute with the tiling matrices below, so any
    row/column-order mistake in the primitive-to-supercell M computation
    (M = sc @ uc^-1 for ASE row-vector cells) changes the result.
    """
    from ase import Atoms
    cell = np.array([[4.0, 0.0, 0.0],
                     [1.3, 3.8, 0.0],
                     [0.4, 0.9, 3.5]])
    return Atoms("Si2", scaled_positions=[[0, 0, 0], [0.27, 0.31, 0.24]],
                 cell=cell, pbc=True)


def test_non_commuting_tiling_recovers_true_m():
    """Regression: M must be sc @ uc^-1 (row-vector cells), not uc^-1 @ sc.

    The two coincide exactly when M commutes with the primitive cell,
    which is true for every uniform (n, n, n) fixture; a skewed cell with
    a non-diagonal tiling tells them apart. ase.build.make_supercell is
    the independent oracle: it constructs the supercell as M @ uc by
    definition.
    """
    from ase.build import make_supercell
    from kaldo.grid import wrap_lattice_vector_to_replica
    from kaldo.interfaces.tdep_io import build_supercell_replica_mapping

    prim = _skewed_primitive()
    uc_cell = np.asarray(prim.cell)
    M0 = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 2]])
    sc = make_supercell(prim, M0)
    n_rep = round(abs(np.linalg.det(M0)))

    mapping = build_supercell_replica_mapping(prim, sc)
    np.testing.assert_allclose(mapping["M"], M0, atol=1e-8)
    assert len(mapping["replica_table"]) == n_rep

    # Every supercell atom must reconstruct from its (primitive atom,
    # replica vector) assignment modulo the TRUE supercell lattice.
    inv_sc = np.linalg.inv(np.asarray(sc.cell))
    for i, rsc in enumerate(sc.positions):
        j = mapping["atom_of_sc"][i]
        R = mapping["replica_vector_of_sc"][i]
        diff = rsc - (prim.positions[j] + R @ uc_cell)
        f = diff @ inv_sc
        np.testing.assert_allclose(f, np.rint(f), atol=1e-6,
                                   err_msg=f"sc atom {i} not on the lattice")

    # Wrapping any replica shifted by a supercell lattice translation
    # (rows of M0 in the primitive basis) must recover the same replica.
    table = mapping["replica_table"]
    for idx, R in enumerate(table):
        for s in ([1, 0, 0], [0, 1, 0], [-1, 1, -1]):
            R_shifted = R + np.array(s) @ M0
            found = wrap_lattice_vector_to_replica(R_shifted, table, mapping["M"])
            assert found == idx, (
                f"replica {R} + {s}@M wrapped to {found}, expected {idx}"
            )


def test_anisotropic_diagonal_on_skewed_cell_resolves_diagonal(tmp_path):
    """Regression: a legitimate (3, 2, 1) diagonal tiling of a skewed cell
    must resolve as diagonal, not fail the integer-M check."""
    import ase.io
    from ase.build import make_supercell
    from kaldo.interfaces.tdep_io import resolve_tdep_supercell

    prim = _skewed_primitive()
    M0 = np.diag([3, 2, 1])
    sc = make_supercell(prim, M0)
    ase.io.write(str(tmp_path / "infile.ucposcar"), prim, format="vasp")
    ase.io.write(str(tmp_path / "infile.ssposcar"), sc, format="vasp")

    _, _, diagonal_supercell = resolve_tdep_supercell(str(tmp_path))
    assert diagonal_supercell == (3, 2, 1)

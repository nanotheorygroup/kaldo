"""Pin the unified TDEP IFC parser API (grid= polymorphism + sigma2 path).

These tests gate the diag/nondiag IFC parser unification: the three
``parse_tdep_*_forceconstant`` functions accept either a Grid (diagonal
supercell) or a NonDiagonalGrid (SNF) and produce the same kaldo
storage shape.
"""
from __future__ import annotations

from pathlib import Path

import ase.io
import numpy as np
import pytest

from kaldo.grid import Grid, NonDiagonalGrid
from kaldo.interfaces.tdep_io import (
    build_supercell_replica_mapping,
    parse_tdep_forceconstant,
    parse_tdep_third_forceconstant,
)


SI_TDEP = Path(__file__).parent / "si-tdep"


@pytest.mark.skipif(not SI_TDEP.exists(), reason="si-tdep fixture missing")
def test_ifc2_grid_polymorphism_diagonal_fixture():
    """On a diagonal Si TDEP fixture, parse_tdep_forceconstant called with
    grid=Grid(supercell) and grid=NonDiagonalGrid(replica_table, M) must
    produce element-wise identical IFC2 tensors."""
    uc = ase.io.read(str(SI_TDEP / "infile.ucposcar"), format="vasp")
    sc = ase.io.read(str(SI_TDEP / "infile.ssposcar"), format="vasp")
    supercell = (5, 5, 5)  # si-tdep ssposcar is a 5x5x5 tiling

    # Diagonal Grid path
    g_diag = Grid(supercell, order="C")
    d2_diag = parse_tdep_forceconstant(
        fc_file=str(SI_TDEP / "infile.forceconstant"),
        primitive=uc,
        grid=g_diag,
    )

    # NonDiagonalGrid path (built from the same diagonal mapping)
    mapping = build_supercell_replica_mapping(uc, sc)
    g_snf = NonDiagonalGrid(
        replica_table=mapping["replica_table"], M=mapping["M"],
    )
    d2_snf = parse_tdep_forceconstant(
        fc_file=str(SI_TDEP / "infile.forceconstant"),
        primitive=uc,
        grid=g_snf,
    )

    # Both shapes must be (1, n_uc, 3, n_rep, n_uc, 3); n_rep = 125
    assert d2_diag.shape == d2_snf.shape == (1, 2, 3, 125, 2, 3)

    # Replica orderings can differ (Grid uses (i,j,k) decomposition; SNF uses
    # the replica_table). Compare via the lattice-vector-keyed mapping.
    d2_diag_arr = (
        d2_diag.todense() if hasattr(d2_diag, "todense") else np.asarray(d2_diag)
    )
    d2_snf_arr = (
        d2_snf.todense() if hasattr(d2_snf, "todense") else np.asarray(d2_snf)
    )

    # For each replica id in g_snf, find the matching id in g_diag via the
    # primitive-basis lattice vector and compare slices. The SNF replica table
    # is in min-Cartesian-norm form (e.g. R = [0, 2, -3] for FCC primitives at
    # 5x5x5), which can fall outside Grid's [0, N) lookup range. Wrap via
    # modular arithmetic before comparing.
    grid_shape = np.array(g_diag.grid_shape)
    for snf_id, R in enumerate(mapping["replica_table"]):
        R_mod = np.array(R) % grid_shape
        diag_ids = g_diag.grid_index_to_id(R_mod, is_wrapping=False)
        assert len(diag_ids) == 1, (
            f"diagonal grid did not resolve replica vector {R} (mod"
            f" {tuple(grid_shape)}) to a single id"
        )
        diag_id = int(diag_ids[0])
        np.testing.assert_allclose(
            d2_snf_arr[0, :, :, snf_id, :, :],
            d2_diag_arr[0, :, :, diag_id, :, :],
            rtol=0, atol=0,
            err_msg=f"IFC2 mismatch at replica R={R}",
        )


@pytest.mark.skipif(not SI_TDEP.exists(), reason="si-tdep fixture missing")
def test_ifc3_grid_polymorphism_diagonal_fixture():
    """IFC3 parser: Grid and NonDiagonalGrid produce equivalent results
    on a diagonal Si TDEP fixture."""
    uc = ase.io.read(str(SI_TDEP / "infile.ucposcar"), format="vasp")
    sc = ase.io.read(str(SI_TDEP / "infile.ssposcar"), format="vasp")
    supercell = (5, 5, 5)  # si-tdep ssposcar is a 5x5x5 tiling

    g_diag = Grid(supercell, order="C")
    d3_diag = parse_tdep_third_forceconstant(
        fc_filename=str(SI_TDEP / "infile.forceconstant_thirdorder"),
        primitive=str(SI_TDEP / "infile.ucposcar"),
        grid=g_diag,
    )

    mapping = build_supercell_replica_mapping(uc, sc)
    g_snf = NonDiagonalGrid(
        replica_table=mapping["replica_table"], M=mapping["M"],
    )
    d3_snf = parse_tdep_third_forceconstant(
        fc_filename=str(SI_TDEP / "infile.forceconstant_thirdorder"),
        primitive=uc,
        grid=g_snf,
    )

    assert d3_diag.shape == d3_snf.shape

    d3_diag_arr = d3_diag.todense()
    d3_snf_arr = d3_snf.todense()

    # Compare slice-by-slice via the lattice-vector mapping for both pair
    # replicas (R2 and R3). SNF replica vectors may lie outside [0, N) — wrap
    # via modular arithmetic before the diagonal-grid lookup.
    grid_shape = np.array(g_diag.grid_shape)
    for snf_r2, R2 in enumerate(mapping["replica_table"]):
        diag_r2 = int(
            g_diag.grid_index_to_id(np.array(R2) % grid_shape, is_wrapping=False)[0]
        )
        for snf_r3, R3 in enumerate(mapping["replica_table"]):
            diag_r3 = int(
                g_diag.grid_index_to_id(np.array(R3) % grid_shape, is_wrapping=False)[0]
            )
            np.testing.assert_allclose(
                d3_snf_arr[:, :, snf_r2, :, :, snf_r3, :, :],
                d3_diag_arr[:, :, diag_r2, :, :, diag_r3, :, :],
                rtol=0, atol=0,
                err_msg=f"IFC3 mismatch at (R2={R2}, R3={R3})",
            )


SIGMA2_FIXTURE = Path(__file__).parent / "sigma2"


@pytest.mark.skipif(
    not SIGMA2_FIXTURE.exists(),
    reason="sigma2 fixture missing",
)
def test_sigma2_two_dim_path():
    """parse_tdep_forceconstant(grid=None, two_dim=True, symmetrize=True)
    must return a (3*n_sc, 3*n_sc) matrix and continue to be the path used
    by sigma2.calculate_sigma2."""
    fc = parse_tdep_forceconstant(
        fc_file=str(SIGMA2_FIXTURE / "infile.forceconstant"),
        primitive=str(SIGMA2_FIXTURE / "infile.ucposcar"),
        supercell=str(SIGMA2_FIXTURE / "infile.ssposcar"),
        symmetrize=True,
        two_dim=True,
    )
    sc = ase.io.read(str(SIGMA2_FIXTURE / "infile.ssposcar"), format="vasp")
    n_sc = len(sc)
    assert fc.shape == (3 * n_sc, 3 * n_sc)
    # Symmetrize should make it Hermitian (real symmetric here).
    np.testing.assert_allclose(fc, fc.T, atol=1e-10)


@pytest.mark.skipif(not SI_TDEP.exists(), reason="si-tdep fixture missing")
def test_grid_none_two_dim_false_returns_4d():
    """grid=None, two_dim=False returns the (n_sc, n_sc, 3, 3) 4D form."""
    fc = parse_tdep_forceconstant(
        fc_file=str(SI_TDEP / "infile.forceconstant"),
        primitive=str(SI_TDEP / "infile.ucposcar"),
        supercell=str(SI_TDEP / "infile.ssposcar"),
        two_dim=False,
    )
    sc = ase.io.read(str(SI_TDEP / "infile.ssposcar"), format="vasp")
    n_sc = len(sc)
    assert fc.shape == (n_sc, n_sc, 3, 3)

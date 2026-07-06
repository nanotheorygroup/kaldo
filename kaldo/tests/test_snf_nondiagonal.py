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

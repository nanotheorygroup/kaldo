"""Regression tests for literal lattice translations in TDEP IFC files."""

import numpy as np
from ase import Atoms

from kaldo.grid import SupercellGrid
from kaldo.interfaces.tdep_io import (
    parse_tdep_forceconstant,
    parse_tdep_third_forceconstant,
)
from kaldo.observables.forceconstant import ForceConstant


def _one_atom():
    return Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)


def test_tdep_ifc2_keeps_translations_in_same_periodic_class(tmp_path):
    path = tmp_path / "infile.forceconstant"
    blocks = []
    for translation, value in [((0, 0, 0), 1.0), ((2, 0, 0), 2.0), ((-2, 0, 0), 3.0)]:
        blocks.extend(["1", "%d %d %d" % translation])
        blocks.extend([f"{value} 0 0", "0 0 0", "0 0 0"])
    path.write_text("1\n10.0\n3\n" + "\n".join(blocks) + "\n")

    tensor, support = parse_tdep_forceconstant(
        fc_file=str(path), primitive=_one_atom(),
        supercell_grid=SupercellGrid(np.diag([2, 1, 1])), return_support=True,
    )

    np.testing.assert_array_equal(
        support.translations, [[0, 0, 0], [-2, 0, 0], [2, 0, 0]]
    )
    assert support.supercell.size == 2
    assert support.size == 3
    np.testing.assert_allclose(tensor[0, 0, 0, :, 0, 0], [1.0, 3.0, 2.0])
    assert len(set(support.class_ids.tolist())) == 1


def test_tdep_ifc3_keeps_literal_translation_axes(tmp_path):
    path = tmp_path / "infile.forceconstant_thirdorder"
    phi = " ".join(str(float(i)) for i in range(27))
    path.write_text(
        "1\n10.0\n2\n"
        "1\n1\n1\n0 0 0\n0 0 0\n2 0 0\n" + phi + "\n"
        "1\n1\n1\n0 0 0\n-2 0 0\n0 0 0\n" + phi + "\n"
    )

    tensor, support = parse_tdep_third_forceconstant(
        fc_filename=str(path), primitive=_one_atom(),
        supercell_grid=SupercellGrid(np.diag([2, 1, 1])), return_support=True,
    )

    np.testing.assert_array_equal(
        support.translations, [[0, 0, 0], [-2, 0, 0], [2, 0, 0]]
    )
    assert tensor.shape == (1, 3, 3, 1, 3, 3, 1, 3)
    assert tensor[0, 2, 0, 0, 2, 2, 0, 2] == 26.0
    assert tensor[0, 2, 1, 0, 2, 0, 0, 2] == 26.0


def test_periodic_loader_support_matches_legacy_replica_order():
    atoms = _one_atom()
    second = ForceConstant.from_supercell(
        atoms=atoms, supercell=(2, 2, 1), grid_type="F", value=None,
        folder="unused",
    )
    assert second.n_replicas == 4
    assert second.n_translations == 4
    np.testing.assert_array_equal(
        second.translation_support.translations,
        second.supercell_grid.representatives,
    )


def test_tdep_rejects_fractional_lattice_vector(tmp_path):
    path = tmp_path / "infile.forceconstant"
    path.write_text("1\n10.0\n1\n1\n0.25 0 0\n1 0 0\n0 1 0\n0 0 1\n")
    with np.testing.assert_raises_regex(ValueError, "integer lattice vector"):
        parse_tdep_forceconstant(
            fc_file=str(path), primitive=_one_atom(),
            supercell_grid=SupercellGrid(np.eye(3, dtype=int)), return_support=True,
        )

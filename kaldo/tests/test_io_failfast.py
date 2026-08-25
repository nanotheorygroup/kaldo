"""Malformed or inconsistent IFC input files must fail fast."""

import ase.build
import numpy as np
import pytest

from kaldo.interfaces.qe_io import read_third_d3q
from kaldo.interfaces.vasp_io import read_second_order_matrix


def test_d3q_supercell_mismatch_raises():
    atoms = ase.build.bulk("Ge", "diamond", a=5.6575)
    with pytest.raises(ValueError, match="third supercell not consistent"):
        read_third_d3q("kaldo/tests/ge-crystal/d3q/FORCE_CONSTANTS_3RD_D3Q", atoms, (2, 2, 2), order="C")


def test_second_order_malformed_index_line_raises(tmp_path):
    path = tmp_path / "FORCE_CONSTANTS_2ND"
    path.write_text("2\nnot an index line\n")
    with pytest.raises(ValueError, match="malformed index line"):
        read_second_order_matrix(str(path), np.array([1, 1, 1]))


def test_second_order_malformed_midfile_line_raises(tmp_path):
    path = tmp_path / "FORCE_CONSTANTS_2ND"
    block = "1 1\n" + "0.0 0.0 0.0\n" * 3
    path.write_text("1\n" + block + "garbage line\n" + block)
    with pytest.raises(ValueError, match="malformed index line"):
        read_second_order_matrix(str(path), np.array([1, 1, 1]))


def test_tdep_accepts_matrix_supercell():
    from kaldo.forceconstants import ForceConstants

    by_tuple = ForceConstants.from_folder("kaldo/tests/si-tdep", supercell=(5, 5, 5), format="tdep", only_second=True)
    by_matrix = ForceConstants.from_folder(
        "kaldo/tests/si-tdep", supercell=np.diag([5, 5, 5]), format="tdep", only_second=True
    )
    np.testing.assert_array_equal(by_tuple.second.value, by_matrix.second.value)

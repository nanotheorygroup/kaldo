"""
Regression tests for the ShengBTE third-order cell-offset bug.

The historical ``read_third_order_matrix`` rounded each Cartesian offset to
an integer lattice index and searched only a bounded set of stored
representatives. An equivalent offset outside that set produced an empty
NumPy fancy index, making assignment a silent no-op. The corrected reader
validates and retains each literal integer translation. Periodic-class
metadata still records finite-supercell equivalence, but off-mesh Fourier
phases no longer collapse distinct literals.

These tests exercise ``kaldo.interfaces.shengbte_io.read_third_order_matrix``
directly, with no pheasy code involved, using a synthetic two-atom cubic
POSCAR and a hand-written FORCE_CONSTANTS_3RD file.
"""

import itertools
from pathlib import Path

import numpy as np
import pytest

POSCAR_2ATOM = (
    "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 3.0\nSi\n2\n"
    "Direct\n0.0 0.0 0.0\n0.5 0.5 0.5\n"
)


def _write_poscar(folder):
    (Path(folder) / "POSCAR").write_text(POSCAR_2ATOM)


def _write_fc3_text(path, blocks):
    """blocks: list of (cart_offset_2 (3,), cart_offset_3 (3,), (i, j, k) 0-based unit atoms, phi (3,3,3))."""
    with open(path, "w") as fd:
        fd.write(f"{len(blocks)}\n")
        for n, (r2, r3, (i, j, k), phi) in enumerate(blocks, start=1):
            fd.write(f"\n{n}\n")
            fd.write("".join(f"{x:25.15f}" for x in r2) + "\n")
            fd.write("".join(f"{x:25.15f}" for x in r3) + "\n")
            fd.write(f"{i + 1:6d}{j + 1:6d}{k + 1:6d}\n")
            for a, b, c in itertools.product([1, 2, 3], repeat=3):
                fd.write(f"{a:4d}{b:4d}{c:4d}{phi[a - 1, b - 1, c - 1]:25.15f}\n")


def test_fc3_offset_sign_convention_preserves_literal_translation(tmp_path):
    """Retain either literal used for an even-supercell half-box image.

    On supercell (2, 1, 1) with a=3.0 cubic cell, +a and -a along x belong to
    the same periodic class. A negative-offset file must retain ``R=-1`` and
    its tensor weight rather than silently dropping the quartet.

    This fails on the pre-fix parser: the -3.0 offset does not match the
    wrapped grid's kept +3.0 representative, the cell lookup returns an empty
    id array, and the fancy-index assignment into the dense tensor becomes a
    silent no-op, leaving that block's slot at its initialized zero.
    """
    import ase.io
    from kaldo.interfaces.shengbte_io import read_third_order_matrix

    rng = np.random.default_rng(11)
    phi = rng.standard_normal((3, 3, 3))
    positive_dir = tmp_path / "positive"
    negative_dir = tmp_path / "negative"
    positive_dir.mkdir()
    negative_dir.mkdir()
    _write_poscar(positive_dir)
    _write_poscar(negative_dir)
    _write_fc3_text(
        positive_dir / "FORCE_CONSTANTS_3RD",
        [(np.array([3.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi)],
    )
    _write_fc3_text(
        negative_dir / "FORCE_CONSTANTS_3RD",
        [(np.array([-3.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi)],
    )
    atoms_positive = ase.io.read(positive_dir / "POSCAR", format="vasp")
    atoms_negative = ase.io.read(negative_dir / "POSCAR", format="vasp")
    third_positive, support_positive = read_third_order_matrix(
        str(positive_dir / "FORCE_CONSTANTS_3RD"),
        atoms_positive,
        (2, 1, 1),
        order="C",
        return_support=True,
    )
    third_negative, support_negative = read_third_order_matrix(
        str(negative_dir / "FORCE_CONSTANTS_3RD"),
        atoms_negative,
        (2, 1, 1),
        order="C",
        return_support=True,
    )
    dense_positive = third_positive.todense()
    dense_negative = third_negative.todense()
    np.testing.assert_allclose(dense_negative, dense_positive, atol=1e-12)
    np.testing.assert_array_equal(support_positive.translations, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(
        support_negative.translations, [[0, 0, 0], [-1, 0, 0]]
    )
    assert support_positive.class_ids[1] == support_negative.class_ids[1]
    np.testing.assert_allclose(dense_negative[0, :, 1, 1, :, 0, 0, :], phi, atol=1e-12)
    assert np.count_nonzero(dense_negative) == np.count_nonzero(phi)


def test_fc3_out_of_representative_range_offset_is_retained(tmp_path):
    """Retain a literal outside the compact supercell representatives.

    On supercell (3, 1, 1), integer translation -2 is outside the old wrapped
    range but belongs to the same periodic class as +1. Its off-mesh phase is
    nevertheless distinct and the reader must preserve ``R=-2`` exactly.

    This also fails on the pre-fix parser: -6.0 rounds to index -2, which is
    not one of the wrapped grid's kept {-1, 0, 1} representatives for a
    (3, 1, 1) grid, so the lookup again returns an empty id array and the
    assignment is silently dropped.
    """
    import ase.io
    from kaldo.interfaces.shengbte_io import read_third_order_matrix

    rng = np.random.default_rng(13)
    phi = rng.standard_normal((3, 3, 3))
    wrapped_dir = tmp_path / "wrapped"
    unwrapped_dir = tmp_path / "unwrapped"
    wrapped_dir.mkdir()
    unwrapped_dir.mkdir()
    _write_poscar(wrapped_dir)
    _write_poscar(unwrapped_dir)
    _write_fc3_text(
        wrapped_dir / "FORCE_CONSTANTS_3RD",
        [(np.array([3.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi)],
    )
    _write_fc3_text(
        unwrapped_dir / "FORCE_CONSTANTS_3RD",
        [(np.array([-6.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi)],
    )
    atoms_wrapped = ase.io.read(wrapped_dir / "POSCAR", format="vasp")
    atoms_unwrapped = ase.io.read(unwrapped_dir / "POSCAR", format="vasp")
    third_wrapped, support_wrapped = read_third_order_matrix(
        str(wrapped_dir / "FORCE_CONSTANTS_3RD"),
        atoms_wrapped,
        (3, 1, 1),
        order="C",
        return_support=True,
    )
    third_unwrapped, support_unwrapped = read_third_order_matrix(
        str(unwrapped_dir / "FORCE_CONSTANTS_3RD"),
        atoms_unwrapped,
        (3, 1, 1),
        order="C",
        return_support=True,
    )
    dense_wrapped = third_wrapped.todense()
    dense_unwrapped = third_unwrapped.todense()
    np.testing.assert_allclose(dense_unwrapped, dense_wrapped, atol=1e-12)
    np.testing.assert_array_equal(support_wrapped.translations[1], [1, 0, 0])
    np.testing.assert_array_equal(support_unwrapped.translations[1], [-2, 0, 0])
    assert support_wrapped.class_ids[1] == support_unwrapped.class_ids[1]
    assert np.count_nonzero(dense_unwrapped) == np.count_nonzero(phi)


def test_fc3_non_lattice_offset_raises(tmp_path):
    """A cell offset that is not an integer multiple of a lattice vector (here
    a half-cell shift along x) cannot correspond to any replica; the parser
    must raise rather than silently rounding to a wrong or empty replica."""
    import ase.io
    from kaldo.interfaces.shengbte_io import read_third_order_matrix

    rng = np.random.default_rng(17)
    phi = rng.standard_normal((3, 3, 3))
    _write_poscar(tmp_path)
    _write_fc3_text(
        tmp_path / "FORCE_CONSTANTS_3RD",
        [(np.array([1.5, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi)],
    )
    atoms = ase.io.read(tmp_path / "POSCAR", format="vasp")
    with pytest.raises(ValueError, match="lattice vector"):
        read_third_order_matrix(
            str(tmp_path / "FORCE_CONSTANTS_3RD"), atoms, (2, 1, 1), order="C"
        )


def test_fc3_same_periodic_class_keeps_distinct_literal_support(tmp_path):
    import ase.io
    from kaldo.interfaces.shengbte_io import read_third_order_matrix

    _write_poscar(tmp_path)
    phi_a = np.ones((3, 3, 3))
    phi_b = np.full((3, 3, 3), 2.0)
    path = tmp_path / "FORCE_CONSTANTS_3RD"
    _write_fc3_text(
        path,
        [
            (np.array([3.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi_a),
            (np.array([-3.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi_b),
        ],
    )
    atoms = ase.io.read(tmp_path / "POSCAR", format="vasp")
    tensor, support = read_third_order_matrix(
        str(path),
        atoms,
        (2, 1, 1),
        return_support=True,
    )

    np.testing.assert_array_equal(
        support.translations, [[0, 0, 0], [-1, 0, 0], [1, 0, 0]]
    )
    assert support.class_ids[1] == support.class_ids[2]
    dense = tensor.todense()
    np.testing.assert_allclose(dense[0, :, 2, 1, :, 0, 0, :], phi_a)
    np.testing.assert_allclose(dense[0, :, 1, 1, :, 0, 0, :], phi_b)


def test_fc3_exact_duplicate_records_are_additive(tmp_path):
    import ase.io
    from kaldo.interfaces.shengbte_io import read_third_order_matrix

    _write_poscar(tmp_path)
    phi = np.arange(27, dtype=float).reshape(3, 3, 3)
    block = (np.array([3.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi)
    path = tmp_path / "FORCE_CONSTANTS_3RD"
    _write_fc3_text(path, [block, block])
    atoms = ase.io.read(tmp_path / "POSCAR", format="vasp")
    tensor, _ = read_third_order_matrix(
        str(path),
        atoms,
        (2, 1, 1),
        return_support=True,
    )

    np.testing.assert_allclose(tensor.todense()[0, :, 1, 1, :, 0, 0, :], 2.0 * phi)


def test_fc3_support_order_is_record_order_independent_and_accepts_matrix(tmp_path):
    import ase.io
    from kaldo.interfaces.shengbte_io import read_third_order_matrix

    _write_poscar(tmp_path)
    atoms = ase.io.read(tmp_path / "POSCAR", format="vasp")
    phi_a = np.ones((3, 3, 3))
    phi_b = np.full((3, 3, 3), 2.0)
    blocks = [
        (np.array([3.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi_a),
        (np.array([-3.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi_b),
    ]
    first = tmp_path / "first.fc3"
    second = tmp_path / "second.fc3"
    _write_fc3_text(first, blocks)
    _write_fc3_text(second, list(reversed(blocks)))
    tensor_a, support_a = read_third_order_matrix(
        str(first),
        atoms,
        np.diag([2, 1, 1]),
        return_support=True,
    )
    tensor_b, support_b = read_third_order_matrix(
        str(second),
        atoms,
        np.diag([2, 1, 1]),
        return_support=True,
    )

    np.testing.assert_array_equal(support_a.translations, support_b.translations)
    np.testing.assert_allclose(tensor_a.todense(), tensor_b.todense())

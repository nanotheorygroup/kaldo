"""
Regression tests for the ShengBTE third-order cell-offset bug.

``read_third_order_matrix`` resolves each quartet's second/third cell offset
(a Cartesian vector in the FORCE_CONSTANTS_3RD file) to a replica id by
rounding it to an integer lattice index and looking it up against kaldo's
minimum-image-wrapped grid (``Grid.grid(is_wrapping=True)``). That grid keeps
only one representative index per replica within the minimum-image range, so
an offset written outside that range, whether from a different sign
convention for a half-box (even-supercell) replica, or from an unwrapped
integer index on an odd supercell, fails the lookup. The lookup returns an
empty array of ids, which is then used directly as a numpy fancy index into
the third-order tensor; indexing with an empty array is not an error, it is a
silent no-op, so the force constants for that quartet are dropped without any
exception, warning, or nonzero-count mismatch surfacing at the call site.

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
            for (a, b, c) in itertools.product([1, 2, 3], repeat=3):
                fd.write(f"{a:4d}{b:4d}{c:4d}{phi[a - 1, b - 1, c - 1]:25.15f}\n")


def test_fc3_offset_sign_convention_resolves_to_same_replica(tmp_path):
    """On supercell (2, 1, 1) with a=3.0 cubic cell, +a and -a along x are the
    same physical replica (id 1): the half-box offset is representable with
    either sign. A file written with the negative-offset convention (-3.0)
    must resolve to the same replica as the positive convention (+3.0), not
    silently drop the force constants for that quartet.

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
    _write_fc3_text(positive_dir / "FORCE_CONSTANTS_3RD",
                    [(np.array([3.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi)])
    _write_fc3_text(negative_dir / "FORCE_CONSTANTS_3RD",
                    [(np.array([-3.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi)])
    atoms_positive = ase.io.read(positive_dir / "POSCAR", format="vasp")
    atoms_negative = ase.io.read(negative_dir / "POSCAR", format="vasp")
    third_positive = read_third_order_matrix(str(positive_dir / "FORCE_CONSTANTS_3RD"), atoms_positive,
                                             (2, 1, 1), order="C")
    third_negative = read_third_order_matrix(str(negative_dir / "FORCE_CONSTANTS_3RD"), atoms_negative,
                                             (2, 1, 1), order="C")
    dense_positive = np.asarray(third_positive).reshape((2, 3, 2, 2, 3, 2, 2, 3))
    dense_negative = np.asarray(third_negative).reshape((2, 3, 2, 2, 3, 2, 2, 3))
    np.testing.assert_allclose(dense_negative, dense_positive, atol=1e-12)
    # block lands at replica 1 (the (1, 0, 0) cell on Grid((2, 1, 1), 'C'))
    np.testing.assert_allclose(dense_negative[0, :, 1, 1, :, 0, 0, :], phi, atol=1e-12)
    assert np.count_nonzero(dense_negative) == np.count_nonzero(phi)


def test_fc3_out_of_representative_range_offset_resolves_on_odd_grid(tmp_path):
    """On supercell (3, 1, 1), the integer cell index -2 (Cartesian -6.0 along
    x) is outside the minimum-image representative range kept by the wrapped
    grid, but is the same replica as index 1 (mod 3): +3.0. The parser must
    resolve -6.0 identically to +3.0.

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
    _write_fc3_text(wrapped_dir / "FORCE_CONSTANTS_3RD",
                    [(np.array([3.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi)])
    _write_fc3_text(unwrapped_dir / "FORCE_CONSTANTS_3RD",
                    [(np.array([-6.0, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi)])
    atoms_wrapped = ase.io.read(wrapped_dir / "POSCAR", format="vasp")
    atoms_unwrapped = ase.io.read(unwrapped_dir / "POSCAR", format="vasp")
    third_wrapped = read_third_order_matrix(str(wrapped_dir / "FORCE_CONSTANTS_3RD"), atoms_wrapped,
                                            (3, 1, 1), order="C")
    third_unwrapped = read_third_order_matrix(str(unwrapped_dir / "FORCE_CONSTANTS_3RD"), atoms_unwrapped,
                                              (3, 1, 1), order="C")
    dense_wrapped = np.asarray(third_wrapped).reshape((2, 3, 3, 2, 3, 3, 2, 3))
    dense_unwrapped = np.asarray(third_unwrapped).reshape((2, 3, 3, 2, 3, 3, 2, 3))
    np.testing.assert_allclose(dense_unwrapped, dense_wrapped, atol=1e-12)
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
    _write_fc3_text(tmp_path / "FORCE_CONSTANTS_3RD",
                    [(np.array([1.5, 0.0, 0.0]), np.zeros(3), (0, 1, 0), phi)])
    atoms = ase.io.read(tmp_path / "POSCAR", format="vasp")
    with pytest.raises(ValueError, match="lattice vector"):
        read_third_order_matrix(str(tmp_path / "FORCE_CONSTANTS_3RD"), atoms, (2, 1, 1), order="C")

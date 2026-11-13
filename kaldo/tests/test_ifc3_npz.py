"""Fast ``third.npz`` loading versus the reference SciPy-canonicalizing conversion."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse import load_npz, save_npz
from sparse import COO

from kaldo.observables import _ifc3_npz
from kaldo.observables._ifc3_npz import load_third_npz

REFERENCE_FILE = "kaldo/tests/data/input/amorphous-si-tersoff-32/third.npz"


def _reference(path, shape):
    return COO.from_scipy_sparse(load_npz(path)).reshape(shape).astype(np.float64)


def _assert_identical(result, expected):
    assert result.shape == expected.shape
    assert result.coords.dtype == expected.coords.dtype and result.data.dtype == np.float64
    np.testing.assert_array_equal(result.coords, expected.coords)
    np.testing.assert_array_equal(result.data, expected.data)


def _assembly_order(rng, n_rows, n_cols, index_dtype, fraction=0.3):
    """Entries as assembly writes them: rows in shuffled chunks, columns strictly increasing within a row."""
    rows, cols = [], []
    for row in rng.permutation(n_rows):
        columns = np.flatnonzero(rng.random(n_cols) < fraction)
        rows.append(np.full(columns.size, row))
        cols.append(columns)
    rows, cols = np.concatenate(rows).astype(index_dtype), np.concatenate(cols).astype(index_dtype)
    return rows, cols, rng.standard_normal(rows.size).astype(np.float32)


def _variant(name, rng):
    """Return a SciPy matrix (12 x 10 -> (3, 4, 10) or (6, 4, 5)) for one input class."""
    rows, cols, values = _assembly_order(rng, 12, 10, np.int32)
    if name == "explicit-zeros":
        values[::5] = 0
    elif name == "duplicates":
        rows, cols, values = np.r_[rows, rows[:3]], np.r_[cols, cols[:3]], np.r_[values, values[:3]]
    elif name == "unsorted-columns":
        rows, cols, values = rows[::-1].copy(), cols[::-1].copy(), values[::-1].copy()
    elif name == "int64-indices":
        rows, cols = rows.astype(np.int64), cols.astype(np.int64)
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(12, 10))
    if name == "csr":
        return matrix.tocsr()
    if name == "canonical":
        matrix.sum_duplicates()
    return matrix


@pytest.mark.parametrize("shape", [(3, 4, 10), (6, 4, 5)], ids=["row-split", "general-unravel"])
@pytest.mark.parametrize(
    "name", ["assembly", "canonical", "explicit-zeros", "duplicates", "unsorted-columns", "int64-indices", "csr"])
def test_load_third_npz_equals_reference(tmp_path, monkeypatch, name, shape):
    path = tmp_path / "third.npz"
    save_npz(path, _variant(name, np.random.default_rng(7)))
    general_calls = []
    general = _ifc3_npz._general
    monkeypatch.setattr(_ifc3_npz, "_general", lambda *args: general_calls.append(1) or general(*args))
    _assert_identical(load_third_npz(str(path), shape), _reference(str(path), shape))
    assert len(general_calls) == (name in ("duplicates", "unsorted-columns")), "fast path taken only when valid"


def test_load_third_npz_equals_reference_on_test_ifc3():
    matrix = load_npz(REFERENCE_FILE)
    n_unit = matrix.shape[1]
    shape = (n_unit, matrix.shape[0] // n_unit, n_unit)
    _assert_identical(load_third_npz(REFERENCE_FILE, shape), _reference(REFERENCE_FILE, shape))


def test_load_third_npz_rejects_wrong_size(tmp_path):
    path = tmp_path / "third.npz"
    save_npz(path, _variant("assembly", np.random.default_rng(7)))
    with pytest.raises(ValueError):
        load_third_npz(str(path), (3, 4, 9))

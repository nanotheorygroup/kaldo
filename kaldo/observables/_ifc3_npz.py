"""Private loading of SciPy ``third.npz`` IFC3 files as canonical pydata/sparse COO arrays."""

import numba
import numpy as np
from scipy.sparse import load_npz
from sparse import COO


def load_third_npz(path: str, shape: tuple[int, ...]) -> COO:
    """Return the float64 COO of a SciPy sparse ``third.npz`` reshaped to ``shape``.

    The result equals ``COO.from_scipy_sparse(load_npz(path)).reshape(shape).astype(np.float64)`` (coordinates,
    their dtype and data). Files written by assembly store each ``(i, j)`` row with strictly increasing columns,
    so a stable counting sort by row yields the canonical order directly, in a fraction of the time and memory of
    the general sort. Files with duplicate entries or unsorted columns within a row take the general path.
    """
    matrix = load_npz(path).asformat("coo")
    shape = tuple(int(dimension) for dimension in shape)
    if matrix.nnz == 0 or int(np.prod(shape, dtype=np.int64)) != matrix.shape[0] * matrix.shape[1]:
        return _general(matrix, shape)
    index_dtype = matrix.row.dtype
    if max(shape) > np.iinfo(index_dtype).max:
        index_dtype = np.min_scalar_type(max(shape))  # as COO.reshape
    coords = np.empty((len(shape), matrix.nnz), dtype=index_dtype)
    data = np.empty(matrix.nnz, dtype=np.float64)
    kept = _canonical_by_rows(
        matrix.row, matrix.col, matrix.data, matrix.shape[0], matrix.shape[1], np.asarray(shape, dtype=np.int64),
        not matrix.has_canonical_format, coords, data,
    )
    if kept < 0:
        del coords, data
        return _general(matrix, shape)
    if kept < matrix.nnz:
        coords, data = np.ascontiguousarray(coords[:, :kept]), data[:kept].copy()
    return COO(coords, data, shape=shape, has_duplicates=False, sorted=True)


def _general(matrix, shape):
    """Reference conversion: SciPy canonicalization, then reshape and float64 data."""
    return COO.from_scipy_sparse(matrix).reshape(shape).astype(np.float64)


@numba.njit(nogil=True, error_model="numpy")
def _canonical_by_rows(row, col, values, n_rows, n_cols, shape, drop_zeros, coords, data):
    """Counting-sort entries by row into ``coords``/``data``; return the kept count, or -1 if columns repeat or
    decrease within a row. Explicit zeros are dropped when ``drop_zeros`` (SciPy does so for non-canonical input).
    """
    start = np.zeros(n_rows + 1, dtype=np.int64)
    for entry in range(row.size):
        if not (drop_zeros and values[entry] == 0):
            start[row[entry] + 1] += 1
    for index in range(n_rows):
        start[index + 1] += start[index]
    rank = shape.size
    for entry in range(row.size):
        if drop_zeros and values[entry] == 0:
            continue
        position = start[row[entry]]
        start[row[entry]] = position + 1
        linear = np.int64(row[entry]) * n_cols + col[entry]
        for axis in range(rank - 1, -1, -1):
            coords[axis, position] = linear % shape[axis]
            linear //= shape[axis]
        data[position] = values[entry]
    kept = start[n_rows - 1] if n_rows else 0
    previous = np.int64(-1)
    for position in range(kept):
        linear = np.int64(0)
        for axis in range(rank):
            linear = linear * shape[axis] + coords[axis, position]
        if linear <= previous:
            return -1
        previous = linear
    return kept

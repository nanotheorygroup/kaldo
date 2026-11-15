"""Private real float64 Gamma eigensolver for amorphous dynamical matrices.

The amorphous Gamma dynamical matrix is real, so it is assembled once from the
IFC2 source directly into the output buffer and diagonalized in place with a
multithreaded LAPACK divide-and-conquer driver, or with MRRR when the 2n^2
divide-and-conquer workspace does not fit in memory. ILP64 oneMKL is used when
it can be loaded, otherwise SciPy's LAPACK. Only the lower triangle in C order
is read, matching the historical TensorFlow solver for finite-difference IFCs
that are not exactly symmetric.
"""

import ctypes
import ctypes.util
import os
import sys

import ase.units as units
import numba
import numpy as np
from psutil import virtual_memory
from scipy.linalg import lapack
from threadpoolctl import threadpool_limits

from kaldo.helpers.logger import get_logger

_logging = get_logger()
_EV_TO_TENJOVERMOL = units.mol / (10 * units.J)
_INT32_MAX = np.iinfo(np.int32).max
_TILE = 64
_AVAILABLE_FRACTION = 0.9
_PR_BLOCK = 256


def is_eligible(second, distance_threshold, is_nac) -> bool:
    """Whether the dense real route reproduces this harmonic request."""
    value = getattr(second, "value", None)
    return (
        not is_nac and distance_threshold is None and type(value) is np.ndarray
        and value.dtype == np.float64 and value.ndim == 6 and value.shape[0] == 1
        and value.shape[1:3] == value.shape[4:6] and value.shape[2] == 3
    )


def _resolve_workers(n_workers) -> int:
    workers = n_workers
    if workers is None:
        workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    if not isinstance(workers, (int, np.integer)) or workers <= 0:
        raise ValueError("eigensolver worker count must be positive")
    return int(workers)


@numba.njit(parallel=True, nogil=True)
def _fill_dynamical_matrix(value, inverse_sqrt_mass, sqrt_mass, scale, out):
    """Mass-weight ``value(n,S,n)`` eV/A^2 into ``out(n,n)`` (rad/ps)^2.

    Element arithmetic matches ``SecondOrder.calculate_dynmat`` followed by the
    Gamma sum over literal translations ``S``.
    """
    n = out.shape[0]
    n_translations = value.shape[1]
    for row in numba.prange(n):
        left = inverse_sqrt_mass[row]
        for column in range(n):
            total = 0.0
            for translation in range(n_translations):
                total += ((value[row, translation, column] * left) / sqrt_mass[column]) * scale
            out[row, column] = total


@numba.njit(parallel=True, nogil=True)
def _transpose_tiles(a, tiles, tile):
    """Swap the ``tiles(m,2)`` upper tile pairs of square ``a`` with their mirrors."""
    n = a.shape[0]
    for pair in numba.prange(tiles.shape[0]):
        i0 = tiles[pair, 0] * tile
        j0 = tiles[pair, 1] * tile
        i1 = min(i0 + tile, n)
        j1 = min(j0 + tile, n)
        for i in range(i0, i1):
            for j in range(max(j0, i + 1), j1):
                swap = a[i, j]
                a[i, j] = a[j, i]
                a[j, i] = swap


def _transpose_square(a, tile=_TILE):
    """Transpose a square, possibly row-strided ``a`` in place."""
    tiles = np.argwhere(np.triu(np.ones(((a.shape[0] + tile - 1) // tile,) * 2, dtype=bool)))
    _transpose_tiles(a, tiles, tile)


class _MKL:
    """Process-local ILP64 MKL ``dsyevd``/``dsyevr`` loaded with private symbol scope."""

    _instance = None
    _failed = False

    def __init__(self, library):
        self.library = library
        self.dsyevd = library.dsyevd_64
        self.dsyevr = library.dsyevr_64
        self.dsyevd.restype = self.dsyevr.restype = None
        self.set_threads = library.MKL_Set_Num_Threads_Local
        self.set_threads.restype = ctypes.c_int
        self.set_threads.argtypes = [ctypes.c_int]

    @classmethod
    def load(cls):
        if cls._instance is None and not cls._failed:
            cls._instance = cls._open()
            cls._failed = cls._instance is None
        return cls._instance

    @classmethod
    def _open(cls):
        library_dir = os.path.join(sys.prefix, "lib")
        names = ["libmkl_rt.so.2", "libmkl_rt.so", "libmkl_rt.2.dylib", "libmkl_rt.dylib"]
        # Environment prefix first (pip or conda ``mkl``), then the loader path (LD_LIBRARY_PATH, modules).
        candidates = [os.path.join(library_dir, name) for name in names] + names
        found = ctypes.util.find_library("mkl_rt")
        if found:
            candidates.append(found)
        for omp in ("libiomp5.so", "libiomp5.dylib"):
            path = os.path.join(library_dir, omp)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path)
                except OSError:
                    pass
        for candidate in candidates:
            if os.path.sep in candidate and not os.path.exists(candidate):
                continue
            try:
                return cls(ctypes.CDLL(candidate))
            except (OSError, AttributeError):
                continue
        return None

    def solve(self, a, vectors, workers, driver):
        """Diagonalize C-lower ``a`` in place; return eigenvalues."""
        n = a.shape[0]
        i64 = ctypes.c_int64
        ref = ctypes.byref
        w = np.empty(n, dtype=np.float64)
        info = i64(0)
        lda = i64(a.strides[0] // a.itemsize)
        query = np.empty(1, dtype=np.float64)
        iquery = np.empty(1, dtype=np.int64)

        def address(array):
            return array.ctypes.data_as(ctypes.c_void_p)

        if driver == "evd":
            jobz = ctypes.c_char_p(b"V" if vectors else b"N")

            def call(work, iwork):
                self.dsyevd(
                    jobz, ctypes.c_char_p(b"U"), ref(i64(n)), address(a), ref(lda), address(w),
                    address(work), ref(i64(work.size if work is not query else -1)), address(iwork),
                    ref(i64(iwork.size if iwork is not iquery else -1)), ref(info),
                )
        else:
            z = np.empty((n, n), dtype=np.float64)
            isuppz = np.empty(2 * n, dtype=np.int64)
            found = i64(0)
            zero = ctypes.c_double(0.0)

            def call(work, iwork):
                self.dsyevr(
                    ctypes.c_char_p(b"V"), ctypes.c_char_p(b"A"), ctypes.c_char_p(b"U"), ref(i64(n)), address(a),
                    ref(lda), ref(zero), ref(zero), ref(i64(0)), ref(i64(0)), ref(zero), ref(found), address(w),
                    address(z), ref(i64(n)), address(isuppz), address(work),
                    ref(i64(work.size if work is not query else -1)), address(iwork),
                    ref(i64(iwork.size if iwork is not iquery else -1)), ref(info),
                )
        previous = self.set_threads(workers)
        try:
            call(query, iquery)
            if info.value == 0:
                call(np.empty(int(query[0]), dtype=np.float64), np.empty(int(iquery[0]), dtype=np.int64))
        finally:
            self.set_threads(previous)
        if info.value != 0:
            raise np.linalg.LinAlgError(f"MKL dsy{driver} failed with info={info.value}")
        if driver == "evr":
            # Column-major z(n,n) holds eigenvector k in its C row k, as dsyevd leaves it in ``a``.
            a[...] = z
        return w


def _scipy_solve(a, vectors, workers, driver):
    """Diagonalize C-lower ``a`` with SciPy LAPACK; return eigenvalues."""
    n = a.shape[0]
    with threadpool_limits(workers, user_api="blas"):
        # The wrappers default to minimal workspace, which forces an unblocked
        # BLAS2 tridiagonal reduction; query the blocked optimum instead.
        if driver == "evd":
            lwork, liwork, info = lapack.dsyevd_lwork(n, compute_v=int(vectors), lower=0)
            if info == 0:
                w, v, info = lapack.dsyevd(
                    a.T, compute_v=int(vectors), lower=0, lwork=int(lwork), liwork=int(liwork), overwrite_a=1,
                )
        else:
            lwork, liwork, info = lapack.dsyevr_lwork(n, lower=0)
            if info == 0:
                w, v, _, _, info = lapack.dsyevr(
                    a.T, compute_v=1, lower=0, lwork=int(lwork), liwork=int(liwork), overwrite_a=1,
                )
    if info != 0:
        raise np.linalg.LinAlgError(f"LAPACK dsy{driver} failed with info={info}")
    if vectors and not np.shares_memory(v, a):
        a.T[...] = v
    return w


def _select_driver(n, vectors, lp64):
    """Choose divide and conquer unless its 2n^2 workspace cannot be held.

    ``dsyevr`` needs a separate n^2 eigenvector array instead. SciPy's LP64
    ``dsyevd`` workspace is addressable only below about 32767 modes.
    """
    if not vectors:
        return "evd"
    evd_bytes = 8 * (1 + 6 * n + 2 * n * n) + 8 * (3 + 5 * n)
    if lp64 and 1 + 6 * n + 2 * n * n > _INT32_MAX:
        return "evr"
    return "evd" if evd_bytes <= _AVAILABLE_FRACTION * virtual_memory().available else "evr"


def _diagonalize(a, vectors, workers):
    """Return eigenvalues; with ``vectors`` leave column eigenvectors in ``a``."""
    mkl = _MKL.load()
    driver = _select_driver(a.shape[0], vectors, lp64=mkl is None)
    _logging.info(
        "Amorphous Gamma eigensolver: %s dsy%s, n=%d, workers=%d",
        "MKL" if mkl is not None else "SciPy", driver, a.shape[0], workers,
    )
    previous = numba.get_num_threads()
    numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))
    try:
        if mkl is not None:
            w = mkl.solve(a, vectors, workers, driver)
        else:
            w = _scipy_solve(a, vectors, workers, driver)
        if vectors:
            # LAPACK leaves eigenvector k in C row k.
            _transpose_square(a)
    finally:
        numba.set_num_threads(previous)
    return w


def _assemble(second, out, workers):
    value = second.value[0]
    n = out.shape[0]
    masses = np.repeat(np.asarray(second.atoms.get_masses(), dtype=np.float64), 3)
    sqrt_mass = np.sqrt(masses)
    inverse_sqrt_mass = 1 / sqrt_mass
    previous = numba.get_num_threads()
    numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))
    try:
        _fill_dynamical_matrix(
            value.reshape(n, value.shape[2], n), inverse_sqrt_mass, sqrt_mass, _EV_TO_TENJOVERMOL, out,
        )
    finally:
        numba.set_num_threads(previous)


def gamma_eigensystem(second, n_workers=1) -> np.ndarray:
    """Return ``(n+1, n)``: eigenvalues (rad/ps)^2 above column eigenvectors."""
    workers = _resolve_workers(n_workers)
    n = second.value.shape[1] * 3
    eigensystem = np.empty((n + 1, n), dtype=np.float64)
    matrix = eigensystem[1:]
    _assemble(second, matrix, workers)
    eigensystem[0] = _diagonalize(matrix, True, workers)
    return eigensystem


def participation_ratio(eigenvectors, block=_PR_BLOCK) -> np.ndarray:
    """Return ``(n,)`` atomic participation of real column eigenvectors ``(n,n)``.

    Equals ``1 / (N sum_i (sum_a e_ia^2)^2)`` for N atoms, evaluated over column
    blocks so that only an ``(n, block)`` temporary is created.
    """
    n = eigenvectors.shape[1]
    n_atoms = eigenvectors.shape[0] // 3
    ratio = np.empty(n, dtype=np.float64)
    for start in range(0, n, block):
        columns = eigenvectors[:, start:start + block]
        atomic = np.square(columns).reshape(n_atoms, 3, -1).sum(axis=1)
        ratio[start:start + block] = 1 / (np.square(atomic).sum(axis=0) * n_atoms)
    return ratio


def gamma_eigenvalues(second, n_workers=1) -> np.ndarray:
    """Return ascending Gamma eigenvalues ``(n,)`` in (rad/ps)^2."""
    workers = _resolve_workers(n_workers)
    n = second.value.shape[1] * 3
    matrix = np.empty((n, n), dtype=np.float64)
    _assemble(second, matrix, workers)
    return _diagonalize(matrix, False, workers)

"""Private TensorFlow-free Bloch harmonics at an arbitrary q point.

The dynamical matrix ``D(q)`` and its Cartesian phase kernels are scattered
from the rows of the ordinary-IFC Fourier plan (``_HarmonicIFCInterpolation``)
by Numba kernels, reproducing ``_HarmonicIFCInterpolation.matrices`` element by
element. ``D(q)`` is real at time-reversal-invariant points (``2q`` integer)
and is then returned as float64 for the real dense eigensolver of
``_amorphous_eigensystem``; otherwise it is diagonalized with ``zheevd`` (ILP64
oneMKL when it can be loaded, SciPy's LAPACK otherwise). Only the lower
triangle in C order is read, as by the historical TensorFlow solver.

Units follow ``HarmonicWithQ``: ``D`` in (rad/ps)^2, frequencies in THz, and
group velocities in kALDo's velocity unit.
"""

from contextlib import contextmanager
import ctypes

import ase.units as units
import numba
import numpy as np
from scipy.linalg import lapack
from threadpoolctl import threadpool_limits

from kaldo.controllers._amorphous_eigensystem import _MKL, _diagonalize, _resolve_workers
from kaldo.grid import TranslationSupport, WignerSeitzImages
from kaldo.helpers.logger import get_logger
from kaldo.observables.harmonic_with_q import _HarmonicIFCInterpolation, resolve_ifc_interpolation
from kaldo.observables.harmonic_with_q_temp import CLASSICAL_HBAR_SCALE

_logging = get_logger()
_EV_TO_TENJOVERMOL = units.mol / (10 * units.J)
_TILE = 64
# Columns per parallel task of the velocity kernel (two cache lines of complex128 per eigenvector row); 8 was
# measured 2.5x faster than 2 and within 15% of 16 at n = 648; wider chunks enlarge each task's (n, chunk) working set.
_VELOCITY_CHUNK = 8
# exp(2 pi i k / 4) for k = 0..3, exact.
_QUARTER_PHASES = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)
_LAYOUT_KEY = "_bloch_row_layout"


def harmonic_plan(second, requested_mode: str) -> _HarmonicIFCInterpolation:
    """Build the ordinary-IFC Fourier plan without reading ``second.dynmat``.

    Mirrors ``_HarmonicIFCInterpolation.build`` (periodic folding, Wigner--Seitz
    images and the pair-displacement rule) with the mass weighting of
    ``SecondOrder.calculate_dynmat`` evaluated in NumPy: ``value / sqrt(m_i) /
    sqrt(m_j) * units.mol / (10 * units.J)`` in (rad/ps)^2.
    """
    support = second.translation_support
    resolved = resolve_ifc_interpolation(second, requested_mode)
    value = second.value
    if hasattr(value, "todense"):
        value = value.todense()
    value = np.asarray(value, dtype=np.float64)[0]
    sqrt_mass = np.sqrt(np.asarray(second.atoms.get_masses(), dtype=np.float64))
    values = value / sqrt_mass[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    values /= sqrt_mass[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    values *= _EV_TO_TENJOVERMOL
    if values.shape[2] != support.size:
        raise ValueError(
            "second-order IFC translation axis does not match its "
            f"TranslationSupport: {values.shape[2]} != {support.size}"
        )

    class_ids = np.asarray(support.class_ids)
    identity_folding = support.size == support.supercell.size and np.array_equal(class_ids, np.arange(support.size))
    if resolved in ("periodic", "wigner-seitz") and not identity_folding:
        folded = np.zeros(
            (values.shape[0], values.shape[1], support.supercell.size, values.shape[3], values.shape[4]),
            dtype=np.float64,
        )
        for source_id, class_id in enumerate(support.class_ids):
            folded[:, :, class_id, :, :] += values[:, :, source_id, :, :]
        if support.provenance == "file" and support.size != support.supercell.size:
            _logging.warning(
                "ifc_interpolation=%r folds %d file-provided translations into %d periodic classes by explicit "
                "user request.", requested_mode, support.size, support.supercell.size,
            )
        values = folded
    if resolved in ("periodic", "wigner-seitz"):
        # An identity folding is a copy in ``build``; skipping it saves one IFC-sized array (0.0 + x == x).
        support = TranslationSupport.periodic(support.supercell, order=support.supercell.order)

    positions = np.array(second.atoms.positions, dtype=np.float64, copy=True)
    cell = np.array(second.atoms.cell, dtype=np.float64, copy=True)
    images = (
        WignerSeitzImages.build(support, positions, cell, pbc=second.atoms.pbc) if resolved == "wigner-seitz" else None
    )
    include_pair_displacement = not (
        resolved == "periodic" and getattr(second, "ifc_interpolation_hint", None) == "periodic"
    )
    return _HarmonicIFCInterpolation(values, support, positions, cell, resolved, images, include_pair_displacement)


class _RowLayout:
    """Plan rows grouped by first atom, keeping the original row order within each group."""

    def __init__(self, plan):
        atom_is, atom_js, blocks, translations, displacements, weights = plan._assembly
        n_atoms = len(plan.positions)
        self.n_modes = 3 * n_atoms
        self.order = np.argsort(atom_is, kind="stable").astype(np.int64)
        self.offsets = np.searchsorted(atom_is[self.order], np.arange(n_atoms + 1)).astype(np.int64)
        self.atom_js = np.ascontiguousarray(atom_js, dtype=np.int64)
        self.atom_is = np.ascontiguousarray(atom_is, dtype=np.int64)
        self.blocks = np.ascontiguousarray(blocks, dtype=np.float64)
        self.translations = np.asarray(translations, dtype=np.float64)
        rounded = np.rint(self.translations)
        self.integer_translations = (
            rounded.astype(np.int64) if np.allclose(self.translations, rounded, rtol=0, atol=1e-12) else None
        )
        self.displacements = np.ascontiguousarray(displacements, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)


def _layout(plan) -> _RowLayout:
    # The plan is a frozen dataclass; like ``functools.cached_property``, store the q-independent layout in its
    # instance dictionary so it lives exactly as long as the plan.
    layout = plan.__dict__.get(_LAYOUT_KEY)
    if layout is None:
        layout = _RowLayout(plan)
        plan.__dict__[_LAYOUT_KEY] = layout
    return layout


def _is_quarter(q_point, multiple):
    scaled = multiple * q_point
    return np.allclose(scaled, np.rint(scaled), rtol=0, atol=1e-12)


def _row_coefficients(layout, q_point) -> np.ndarray:
    """Return complex128 ``weight * exp(2 pi i R.q)`` per plan row, exact on the quarter grid."""
    if layout.integer_translations is not None and _is_quarter(q_point, 4):
        quarter = np.rint(4 * q_point).astype(np.int64)
        phases = _QUARTER_PHASES[np.mod(layout.integer_translations @ quarter, 4)]
    else:
        phases = np.exp(2j * np.pi * (layout.translations @ q_point))
    return layout.weights * phases


@contextmanager
def _numba_threads(n_workers):
    workers = _resolve_workers(n_workers)
    previous = numba.get_num_threads()
    numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))
    try:
        yield workers
    finally:
        numba.set_num_threads(previous)


@numba.njit(parallel=True, nogil=True)
def _scatter_dynamical(order, offsets, atom_js, blocks, coefficients, out):
    """``out[3i+a, 3j+b] += c_r * B_r[a, b]`` over the rows ``r`` of each atom ``i``, in row order."""
    for atom_i in numba.prange(offsets.size - 1):
        for position in range(offsets[atom_i], offsets[atom_i + 1]):
            row = order[position]
            column = 3 * atom_js[row]
            coefficient = coefficients[row]
            for a in range(3):
                for b in range(3):
                    out[3 * atom_i + a, column + b] += coefficient * blocks[row, a, b]


@numba.njit(parallel=True, nogil=True)
def _scatter_derivative(order, offsets, atom_js, blocks, coefficients, displacement, out):
    """``out[3i+a, 3j+b] -= d_r * (c_r * B_r[a, b])``: the stored phase kernel without the factor ``i``."""
    for atom_i in numba.prange(offsets.size - 1):
        for position in range(offsets[atom_i], offsets[atom_i + 1]):
            row = order[position]
            column = 3 * atom_js[row]
            coefficient = coefficients[row]
            distance = displacement[row]
            for a in range(3):
                for b in range(3):
                    out[3 * atom_i + a, column + b] -= distance * (coefficient * blocks[row, a, b])


def dynamical_matrix(plan, q_point: np.ndarray, n_workers=None) -> np.ndarray:
    """Return ``D(q)`` ``(n, n)`` in (rad/ps)^2: float64 when ``2q`` is integer, complex128 otherwise."""
    q_point = np.asarray(q_point, dtype=np.float64)
    layout = _layout(plan)
    coefficients = _row_coefficients(layout, q_point)
    if layout.integer_translations is not None and _is_quarter(q_point, 2):
        # Every phase is exactly +-1 here, so the matrix is real.
        coefficients = np.ascontiguousarray(coefficients.real)
        out = np.zeros((layout.n_modes, layout.n_modes), dtype=np.float64)
    else:
        out = np.zeros((layout.n_modes, layout.n_modes), dtype=np.complex128)
    with _numba_threads(n_workers):
        _scatter_dynamical(layout.order, layout.offsets, layout.atom_js, layout.blocks, coefficients, out)
    return out


def derivative_matrix(plan, q_point, direction: int, n_workers=None) -> np.ndarray:
    """Return the complex128 ``(n, n)`` Cartesian phase kernel ``plan.matrices(q)[1][direction]``."""
    q_point = np.asarray(q_point, dtype=np.float64)
    layout = _layout(plan)
    coefficients = _row_coefficients(layout, q_point)
    displacement = np.ascontiguousarray(layout.displacements[:, direction])
    out = np.zeros((layout.n_modes, layout.n_modes), dtype=np.complex128)
    with _numba_threads(n_workers):
        _scatter_derivative(
            layout.order, layout.offsets, layout.atom_js, layout.blocks, coefficients, displacement, out,
        )
    return out


@numba.njit(parallel=True, nogil=True)
def _conjugate_transpose_tiles(a, tiles, tile):
    """Replace square ``a`` by its conjugate transpose over the ``tiles(m,2)`` upper tile pairs."""
    n = a.shape[0]
    for pair in numba.prange(tiles.shape[0]):
        i0 = tiles[pair, 0] * tile
        j0 = tiles[pair, 1] * tile
        i1 = min(i0 + tile, n)
        j1 = min(j0 + tile, n)
        for i in range(i0, i1):
            for j in range(max(j0, i), j1):
                if j == i:
                    a[i, i] = np.conj(a[i, i])
                else:
                    swap = a[i, j]
                    a[i, j] = np.conj(a[j, i])
                    a[j, i] = np.conj(swap)


def _conjugate_transpose_square(a, tile=_TILE):
    tiles = np.argwhere(np.triu(np.ones(((a.shape[0] + tile - 1) // tile,) * 2, dtype=bool)))
    _conjugate_transpose_tiles(a, tiles, tile)


_ZHEEVD = {}


def _mkl_zheevd(mkl, a, workers):
    """Diagonalize C-lower complex ``a`` in place with ILP64 ``zheevd_64``; return eigenvalues.

    The column-major view of C-ordered ``a`` is ``a^T = conj(a)``; LAPACK leaves eigenvector ``k`` of
    that matrix, i.e. the conjugate of eigenvector ``k`` of ``a``, in C row ``k``.
    """
    zheevd = _ZHEEVD.get(id(mkl.library))
    if zheevd is None:
        zheevd = mkl.library.zheevd_64
        zheevd.restype = None
        _ZHEEVD[id(mkl.library)] = zheevd
    n = a.shape[0]
    i64 = ctypes.c_int64
    ref = ctypes.byref
    w = np.empty(n, dtype=np.float64)
    info = i64(0)
    lda = i64(a.strides[0] // a.itemsize)

    def address(array):
        return array.ctypes.data_as(ctypes.c_void_p)

    def call(work, rwork, iwork, query):
        sizes = (-1, -1, -1) if query else (work.size, rwork.size, iwork.size)
        zheevd(
            ctypes.c_char_p(b"V"), ctypes.c_char_p(b"U"), ref(i64(n)), address(a), ref(lda), address(w),
            address(work), ref(i64(sizes[0])), address(rwork), ref(i64(sizes[1])), address(iwork),
            ref(i64(sizes[2])), ref(info),
        )

    previous = mkl.set_threads(workers)
    try:
        work = np.empty(1, dtype=np.complex128)
        rwork = np.empty(1, dtype=np.float64)
        iwork = np.empty(1, dtype=np.int64)
        call(work, rwork, iwork, query=True)
        if info.value == 0:
            call(
                np.empty(int(work[0].real), dtype=np.complex128), np.empty(int(rwork[0]), dtype=np.float64),
                np.empty(int(iwork[0]), dtype=np.int64), query=False,
            )
    finally:
        mkl.set_threads(previous)
    if info.value != 0:
        raise np.linalg.LinAlgError(f"MKL zheevd failed with info={info.value}")
    return w


def _scipy_zheevd(a, workers):
    """Diagonalize C-lower complex ``a`` with SciPy ``zheevd``; same in-place layout as :func:`_mkl_zheevd`."""
    n = a.shape[0]
    with threadpool_limits(workers, user_api="blas"):
        # The wrapper's default workspace forces the unblocked tridiagonal reduction; query the blocked optimum.
        work, iwork, rwork, info = lapack.zheevd_lwork(n, compute_v=1, lower=0)
        if info == 0:
            w, v, info = lapack.zheevd(
                a.T, compute_v=1, lower=0, lwork=int(np.real(work)), liwork=int(iwork), lrwork=int(rwork),
                overwrite_a=1,
            )
    if info != 0:
        raise np.linalg.LinAlgError(f"LAPACK zheevd failed with info={info}")
    if not np.shares_memory(v, a):
        a.T[...] = v
    return w


def eigensystem(matrix: np.ndarray, n_workers) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending eigenvalues ``(n,)`` and column eigenvectors ``(n, n)`` of ``D(q)``.

    ``matrix`` is overwritten: the eigenvectors are returned in its storage. Only its lower triangle in C order
    is read. Float64 input takes the real dense solver of ``_amorphous_eigensystem``; complex128 input takes
    ``zheevd``.
    """
    workers = _resolve_workers(n_workers)
    if matrix.dtype == np.float64:
        return _diagonalize(matrix, True, workers), matrix
    if matrix.dtype != np.complex128 or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square float64 or complex128 matrix, got {matrix.dtype} {matrix.shape}")
    if matrix.strides[1] != matrix.itemsize:
        raise ValueError("eigensystem needs unit-stride rows")
    mkl = _MKL.load()
    _logging.info(
        "Bloch eigensolver: %s zheevd, n=%d, workers=%d", "MKL" if mkl is not None else "SciPy", matrix.shape[0],
        workers,
    )
    with _numba_threads(workers):
        w = _mkl_zheevd(mkl, matrix, workers) if mkl is not None else _scipy_zheevd(matrix, workers)
        _conjugate_transpose_square(matrix)
    return w, matrix


def frequency_from_eigenvalues(w) -> np.ndarray:
    """Signed THz frequencies ``sign(w) sqrt(|w|) / (2 pi)``; negative eigenvalues mark imaginary modes."""
    w = np.asarray(w, dtype=np.float64)
    return np.abs(w) ** 0.5 * np.sign(w) / (np.pi * 2.0)


@numba.njit(parallel=True, nogil=True)
def _velocity_numerators(atom_is, atom_js, blocks, coefficients, displacements, vectors, chunk, out):
    """``out[m, x] = Im(sum_r conj(E[3i+a, m]) (-d_rx c_r B_r[a, b]) E[3j+b, m])`` over plan rows ``r``.

    Each parallel task owns ``chunk`` adjacent columns, so ``out`` rows are written by one task only.
    """
    n = vectors.shape[1]
    n_rows = atom_is.size
    for task in numba.prange((n + chunk - 1) // chunk):
        start = task * chunk
        stop = min(start + chunk, n)
        width = stop - start
        accumulator = np.zeros((width, 3), dtype=np.float64)
        element = np.empty((3, 3), dtype=coefficients.dtype)
        for row in range(n_rows):
            left = 3 * atom_is[row]
            right = 3 * atom_js[row]
            coefficient = coefficients[row]
            for a in range(3):
                for b in range(3):
                    element[a, b] = coefficient * blocks[row, a, b]
            dx = displacements[row, 0]
            dy = displacements[row, 1]
            dz = displacements[row, 2]
            for k in range(width):
                column = start + k
                y0 = vectors[right, column]
                y1 = vectors[right + 1, column]
                y2 = vectors[right + 2, column]
                # Im(x^H (c B) y) with the 3x3 block applied to y first.
                value = (
                    np.conj(vectors[left, column]) * (element[0, 0] * y0 + element[0, 1] * y1 + element[0, 2] * y2)
                    + np.conj(vectors[left + 1, column])
                    * (element[1, 0] * y0 + element[1, 1] * y1 + element[1, 2] * y2)
                    + np.conj(vectors[left + 2, column])
                    * (element[2, 0] * y0 + element[2, 1] * y1 + element[2, 2] * y2)
                ).imag
                accumulator[k, 0] -= dx * value
                accumulator[k, 1] -= dy * value
                accumulator[k, 2] -= dz * value
        for k in range(width):
            for x in range(3):
                out[start + k, x] = accumulator[k, x]


def diagonal_velocity(plan, q_point, eigenvectors, frequency, n_workers) -> np.ndarray:
    """Return group velocities ``(n, 3)``: ``Im(diag(E^H D_x E)) / (4 pi f)`` for each direction ``x``.

    ``D_x`` is the stored phase kernel of :func:`derivative_matrix`. The diagonal is contracted directly from the
    plan rows in parallel column chunks, so neither ``D_x`` nor any ``(n, n)`` flux operator is formed. Modes with
    nonpositive frequency get zero velocity.
    """
    q_point = np.asarray(q_point, dtype=np.float64)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    layout = _layout(plan)
    coefficients = _row_coefficients(layout, q_point)
    numerators = np.empty((eigenvectors.shape[1], 3), dtype=np.float64)
    with _numba_threads(n_workers):
        _velocity_numerators(
            layout.atom_is, layout.atom_js, layout.blocks, coefficients, layout.displacements, eigenvectors,
            _VELOCITY_CHUNK, numerators,
        )
    velocity = np.zeros_like(numerators)
    positive = frequency > 0
    velocity[positive] = numerators[positive] / (4 * np.pi * frequency[positive, np.newaxis])
    return velocity


def _hbar(is_classic):
    return units._hbar * CLASSICAL_HBAR_SCALE if is_classic else units._hbar


def population(frequency, physical, temperature, is_classic) -> np.ndarray:
    """Bose--Einstein occupation, zero on unphysical modes, as ``HarmonicWithQTemp._calculate_population``."""
    frequency = np.asarray(frequency, dtype=np.float64)
    kelvintothz = units.kB / units.J / (2 * np.pi * _hbar(is_classic)) * 1e-12
    temp = temperature * kelvintothz
    occupation = np.zeros_like(frequency)
    physical = np.asarray(physical, dtype=bool).reshape(frequency.shape)
    occupation[physical] = 1. / (np.exp(frequency[physical] / temp) - 1.)
    return occupation


def heat_capacity(frequency, physical, temperature, is_classic) -> np.ndarray:
    """Mode heat capacity in J/K, as ``HarmonicWithQTemp._calculate_heat_capacity``."""
    frequency = np.asarray(frequency, dtype=np.float64)
    c_v = np.zeros_like(frequency)
    kelvintothz = units.kB / units.J / (2 * np.pi * _hbar(is_classic)) * 1e-12
    scaled_temperature = temperature * kelvintothz
    physical = np.asarray(physical, dtype=bool).reshape(frequency.shape)
    kelvintojoule = units.kB / units.J
    f_be = population(frequency, physical, temperature, is_classic)
    c_v[physical] = kelvintojoule * f_be[physical] * (f_be[physical] + 1) * frequency[physical] ** 2 / (
        scaled_temperature ** 2)
    return c_v

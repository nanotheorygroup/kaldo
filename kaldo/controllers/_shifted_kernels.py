"""Private factorized Wigner-Seitz IFC3 legs and Numba kernels for shifted-grid projection.

For a single-cell source, Wigner-Seitz compilation weights each Gamma-folded
IFC3 entry by independent leg images, ``Phi(i,jR',kR'') = PhiG[i,j,k] w_ij(R')
w_ik(R'')``. The q-folded tensor is therefore ``PhiG[i,j,k] alpha_ij(q')
beta_ik(q'')`` with per-atom-pair tables built here. The kernels contract the
Gamma pair CSR with those tables (stage 1) and reduce complex energy-window
tiles (stage 3) for the shifted-grid projection driver.
"""

from typing import NamedTuple

import numba
import numpy as np

from kaldo.controllers._projection_kernels import _BLAS_MAX, _dgemm, _gemm_into, _validate_matrix
from kaldo.grid import TranslationSupport, WignerSeitzImages
from kaldo.observables.thirdorder import _IFC_INTERPOLATION_MODES, _GammaPairCSR

_ROW_BLOCK = 128  # rows per tile, as in _amorphous_projection._plan_tiles
_IMAGINARY_TOLERANCE = 1e-12


def leg_tables(
    third, layout: _GammaPairCSR, q_rows: np.ndarray, q_cols: np.ndarray, mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-atom-pair IFC3 leg phase tables for a single-cell source.

    ``q_rows`` ``(m,3)`` and ``q_cols`` ``(m',3)`` are fractional reciprocal
    coordinates of the j leg (q') and the k leg (q''). The dimensionless
    results are complex128 ``alpha(m,N,N)`` with
    ``alpha[a,i,j] = sum_t w_t exp(2 pi i q_rows[a].R_t)`` and float64
    ``beta(m',N,N)`` with ``beta[b,i,k] = Re sum_t w_t exp(-2 pi i q_cols[b].R_t)``,
    summed over the tied Wigner-Seitz images ``R_t`` of atom pair ``(i,j)``
    or ``(i,k)``. ``mode`` resolves as ``ThirdOrder.get_interpolation`` does,
    without compiling the rank-eight tensor: periodic interpolation, or a
    file-provided single zero translation under ``'auto'``, gives unit legs.
    Images are evaluated only for atom pairs present in ``layout``; absent
    pairs hold zero. ``ValueError`` is raised for a multi-cell supercell,
    other literal file translations, or ``|Im beta| > 1e-12``.
    """
    if mode not in _IFC_INTERPOLATION_MODES:
        raise ValueError(f"interpolation mode {mode!r} is invalid; choose one of {_IFC_INTERPOLATION_MODES}")
    q_rows, q_cols = _q_points(q_rows, "q_rows"), _q_points(q_cols, "q_cols")
    n_atoms = len(third.atoms)
    if layout.n_modes != 3 * n_atoms:
        raise ValueError("Gamma pair layout does not match the IFC3 atom count")
    support = third.translation_support
    if support.supercell.size != 1:
        raise ValueError("factorized IFC3 legs require a single-cell (supercell size 1) source")
    alpha_shape, beta_shape = (len(q_rows), n_atoms, n_atoms), (len(q_cols), n_atoms, n_atoms)
    if mode == "auto" and support.provenance == "file":
        if support.size != 1 or np.any(support.translations[0] != 0):
            raise ValueError("literal file-provided IFC3 translations other than one zero translation are unsupported")
        resolved = "periodic"
    else:
        resolved = (getattr(third, "ifc_interpolation_hint", None) or "wigner-seitz") if mode == "auto" else mode
    if resolved == "periodic":
        return np.ones(alpha_shape, dtype=np.complex128), np.ones(beta_shape, dtype=np.float64)

    present = _present_atom_pairs(layout, n_atoms)
    atom_i, atom_j = np.nonzero(present)
    del present
    alpha = np.zeros(alpha_shape, dtype=np.complex128)
    beta = np.zeros(beta_shape, dtype=np.float64)
    if not atom_i.size:
        return alpha, beta
    images = WignerSeitzImages.build(
        TranslationSupport.periodic(support.supercell, order=support.supercell.order),
        np.asarray(third.atoms.positions), np.asarray(third.atoms.cell), pbc=third.atoms.pbc,
    )
    pair_ids, translations, _, weights = images.image_table(np.zeros(atom_i.size, dtype=np.int64), atom_i, atom_j)
    alpha[:, atom_i, atom_j] = _image_sums(q_rows, translations, weights, pair_ids, atom_i.size, 1.)
    beta_complex = _image_sums(q_cols, translations, weights, pair_ids, atom_i.size, -1.)
    if np.any(np.abs(beta_complex.imag) > _IMAGINARY_TOLERANCE):
        raise ValueError("beta legs are not real; q_cols must lie on the Gamma-centred grid {0,1/2}^3")
    beta[:, atom_i, atom_j] = beta_complex.real
    return alpha, beta


def _q_points(q_points: np.ndarray, name: str) -> np.ndarray:
    """Validate finite fractional ``(m,3)`` q-points as float64."""
    q_points = np.asarray(q_points, dtype=np.float64)
    if q_points.ndim != 2 or q_points.shape[1] != 3 or not np.all(np.isfinite(q_points)):
        raise ValueError(name + " must be finite fractional coordinates with shape (m,3)")
    return q_points


def _present_atom_pairs(layout: _GammaPairCSR, n_atoms: int) -> np.ndarray:
    """Mark boolean ``(N,N)`` atom pairs ``(i,j)`` and ``(i,k)`` carried by the pair CSR."""
    present = np.zeros((n_atoms, n_atoms), dtype=np.bool_)
    pair_atom_i = (layout.pair_i // 3).astype(np.int32)
    partner_atom_j = np.repeat(np.arange(layout.n_modes, dtype=np.int64) // 3, np.diff(layout.j_ptr))
    present[pair_atom_i, partner_atom_j] = True
    del partner_atom_j
    for start, _, matrix in layout.chunks:
        _mark_partner_atoms(matrix.indptr, matrix.indices, start, pair_atom_i, present)
    return present


@numba.njit(nogil=True, error_model="numpy")
def _mark_partner_atoms(
    indptr: np.ndarray, indices: np.ndarray, row_offset: int, pair_atom_i: np.ndarray, present: np.ndarray,
) -> None:
    """Set ``present[atom_i, k // 3]`` for every entry of one CSR chunk."""
    for row in range(indptr.size - 1):
        atom_i = pair_atom_i[row_offset + row]
        for entry in range(indptr[row], indptr[row + 1]):
            present[atom_i, indices[entry] // 3] = True


def _image_sums(
    q_points: np.ndarray, translations: np.ndarray, weights: np.ndarray, pair_ids: np.ndarray, n_pairs: int,
    sign: float,
) -> np.ndarray:
    """Sum tied-image weights times ``exp(sign 2 pi i q.R)`` into complex ``(m,pairs)``."""
    angles = sign * 2 * np.pi * np.einsum("ax,tx->at", q_points, translations.astype(np.float64))
    sums = np.empty((len(q_points), n_pairs), dtype=np.complex128)
    for index, angle in enumerate(angles):
        sums[index].real = np.bincount(pair_ids, weights=weights * np.cos(angle), minlength=n_pairs)
        sums[index].imag = np.bincount(pair_ids, weights=weights * np.sin(angle), minlength=n_pairs)
    return sums


def stage1(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, row_offset: int, pair_atom_i: np.ndarray,
    beta_b: np.ndarray, columns: np.ndarray, out: np.ndarray,
) -> None:
    """Contract one Gamma pair CSR chunk with beta-scaled mode columns.

    ``indptr``, ``indices`` and float64 ``data`` (eV/Angstrom^3) are one
    chunk's CSR arrays over Cartesian k. For local row ``r`` this writes
    ``out[row_offset+r] = sum_k data * beta_b[pair_atom_i[row_offset+r], k//3]
    * columns[k]``. ``pair_atom_i`` is the int32 ``(pairs,)`` atom of each
    pair row, ``beta_b`` the dimensionless float64 ``(N,N)`` leg table,
    ``columns`` float64 ``(n,w)`` and ``out`` float64 ``(pairs,w)`` with unit
    column stride. Rows run in parallel; IFC values are never copied.
    """
    if not (
        isinstance(indptr, np.ndarray) and isinstance(indices, np.ndarray) and indptr.ndim == indices.ndim == 1
        and indptr.dtype.kind == indices.dtype.kind == "i" and indptr.size >= 1
    ):
        raise TypeError("CSR indptr and indices must be one-dimensional integer arrays")
    if not isinstance(data, np.ndarray) or data.dtype != np.float64 or data.shape != indices.shape:
        raise TypeError("CSR data must be a float64 vector matching indices")
    if not isinstance(pair_atom_i, np.ndarray) or pair_atom_i.dtype != np.int32 or pair_atom_i.ndim != 1:
        raise TypeError("pair_atom_i must be a one-dimensional int32 array")
    if not isinstance(beta_b, np.ndarray) or beta_b.dtype != np.float64 or beta_b.ndim != 2:
        raise TypeError("beta_b must be a two-dimensional float64 array")
    n_atoms = beta_b.shape[0]
    for name, matrix in (("columns", columns), ("out", out)):
        if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float64 or matrix.ndim != 2:
            raise TypeError(name + " must be a two-dimensional float64 array")
        if matrix.size and (matrix.strides[1] != matrix.itemsize or not matrix.flags.aligned):
            raise ValueError(name + " requires aligned unit column stride")
    if beta_b.shape != (n_atoms, n_atoms) or columns.shape[0] != 3 * n_atoms or out.shape[1] != columns.shape[1]:
        raise ValueError("beta_b must be (N,N), columns (3N,w) and out (pairs,w)")
    if not isinstance(row_offset, (int, np.integer)):
        raise TypeError("row_offset must be an integer")
    n_rows = indptr.size - 1
    row_offset = int(row_offset)
    if row_offset < 0 or row_offset + n_rows > min(out.shape[0], pair_atom_i.size):
        raise ValueError("CSR chunk rows exceed the pair rows of pair_atom_i or out")
    atoms = pair_atom_i[row_offset:row_offset + n_rows]
    if n_rows and (atoms.min() < 0 or atoms.max() >= n_atoms):
        raise ValueError("pair_atom_i contains atoms outside beta_b")
    if _stage1_kernel(indptr, indices, data, row_offset, pair_atom_i, beta_b, columns, out):
        raise ValueError("CSR chunk has invalid row pointers or column indices")


@numba.njit(nogil=True, parallel=True, error_model="numpy")
def _stage1_kernel(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, row_offset: int, pair_atom_i: np.ndarray,
    beta_b: np.ndarray, columns: np.ndarray, out: np.ndarray,
) -> int:
    """Run checked ``stage1`` rows in parallel; return the count of invalid pointers or indices.

    Four entries share one pass over the target row, which halves its
    load/store traffic (measured 1.4-1.6x faster than one entry per pass).
    """
    n_modes, width, n_entries = columns.shape[0], columns.shape[1], data.size
    invalid = 0
    for row in numba.prange(indptr.size - 1):
        target = out[row_offset + row]
        target[:] = 0.
        start, stop = indptr[row], indptr[row + 1]
        if start < 0 or stop < start or stop > n_entries:
            invalid += 1
            continue
        weights = beta_b[pair_atom_i[row_offset + row]]
        entry = start
        while entry + 4 <= stop:
            k0, k1, k2, k3 = indices[entry], indices[entry + 1], indices[entry + 2], indices[entry + 3]
            entry += 4
            if min(k0, k1, k2, k3) < 0 or max(k0, k1, k2, k3) >= n_modes:
                invalid += 1
                continue
            scale0 = data[entry - 4] * weights[k0 // 3]
            scale1 = data[entry - 3] * weights[k1 // 3]
            scale2 = data[entry - 2] * weights[k2 // 3]
            scale3 = data[entry - 1] * weights[k3 // 3]
            source0, source1, source2, source3 = columns[k0], columns[k1], columns[k2], columns[k3]
            for column in range(width):
                target[column] += (
                    (scale0 * source0[column] + scale1 * source1[column])
                    + (scale2 * source2[column] + scale3 * source3[column])
                )
        for entry in range(entry, stop):
            k = indices[entry]
            if k < 0 or k >= n_modes:
                invalid += 1
                continue
            scale = data[entry] * weights[k // 3]
            source = columns[k]
            for column in range(width):
                target[column] += scale * source[column]
    return invalid


class Stage2Plan(NamedTuple):
    """Validated pair layout for ``stage2``: int64 ``j_ptr (n+1,)``, int64 ``pair_i (pairs,)``, int32
    ``pair_atom_i (pairs,)``, boolean ``reuse (n,)`` (partner ``j`` has the pair rows of ``j-1`` within one atom, so
    its weights are reused), the largest pair-row count ``max_rows`` and the atom count."""
    j_ptr: np.ndarray
    pair_i: np.ndarray
    pair_atom_i: np.ndarray
    reuse: np.ndarray
    max_rows: int
    n_atoms: int


def stage2_plan(j_ptr: np.ndarray, pair_i: np.ndarray, n_atoms: int) -> Stage2Plan:
    """Validate the Gamma pair layout once and mark partners whose weights repeat their predecessor's.

    The q' leg ``alpha_ij`` depends only on atoms, so the three Cartesian partners of one atom share stage-2
    weights whenever their pair rows coincide (as they do for dense 3x3 IFC3 blocks).
    """
    j_ptr = np.ascontiguousarray(j_ptr, dtype=np.int64)
    pair_i = np.ascontiguousarray(pair_i, dtype=np.int64)
    n_modes = 3 * int(n_atoms)
    if j_ptr.shape != (n_modes + 1,) or j_ptr[0] != 0 or j_ptr[-1] != pair_i.size or np.any(np.diff(j_ptr) < 0):
        raise ValueError("j_ptr must be nondecreasing row pointers (3N+1,) over all pair rows")
    if pair_i.ndim != 1 or (pair_i.size and (pair_i.min() < 0 or pair_i.max() >= n_modes)):
        raise ValueError("pair_i must hold Cartesian indices of the N atoms")
    counts = np.diff(j_ptr)
    reuse = np.zeros(n_modes, dtype=np.bool_)
    for partner_j in range(n_modes):
        if partner_j % 3 and counts[partner_j] == counts[partner_j - 1]:
            start, previous = j_ptr[partner_j], j_ptr[partner_j - 1]
            reuse[partner_j] = np.array_equal(pair_i[start:start + counts[partner_j]],
                                              pair_i[previous:previous + counts[partner_j]])
    return Stage2Plan(j_ptr, pair_i, (pair_i // 3).astype(np.int32), reuse, int(counts.max(initial=0)),
                      int(n_atoms))


def stage2(
    plan: Stage2Plan, alpha_q: np.ndarray, initial_re: np.ndarray, initial_im: np.ndarray, contracted: np.ndarray,
    inter_re: np.ndarray, inter_im: np.ndarray, inter_sum: np.ndarray, pool, parts: int,
) -> None:
    """Fill ``inter[j, nu, :] = sum_i e_i(q nu) alpha_ij(q') C[(i,j), :]`` as float64 partner-major planes.

    ``alpha_q`` is the complex128 ``(N,N)`` q' leg, ``initial_re/initial_im`` the float64 ``(n, block)``
    mass-rescaled initial vectors, ``contracted`` the stage-1 output ``(pairs, w)`` (unit column stride, any row
    stride) and ``inter_re``, ``inter_im`` and ``inter_sum = inter_re + inter_im`` contiguous ``(n, block, w)``
    planes (eV/(Angstrom^3*amu)); partner-major planes make each GEMM output one contiguous slab (measured 17%
    faster than ``(block, n, w)`` at 1920 atoms), and ``plane.transpose(1, 0, 2)`` is the ``(block, n, w)`` view
    ``reduce_tiles_split`` reads. Partners are split into ``parts`` contiguous groups of whole atoms run on
    ``pool``. Per partner, two GEMMs with weights ``(rows, block)`` built from contiguous initial rows.
    """
    n_modes = plan.j_ptr.size - 1
    block = initial_re.shape[1] if initial_re.ndim == 2 else -1
    for name, vectors in (("initial_re", initial_re), ("initial_im", initial_im)):
        if (
            not isinstance(vectors, np.ndarray) or vectors.dtype != np.float64 or vectors.shape != (n_modes, block)
            or not vectors.flags.c_contiguous
        ):
            raise TypeError(name + " must be contiguous float64 (n, block)")
    if not 0 < block <= _BLAS_MAX:
        raise ValueError("stage-2 block must be positive")
    if not isinstance(alpha_q, np.ndarray) or alpha_q.dtype != np.complex128 or alpha_q.shape != (plan.n_atoms,) * 2:
        raise TypeError("alpha_q must be complex128 (N,N)")
    _validate_matrix(contracted, "contracted")
    if contracted.shape[0] != plan.pair_i.size:
        raise ValueError("contracted must hold one row per pair")
    width = contracted.shape[1]
    for name, plane in (("inter_re", inter_re), ("inter_im", inter_im), ("inter_sum", inter_sum)):
        if (
            not isinstance(plane, np.ndarray) or plane.dtype != np.float64 or plane.shape != (n_modes, block, width)
            or not plane.flags.c_contiguous or block * width > _BLAS_MAX
        ):
            raise TypeError(name + " must be contiguous float64 (n, block, w) with an LP64 slot stride")
    edges = np.unique(np.linspace(0, plan.n_atoms, max(1, min(int(parts), plan.n_atoms)) + 1).astype(np.int64))

    def run(bounds):
        weights = np.empty((2, max(plan.max_rows, 1) * block), dtype=np.float64)
        _stage2_kernel(3 * int(bounds[0]), 3 * int(bounds[1]), plan.j_ptr, plan.pair_i, plan.pair_atom_i, plan.reuse,
                       alpha_q, initial_re, initial_im, contracted, inter_re, inter_im, inter_sum, weights)

    list(pool.map(run, zip(edges[:-1], edges[1:])))


@numba.njit(nogil=True)
def _gemm_tn(
    weights: np.ndarray, right: np.ndarray, output: np.ndarray, flags: np.ndarray, dimensions: np.ndarray,
    scalars: np.ndarray,
) -> None:
    """Row-major ``output(M,N) = weights(K,M)^T right(K,N)`` as column-major ``dgemm('N','T')``."""
    dimensions[0] = right.shape[1]
    dimensions[1] = weights.shape[1]
    dimensions[2] = weights.shape[0]
    dimensions[3] = right.strides[0] // right.itemsize
    dimensions[4] = weights.strides[0] // weights.itemsize
    dimensions[5] = output.strides[0] // output.itemsize
    _dgemm(
        flags[0:].ctypes, flags[1:].ctypes, dimensions[0:].ctypes, dimensions[1:].ctypes, dimensions[2:].ctypes,
        scalars[0:].ctypes, right.ctypes, dimensions[3:].ctypes, weights.ctypes, dimensions[4:].ctypes,
        scalars[1:].ctypes, output.ctypes, dimensions[5:].ctypes,
    )


@numba.njit(nogil=True, error_model="numpy")
def _stage2_kernel(
    j_start: int, j_stop: int, j_ptr: np.ndarray, pair_i: np.ndarray, pair_atom_i: np.ndarray, reuse: np.ndarray,
    alpha_q: np.ndarray, initial_re: np.ndarray, initial_im: np.ndarray, contracted: np.ndarray,
    inter_re: np.ndarray, inter_im: np.ndarray, inter_sum: np.ndarray, weights: np.ndarray,
) -> None:
    """Run checked ``stage2`` partners ``[j_start, j_stop)``; ``weights`` is ``(2, max_rows * block)`` scratch."""
    flags = np.array([ord("N"), ord("T")], dtype=np.uint8)
    dimensions = np.empty(6, dtype=np.int32)
    assign = np.array([1., 0.])
    block = initial_re.shape[1]
    for partner_j in range(j_start, j_stop):
        start, stop = j_ptr[partner_j], j_ptr[partner_j + 1]
        rows = stop - start
        if rows == 0:
            inter_re[partner_j] = 0.
            inter_im[partner_j] = 0.
            inter_sum[partner_j] = 0.
            continue
        weight_re = weights[0, :rows * block].reshape((rows, block))
        weight_im = weights[1, :rows * block].reshape((rows, block))
        if partner_j == j_start or not reuse[partner_j]:
            atom_j = partner_j // 3
            for row in range(rows):
                leg = alpha_q[pair_atom_i[start + row], atom_j]
                leg_re, leg_im = leg.real, leg.imag
                source_re, source_im = initial_re[pair_i[start + row]], initial_im[pair_i[start + row]]
                target_re, target_im = weight_re[row], weight_im[row]
                for nu in range(block):
                    target_re[nu] = source_re[nu] * leg_re - source_im[nu] * leg_im
                    target_im[nu] = source_re[nu] * leg_im + source_im[nu] * leg_re
        slab = contracted[start:stop]
        out_re, out_im, out_sum = inter_re[partner_j], inter_im[partner_j], inter_sum[partner_j]
        _gemm_tn(weight_re, slab, out_re, flags, dimensions, assign)
        _gemm_tn(weight_im, slab, out_im, flags, dimensions, assign)
        for nu in range(block):
            for column in range(out_sum.shape[1]):
                out_sum[nu, column] = out_re[nu, column] + out_im[nu, column]


def plan_tiles(
    modes: np.ndarray, omega_initial: np.ndarray, omega_rows: np.ndarray, omega_cols: np.ndarray,
    physical_rows: np.ndarray, physical_cols: np.ndarray, threshold: float,
) -> np.ndarray:
    """Plan int64 ``(tiles,7)`` slot/mode/channel/row/column spans in rad/ps.

    ``modes`` holds ``(block,)`` initial indices into ``omega_initial`` (q);
    rows index ``omega_rows`` (q') and columns ``omega_cols`` (q''), each
    ``(n,)`` in rad/ps and sorted ascending. Boolean ``physical_rows`` and
    ``physical_cols`` are ``(n,)`` masks. Columns conservatively enclose
    ``|omega + (2 channel - 1) omega' - omega''| < threshold``; a row is kept
    when it is physical and its window holds a physical column. Exact strict
    selection is left to ``reduce_tiles_split``. Tiles of one slot and
    channel are disjoint.
    """
    for name, omega in (("omega_rows", omega_rows), ("omega_cols", omega_cols)):
        if np.any(omega[1:] < omega[:-1]):
            raise ValueError(name + " must be sorted ascending")
    modes = np.ascontiguousarray(modes, dtype=np.int64)
    omega_initial, omega_rows, omega_cols = (
        np.ascontiguousarray(omega, dtype=np.float64) for omega in (omega_initial, omega_rows, omega_cols))
    physical_rows = np.ascontiguousarray(physical_rows, dtype=np.bool_)
    physical_before = np.concatenate(([0], np.cumsum(physical_cols, dtype=np.int64)))
    padding = threshold * 1e-6 + 1e-12
    arguments = (modes, omega_initial, omega_rows, omega_cols, physical_rows, physical_before, float(threshold),
                 float(padding))
    plans = np.empty((_plan_tiles_kernel(*arguments, np.empty((0, 7), dtype=np.int64)), 7), dtype=np.int64)
    _plan_tiles_kernel(*arguments, plans)
    return plans


@numba.njit(nogil=True, error_model="numpy")
def _plan_tiles_kernel(
    modes: np.ndarray, omega_initial: np.ndarray, omega_rows: np.ndarray, omega_cols: np.ndarray,
    physical_rows: np.ndarray, physical_before: np.ndarray, threshold: float, padding: float, plans: np.ndarray,
) -> int:
    """Count ``plan_tiles`` rows, and write them when ``plans`` has room; return the count.

    Targets are monotone in the sorted rows, so the ``searchsorted`` bounds (left for ``lower``, right for
    ``upper``) are found by pointer walks in the same floating-point order as the vectorized definition.
    """
    n_rows, n_cols = omega_rows.size, omega_cols.size
    lower = np.empty(n_rows, dtype=np.int64)
    upper = np.empty(n_rows, dtype=np.int64)
    write = plans.shape[0] > 0
    count = 0
    for slot in range(modes.size):
        initial = modes[slot]
        for channel in range(2):
            sign = 2 * channel - 1
            low_pointer, high_pointer = 0, 0
            for step in range(n_rows):
                row = step if channel else n_rows - 1 - step  # ascending targets
                target = omega_initial[initial] + sign * omega_rows[row]
                low_value = (target - threshold) - padding
                high_value = (target + threshold) + padding
                while low_pointer < n_cols and omega_cols[low_pointer] < low_value:
                    low_pointer += 1
                while high_pointer < n_cols and omega_cols[high_pointer] <= high_value:
                    high_pointer += 1
                lower[row], upper[row] = low_pointer, high_pointer
            kept, row_start, row_last = 0, -1, -1
            for row in range(n_rows + 1):
                keep = row < n_rows and physical_rows[row]
                keep = keep and physical_before[upper[row]] > physical_before[lower[row]]
                if keep:
                    if kept == 0:
                        row_start = row
                    row_last = row
                    kept += 1
                if kept and (kept == _ROW_BLOCK or row == n_rows):
                    if write:
                        plans[count, 0], plans[count, 1], plans[count, 2] = slot, initial, channel
                        plans[count, 3], plans[count, 4] = row_start, row_last + 1
                        plans[count, 5] = min(lower[row_start], lower[row_last])
                        plans[count, 6] = max(upper[row_start], upper[row_last])
                    count += 1
                    kept = 0
    return count


def reduce_tiles_split(
    tiles: np.ndarray, rows_re: np.ndarray, rows_im: np.ndarray, inter_re: np.ndarray, inter_im: np.ndarray,
    column_offset: int, omega_initial: np.ndarray, omega_rows: np.ndarray, omega_cols: np.ndarray,
    physical_rows: np.ndarray, physical_cols: np.ndarray, pop_rows: np.ndarray, pop_cols: np.ndarray,
    hbar_factors: np.ndarray, threshold: float, sigma: float, scales: np.ndarray,
    rows_sum: np.ndarray | None = None, inter_sum: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce complex energy-window tiles into phase space and linewidth sums.

    ``tiles`` is int64 ``(t,7)``: slot, initial, channel, row_start,
    row_stop, column_start, column_stop (``plan_tiles``, columns trimmed to
    ``[column_offset, column_offset+w)``). Float64 planes ``rows_re/rows_im``
    ``(n,n)`` hold transposed q' eigenvectors (inverse square-root mass) and
    ``inter_re/inter_im`` ``(block,n,w)`` the stage-2 intermediate
    (eV/(Angstrom^3*amu)), possibly a strided view of partner-major planes
    (unit column stride). Per tile, ``|A|^2 = Re^2 + Im^2`` of
    ``A = rows[row] . inter[slot,:,column]`` comes from three GEMMs (Gauss):
    ``T1 = Lr Rr``, ``T2 = Li Ri``, ``T3 = (Lr + Li)(Rr + Ri)``, ``Re = T1 - T2``
    and ``Im = T3 - T1 - T2``. ``rows_sum = rows_re + rows_im`` and
    ``inter_sum = inter_re + inter_im`` may be passed precomputed (same
    layouts); they are formed here otherwise. Precomputed sums are trusted,
    not checked: a sum that differs from ``re + im`` silently corrupts ``Im``.
    Frequencies ``(n,)`` and ``threshold``/``sigma``
    are rad/ps; channel 1 selects ``w + w' - w''`` with ``n' - n''`` and
    channel 0 ``w - w' - w''`` with ``(1 + n' + n'')/2``. Dimensionless
    ``pop_rows/pop_cols`` ``(P,n)`` pair with ``hbar_factors (P,)``;
    ``scales (n,)`` carries the package hbar/frequency conversion of the
    initial mode. Returns float64 ``(P,block,2)`` phase space and THz
    linewidth sums, and int64 ``(block,2)`` selected-pair counts per channel.
    """
    for name, matrix in (("rows_re", rows_re), ("rows_im", rows_im)):
        _validate_matrix(matrix, name)
    n_modes = rows_re.shape[0]
    if rows_re.shape != (n_modes, n_modes) or rows_im.shape != rows_re.shape:
        raise ValueError("rows_re and rows_im must be square with equal shapes")
    for name, inter in (("inter_re", inter_re), ("inter_im", inter_im)):
        if (
            not isinstance(inter, np.ndarray) or inter.ndim != 3 or inter.dtype != np.float64
            or inter.shape[1] != n_modes or inter.shape != inter_re.shape
        ):
            raise TypeError(name + " must have float64 shape (block,n,w) matching rows and the other plane")
        if inter.shape[0] > _BLAS_MAX or not inter.flags.aligned:
            raise ValueError(name + " must be aligned with an LP64 block dimension")
        if inter.shape[0]:
            _validate_matrix(inter[0], name)
    if rows_sum is None:
        rows_sum = rows_re + rows_im
    if inter_sum is None:
        inter_sum = inter_re + inter_im
    _validate_matrix(rows_sum, "rows_sum")
    if rows_sum.shape != rows_re.shape:
        raise ValueError("rows_sum must match rows_re")
    if (
        not isinstance(inter_sum, np.ndarray) or inter_sum.dtype != np.float64 or inter_sum.shape != inter_re.shape
        or not inter_sum.flags.aligned
    ):
        raise TypeError("inter_sum must be float64 with the shape of inter_re")
    if inter_sum.shape[0]:
        _validate_matrix(inter_sum[0], "inter_sum")
    block, width = inter_re.shape[0], inter_re.shape[2]
    for name, vector, dtype in (
        ("omega_initial", omega_initial, np.float64), ("omega_rows", omega_rows, np.float64),
        ("omega_cols", omega_cols, np.float64), ("physical_rows", physical_rows, np.bool_),
        ("physical_cols", physical_cols, np.bool_), ("scales", scales, np.float64),
    ):
        if (
            not isinstance(vector, np.ndarray) or vector.shape != (n_modes,) or vector.dtype != dtype
            or not vector.flags.c_contiguous or not vector.flags.aligned
        ):
            raise TypeError(name + " must be a contiguous vector of the required dtype and length")
    for name, populations in (("pop_rows", pop_rows), ("pop_cols", pop_cols)):
        if (
            not isinstance(populations, np.ndarray) or populations.ndim != 2 or populations.dtype != np.float64
            or populations.shape[1] != n_modes or not populations.shape[0] or not populations.flags.c_contiguous
            or populations.shape != pop_rows.shape
        ):
            raise TypeError(name + " must be a contiguous float64 (P,n) stack matching the other")
    if (
        not isinstance(hbar_factors, np.ndarray) or hbar_factors.dtype != np.float64
        or hbar_factors.shape != pop_rows.shape[:1] or not np.all(np.isfinite(hbar_factors))
    ):
        raise ValueError("hbar_factors must be finite float64 (P,)")
    if not isinstance(tiles, np.ndarray) or tiles.dtype != np.int64 or tiles.ndim != 2 or tiles.shape[1] != 7:
        raise TypeError("tile table must have int64 shape (tiles,7)")
    if not isinstance(column_offset, (int, np.integer)) or column_offset < 0:
        raise ValueError("tile column offset must be a nonnegative integer")
    if tiles.size and (
        np.any(tiles[:, 0] < 0) or np.any(tiles[:, 0] >= block)
        or np.any(tiles[:, 1] < 0) or np.any(tiles[:, 1] >= n_modes)
        or np.any((tiles[:, 2] != 0) & (tiles[:, 2] != 1))
        or np.any(tiles[:, 3] < 0) or np.any(tiles[:, 4] <= tiles[:, 3]) or np.any(tiles[:, 4] > n_modes)
        or np.any(tiles[:, 5] < column_offset) or np.any(tiles[:, 6] <= tiles[:, 5])
        or np.any(tiles[:, 6] > n_modes) or np.any(tiles[:, 6] - column_offset > width)
    ):
        raise ValueError("tile table contains invalid slots, modes, channels or bounds")
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Gaussian sigma must be positive and threshold nonnegative")
    reduced = np.zeros((pop_rows.shape[0], block, 2), dtype=np.float64)
    counts = np.zeros((block, 2), dtype=np.int64)
    if tiles.size and threshold > 0:
        tiles = np.ascontiguousarray(tiles)
        area = int(np.max((tiles[:, 4] - tiles[:, 3]) * (tiles[:, 6] - tiles[:, 5])))
        _reduce_split_kernel(
            tiles, rows_re, rows_im, rows_sum, inter_re, inter_im, inter_sum, int(column_offset), omega_initial,
            omega_rows, omega_cols, physical_rows, physical_cols, pop_rows, pop_cols, hbar_factors, float(threshold),
            float(sigma), scales, np.empty((3, area), dtype=np.float64), reduced, counts,
        )
    return reduced, counts


@numba.njit(nogil=True, error_model="numpy")
def _reduce_split_kernel(
    tiles: np.ndarray, rows_re: np.ndarray, rows_im: np.ndarray, rows_sum: np.ndarray, inter_re: np.ndarray,
    inter_im: np.ndarray, inter_sum: np.ndarray, column_offset: int, omega_initial: np.ndarray,
    omega_rows: np.ndarray, omega_cols: np.ndarray, physical_rows: np.ndarray, physical_cols: np.ndarray,
    pop_rows: np.ndarray, pop_cols: np.ndarray, hbar_factors: np.ndarray, threshold: float, sigma: float,
    scales: np.ndarray, buffers: np.ndarray, reduced: np.ndarray, counts: np.ndarray,
) -> None:
    """Fuse checked ``reduce_tiles_split`` operands; ``buffers`` is ``(3, max_tile_area)`` float64."""
    transpose = np.array([ord("N")], dtype=np.uint8)
    dimensions = np.empty(6, dtype=np.int32)
    assign = np.array([1., 0.])
    normalization = 1 / np.sqrt(np.pi * sigma ** 2)
    sigma_squared = sigma ** 2
    n_populations = pop_rows.shape[0]
    phase_sums = np.empty(n_populations)
    gamma_sums = np.empty(n_populations)
    for tile in tiles:
        slot, initial, channel, row_start, row_stop, column_start, column_stop = tile
        rows, columns = row_stop - row_start, column_stop - column_start
        real_real = buffers[0, :rows * columns].reshape((rows, columns))
        imag_imag = buffers[1, :rows * columns].reshape((rows, columns))
        sum_sum = buffers[2, :rows * columns].reshape((rows, columns))
        lower, upper = column_start - column_offset, column_stop - column_offset
        _gemm_into(rows_re[row_start:row_stop], inter_re[slot, :, lower:upper], real_real, transpose, dimensions,
                   assign)
        _gemm_into(rows_im[row_start:row_stop], inter_im[slot, :, lower:upper], imag_imag, transpose, dimensions,
                   assign)
        _gemm_into(rows_sum[row_start:row_stop], inter_sum[slot, :, lower:upper], sum_sum, transpose, dimensions,
                   assign)
        phase_sums[:] = 0.
        gamma_sums[:] = 0.
        selected = 0
        for row in range(rows):
            row_mode = row_start + row
            if not physical_rows[row_mode]:
                continue
            base = omega_initial[initial] + (2 * channel - 1) * omega_rows[row_mode]
            for column in range(columns):
                column_mode = column_start + column
                difference = base - omega_cols[column_mode]
                if physical_cols[column_mode] and abs(difference) < threshold:
                    phase = normalization * np.exp(-difference * difference / sigma_squared)
                    real = real_real[row, column] - imag_imag[row, column]
                    imag = sum_sum[row, column] - real_real[row, column] - imag_imag[row, column]
                    squared = real * real + imag * imag
                    squared /= omega_rows[row_mode] * omega_cols[column_mode]
                    squared *= scales[initial]
                    for index in range(n_populations):
                        row_population = pop_rows[index, row_mode]
                        column_population = pop_cols[index, column_mode]
                        if channel:
                            population_delta = row_population - column_population
                        else:
                            population_delta = 0.5 * (1 + row_population + column_population)
                        phase_sums[index] += phase * population_delta
                        gamma_sums[index] += ((squared * hbar_factors[index]) * phase) * population_delta
                    selected += 1
        for index in range(n_populations):
            reduced[index, slot, 0] += phase_sums[index]
            reduced[index, slot, 1] += gamma_sums[index]
        counts[slot, channel] += selected

"""Mode-pair kernels and the blocked real Gamma-point QHGK reducer.

The kernels are shared by every QHGK diffusivity and conductivity path,
including ``Conductivity(method='qhgk')``. Angular frequencies ``omega`` and the QHGK ``diffusivity_bandwidth`` are in
rad/ps. Each function takes two mode blocks, so a caller can evaluate a full
``(M, M)`` pair field or a bounded ``(B_i, B_j)`` tile with identical values.
Only modes that are physical and have positive frequency contribute.
"""

from time import perf_counter

import ase.units as units
import numpy as np

from kaldo.controllers.dirac_kernel import gaussian_delta, lorentz_delta, triangular_delta
from kaldo.helpers.units import KB_J_PER_K
from kaldo.observables.eigenvector_source import NumpyEigenvectorSource

# x = _HBAR_OVER_KB * omega / T is dimensionless for omega in rad/ps and T in K.
_HBAR_OVER_KB = units._hbar * 1e12 / KB_J_PER_K
_DEGENERACY_RTOL = 1e-6
_CURVES = {"gauss": gaussian_delta, "triangle": triangular_delta,
           "lorentz": lorentz_delta}


def _active(omega, physical_mode):
    return np.asarray(physical_mode, dtype=bool).reshape(omega.shape) & (omega > 0)


def diffusivity_kernel(omega_i, omega_j, bandwidth_i, bandwidth_j, physical_i, physical_j, curve, *,
                       is_including_antiresonant=False, diffusivity_threshold=None):
    """Return ``pi * delta(omega_i - omega_j) / (4 omega_i omega_j)`` for one mode block.

    The broadening width is ``sigma_ij = 2 (b_i + b_j)``. ``diffusivity_threshold``
    keeps the historical rule: pair ``(i, j)`` is kept when
    ``|omega_i - omega_j| < threshold * 2 pi * b_j``, using the partner
    bandwidth, and it applies only to the resonant term. Pairs with an
    inactive mode or zero width contribute zero.
    """
    wi = np.asarray(omega_i, dtype=np.float64).reshape(-1)
    wj = np.asarray(omega_j, dtype=np.float64).reshape(-1)
    bi = np.broadcast_to(np.asarray(bandwidth_i, dtype=np.float64).reshape(-1), wi.shape)
    bj = np.broadcast_to(np.asarray(bandwidth_j, dtype=np.float64).reshape(-1), wj.shape)
    sigma = 2 * (bi[:, np.newaxis] + bj[np.newaxis, :])
    delta = wi[:, np.newaxis] - wj[np.newaxis, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kernel = np.array(curve(delta, sigma), dtype=np.float64)
        if diffusivity_threshold is not None:
            if not np.isfinite(diffusivity_threshold) or diffusivity_threshold < 0:
                raise ValueError("diffusivity_threshold must be finite and nonnegative")
            kernel[np.abs(delta) >= diffusivity_threshold * 2 * np.pi * bj[np.newaxis, :]] = 0
        if is_including_antiresonant:
            kernel += curve(wi[:, np.newaxis] + wj[np.newaxis, :], sigma)
        kernel *= np.pi / 4
        kernel /= wi[:, np.newaxis]
        kernel /= wj[np.newaxis, :]
    valid = _active(wi, physical_i)[:, np.newaxis] & _active(wj, physical_j)[np.newaxis, :] & (sigma > 0)
    kernel[~(valid & np.isfinite(kernel))] = 0
    return kernel


def _single_mode_heat_capacity(x):
    """Stable dimensionless Bose heat capacity ``x^2 e^x / (e^x - 1)^2``."""
    result = np.zeros_like(x)
    small = x < 1e-4
    result[small] = 1 - x[small] ** 2 / 12 + x[small] ** 4 / 240
    moderate = (x >= 1e-4) & (x < 700)
    z = np.exp(-x[moderate])
    result[moderate] = x[moderate] ** 2 * z / (-np.expm1(-x[moderate])) ** 2
    return result


def generalized_heat_capacity(omega_i, omega_j, physical_i, physical_j, *, temperature, is_classic):
    """Return the QHGK pair heat capacity ``C_ij / k_B`` for one mode block.

    ``C_ij = x_i x_j (n_i - n_j) / (x_j - x_i)`` with ``x = hbar omega / (k_B T)``;
    it equals one in the classical limit. Pairs with
    ``|x_i - x_j| <= 1e-6 max(1, |x_i|, |x_j|)`` use the mean of the two
    single-mode heat capacities, which avoids subtracting nearly equal Bose
    occupations. The result is ``float64``.
    """
    wi = np.asarray(omega_i, dtype=np.float64).reshape(-1)
    wj = np.asarray(omega_j, dtype=np.float64).reshape(-1)
    active_i, active_j = _active(wi, physical_i), _active(wj, physical_j)
    valid = active_i[:, np.newaxis] & active_j[np.newaxis, :]
    heat = np.zeros((wi.size, wj.size), dtype=np.float64)
    if is_classic:
        heat[valid] = 1
        return heat
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("quantum temperature must be finite and positive")
    xi = wi * (_HBAR_OVER_KB / temperature)
    xj = wj * (_HBAR_OVER_KB / temperature)
    ni = np.zeros_like(xi)
    nj = np.zeros_like(xj)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ni[active_i] = 1 / np.expm1(xi[active_i])
        nj[active_j] = 1 / np.expm1(xj[active_j])
    dx = xj[np.newaxis, :] - xi[:, np.newaxis]
    scale = np.maximum(1, np.maximum(np.abs(xi[:, np.newaxis]), np.abs(xj[np.newaxis, :])))
    near = np.abs(dx) <= _DEGENERACY_RTOL * scale
    np.divide(xi[:, np.newaxis] * xj[np.newaxis, :] * (ni[:, np.newaxis] - nj[np.newaxis, :]),
              dx, out=heat, where=valid & ~near)
    degenerate = valid & near
    if np.any(degenerate):
        ci = _single_mode_heat_capacity(np.maximum(xi, np.finfo(float).tiny))
        cj = _single_mode_heat_capacity(np.maximum(xj, np.finfo(float).tiny))
        heat[degenerate] = ((ci[:, np.newaxis] + cj[np.newaxis, :]) / 2)[degenerate]
    return heat


def _vector(value, n_modes, name, dtype):
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = np.full(n_modes, array, dtype=dtype)
    elif array.size == n_modes:
        array = np.ascontiguousarray(array.reshape(n_modes), dtype=dtype)
    else:
        raise ValueError(f"{name} must be scalar or have {n_modes} entries")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _physical(value, n_modes):
    array = np.asarray(value)
    if array.size != n_modes:
        raise ValueError(f"physical_mode must have {n_modes} entries")
    return np.ascontiguousarray(array.reshape(n_modes), dtype=bool)


def qhgk_pair_factors(omega_i, omega_j, bandwidth_i, bandwidth_j,
                      physical_i, physical_j, *, temperature, is_classic,
                      broadening_shape="triangle", diffusivity_threshold=None,
                      dtype=np.float32):
    """Return the resonant diffusivity kernel and ``C_ij/k_B`` for one block in ``dtype``."""
    try:
        curve = _CURVES[broadening_shape]
    except KeyError as exc:
        raise ValueError(f"unknown broadening_shape {broadening_shape!r}") from exc
    kernel = diffusivity_kernel(omega_i, omega_j, bandwidth_i, bandwidth_j, physical_i, physical_j,
                                curve, diffusivity_threshold=diffusivity_threshold)
    heat = generalized_heat_capacity(omega_i, omega_j, physical_i, physical_j,
                                     temperature=temperature, is_classic=is_classic)
    return kernel.astype(dtype, copy=False), heat.astype(dtype, copy=False)


def _inputs(omega, diffusivity_bandwidth, physical_mode, volume, eigenvectors,
            harmonic_operator, compute_dtype, accumulator_dtype):
    work = np.dtype(compute_dtype)
    accum = np.dtype(accumulator_dtype)
    supported = (np.dtype("float32"), np.dtype("float64"))
    if work not in supported or accum not in supported:
        raise ValueError("compute_dtype and accumulator_dtype must be float32 or float64")
    w = np.asarray(omega, dtype=np.float64)
    if w.ndim != 1 or not w.size or not np.all(np.isfinite(w)) or np.any(np.diff(w) < 0):
        raise ValueError("omega must be a finite, nonempty, sorted 1D array")
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("volume must be finite and positive")
    source = eigenvectors if hasattr(eigenvectors, "read_modes") else NumpyEigenvectorSource(eigenvectors)
    if source.n_modes != w.size or source.n_coordinates != w.size:
        raise ValueError("Gamma-point eigenvectors must have shape (M, M)")
    if len(harmonic_operator) != 3:
        raise ValueError("harmonic_operator must contain three derivative operators")
    for operator in harmonic_operator:
        if operator.shape != (w.size, w.size) or not hasattr(operator, "__matmul__"):
            raise ValueError("each derivative operator must support @ panel and have shape (M, M)")
    bw = _vector(diffusivity_bandwidth, w.size, "diffusivity_bandwidth", np.float64)
    if np.any(bw < 0):
        raise ValueError("diffusivity_bandwidth must be nonnegative")
    return w, bw, _physical(physical_mode, w.size), source, work, accum


def _finish(conductivity, diffusivity, volume):
    return conductivity * (KB_J_PER_K * 1e22 / volume), diffusivity / 100


def _work_operators(operators, dtype):
    """Cast operators once, so sparse/panel products stay in the work dtype."""
    # A future matrix-panel operator may implement only shape and @. Such an
    # operator is responsible for its own internal precision; cast its output.
    return tuple(operator.astype(dtype, copy=False) if hasattr(operator, "astype")
                 else operator for operator in operators)


def _read_panel(source, start, stop, dtype, n_coordinates):
    panel = np.asarray(source.read_modes(start, stop, dtype=dtype))
    if panel.shape != (n_coordinates, stop - start) or np.iscomplexobj(panel):
        raise ValueError("eigenvector source returned a non-real or incorrectly shaped panel")
    return panel.astype(dtype, copy=False)


def _triangle_block_outside_support(wi, wj, bi, bj):
    """Conservative block test; equality is outside triangular support."""
    separation = max(0., float(wj[0] - wi[-1]), float(wi[0] - wj[-1]))
    return separation >= 2 * (float(np.max(bi)) + float(np.max(bj)))


def _zero_self_pairs(pair_block, core_start, partner_start):
    """Enforce the zero diagonal of a real skew-derivative projection."""
    lo = max(core_start, partner_start)
    hi = min(core_start + pair_block.shape[0],
             partner_start + pair_block.shape[1])
    if hi > lo:
        modes = np.arange(lo, hi)
        pair_block[modes-core_start, modes-partner_start] = 0


def calculate_qhgk_isotropic_blocked(omega, diffusivity_bandwidth, physical_mode,
                                      temperature, is_classic, volume, eigenvectors,
                                      harmonic_operator, *, core_start=0, core_stop=None,
                                      broadening_shape="triangle", block_size=1024,
                                      diffusivity_threshold=None,
                                      compute_dtype=np.float32,
                                      accumulator_dtype=np.float64, stats=None):
    """Return per-mode isotropic conductivity and diffusivity for modes ``[core_start, core_stop)``.

    Only coordinate-by-``B`` eigenvector panels and ``B×B`` pair scratch live
    during a partner iteration. Even ``block_size >= M`` is capped at ``M-1``
    for ``M>1`` so no global ``M×M`` pair field is formed. Operators may be
    NumPy arrays, SciPy CSR/BSR matrices, or objects supporting ``@ panel``.
    They should use ``compute_dtype`` to avoid promotion during projection.
    """
    w, bw, physical, source, work, accum = _inputs(
        omega, diffusivity_bandwidth, physical_mode, volume,
        eigenvectors, harmonic_operator, compute_dtype, accumulator_dtype)
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    m = w.size
    if core_stop is None:
        core_stop = m
    if not 0 <= core_start <= core_stop <= m:
        raise ValueError("invalid core mode range")
    block = min(block_size, max(1, m - 1))
    operators = _work_operators(harmonic_operator, work)
    conductivity = np.zeros(core_stop - core_start, dtype=accum)
    diffusivity = np.zeros(core_stop - core_start, dtype=accum)
    if stats is not None:
        stats.update(evaluated_pairs=0, skipped_blocks=0, panel_io_seconds=0.,
                     panel_io_bytes=0,
                     operator_seconds=0., contraction_seconds=0.,
                     pair_seconds=0., reduction_seconds=0.)
    # TODO(distributed-qhgk): exploit symmetric unordered-pair ownership so
    # (m, n) is evaluated once and contributes to both mode accumulators.
    for i0 in range(core_start, core_stop, block):
        i1 = min(i0 + block, core_stop)
        t = perf_counter()
        ei = _read_panel(source, i0, i1, work, m)
        if stats is not None:
            stats["panel_io_seconds"] += perf_counter() - t
            stats["panel_io_bytes"] += m * (i1 - i0) * work.itemsize
        local_c = np.zeros(i1 - i0, dtype=accum)
        local_d = np.zeros(i1 - i0, dtype=accum)
        for j0 in range(0, m, block):
            j1 = min(j0 + block, m)
            if broadening_shape == "triangle" and _triangle_block_outside_support(
                    w[i0:i1], w[j0:j1], bw[i0:i1], bw[j0:j1]):
                if stats is not None:
                    stats["skipped_blocks"] += 1
                continue
            t = perf_counter()
            ej = _read_panel(source, j0, j1, work, m)
            if stats is not None:
                stats["panel_io_seconds"] += perf_counter() - t
                stats["panel_io_bytes"] += m * (j1 - j0) * work.itemsize
                stats["evaluated_pairs"] += (i1 - i0) * (j1 - j0)
            t = perf_counter()
            kernel, heat = qhgk_pair_factors(
                w[i0:i1], w[j0:j1], bw[i0:i1], bw[j0:j1],
                physical[i0:i1], physical[j0:j1], temperature=temperature,
                is_classic=is_classic, broadening_shape=broadening_shape,
                diffusivity_threshold=diffusivity_threshold, dtype=work)
            if stats is not None:
                stats["pair_seconds"] += perf_counter() - t
            s2 = np.zeros((i1 - i0, j1 - j0), dtype=work)
            for operator in operators:
                t = perf_counter()
                y = np.asarray(operator @ ej)
                if stats is not None:
                    stats["operator_seconds"] += perf_counter() - t
                if y.shape != (m, j1 - j0):
                    raise ValueError("derivative operator returned an invalid panel shape")
                if y.dtype != work:
                    y = y.astype(work, copy=False)
                t = perf_counter()
                s = ei.T @ y
                if stats is not None:
                    stats["contraction_seconds"] += perf_counter() - t
                np.multiply(s, s, out=s)
                s2 += s
            s2 /= 3
            # Real Gamma skew derivatives have exactly zero self projection.
            _zero_self_pairs(s2, i0, j0)
            t = perf_counter()
            np.multiply(kernel, s2, out=kernel)
            local_d += np.sum(kernel, axis=1, dtype=accum)
            np.multiply(kernel, heat, out=kernel)
            local_c += np.sum(kernel, axis=1, dtype=accum)
            if stats is not None:
                stats["reduction_seconds"] += perf_counter() - t
            # All pair and partner-panel scratch dies at the next iteration.
            del ej, kernel, heat, s2, y, s
        conductivity[i0-core_start:i1-core_start] = local_c
        diffusivity[i0-core_start:i1-core_start] = local_d
    return _finish(conductivity, diffusivity, volume)


def calculate_qhgk_isotropic_dense(omega, diffusivity_bandwidth, physical_mode,
                                    temperature, is_classic, volume, eigenvectors,
                                    harmonic_operator, *, broadening_shape="triangle",
                                    diffusivity_threshold=None,
                                    compute_dtype=np.float64,
                                    accumulator_dtype=np.float64):
    """Small-system NumPy reference; deliberately forms complete pair fields."""
    w, bw, physical, source, work, accum = _inputs(
        omega, diffusivity_bandwidth, physical_mode, volume,
        eigenvectors, harmonic_operator, compute_dtype, accumulator_dtype)
    e = _read_panel(source, 0, w.size, work, w.size)
    operators = _work_operators(harmonic_operator, work)
    kernel, heat = qhgk_pair_factors(
        w, w, bw, bw, physical, physical, temperature=temperature,
        is_classic=is_classic, broadening_shape=broadening_shape,
        diffusivity_threshold=diffusivity_threshold, dtype=work)
    s2 = np.zeros((w.size, w.size), dtype=work)
    for operator in operators:
        s = e.T @ (operator @ e)
        s2 += s * s
    s2 /= 3
    _zero_self_pairs(s2, 0, 0)
    diffusivity = np.sum(kernel * s2, axis=1, dtype=accum)
    conductivity = np.sum(heat * kernel * s2, axis=1, dtype=accum)
    return _finish(conductivity, diffusivity, volume)

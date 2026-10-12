"""Mode-pair kernels and the blocked real Gamma-point QHGK reducer.

The kernels are shared by every QHGK diffusivity and conductivity path,
including ``Conductivity(method='qhgk')``. Angular frequencies ``omega`` and
the QHGK ``diffusivity_bandwidth`` are in rad/ps. Each kernel takes two mode
blocks, so a caller can evaluate a full ``(M, M)`` pair field or a bounded
``(B_i, B_j)`` tile with identical values. Only modes that are physical and
have positive frequency contribute.

The reducer, :func:`reduce_qhgk_blocks`, iterates core-mode blocks against
partner-mode blocks and keeps only ``B x B`` pair scratch. It asks a *flux
projector* for each flux block ``e_i^T (dD/dx_a) e_j``: a
:class:`LocalFluxProjector` holds every coordinate row in one process, and
``kaldo.parallel.distributed_qhgk.DistributedFluxProjector`` holds one MPI
rank's rows. Volume is in Å³, conductivity in W/(m K), diffusivity in mm²/s.
Eigenvector panels are coordinate-major ``(3*N, modes)``.
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
                      is_including_antiresonant=False, dtype=np.float32):
    """Return the diffusivity kernel and ``C_ij/k_B`` for one block in ``dtype``."""
    try:
        curve = _CURVES[broadening_shape]
    except KeyError as exc:
        raise ValueError(f"unknown broadening_shape {broadening_shape!r}") from exc
    kernel = diffusivity_kernel(omega_i, omega_j, bandwidth_i, bandwidth_j, physical_i, physical_j,
                                curve, is_including_antiresonant=is_including_antiresonant,
                                diffusivity_threshold=diffusivity_threshold)
    heat = generalized_heat_capacity(omega_i, omega_j, physical_i, physical_j,
                                     temperature=temperature, is_classic=is_classic)
    return kernel.astype(dtype, copy=False), heat.astype(dtype, copy=False)


def check_dtypes(compute_dtype, accumulator_dtype):
    work, accum = np.dtype(compute_dtype), np.dtype(accumulator_dtype)
    supported = (np.dtype("float32"), np.dtype("float64"))
    if work not in supported or accum not in supported:
        raise ValueError("compute_dtype and accumulator_dtype must be float32 or float64")
    return work, accum


def check_mode_inputs(omega, diffusivity_bandwidth, physical_mode, volume, temperature):
    """Validate sorted rad/ps frequencies, nonnegative bandwidths, and the cell volume."""
    w = np.asarray(omega, dtype=np.float64)
    if w.ndim != 1 or not w.size or not np.all(np.isfinite(w)) or np.any(np.diff(w) < 0):
        raise ValueError("omega must be a finite, nonempty, sorted 1D array")
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("volume must be finite and positive")
    if not np.isfinite(temperature):
        raise ValueError("temperature must be finite")
    bw = _vector(diffusivity_bandwidth, w.size, "diffusivity_bandwidth", np.float64)
    if np.any(bw < 0):
        raise ValueError("diffusivity_bandwidth must be nonnegative")
    return w, bw, _physical(physical_mode, w.size)


def _finish(conductivity, diffusivity, volume):
    return conductivity * (KB_J_PER_K * 1e22 / volume), diffusivity / 100


def _work_operators(operators, dtype):
    """Cast operators once, so sparse/panel products stay in the work dtype."""
    # A future matrix-panel operator may implement only shape and @. Such an
    # operator is responsible for its own internal precision; cast its output.
    return tuple(operator.astype(dtype, copy=False) if hasattr(operator, "astype")
                 else operator for operator in operators)


def read_panel(source, start, stop, dtype):
    """Read a real ``(source.n_coordinates, stop-start)`` eigenvector panel in ``dtype``."""
    panel = np.asarray(source.read_modes(start, stop, dtype=dtype))
    if panel.shape != (source.n_coordinates, stop - start) or np.iscomplexobj(panel):
        raise ValueError("eigenvector source returned a non-real or incorrectly shaped panel")
    return panel.astype(dtype, copy=False)


def _triangle_block_outside_support(wi, wj, bi, bj):
    """Conservative block test; equality is outside triangular support."""
    separation = max(0., float(wj[0] - wi[-1]), float(wi[0] - wj[-1]))
    return separation >= 2 * (float(np.max(bi)) + float(np.max(bj)))


def _zero_self_pairs(pair_block, core_start, partner_start):
    """Enforce the zero diagonal of a real skew-derivative projection (last two axes)."""
    lo = max(core_start, partner_start)
    hi = min(core_start + pair_block.shape[-2],
             partner_start + pair_block.shape[-1])
    if hi > lo:
        modes = np.arange(lo, hi)
        pair_block[..., modes-core_start, modes-partner_start] = 0


class LocalFluxProjector:
    """Flux blocks ``e_i^T (dD/dx_a) e_j`` from eigenvectors and operators held by one process.

    Operators may be NumPy arrays, SciPy CSR/BSR matrices, or objects
    supporting ``@ panel``; they are cast once to ``dtype``.
    """

    rank = 0

    def __init__(self, eigenvectors, harmonic_operator, *, dtype):
        self.source = eigenvectors if hasattr(eigenvectors, "read_modes") else NumpyEigenvectorSource(eigenvectors)
        self.n_modes = self.source.n_modes
        self.dtype = np.dtype(dtype)
        if self.source.n_coordinates != self.n_modes:
            raise ValueError("Gamma-point eigenvectors must have shape (M, M)")
        if len(harmonic_operator) != 3:
            raise ValueError("harmonic_operator must contain three derivative operators")
        for operator in harmonic_operator:
            if operator.shape != (self.n_modes, self.n_modes) or not hasattr(operator, "__matmul__"):
                raise ValueError("each derivative operator must support @ panel and have shape (M, M)")
        self.operators = _work_operators(harmonic_operator, self.dtype)
        self.stats = {"panel_io_seconds": 0., "panel_io_bytes": 0,
                      "operator_seconds": 0., "contraction_seconds": 0.}

    def owner(self, block_index):
        return 0

    def core_panel(self, start, stop):
        t = perf_counter()
        panel = read_panel(self.source, start, stop, self.dtype)
        self.stats["panel_io_seconds"] += perf_counter() - t
        self.stats["panel_io_bytes"] += panel.nbytes
        return panel

    def flux(self, core_panel, start, stop, owner=0):
        """Return the ``(3, core modes, stop-start)`` flux block."""
        partner = self.core_panel(start, stop)
        flux = np.empty((3, core_panel.shape[1], stop - start), dtype=self.dtype)
        for axis, operator in enumerate(self.operators):
            t = perf_counter()
            y = np.asarray(operator @ partner)
            self.stats["operator_seconds"] += perf_counter() - t
            if y.shape != partner.shape:
                raise ValueError("derivative operator returned an invalid panel shape")
            t = perf_counter()
            flux[axis] = core_panel.T @ y.astype(self.dtype, copy=False)
            self.stats["contraction_seconds"] += perf_counter() - t
        return flux


def reduce_qhgk_blocks(projector, omega, diffusivity_bandwidth, physical_mode, *, temperature,
                       is_classic, volume, core_start=0, core_stop=None, block_size=1024,
                       broadening_shape="triangle", diffusivity_threshold=None,
                       is_including_antiresonant=False, is_tensor=False, accumulator_dtype=np.float64,
                       checkpoint=None, max_blocks=None):
    """Reduce core-mode blocks of ``[core_start, core_stop)`` and return this rank's results.

    Every rank must call this with identical arguments: all ranks visit the
    same blocks in the same order, so a distributed projector's collectives
    match. Only a block's owner forms pair factors and accumulates.

    Returns ``(blocks, stats)``. ``blocks`` lists ``(start, stop, conductivity,
    diffusivity, evaluated_pairs)`` for the blocks this rank owns and
    finished, from ``checkpoint`` or computed now; ``conductivity`` has shape
    ``(modes,)`` or, with ``is_tensor``, ``(modes, 3, 3)``. Even
    ``block_size >= M`` is capped at ``M-1`` so no global pair field forms.
    ``max_blocks`` caps blocks newly computed in this call, for restart tests.
    """
    w, bw, physical = omega, diffusivity_bandwidth, physical_mode
    m = projector.n_modes
    if core_stop is None:
        core_stop = m
    if not 0 <= core_start <= core_stop <= m or w.shape != (m,):
        raise ValueError("invalid core mode range")
    if (not isinstance(block_size, (int, np.integer)) or block_size < 1 or
            max_blocks is not None and (not isinstance(max_blocks, (int, np.integer)) or max_blocks < 0)):
        raise ValueError("block_size must be a positive integer and max_blocks nonnegative")
    if broadening_shape not in _CURVES:
        raise ValueError(f"unknown broadening_shape {broadening_shape!r}")
    work, accum = projector.dtype, np.dtype(accumulator_dtype)
    # The triangle has compact support only for the resonant term.
    skip_blocks = broadening_shape == "triangle" and not is_including_antiresonant
    block = min(block_size, max(1, m - 1))
    stats = {"evaluated_pairs": 0, "skipped_blocks": 0, "pair_seconds": 0., "reduction_seconds": 0.}
    blocks = []
    computed = 0
    for index, i0 in enumerate(range(core_start, core_stop, block)):
        i1 = min(i0 + block, core_stop)
        owner = projector.owner(index)
        is_owner = owner == projector.rank
        c_shape = (i1 - i0, 3, 3) if is_tensor else (i1 - i0,)
        if checkpoint is not None:
            saved = checkpoint.load_block(i0, i1, owner=owner,
                                          shapes={"conductivity": c_shape, "diffusivity": (i1 - i0,)})
            if saved is not None:
                if is_owner:
                    blocks.append((i0, i1, saved["conductivity"], saved["diffusivity"],
                                   int(saved["evaluated_pairs"])))
                continue
        if max_blocks is not None and computed >= max_blocks:
            continue
        ei = projector.core_panel(i0, i1)
        if is_owner:
            conductivity = np.zeros(c_shape, dtype=accum)
            diffusivity = np.zeros(i1 - i0, dtype=accum)
            pairs = 0
        for j0 in range(0, m, block):
            j1 = min(j0 + block, m)
            if skip_blocks and _triangle_block_outside_support(w[i0:i1], w[j0:j1], bw[i0:i1], bw[j0:j1]):
                stats["skipped_blocks"] += is_owner
                continue
            flux = projector.flux(ei, j0, j1, owner)
            if not is_owner:
                continue
            t = perf_counter()
            kernel, heat = qhgk_pair_factors(
                w[i0:i1], w[j0:j1], bw[i0:i1], bw[j0:j1], physical[i0:i1], physical[j0:j1],
                temperature=temperature, is_classic=is_classic, broadening_shape=broadening_shape,
                diffusivity_threshold=diffusivity_threshold,
                is_including_antiresonant=is_including_antiresonant, dtype=work)
            stats["pair_seconds"] += perf_counter() - t
            t = perf_counter()
            # Real Gamma skew derivatives have exactly zero self projection.
            _zero_self_pairs(flux, i0, j0)
            s2 = np.einsum("aij,aij->ij", flux, flux) / 3
            diffusivity += np.sum(kernel * s2, axis=1, dtype=accum)
            weight = kernel * heat
            if is_tensor:
                for a in range(3):
                    for b in range(a, 3):
                        conductivity[:, a, b] += np.sum(weight * flux[a] * flux[b], axis=1, dtype=accum)
                        conductivity[:, b, a] = conductivity[:, a, b]
            else:
                conductivity += np.sum(weight * s2, axis=1, dtype=accum)
            stats["reduction_seconds"] += perf_counter() - t
            stats["evaluated_pairs"] += (i1 - i0) * (j1 - j0)
            pairs += (i1 - i0) * (j1 - j0)
            # All pair and partner-panel scratch dies at the next iteration.
            del flux, kernel, heat, s2, weight
        if is_owner:
            conductivity, diffusivity = _finish(conductivity, diffusivity, volume)
            if checkpoint is not None:
                checkpoint.save_block(i0, i1, conductivity=conductivity, diffusivity=diffusivity,
                                      evaluated_pairs=np.int64(pairs))
            blocks.append((i0, i1, conductivity, diffusivity, pairs))
        computed += 1
    return blocks, {**stats, **projector.stats}


def calculate_qhgk_blocked(omega, diffusivity_bandwidth, physical_mode, temperature, is_classic, volume,
                           eigenvectors, harmonic_operator, *, core_start=0, core_stop=None,
                           broadening_shape="triangle", block_size=1024, diffusivity_threshold=None,
                           is_including_antiresonant=False, is_tensor=False, compute_dtype=np.float32,
                           accumulator_dtype=np.float64, stats=None):
    """Return per-mode conductivity and diffusivity for modes ``[core_start, core_stop)``.

    Conductivity is the isotropic trace/3 with shape ``(modes,)``, or the full
    ``(modes, 3, 3)`` tensor with ``is_tensor=True``. ``eigenvectors`` is an
    ``(M, M)`` array or any eigenvector source; ``harmonic_operator`` holds the
    three real derivative operators. Operators should use ``compute_dtype`` to
    avoid promotion during projection. ``stats``, when given, receives pair
    counts and timings.
    """
    work, accum = check_dtypes(compute_dtype, accumulator_dtype)
    w, bw, physical = check_mode_inputs(omega, diffusivity_bandwidth, physical_mode, volume, temperature)
    projector = LocalFluxProjector(eigenvectors, harmonic_operator, dtype=work)
    if projector.n_modes != w.size:
        raise ValueError("Gamma-point eigenvectors must have shape (M, M)")
    core_stop = w.size if core_stop is None else core_stop
    blocks, measured = reduce_qhgk_blocks(
        projector, w, bw, physical, temperature=temperature, is_classic=is_classic, volume=volume,
        core_start=core_start, core_stop=core_stop, block_size=block_size,
        broadening_shape=broadening_shape, diffusivity_threshold=diffusivity_threshold,
        is_including_antiresonant=is_including_antiresonant, is_tensor=is_tensor,
        accumulator_dtype=accum)
    if stats is not None:
        stats.update(measured)
    shape = (core_stop - core_start, 3, 3) if is_tensor else (core_stop - core_start,)
    conductivity = np.zeros(shape, dtype=accum)
    diffusivity = np.zeros(core_stop - core_start, dtype=accum)
    for i0, i1, c, d, _ in blocks:
        conductivity[i0 - core_start:i1 - core_start] = c
        diffusivity[i0 - core_start:i1 - core_start] = d
    return conductivity, diffusivity


def calculate_qhgk_isotropic_dense(omega, diffusivity_bandwidth, physical_mode,
                                    temperature, is_classic, volume, eigenvectors,
                                    harmonic_operator, *, broadening_shape="triangle",
                                    diffusivity_threshold=None,
                                    compute_dtype=np.float64,
                                    accumulator_dtype=np.float64):
    """Small-system NumPy reference; deliberately forms complete pair fields."""
    work, accum = check_dtypes(compute_dtype, accumulator_dtype)
    w, bw, physical = check_mode_inputs(omega, diffusivity_bandwidth, physical_mode, volume, temperature)
    projector = LocalFluxProjector(eigenvectors, harmonic_operator, dtype=work)
    e = projector.core_panel(0, w.size)
    kernel, heat = qhgk_pair_factors(
        w, w, bw, bw, physical, physical, temperature=temperature,
        is_classic=is_classic, broadening_shape=broadening_shape,
        diffusivity_threshold=diffusivity_threshold, dtype=work)
    s2 = np.zeros((w.size, w.size), dtype=work)
    for operator in projector.operators:
        s = e.T @ (operator @ e)
        s2 += s * s
    s2 /= 3
    _zero_self_pairs(s2, 0, 0)
    diffusivity = np.sum(kernel * s2, axis=1, dtype=accum)
    conductivity = np.sum(heat * kernel * s2, axis=1, dtype=accum)
    return _finish(conductivity, diffusivity, volume)

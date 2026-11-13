"""Private three-phonon RTA projection on the half-step shifted 2x2x2 grid.

The shifted points ``S = (+-1/4)^3`` carry the initial and ``q'`` phonons, which are complex. Their sums
``q'' = q + q'`` lie on the Gamma-centred partner grid ``P = {0,1/2}^3`` of time-reversal-invariant points, where
eigenvectors and the Wigner-Seitz ``q''`` leg are real. For a single-cell source the q-folded IFC3 factorizes as
``PhiG[i,j,k] alpha_ij(q') beta_ik(q'')`` (``_shifted_kernels.leg_tables``), so one pass ``(q, q')`` evaluates

    A[nu,nu',nu''] = sum_{i,j,k} PhiG[i,j,k] alpha_ij(q') beta_ik(q'') e_i(q nu) e_j(q' nu') e_k(q'' nu'')

with mass-rescaled eigenvectors, in three stages: a real sparse contraction ``C[(i,j),nu''] = sum_k PhiG beta e``
shared by the passes of one ``q''``; ``I[nu,j,nu''] = sum_i e_i(q nu) alpha_ij(q') C`` in real, imaginary and
summed planes; and energy-window tiles ``A = e(q')^T I`` (three real GEMMs per tile, Gauss). Both channels use the
same tiles: kALDo's decay at ``-q'`` (conjugated ``e`` and phases, ``q'' = q - (-q')``) equals this absorption-form
amplitude, with equal frequency and occupation at ``+-q'``, so summing over all ``q'`` counts each absorption and
each decay exactly once. Only time-reversal representatives of ``q`` are projected; ``Gamma(-q) = Gamma(q)`` fills
the others.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import time

import ase.units as units
import numba
import numpy as np
from psutil import virtual_memory
from threadpoolctl import threadpool_limits

from kaldo.controllers._shifted_kernels import leg_tables, plan_tiles, reduce_tiles_split, stage1, stage2, stage2_plan
from kaldo.controllers.anharmonic import GAMMA_TO_THZ
from kaldo.helpers.logger import get_logger

_logging = get_logger()
_COLUMN_BLOCK = 512
_MODE_BLOCK = 256
_DELTA_THRESHOLD = 2  # Gaussian cutoff in widths, as kALDo's crystal delta
_WORKSPACE_FRACTION = 0.25
_WORKSPACE_MULTIPLIER = 8
_MIN_WORKSPACE_BYTES = 128 * 1024 ** 2
_WEIGHT_ARRAYS = 2  # stage-2 per-worker (pair rows, block) float64 weights
_PLANES = 3  # stage-2 re, im and re + im planes (Gauss three-multiplication tiles)


def project_shifted(
    third, harmonics, grid, mode: str, sigma_thz: float, population: np.ndarray, partner_population: np.ndarray,
    hbar_factor: float | np.ndarray, n_workers: int | None = None, all_initial: bool = False,
) -> np.ndarray:
    """Return three-phonon phase space and linewidth (THz) of every mode on a shifted grid.

    Parameters
    ----------
    third : kaldo.observables.thirdorder.ThirdOrder
        Single-cell IFC3 in eV/Angstrom^3 (``third._gamma_pair_csr()`` and ``leg_tables`` must accept it).
    harmonics : kaldo.controllers._shifted_harmonics.ShiftedHarmonics
        ``frequency (m,n)`` THz sorted per point, complex ``eigenvectors (m,n,n)`` (columns are modes, not
        mass-rescaled), ``physical (m,n)`` and ``representatives``; ``harmonics.partner()`` gives the real
        partner-grid harmonics in ``grid.partner_grid()`` order.
    grid : kaldo.grid.QGrid
        The shifted mesh of ``harmonics``; its ``partner_grid()`` points must be time-reversal invariant.
    mode : str
        IFC interpolation mode, resolved as ``leg_tables`` does.
    sigma_thz : float
        Fixed Gaussian bandwidth in THz (positive); triplets with ``|dw| < 2 * 2 pi sigma`` are kept.
    population, partner_population : np.ndarray
        Dimensionless occupations ``(m,n)`` on the shifted grid and ``(m',n)`` on the partner grid, or stacks
        ``(P,m,n)`` and ``(P,m',n)``. Only physical entries are read.
    hbar_factor : float or np.ndarray
        Package hbar compensation (``CLASSICAL_HBAR_SCALE`` for classical occupations, else 1); ``(P,)`` for
        stacks.
    n_workers : int or None
        Worker threads; None honours CPU affinity. Results do not depend on it beyond summation order.
    all_initial : bool
        Project every initial point instead of copying non-representatives from their time-reversal partner.

    Returns
    -------
    np.ndarray
        float64 ``(m*n, 2)`` (``(P, m*n, 2)`` for stacks) in ``ik*n + nu`` order: phase space divided by ``m``
        and linewidth in THz, both channels summed. Unphysical modes are zero.
    """
    started = time.perf_counter()
    frequency = np.asarray(harmonics.frequency)
    n_points, n_modes = frequency.shape
    partner = harmonics.partner()
    if not grid.is_shifted or grid.size != n_points:
        raise ValueError("shifted projection requires the shifted grid of the harmonics")
    if 3 * len(third.atoms) != n_modes:
        raise ValueError("harmonics do not match the IFC3 atom count")
    partner_points = np.asarray(grid.partner_grid().fractional_points, dtype=np.float64)
    n_partner = len(partner_points)
    if (
        harmonics.eigenvectors.shape != (n_points, n_modes, n_modes)
        or harmonics.eigenvectors.dtype != np.complex128
        or partner.eigenvectors.shape != (n_partner, n_modes, n_modes) or partner.eigenvectors.dtype != np.float64
        or partner.frequency.shape != (n_partner, n_modes)
    ):
        raise ValueError("shifted projection requires complex128 shifted and float64 partner eigenvectors")
    if np.ndim(sigma_thz) != 0 or not np.isfinite(sigma_thz) or sigma_thz <= 0:
        raise ValueError("shifted projection requires a fixed positive Gaussian bandwidth (third_bandwidth)")

    omega = np.ascontiguousarray(2 * np.pi * frequency, dtype=np.float64)
    partner_omega = np.ascontiguousarray(2 * np.pi * np.asarray(partner.frequency), dtype=np.float64)
    physical = np.ascontiguousarray(harmonics.physical, dtype=np.bool_).reshape(n_points, n_modes)
    partner_physical = np.ascontiguousarray(partner.physical, dtype=np.bool_).reshape(n_partner, n_modes)
    for values, mask in ((omega, physical), (partner_omega, partner_physical)):
        if not np.all(np.isfinite(values)) or np.any(values[:, 1:] < values[:, :-1]) or np.any(values[mask] <= 0):
            raise ValueError("shifted projection requires sorted finite frequencies, positive on physical modes")
    populations, partner_populations, hbar_factors, is_stack = _occupations(
        population, partner_population, hbar_factor, physical, partner_physical,
    )
    workers = _worker_count(n_workers)

    representatives = np.asarray(harmonics.representatives, dtype=np.int64)
    initial_points = np.arange(n_points, dtype=np.int64) if all_initial else representatives
    q_prime_of = {}
    for q in initial_points.tolist():
        sums = np.asarray(grid.shifted_partner_ids(q, True), dtype=np.int64)
        if np.unique(sums).size != n_partner or sums.size != n_points:
            raise ValueError("q + q' must map the shifted grid one-to-one onto its partner grid")
        q_prime_of[q] = np.empty(n_partner, dtype=np.int64)
        q_prime_of[q][sums] = np.arange(n_points)  # Ruling R2: invert q' -> q + q'
    passes = [(q, b, int(q_prime_of[q][b])) for b in range(n_partner) for q in initial_points.tolist()]

    reduced = np.zeros((populations.shape[0], n_points, n_modes, 2), dtype=np.float64)
    angular_sigma = 2 * np.pi * float(sigma_thz)
    threshold = _DELTA_THRESHOLD * angular_sigma
    initial_modes = {q: np.flatnonzero(physical[q]) for q in initial_points.tolist()}
    scales = np.zeros((n_points, n_modes), dtype=np.float64)
    scales[physical] = np.pi * units._hbar / 4 * GAMMA_TO_THZ / omega[physical] / n_points
    masses = np.asarray(third.atoms.get_masses(), dtype=np.float64)
    inverse_sqrt_mass = np.repeat(1 / np.sqrt(masses), 3)[:, np.newaxis]

    _logging.info("Shifted projection started: %d passes over %d partner points, workers=%d",
                  len(passes), n_partner, workers)
    layout = third._gamma_pair_csr()
    pair_plan = stage2_plan(layout.j_ptr, layout.pair_i, len(third.atoms))
    pair_atom_i = pair_plan.pair_atom_i
    alpha, beta = leg_tables(third, layout, np.asarray(grid.fractional_points, dtype=np.float64), partner_points,
                             mode)
    with ThreadPoolExecutor(workers) as pool:
        tables = list(pool.map(lambda job: plan_tiles(
            initial_modes[job[0]], omega[job[0]], omega[job[2]], partner_omega[job[1]], physical[job[2]],
            partner_physical[job[1]], threshold,
        ), passes))
    max_row_span = max((int(np.max(table[:, 4] - table[:, 3])) for table in tables if table.size), default=0)
    _logging.info("Shifted projection: legs and %d tiles planned in %.2f s",
                  sum(len(table) for table in tables), time.perf_counter() - started)
    if not max_row_span:
        return _finish(reduced, grid, representatives, all_initial, is_stack)

    fixed_bytes = alpha.nbytes + beta.nbytes + sum(table.nbytes for table in tables) + pair_plan.pair_i.nbytes
    width, block = _block_sizes(
        layout, n_modes, workers, max_row_span, max(1, pair_plan.max_rows), populations.shape[0],
        harmonics.eigenvectors.nbytes + partner.eigenvectors.nbytes, fixed_bytes,
    )
    _logging.info("Shifted projection: columns=%d, modes=%d", width, block)
    contracted_buffer = np.empty(layout.pair_i.size * width, dtype=np.float64)
    planes = [np.empty(block * n_modes * width, dtype=np.float64) for _ in range(_PLANES)]
    timings = dict(stage1=0., stage2=0., stage3=0.)
    selected_pairs = 0

    with ThreadPoolExecutor(workers) as pool, threadpool_limits(1), _numba_threads(workers):
        for b in range(n_partner):
            b_passes = [(index, q, q_prime) for index, (q, pass_b, q_prime) in enumerate(passes) if pass_b == b]
            for column_start in range(0, n_modes, width):
                column_stop = min(n_modes, column_start + width)
                column_tiles = [_clip_columns(tables[index], column_start, column_stop) for index, _, _ in b_passes]
                if not any(tiles.size for tiles in column_tiles):
                    continue
                tick = time.perf_counter()
                contracted = contracted_buffer[:layout.pair_i.size * (column_stop - column_start)].reshape(
                    layout.pair_i.size, column_stop - column_start)
                columns = np.ascontiguousarray(
                    partner.eigenvectors[b][:, column_start:column_stop] * inverse_sqrt_mass, dtype=np.float64)
                for start, _, matrix in layout.chunks:
                    stage1(matrix.indptr, matrix.indices, matrix.data, start, pair_atom_i, beta[b], columns,
                           contracted)
                timings["stage1"] += time.perf_counter() - tick

                for (_, q, q_prime), tiles_q in zip(b_passes, column_tiles):
                    if not tiles_q.size:
                        continue
                    rows = (harmonics.eigenvectors[q_prime] * inverse_sqrt_mass).T
                    rows_re, rows_im = np.ascontiguousarray(rows.real), np.ascontiguousarray(rows.imag)
                    del rows
                    rows_sum = rows_re + rows_im
                    pop_rows = np.ascontiguousarray(populations[:, q_prime])
                    pop_cols = np.ascontiguousarray(partner_populations[:, b])
                    modes_q = initial_modes[q]
                    for slot_start in range(0, modes_q.size, block):
                        in_block = (tiles_q[:, 0] >= slot_start) & (tiles_q[:, 0] < slot_start + block)
                        tiles = tiles_q[in_block]
                        if not tiles.size:
                            continue
                        tiles[:, 0] -= slot_start
                        block_modes = modes_q[slot_start:slot_start + block]
                        trim_start, trim_stop = int(tiles[:, 5].min()), int(tiles[:, 6].max())
                        partner_major = [
                            plane[:block_modes.size * n_modes * (trim_stop - trim_start)].reshape(
                                n_modes, block_modes.size, trim_stop - trim_start) for plane in planes]
                        tick = time.perf_counter()
                        initial = harmonics.eigenvectors[q][:, block_modes] * inverse_sqrt_mass
                        stage2(
                            pair_plan, alpha[q_prime], np.ascontiguousarray(initial.real),
                            np.ascontiguousarray(initial.imag),
                            contracted[:, trim_start - column_start:trim_stop - column_start], *partner_major, pool,
                            workers * 4,
                        )
                        del initial
                        inter_re, inter_im, inter_sum = (plane.transpose(1, 0, 2) for plane in partner_major)
                        timings["stage2"] += time.perf_counter() - tick
                        tick = time.perf_counter()
                        areas = np.cumsum((tiles[:, 4] - tiles[:, 3]) * (tiles[:, 6] - tiles[:, 5]))
                        jobs = min(tiles.shape[0], workers * 4)
                        cuts = np.unique(np.searchsorted(areas, np.linspace(0, areas[-1], jobs + 1)[1:-1]))

                        def reduce_part(part, q=q, q_prime=q_prime, b=b, inter_re=inter_re, inter_im=inter_im,
                                        inter_sum=inter_sum, trim_start=trim_start, rows_re=rows_re, rows_im=rows_im,
                                        rows_sum=rows_sum, pop_rows=pop_rows, pop_cols=pop_cols):
                            """Reduce one group of tiles of this pass and mode block."""
                            return reduce_tiles_split(
                                part, rows_re, rows_im, inter_re, inter_im, trim_start, omega[q], omega[q_prime],
                                partner_omega[b], physical[q_prime], partner_physical[b], pop_rows, pop_cols,
                                hbar_factors, threshold, angular_sigma, scales[q], rows_sum=rows_sum,
                                inter_sum=inter_sum,
                            )

                        for partial, counts in pool.map(reduce_part, np.split(tiles, cuts)):
                            reduced[:, q, block_modes] += partial
                            selected_pairs += int(counts.sum())
                        timings["stage3"] += time.perf_counter() - tick
                _logging.info(
                    "Shifted projection: partner point %d/%d, columns %d-%d of %d done after %.1f s "
                    "(stage1 %.1f s, stage2 %.1f s, tiles %.1f s)", b + 1, n_partner, column_start, column_stop,
                    n_modes, time.perf_counter() - started, timings["stage1"], timings["stage2"], timings["stage3"],
                )
            _logging.info("Shifted projection: partner point %d/%d done after %.1f s", b + 1, n_partner,
                          time.perf_counter() - started)
    _logging.info(
        "Shifted projection finished in %.2f s: stage1 %.2f s, stage2 %.2f s, tiles %.2f s; %d selected pairs",
        time.perf_counter() - started, timings["stage1"], timings["stage2"], timings["stage3"], selected_pairs,
    )
    return _finish(reduced, grid, representatives, all_initial, is_stack)


def _occupations(population, partner_population, hbar_factor, physical, partner_physical):
    """Return float64 ``(P,m,n)`` and ``(P,m',n)`` occupation stacks, ``(P,)`` hbar factors and the stack flag."""
    population = np.asarray(population)
    partner_population = np.asarray(partner_population)
    is_stack = population.ndim == 3
    hbar_factor = np.asarray(hbar_factor)
    if is_stack:
        if (
            population.shape[1:] != physical.shape or not population.shape[0]
            or partner_population.shape != population.shape[:1] + partner_physical.shape
            or hbar_factor.shape != population.shape[:1]
        ):
            raise ValueError("occupation stacks must be (P,m,n) and (P,m',n) with (P,) hbar factors")
    elif (
        population.shape != physical.shape or partner_population.shape != partner_physical.shape
        or hbar_factor.ndim != 0
    ):
        raise ValueError("occupations must be (m,n) and (m',n) with a scalar hbar factor")
    for values in (population, partner_population, hbar_factor):
        if values.dtype.kind not in "fiu":
            raise TypeError("occupations and hbar factors must be real numbers")
    populations = np.ascontiguousarray(population.reshape((-1,) + physical.shape), dtype=np.float64)
    partner_populations = np.ascontiguousarray(
        partner_population.reshape((-1,) + partner_physical.shape), dtype=np.float64)
    hbar_factors = np.ascontiguousarray(hbar_factor.reshape(-1), dtype=np.float64)
    if (
        not np.all(np.isfinite(populations[:, physical])) or not np.all(np.isfinite(hbar_factors))
        or not np.all(np.isfinite(partner_populations[:, partner_physical]))
    ):
        raise ValueError("occupations of physical modes and hbar factors must be finite")
    return populations, partner_populations, hbar_factors, is_stack


def _worker_count(n_workers):
    """Return a positive worker count; None honours CPU affinity."""
    workers = n_workers
    if workers is None:
        workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    if not isinstance(workers, (int, np.integer)) or workers <= 0:
        raise ValueError("shifted projection worker count must be positive")
    return int(workers)


class _numba_threads:
    """Context limiting Numba's parallel kernels (stage 1) to the worker count of this thread."""

    def __init__(self, workers):
        self._workers = min(workers, numba.config.NUMBA_NUM_THREADS)

    def __enter__(self):
        self._previous = numba.get_num_threads()
        numba.set_num_threads(self._workers)

    def __exit__(self, *exc_info):
        numba.set_num_threads(self._previous)


def _block_sizes(layout, n_modes, workers, max_row_span, max_pair_rows, n_populations, input_bytes, fixed_bytes=0):
    """Return column width and initial-mode block whose float64 scratch fits the memory budget.

    Scratch counts the stage-1 output ``(pairs, w)``, the stage-2 re/im/sum planes ``3 (block, n, w)``, one pass's
    re/im/sum ``q'`` rows ``3 (n, n)`` plus their complex ``(n, n)`` temporary, the stage-1 columns and initial
    vectors, and per worker the tile buffers ``3 (max_row_span, w)``, the stage-2 weights and the reducer output.
    ``fixed_bytes`` (leg tables, tile tables) is added to every requirement. The budget is a fraction of available
    memory, capped relative to the IFC3 and eigenvector inputs but never below the 1 x 1 requirement.
    """
    width, block = min(_COLUMN_BLOCK, n_modes), min(_MODE_BLOCK, n_modes)
    pairs = layout.pair_i.size

    def required_bytes(column_count, mode_count):
        entries = (
            pairs * column_count + _PLANES * mode_count * n_modes * column_count + 5 * n_modes * n_modes
            + n_modes * column_count + 4 * n_modes * mode_count
            + workers * (
                _PLANES * max_row_span * column_count + _WEIGHT_ARRAYS * max_pair_rows * mode_count
                + 2 * n_populations * mode_count
            )
        )
        return int(entries * np.dtype(np.float64).itemsize) + int(fixed_bytes)

    input_bytes += sum(
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes for _, _, matrix in layout.chunks
    )
    budget = min(
        int(virtual_memory().available * _WORKSPACE_FRACTION),
        max(_MIN_WORKSPACE_BYTES, _WORKSPACE_MULTIPLIER * input_bytes, required_bytes(1, 1)),
    )
    while required_bytes(width, block) > budget:
        if width > 1:
            width = max(1, width // 2)
        elif block > 1:
            block = max(1, block // 2)
        else:
            raise MemoryError("insufficient available memory for the shifted projection workspace")
    return width, block


def _clip_columns(table, column_start, column_stop):
    """Return int64 ``(t,7)`` tiles overlapping ``[column_start, column_stop)``, clipped to it."""
    tiles = table[(table[:, 5] < column_stop) & (table[:, 6] > column_start)].copy()
    np.maximum(tiles[:, 5], column_start, out=tiles[:, 5])
    np.minimum(tiles[:, 6], column_stop, out=tiles[:, 6])
    return tiles


def _finish(reduced, grid, representatives, all_initial, is_stack):
    """Normalize phase space, copy time-reversal partners and flatten to ``ik*n + nu`` rows."""
    n_populations, n_points, n_modes, _ = reduced.shape
    reduced[..., 0] /= n_points
    if not all_initial:
        for point in sorted(set(range(n_points)) - set(representatives.tolist())):
            reduced[:, point] = reduced[:, grid.partner_id(point)]
    flat = reduced.reshape(n_populations, n_points * n_modes, 2)
    return flat if is_stack else flat[0]

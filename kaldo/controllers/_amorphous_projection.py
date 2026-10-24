"""Private float64 pair-first Gamma projection for scalar Gaussian results.

No permutation symmetry is assumed. Unsupported public options stay on the
compatible projector; this module does not change its sparse output contract.
"""

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
import os

import ase.units as units
import numpy as np
from psutil import virtual_memory
from sparse import COO as _COO
from threadpoolctl import threadpool_limits

from kaldo.controllers.anharmonic import GAMMA_TO_THZ
from kaldo.controllers.dirac_kernel import gaussian_delta
from kaldo.helpers.logger import get_logger
from kaldo.observables.thirdorder import _GammaPairCSR, ThirdOrder as _ThirdOrder

_logging = get_logger()
_COLUMN_BLOCK = 512
_MODE_BLOCK = 256
_ROW_BLOCK = 128
_STAGE1_COLUMNS = 128
_WORKSPACE_FRACTION = 0.25
_WORKSPACE_MULTIPLIER = 8
_MIN_WORKSPACE_BYTES = 128 * 1024 ** 2


def _try_project(
    third: _ThirdOrder, eigenvectors: np.ndarray, frequency: np.ndarray, physical: np.ndarray,
    sigma: float | None, population: np.ndarray, hbar_factor: float, n_workers: int | None,
) -> np.ndarray | None:
    """Return scalar ``(n,2)`` phase space/THz or retain legacy dispatch.

    Source IFC3 is in eV/Angstrom^3. Float64 mass-rescaled eigenvectors
    ``(n,n)`` have inverse square-root atomic-mass units, frequency ``(1,n)``
    and Gaussian width are in THz, and the boolean physical mask has ``n``
    entries. Population ``(n,)`` is dimensionless with the package's hbar
    compensation passed separately. Packing never changes source buffers.
    """
    value = third.value
    if type(value) not in (np.ndarray, _COO) or value.dtype != np.float64:
        return None
    if (
        any(type(array) is not np.ndarray for array in (eigenvectors, frequency, physical, population))
        or eigenvectors.dtype != np.float64 or frequency.dtype != np.float64
        or population.dtype != np.float64 or eigenvectors.ndim != 2
        or eigenvectors.shape[0] != eigenvectors.shape[1]
        or frequency.shape != (1, eigenvectors.shape[0])
        or physical.size != eigenvectors.shape[0] or population.shape != (eigenvectors.shape[0],)
    ):
        return None
    omega = 2 * np.pi * frequency.reshape(-1)
    physical = physical.reshape(-1)
    if (
        not _is_eligible(omega, physical, sigma) or not np.all(np.isfinite(eigenvectors))
        or not np.all(np.isfinite(population[physical]))
    ):
        return None
    _logging.info("Projection started (amorphous pair-first)")
    try:
        layout = third._gamma_pair_csr()
    except (TypeError, ValueError):
        return None
    return _project(layout, eigenvectors, omega, physical, sigma, population, hbar_factor, n_workers)[0]


def _is_eligible(omega: np.ndarray, physical: np.ndarray, sigma: float) -> bool:
    """Check sorted angular frequencies ``(n,)`` in rad/ps and width in THz.

    ``physical`` is the dimensionless boolean ``(n,)`` physical-mode mask.
    """
    if not (
        omega.ndim == physical.ndim == 1 and omega.shape == physical.shape
        and physical.dtype == np.bool_ and np.ndim(sigma) == 0
        and omega.dtype.kind in "fiu" and np.asarray(sigma).dtype.kind in "fiu"
        and np.isfinite(sigma) and sigma >= 0 and np.all(np.isfinite(omega))
        and np.all(omega[1:] >= omega[:-1]) and np.all(omega[physical] > 0)
    ):
        return False
    with np.errstate(over="ignore", invalid="ignore"):
        angular_sigma = 2 * np.pi * np.float64(sigma)
        threshold = 2 * angular_sigma
    return bool(np.isfinite(threshold))


def _block_sizes(layout: _GammaPairCSR, workers: int) -> tuple[int, int]:
    """Budget float64 scratch bytes against free memory and CSR/eigenvectors.

    The input-relative ceiling permits the minimum feasible block sizes.
    """
    n_modes = layout.n_modes
    columns, modes = min(_COLUMN_BLOCK, n_modes), min(_MODE_BLOCK, n_modes)
    max_chunk = max((stop - start for start, stop, _ in layout.chunks), default=0)
    value_bytes = np.dtype(np.float64).itemsize

    def required_bytes(column_count: int, mode_count: int) -> int:
        """Estimate float64 contraction scratch in bytes for block dimensions."""
        entries = (
            2 * n_modes * n_modes + layout.pair_i.size * column_count + mode_count * n_modes * column_count
            + n_modes * (column_count + mode_count)
            + workers * (
                max_chunk * min(column_count, _STAGE1_COLUMNS) + n_modes * (mode_count + 16 * column_count)
            )
        )
        return int(entries * value_bytes)

    input_bytes = n_modes * n_modes * value_bytes + sum(
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes for _, _, matrix in layout.chunks
    )
    budget = min(
        int(virtual_memory().available * _WORKSPACE_FRACTION),
        max(_MIN_WORKSPACE_BYTES, _WORKSPACE_MULTIPLIER * input_bytes, required_bytes(1, 1)),
    )
    while True:
        if required_bytes(columns, modes) <= budget:
            return columns, modes
        if columns > 1:
            columns = max(1, columns // 2)
        elif modes > 1:
            modes = max(1, modes // 2)
        else:
            raise MemoryError("insufficient available memory for pair-first projection workspace")


def _plan_tiles(modes: np.ndarray, omega: np.ndarray, physical: np.ndarray, threshold: float) -> np.ndarray:
    """Plan int64 ``(tiles,7)`` mode/channel/row/column spans in rad/ps.

    Columns conservatively enclose the cutoff. Exact strict selection is
    applied in the reducer, including degeneracies and holes in the mask.
    ``modes`` contains ``(block,)`` initial indices, ``omega`` is ``(n,)``
    in rad/ps, and ``physical`` is a dimensionless boolean ``(n,)`` mask.
    """
    plans = []
    padding = threshold * 1e-6 + 1e-12
    for slot, initial in enumerate(modes):
        for channel in (0, 1):
            targets = omega[initial] + (2 * channel - 1) * omega
            lower = np.searchsorted(omega, targets - threshold - padding, side="left")
            upper = np.searchsorted(omega, targets + threshold + padding, side="right")
            rows = np.flatnonzero((upper > lower) & physical)
            for offset in range(0, rows.size, _ROW_BLOCK):
                row_start = rows[offset]
                row_stop = rows[min(offset + _ROW_BLOCK, rows.size) - 1] + 1
                column_start = min(lower[row_start], lower[row_stop - 1])
                column_stop = max(upper[row_start], upper[row_stop - 1])
                plans.append((slot, initial, channel, row_start, row_stop, column_start, column_stop))
    return np.asarray(plans, dtype=np.int64).reshape(-1, 7)


def _run_tiles_numpy(
    tiles: np.ndarray, eigenvectors_t: np.ndarray, intermediate: np.ndarray, column_offset: int,
    omega: np.ndarray, physical: np.ndarray, population: np.ndarray, hbar_factor: float,
    threshold: float, sigma: float, scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce int64 ``(tiles,7)`` plans into block-local scalar outputs.

    ``eigenvectors_t`` is ``(n,n)`` in inverse square-root mass units and
    ``intermediate`` is ``(block,n,columns)`` in eV/(Angstrom^3*amu) after
    two IFC3 contractions.
    ``omega`` is ``(n,)`` in rad/ps; ``sigma`` and ``threshold`` are rad/ps.
    Dimensionless occupations ``(n,)`` and the package hbar conversion yield
    phase space and linewidth in the existing package units and THz.
    ``scales`` is ``(n,)`` with ``pi*hbar*GAMMA_TO_THZ/(4*omega)`` on
    physical modes and zero elsewhere, using the package unit conversion.
    A ``(populations,n)`` occupation stack with ``(populations,)`` hbar
    factors returns ``(populations,block,2)``; each slice repeats the
    single-occupation arithmetic exactly.
    """
    is_stack = np.ndim(population) == 2
    populations = population if is_stack else population[None]
    hbar_factors = np.reshape(hbar_factor, -1) if is_stack else (hbar_factor,)
    reduced = np.zeros((populations.shape[0], intermediate.shape[0], 2), dtype=np.float64)
    counts = np.zeros((intermediate.shape[0], 2), dtype=np.int64)
    for slot, initial, channel, row_start, row_stop, column_start, column_stop in tiles:
        potential = eigenvectors_t[row_start:row_stop] @ intermediate[
            slot, :, column_start - column_offset:column_stop - column_offset
        ]
        difference = (
            omega[initial] + (2 * channel - 1) * omega[row_start:row_stop, None]
            - omega[None, column_start:column_stop]
        )
        selected = (
            (np.abs(difference) < threshold) & physical[row_start:row_stop, None]
            & physical[None, column_start:column_stop]
        )
        row, column = np.nonzero(selected)
        if not row.size:
            continue
        phase = gaussian_delta(np.abs(difference[row, column]), sigma)
        squared = potential[row, column] ** 2
        row_modes, column_modes = row + row_start, column + column_start
        squared /= omega[row_modes] * omega[column_modes]
        squared *= scales[initial]
        for index, (occupation, factor) in enumerate(zip(populations, hbar_factors)):
            if channel:
                population_delta = occupation[row_modes] - occupation[column_modes]
            else:
                population_delta = 0.5 * (1 + occupation[row_modes] + occupation[column_modes])
            reduced[index, slot, 0] += np.sum(phase * population_delta)
            reduced[index, slot, 1] += np.sum(((squared * factor) * phase) * population_delta)
        counts[slot, channel] += row.size
    return (reduced if is_stack else reduced[0]), counts


def _load_tile_reducer() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Load the process-local native reducer or its genuine NumPy fallback."""
    try:
        from kaldo.controllers._projection_kernels import _run_tiles
    except ImportError:
        return _run_tiles_numpy
    return _run_tiles


def _project(
    layout: _GammaPairCSR, eigenvectors: np.ndarray, omega: np.ndarray, physical: np.ndarray,
    sigma: float, population: np.ndarray, hbar_factor: float | np.ndarray = 1.0, n_workers: int | None = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Project Gamma IFC3 to ``(n,2)`` phase-space/THz and channel counts.

    ``layout`` owns IFC3 in eV/Angstrom^3. Real mass-rescaled eigenvectors
    ``(n,n)`` are float64, angular frequencies ``(n,)`` are rad/ps, and
    ``physical`` is boolean ``(n,)``. ``sigma`` is a Gaussian width in THz.
    ``population`` is the package's dimensionless ``(n,)`` occupation, with
    its classical hbar compensation supplied separately. A ``(populations,n)``
    stack, e.g. several temperatures or statistics, with ``(populations,)``
    hbar factors shares the occupation-independent contractions and returns
    ``(populations,n,2)``; each slice equals the single-occupation result
    bit for bit, and the channel counts do not depend on occupation. A worker count of
    None honors CPU affinity. BLAS thread limits are process-global contexts.
    Concurrent projections can therefore interfere with each other's limits.
    """
    n_modes = layout.n_modes
    if not _is_eligible(omega, physical, sigma):
        raise ValueError("pair-first projection requires sorted finite positive physical frequencies")
    sigma = float(sigma)
    is_stack = np.ndim(population) == 2
    if is_stack:
        hbar_factor = np.asarray(hbar_factor)
        if (
            population.shape[1:] != (n_modes,) or not population.shape[0]
            or hbar_factor.shape != population.shape[:1] or hbar_factor.dtype.kind not in "fiu"
            or not np.all(np.isfinite(hbar_factor))
        ):
            raise ValueError("pair-first population stack requires (populations,n) and matching finite hbar factors")
        hbar_factor = np.ascontiguousarray(hbar_factor, dtype=np.float64)
    elif np.ndim(hbar_factor) != 0:
        raise ValueError("pair-first projection of one population requires a scalar hbar factor")
    if (
        eigenvectors.shape != (n_modes, n_modes) or eigenvectors.dtype != np.float64
        or omega.shape != (n_modes,) or population.shape[-1:] != (n_modes,)
        or not np.all(np.isfinite(eigenvectors)) or not np.all(np.isfinite(population[..., physical]))
    ):
        raise ValueError("pair-first projection requires finite float64 eigenvectors and physical occupations")
    reduced = np.zeros((population.shape[0] if is_stack else 1, n_modes, 2), dtype=np.float64)
    counts = np.zeros((n_modes, 2), dtype=np.int64)

    def result() -> tuple[np.ndarray, np.ndarray]:
        """Return the stack, or the ``(n,2)`` single-occupation contract."""
        return (reduced if is_stack else reduced[0]), counts

    if sigma == 0 or not np.any(physical):
        return result()
    workers = n_workers
    if workers is None:
        workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    if not isinstance(workers, (int, np.integer)) or workers <= 0:
        raise ValueError("pair-first projection worker count must be positive")
    column_block, mode_block = _block_sizes(layout, workers)
    mode_indices = np.flatnonzero(physical)
    blocks = [mode_indices[start:start + mode_block] for start in range(0, mode_indices.size, mode_block)]
    angular_sigma = 2 * np.pi * sigma
    threshold = 2 * angular_sigma
    tables = [_plan_tiles(block, omega, physical, threshold) for block in blocks]
    if not any(table.size for table in tables):
        return result()
    eigenvectors = np.ascontiguousarray(eigenvectors)
    eigenvectors_t = np.ascontiguousarray(eigenvectors.T)
    omega = np.ascontiguousarray(omega, dtype=np.float64)
    population = np.ascontiguousarray(population, dtype=np.float64)
    physical = np.ascontiguousarray(physical)
    tile_reducer = _load_tile_reducer()
    scales = np.zeros(n_modes, dtype=np.float64)
    scales[physical] = np.pi * units._hbar / 4 * GAMMA_TO_THZ / omega[physical]
    contracted_buffer = np.empty(layout.pair_i.size * column_block, dtype=np.float64)
    intermediate_buffer = np.empty(mode_block * n_modes * column_block, dtype=np.float64)
    edges = np.linspace(0, n_modes, min(n_modes, workers * 4) + 1, dtype=int)
    backend = "NumPy" if tile_reducer is _run_tiles_numpy else "Numba"
    _logging.info(
        "Amorphous pair-first %s projection: workers=%d, columns=%d, modes=%d",
        backend, workers, column_block, mode_block,
    )
    with ThreadPoolExecutor(workers) as pool, threadpool_limits(1):
        for column_start in range(0, n_modes, column_block):
            column_stop = min(n_modes, column_start + column_block)
            width = column_stop - column_start
            contracted = contracted_buffer[:layout.pair_i.size * width].reshape(-1, width)
            column_slices = [
                (offset, np.ascontiguousarray(eigenvectors[
                    :, column_start + offset:min(column_stop, column_start + offset + _STAGE1_COLUMNS)
                ]))
                for offset in range(0, width, _STAGE1_COLUMNS)
            ]

            def contract_chunk(chunk: tuple[int, int, object]) -> None:
                """Contract owned CSR pair rows with dimensioned mode columns."""
                start, stop, matrix = chunk
                for offset, columns in column_slices:
                    contracted[start:stop, offset:offset + columns.shape[1]] = matrix @ columns

            list(pool.map(contract_chunk, layout.chunks))
            for block, table in zip(blocks, tables):
                selected = (table[:, 5] < column_stop) & (table[:, 6] > column_start)
                tiles = table[selected].copy()
                if not tiles.size:
                    continue
                np.maximum(tiles[:, 5], column_start, out=tiles[:, 5])
                np.minimum(tiles[:, 6], column_stop, out=tiles[:, 6])
                trim_start, trim_stop = int(tiles[:, 5].min()), int(tiles[:, 6].max())
                trim_width = trim_stop - trim_start
                intermediate = intermediate_buffer[:block.size * n_modes * trim_width].reshape(
                    block.size, n_modes, trim_width
                )
                initial_vectors = np.ascontiguousarray(eigenvectors[:, block])

                def contract_partners(bounds: tuple[int, int]) -> None:
                    """Contract central indices into ``(block,j,columns)``."""
                    for partner_j in range(*bounds):
                        start, stop = layout.j_ptr[partner_j:partner_j + 2]
                        np.matmul(
                            initial_vectors[layout.pair_i[start:stop]].T,
                            contracted[start:stop, trim_start - column_start:trim_stop - column_start],
                            out=intermediate[:, partner_j, :],
                        )

                list(pool.map(contract_partners, zip(edges[:-1], edges[1:])))
                areas = np.cumsum((tiles[:, 4] - tiles[:, 3]) * (tiles[:, 6] - tiles[:, 5]))
                jobs = min(tiles.shape[0], workers * 4)
                cuts = np.unique(np.searchsorted(areas, np.linspace(0, areas[-1], jobs + 1)[1:-1]))

                def reduce_tiles(part: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    """Reduce one ordered group of int64 ``(tiles,7)`` plans."""
                    return tile_reducer(
                        part, eigenvectors_t, intermediate, trim_start, omega, physical, population,
                        hbar_factor, threshold, angular_sigma, scales,
                    )

                for partial, partial_counts in pool.map(reduce_tiles, np.split(tiles, cuts)):
                    reduced[:, block] += partial if is_stack else partial[None]
                    counts[block] += partial_counts
            _logging.info("Amorphous pair-first projection: columns %d/%d", column_stop, n_modes)
    return result()

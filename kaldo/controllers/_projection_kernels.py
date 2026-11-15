"""Private checked float64 BLAS bridge and fused Gaussian tile reduction."""

import ctypes

import numba
from numba.extending import get_cython_function_address as _get_cython_function_address
import numpy as np
from scipy.linalg import cython_blas as _cython_blas

_BLAS_MAX = np.iinfo(np.int32).max


def _load_dgemm() -> ctypes._CFuncPtr:
    """Resolve a process-local SciPy LP64 float64 GEMM function pointer."""
    try:
        capsule_name = ctypes.pythonapi.PyCapsule_GetName
        capsule_name.argtypes = [ctypes.py_object]
        capsule_name.restype = ctypes.c_char_p
        signature = capsule_name(_cython_blas.__pyx_capi__["dgemm"]).decode().replace(" ", "")
        scalar = "__pyx_t_5scipy_6linalg_11cython_blas_d*"
        arguments = ["char*", "char*", "int*", "int*", "int*", scalar, scalar,
                     "int*", scalar, "int*", scalar, scalar, "int*"]
        expected = "void(" + ",".join(arguments) + ")"
        if signature not in (expected, expected.replace(scalar, "double*")):
            raise ValueError("unsupported SciPy dgemm capsule signature")
        address = _get_cython_function_address("scipy.linalg.cython_blas", "dgemm")
        if not address:
            raise ValueError("null SciPy dgemm address")
        return ctypes.CFUNCTYPE(None, *([ctypes.c_void_p] * 13))(address)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ImportError("SciPy float64 LP64 BLAS bridge is unavailable") from error


_dgemm = _load_dgemm()


def _validate_matrix(matrix: np.ndarray, name: str) -> None:
    """Validate a float64 ``(rows,columns)`` matrix before native access."""
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.dtype != np.float64:
        raise TypeError(name + " must be a two-dimensional float64 array")
    if any(dimension > _BLAS_MAX for dimension in matrix.shape):
        raise ValueError(name + " dimensions exceed LP64 BLAS bounds")
    if matrix.size and (
        not matrix.flags.aligned or matrix.strides[1] != matrix.itemsize
        or matrix.strides[0] % matrix.itemsize
        or not matrix.shape[1] <= matrix.strides[0] // matrix.itemsize <= _BLAS_MAX
    ):
        raise ValueError(name + " requires unit column stride and a positive LP64 leading dimension")


@numba.njit(nogil=True)
def _gemm_into(
    left: np.ndarray, right: np.ndarray, output: np.ndarray, transpose: np.ndarray,
    dimensions: np.ndarray, scalars: np.ndarray,
) -> None:
    """Multiply checked ``left(rows,inner)`` and ``right(inner,columns)``.

    ``output(rows,columns)`` has product units. Dimensionless scratch arrays
    are ``transpose(1)`` uint8, ``dimensions(6)`` int32 and ``scalars(2)`` float64.
    """
    dimensions[0] = right.shape[1]
    dimensions[1] = left.shape[0]
    dimensions[2] = left.shape[1]
    dimensions[3] = right.strides[0] // right.itemsize
    dimensions[4] = left.strides[0] // left.itemsize
    dimensions[5] = output.strides[0] // output.itemsize
    _dgemm(
        transpose.ctypes, transpose.ctypes, dimensions[0:].ctypes, dimensions[1:].ctypes,
        dimensions[2:].ctypes, scalars[0:].ctypes, right.ctypes, dimensions[3:].ctypes,
        left.ctypes, dimensions[4:].ctypes, scalars[1:].ctypes, output.ctypes, dimensions[5:].ctypes,
    )


@numba.njit(nogil=True, error_model="numpy")
def _reduce_tiles(
    tiles: np.ndarray, eigenvectors_t: np.ndarray, intermediate: np.ndarray, column_offset: int,
    omega: np.ndarray, physical: np.ndarray, populations: np.ndarray, hbar_factors: np.ndarray,
    threshold: float, sigma: float, scales: np.ndarray, symmetric: bool, partner_offset: int,
    potential_buffer: np.ndarray, reduced: np.ndarray, counts: np.ndarray,
    partner: np.ndarray, partner_counts: np.ndarray,
) -> None:
    """Fuse operands with the shapes and units checked by ``_run_tiles``.

    ``potential_buffer(max_tile_area,)`` owns float64 eV/(Angstrom^3*amu^1.5)
    contractions. Dimensionless ``populations(populations,n)`` pair with
    ``hbar_factors(populations,)``; ``reduced(populations,block,2)`` and
    ``counts(block,2)`` are owned outputs. Each occupation keeps the
    sequential per-tile sums of the single-occupation reduction. With
    ``symmetric`` IFC3, absorption pairs whose row mode exceeds the initial
    mode are also credited to ``partner(populations,span,2)`` and
    ``partner_counts(span,)`` at row ``mode - partner_offset``.
    """
    transpose = np.array([ord("N")], dtype=np.uint8)
    dimensions = np.empty(6, dtype=np.int32)
    scalars = np.array([1., 0.])
    normalization = 1 / np.sqrt(np.pi * sigma ** 2)
    sigma_squared = sigma ** 2
    n_populations = populations.shape[0]
    phase_sums = np.empty(n_populations)
    gamma_sums = np.empty(n_populations)
    for tile in tiles:
        slot, initial, channel, row_start, row_stop, column_start, column_stop = tile
        rows, columns = row_stop - row_start, column_stop - column_start
        potential = potential_buffer[:rows * columns].reshape((rows, columns))
        _gemm_into(
            eigenvectors_t[row_start:row_stop],
            intermediate[slot, :, column_start - column_offset:column_stop - column_offset],
            potential, transpose, dimensions, scalars,
        )
        phase_sums[:] = 0.
        gamma_sums[:] = 0.
        selected = 0
        for row in range(rows):
            row_mode = row_start + row
            if not physical[row_mode]:
                continue
            base = omega[initial] + (2 * channel - 1) * omega[row_mode]
            for column in range(columns):
                column_mode = column_start + column
                difference = base - omega[column_mode]
                if physical[column_mode] and abs(difference) < threshold:
                    phase = normalization * np.exp(-difference * difference / sigma_squared)
                    squared = potential[row, column] ** 2
                    squared /= omega[row_mode] * omega[column_mode]
                    squared *= scales[initial]
                    is_partner = symmetric and channel == 1 and row_mode > initial
                    for index in range(n_populations):
                        row_population = populations[index, row_mode]
                        column_population = populations[index, column_mode]
                        if channel:
                            population_delta = row_population - column_population
                        else:
                            population_delta = 0.5 * (1 + row_population + column_population)
                        weight = (squared * hbar_factors[index]) * phase
                        phase_sums[index] += phase * population_delta
                        gamma_sums[index] += weight * population_delta
                        if is_partner:
                            partner_delta = populations[index, initial] - column_population
                            partner[index, row_mode - partner_offset, 0] += phase * partner_delta
                            partner[index, row_mode - partner_offset, 1] += weight * partner_delta
                    if is_partner:
                        partner_counts[row_mode - partner_offset] += 1
                    selected += 1
        for index in range(n_populations):
            reduced[index, slot, 0] += phase_sums[index]
            reduced[index, slot, 1] += gamma_sums[index]
        counts[slot, channel] += selected


def _run_tiles(
    tiles: np.ndarray, eigenvectors_t: np.ndarray, intermediate: np.ndarray, column_offset: int,
    omega: np.ndarray, physical: np.ndarray, population: np.ndarray, hbar_factor: float,
    threshold: float, sigma: float, scales: np.ndarray, symmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[int, np.ndarray, np.ndarray]]:
    """Check one job before fused native projection, with no shared writes.

    Int64 ``tiles(tiles,7)`` hold slot, mode, channel, row and column bounds.
    Float64 ``eigenvectors_t(n,n)`` has inverse square-root mass units and
    ``intermediate(block,n,columns)`` has eV/(Angstrom^3*amu) units.
    ``omega(n,)``, ``threshold`` and ``sigma`` are in rad/ps; boolean
    ``physical(n,)`` and float64 ``population(n,)`` are dimensionless.
    ``scales(n,)`` contains the package hbar/angular-frequency conversion;
    ``hbar_factor`` compensates the classical occupation convention.
    Return ``(block,2)`` phase-space/THz and int64 channel counts. A
    contiguous ``(populations,n)`` occupation stack with finite float64
    ``hbar_factor(populations,)`` returns ``(populations,block,2)``.
    Boolean ``symmetric`` asserts i<->j-symmetric IFC3; the third result
    ``(offset, partner, partner_counts)`` then credits absorption pairs to
    their row modes, as in ``_run_tiles_numpy``.
    """
    _validate_matrix(eigenvectors_t, "eigenvectors_t")
    n_modes = eigenvectors_t.shape[0]
    if eigenvectors_t.shape != (n_modes, n_modes):
        raise ValueError("eigenvectors_t must be square")
    if (
        not isinstance(intermediate, np.ndarray) or intermediate.ndim != 3
        or intermediate.dtype != np.float64 or intermediate.shape[1] != n_modes
    ):
        raise TypeError("intermediate must have float64 shape (block,n,columns)")
    if intermediate.shape[0] > _BLAS_MAX:
        raise ValueError("intermediate block dimension exceeds LP64 bounds")
    if not intermediate.flags.aligned:
        raise ValueError("intermediate must be aligned across all block slots")
    if intermediate.shape[0]:
        _validate_matrix(intermediate[0], "intermediate")
    for name, vector, dtype in (
        ("omega", omega, np.float64), ("physical", physical, np.bool_),
        ("scales", scales, np.float64),
    ):
        if (
            not isinstance(vector, np.ndarray) or vector.shape != (n_modes,)
            or vector.dtype != dtype or not vector.flags.c_contiguous or not vector.flags.aligned
        ):
            raise TypeError(name + " must be a contiguous vector of the required dtype and length")
    if (
        not isinstance(tiles, np.ndarray) or tiles.dtype != np.int64
        or tiles.ndim != 2 or tiles.shape[1] != 7
    ):
        raise TypeError("tile table must have int64 shape (tiles,7)")
    if not tiles.flags.aligned:
        raise ValueError("tile table must be aligned for compiled access")
    if not isinstance(column_offset, (int, np.integer)) or column_offset < 0:
        raise ValueError("tile column offset must be a nonnegative integer")
    if tiles.size and (
        np.any(tiles[:, 0] < 0) or np.any(tiles[:, 0] >= intermediate.shape[0])
        or np.any(tiles[:, 1] < 0) or np.any(tiles[:, 1] >= n_modes)
        or np.any((tiles[:, 2] != 0) & (tiles[:, 2] != 1))
        or np.any(tiles[:, 3] < 0) or np.any(tiles[:, 4] <= tiles[:, 3])
        or np.any(tiles[:, 4] > n_modes) or np.any(tiles[:, 5] < column_offset)
        or np.any(tiles[:, 6] <= tiles[:, 5]) or np.any(tiles[:, 6] > n_modes)
        or np.any(tiles[:, 6] - column_offset > intermediate.shape[2])
    ):
        raise ValueError("tile table contains invalid slots, modes, channels or bounds")
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("native Gaussian sigma must be positive and threshold nonnegative")
    if not isinstance(symmetric, (bool, np.bool_)):
        raise TypeError("symmetric must be a boolean")
    is_stack = isinstance(population, np.ndarray) and population.ndim == 2
    if (
        not isinstance(population, np.ndarray) or population.shape[-1:] != (n_modes,)
        or population.ndim not in (1, 2) or population.dtype != np.float64
        or not population.flags.c_contiguous or not population.flags.aligned
    ):
        raise TypeError("population must be a contiguous float64 (n,) vector or (populations,n) stack")
    if is_stack:
        if (
            not population.shape[0] or not isinstance(hbar_factor, np.ndarray)
            or hbar_factor.shape != population.shape[:1] or hbar_factor.dtype != np.float64
            or not hbar_factor.flags.c_contiguous or not np.all(np.isfinite(hbar_factor))
        ):
            raise ValueError("a population stack requires matching finite float64 hbar factors")
        populations, hbar_factors = population, hbar_factor
    else:
        populations, hbar_factors = population[None], np.array([hbar_factor], dtype=np.float64)
    reduced = np.zeros((populations.shape[0], intermediate.shape[0], 2), dtype=np.float64)
    counts = np.zeros((intermediate.shape[0], 2), dtype=np.int64)
    absorption = tiles[tiles[:, 2] == 1] if symmetric else tiles[:0]
    offset = int(absorption[:, 3].min()) if absorption.size else 0
    span = int(absorption[:, 4].max()) - offset if absorption.size else 0
    partner = np.zeros((populations.shape[0], span, 2), dtype=np.float64)
    partner_counts = np.zeros(span, dtype=np.int64)
    if tiles.size and threshold != 0:
        area = int(np.max((tiles[:, 4] - tiles[:, 3]) * (tiles[:, 6] - tiles[:, 5])))
        _reduce_tiles(
            np.ascontiguousarray(tiles), eigenvectors_t, intermediate, column_offset, omega, physical,
            populations, hbar_factors, threshold, sigma, scales, bool(symmetric), offset,
            np.empty(area, dtype=np.float64), reduced, counts, partner, partner_counts,
        )
    return (reduced if is_stack else reduced[0]), counts, (offset, partner if is_stack else partner[0], partner_counts)

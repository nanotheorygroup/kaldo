"""Checked native float64 GEMM and fused projection reducer contracts."""

import importlib
import ctypes
import os
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest


def _kernels():
    return importlib.import_module("kaldo.controllers._projection_kernels")


def _job() -> tuple:
    """Build dimensionless operands and angular frequencies in rad/ps."""
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    random = np.random.default_rng(605)
    omega = np.array([0., 1., 1., 2., 3., 4., 6.])
    physical = np.array([False, True, True, False, True, True, True])
    modes = np.array([1, 4, 6])
    tiles = projection._plan_tiles(modes, omega, physical, 2.)
    eigenvectors_t = random.normal(size=(14, 7))[::2]
    intermediate = random.normal(size=(3, 14, 7))[:, ::2]
    return (
        tiles, eigenvectors_t, intermediate, 0, omega, physical,
        np.linspace(2., .1, 7), 1., 2., 1., np.linspace(.4, 1.2, 7),
    )


def test_native_projection_kernel_exists():
    assert importlib.util.find_spec("kaldo.controllers._projection_kernels") is not None


@pytest.mark.parametrize("case", ["float32", "columns", "negative", "leading", "shape", "overflow"])
def test_native_reducer_rejects_unsafe_operands_before_blas(case):
    arguments = list(_job())
    eigenvectors_t = np.ones((7, 7))
    if case == "float32":
        eigenvectors_t = eigenvectors_t.astype(np.float32)
    elif case == "columns":
        eigenvectors_t = np.ones((7, 14))[:, ::2]
    elif case == "negative":
        eigenvectors_t = eigenvectors_t[::-1]
    elif case == "leading":
        eigenvectors_t = np.lib.stride_tricks.as_strided(eigenvectors_t, shape=(7, 7), strides=(8, 8))
    elif case == "shape":
        eigenvectors_t = np.ones((7, 6))
    else:
        eigenvectors_t = np.lib.stride_tricks.as_strided(
            eigenvectors_t, shape=(7, 7), strides=(2 ** 31 * 8, 8),
        )
    arguments[1] = eigenvectors_t
    with pytest.raises((ValueError, TypeError), match="float64|stride|dimension|square"):
        _kernels()._run_tiles(*arguments)


@pytest.mark.parametrize("case", ["intermediate", "tiles"])
def test_native_reducer_rejects_unaligned_arrays_before_compiled_access(monkeypatch, case):
    kernels = _kernels()
    arguments = list(_job())
    if case == "intermediate":
        intermediate = np.ndarray(
            (3, 7, 7), dtype=np.float64, buffer=np.zeros(800, np.uint8), strides=(1, 56, 8),
        )
        assert intermediate[0].flags.aligned
        assert not intermediate.flags.aligned
        arguments[2] = intermediate
    else:
        tiles = arguments[0]
        unaligned = np.ndarray(tiles.shape, dtype=np.int64, buffer=np.empty(tiles.nbytes + 1, np.uint8), offset=1)
        unaligned[:] = tiles
        assert unaligned.flags.c_contiguous and not unaligned.flags.aligned
        arguments[0] = unaligned

    native_spy = Mock(side_effect=AssertionError("unaligned array reached the compiled kernel"))
    monkeypatch.setattr(kernels, "_reduce_tiles", native_spy)
    with pytest.raises(ValueError, match="aligned"):
        kernels._run_tiles(*arguments)
    native_spy.assert_not_called()


@pytest.mark.parametrize("hbar_factor", [1., 1e-6])
def test_native_reducer_matches_numpy_with_mask_gaps_and_repeated_jobs(hbar_factor):
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    arguments = list(_job())
    arguments[7] = hbar_factor
    expected, expected_counts = projection._run_tiles_numpy(*arguments)
    for _ in range(2):
        actual, counts = _kernels()._run_tiles(*arguments)
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("case", ["width", "frequencies"])
def test_native_reducer_preserves_numpy_ieee_semantics_on_underflow(case):
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    arguments = list(_job())
    if case == "width":
        arguments[9] = 1e-200
    else:
        arguments[4] = arguments[4] * 1e-200
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        expected, expected_counts = projection._run_tiles_numpy(*arguments)
        actual, counts = _kernels()._run_tiles(*arguments)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13, equal_nan=True)


@pytest.mark.parametrize("case", ["dtype", "shape", "slot", "mode", "channel", "row", "column"])
def test_native_reducer_rejects_invalid_tiles_before_blas(case):
    arguments = list(_job())
    tiles = arguments[0].copy()
    if case == "dtype":
        tiles = tiles.astype(np.int32)
    elif case == "shape":
        tiles = tiles[:, :6]
    else:
        column, invalid = {
            "slot": (0, 3), "mode": (1, 7), "channel": (2, 2),
            "row": (3, -1), "column": (6, 8),
        }[case]
        tiles[0, column] = invalid
    arguments[0] = tiles
    with pytest.raises((TypeError, ValueError), match="tile"):
        _kernels()._run_tiles(*arguments)


def test_native_reducer_rejects_unsafe_intermediate_and_vectors():
    for argument, replacement in (
        (2, np.ones((3, 7, 14))[:, :, ::2]),
        (4, np.arange(7., dtype=np.float32)),
        (5, np.ones(7, dtype=np.int64)),
        (6, np.ones(6)),
    ):
        arguments = list(_job())
        arguments[argument] = replacement
        with pytest.raises((TypeError, ValueError)):
            _kernels()._run_tiles(*arguments)


def test_native_empty_plan_returns_owned_zeros():
    arguments = list(_job())
    arguments[0] = np.empty((0, 7), dtype=np.int64)
    actual, counts = _kernels()._run_tiles(*arguments)
    np.testing.assert_array_equal(actual, np.zeros((3, 2)))
    np.testing.assert_array_equal(counts, np.zeros((3, 2), dtype=np.int64))


@pytest.mark.parametrize("case", ["invalid_capsule", "ilp64", "address_lookup"])
def test_native_blas_initialization_rejects_unknown_capsule(monkeypatch, case):
    kernels = _kernels()
    if case == "address_lookup":
        def unavailable(*arguments):
            raise ValueError("missing Cython address")
        monkeypatch.setattr(kernels, "_get_cython_function_address", unavailable)
    else:
        capsule = object()
        if case == "ilp64":
            signature = b"void (char *, char *, long *, long *, long *, double *)"
            new_capsule = ctypes.pythonapi.PyCapsule_New
            new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
            new_capsule.restype = ctypes.py_object
            capsule = new_capsule(1, signature, None)
        monkeypatch.setitem(kernels._cython_blas.__pyx_capi__, "dgemm", capsule)
    with pytest.raises(ImportError, match="BLAS"):
        kernels._load_dgemm()


@pytest.mark.parametrize("blocked", [False, True])
def test_projection_backend_initializes_in_fresh_process_and_falls_back(blocked):
    script = '''
import sys
import numpy as np
from scipy.sparse import csr_matrix
from types import SimpleNamespace
from kaldo.controllers import _amorphous_projection as projection
assert "kaldo.controllers._projection_kernels" not in sys.modules
blocked = sys.argv[1] == "True"
if blocked:
    sys.modules["kaldo.controllers._projection_kernels"] = None
reducer = projection._load_tile_reducer()
assert reducer.__name__ == ("_run_tiles_numpy" if blocked else "_run_tiles")
layout = SimpleNamespace(n_modes=3, pair_i=np.array([0], np.int32),
    j_ptr=np.array([0, 1, 1, 1], np.int64),
    chunks=((0, 1, csr_matrix([[1., 0., 0.]])),))
arguments = (layout, np.eye(3), np.arange(1., 4.), np.ones(3, bool), .1, np.ones(3))
actual, counts = projection._project(*arguments)
projection._load_tile_reducer = lambda: projection._run_tiles_numpy
expected, expected_counts = projection._project(*arguments)
np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
np.testing.assert_array_equal(counts, expected_counts)
assert np.any(counts)
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(blocked)], capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _population_stack() -> tuple[np.ndarray, np.ndarray]:
    """Dimensionless ``(populations,n)`` occupations and their hbar factors."""
    populations = np.stack((np.linspace(2., .1, 7), np.linspace(.3, 1.7, 7), np.full(7, .9)))
    return populations, np.array([1., 1e-6, .5])


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_reducer_population_stack_equals_separate_jobs(backend):
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    reducer = projection._run_tiles_numpy if backend == "numpy" else _kernels()._run_tiles
    arguments = list(_job())
    populations, hbar_factors = _population_stack()
    separate = [
        reducer(*arguments[:6], population, hbar_factor, *arguments[8:])
        for population, hbar_factor in zip(populations, hbar_factors)
    ]
    actual, counts = reducer(*arguments[:6], populations, hbar_factors, *arguments[8:])
    assert actual.shape == (3, 3, 2) and counts.shape == (3, 2)
    for stacked, (expected, expected_counts) in zip(actual, separate):
        np.testing.assert_array_equal(stacked, expected)
        np.testing.assert_array_equal(counts, expected_counts)


@pytest.mark.parametrize("case", ["hbar_length", "hbar_scalar", "hbar_nonfinite", "empty", "rank", "length"])
def test_native_reducer_rejects_invalid_population_stacks(case):
    arguments = list(_job())
    populations, hbar_factors = _population_stack()
    if case == "hbar_length":
        hbar_factors = hbar_factors[:2]
    elif case == "hbar_scalar":
        hbar_factors = 1.
    elif case == "hbar_nonfinite":
        hbar_factors[1] = np.nan
    elif case == "empty":
        populations, hbar_factors = populations[:0], hbar_factors[:0]
    elif case == "rank":
        populations = populations[None]
    else:
        populations = populations[:, :6]
    arguments[6], arguments[7] = populations, hbar_factors
    with pytest.raises((TypeError, ValueError), match="population|hbar"):
        _kernels()._run_tiles(*arguments)

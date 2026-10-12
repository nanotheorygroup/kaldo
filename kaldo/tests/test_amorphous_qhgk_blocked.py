"""TensorFlow-free checks of the real Gamma-point blocked QHGK reducer."""

import subprocess
import sys
from pathlib import Path

import ase.units as units
import numpy as np
import pytest
from scipy import sparse
from ase.io import read

from kaldo.amorphous import (NpyEigensystemSource,
                             calculate_qhgk_isotropic_blocked as blocked,
                             calculate_qhgk_isotropic_dense as dense,
                             qhgk_pair_factors)
from kaldo.controllers.qhgk import generalized_heat_capacity


@pytest.fixture
def system():
    rng = np.random.default_rng(31415)
    m = 9
    symmetric = rng.normal(size=(m, m))
    symmetric = (symmetric + symmetric.T) / 2
    _, e = np.linalg.eigh(symmetric)
    # A real symmetric harmonic matrix and antisymmetric pair displacement
    # generate real skew derivative operators, as at amorphous Gamma.
    coordinates = rng.normal(size=(m, 3))
    operators = tuple(symmetric * (coordinates[:, a, None] - coordinates[None, :, a])
                      for a in range(3))
    omega = np.array([0, 0.8, 0.8000000001, 1.7, 2.4, 3.2, 4.0, 5.1, 6.8])
    physical = np.array([False, True, True, False, True, True, True, True, True])
    bandwidth = np.linspace(0.04, 0.17, m)
    return omega, bandwidth, physical, 300., e, operators


def kwargs(system, *, shape="triangle", bandwidth=None, is_classic=False,
           threshold=None, dtype=np.float64):
    omega, bw, physical, temperature, e, operators = system
    return dict(omega=omega, diffusivity_bandwidth=bw if bandwidth is None else bandwidth,
                physical_mode=physical, temperature=temperature, is_classic=is_classic,
                volume=125., eigenvectors=e, harmonic_operator=operators,
                broadening_shape=shape, diffusivity_threshold=threshold,
                compute_dtype=dtype)


@pytest.mark.parametrize("shape", ["gauss", "triangle", "lorentz"])
@pytest.mark.parametrize("bandwidth", [None, 0.08])
@pytest.mark.parametrize("is_classic", [False, True])
@pytest.mark.parametrize("threshold", [None, 2.0])
def test_dense_blocked_parity(system, shape, bandwidth, is_classic, threshold):
    inputs = kwargs(system, shape=shape, bandwidth=bandwidth,
                    is_classic=is_classic, threshold=threshold)
    reference = dense(**inputs)
    for block_size in (1, 4, 9):
        actual = blocked(**inputs, block_size=block_size)
        for got, expected in zip(actual, reference):
            assert got.shape == (9,) and got.dtype == np.float64
            assert np.isfinite(got).all()
            np.testing.assert_allclose(got, expected, rtol=2e-14, atol=1e-14)
            assert got[0] == 0 and got[3] == 0


def test_full_tensor_trace_reference(system):
    inputs = kwargs(system)
    omega, bw, physical, _, e, operators = system
    kernel, heat = qhgk_pair_factors(omega, omega, bw, bw, physical, physical,
                                     temperature=300., is_classic=False, dtype=np.float64)
    flux = [e.T @ operator @ e for operator in operators]
    tensor = np.empty((9, 3, 3))
    for a in range(3):
        for b in range(3):
            tensor[:, a, b] = np.sum(heat * kernel * flux[a] * flux[b], axis=1)
    tensor *= (units.kB / units.J) * 1e22 / 125.
    np.testing.assert_allclose(blocked(**inputs, block_size=4)[0],
                               np.trace(tensor, axis1=1, axis2=2) / 3,
                               rtol=1e-14, atol=1e-14)


def test_quantum_heat_capacity_and_masks():
    omega = np.array([0., 1., 1. + 1e-11, 2.])
    physical = np.ones(4, dtype=bool)
    classical = generalized_heat_capacity(omega, omega, physical, physical,
                                          temperature=300, is_classic=True)
    assert np.all(classical[0] == 0)
    assert np.all(classical[1:, 1:] == 1)
    quantum = generalized_heat_capacity(omega, omega, physical, physical,
                                        temperature=300, is_classic=False)
    assert np.isfinite(quantum).all()
    np.testing.assert_allclose(quantum[1, 2], quantum[1, 1], rtol=1e-9)
    np.testing.assert_allclose(quantum, quantum.T, rtol=1e-14, atol=1e-14)


def test_quantum_heat_capacity_matches_legacy_equation():
    omega = np.array([0.5, 1.7, 4.3])
    physical = np.ones(3, dtype=bool)
    temperature = 125.
    x = units._hbar * 1e12 * omega / ((units.kB / units.J) * temperature)
    population = 1 / np.expm1(x)
    with np.errstate(invalid="ignore"):
        expected = omega[:, None] * omega[None, :] * (
            population[:, None] - population[None, :]) / (
            (omega[None, :] - omega[:, None]) * temperature
            * (units.kB / units.J) / (units._hbar * 1e12))
    np.fill_diagonal(expected, (x / (2 * np.sinh(x / 2))) ** 2)
    actual = generalized_heat_capacity(omega, omega, physical, physical,
                                       temperature=temperature, is_classic=False)
    np.testing.assert_allclose(actual, expected, rtol=1e-13)


def test_legacy_threshold_uses_partner_bandwidth():
    omega = np.array([1., 1.1])
    bw = np.array([0.001, 0.2])
    physical = np.ones(2, dtype=bool)
    kernel, _ = qhgk_pair_factors(omega, omega, bw, bw, physical, physical,
                                  temperature=300, is_classic=True,
                                  broadening_shape="lorentz", diffusivity_threshold=1,
                                  dtype=np.float64)
    assert kernel[0, 1] > 0 and kernel[1, 0] == 0


def test_sparse_mmap_and_mixed_precision(system, tmp_path):
    inputs = kwargs(system)
    reference = dense(**inputs)
    omega, bw, physical, temperature, e, operators = system
    path = tmp_path / "_eigensystem.npy"
    np.save(path, np.vstack([omega ** 2, e]))
    source = NpyEigensystemSource(path)
    assert isinstance(source._array.base, np.memmap)
    np.testing.assert_array_equal(source.read_rows_modes(2, 7, 1, 4,
                                                         dtype=np.float32),
                                  e[2:7, 1:4].astype(np.float32))
    sparse_ops = tuple(sparse.csr_matrix(operator) for operator in operators)
    mixed = blocked(omega, bw, physical, temperature, False, 125., source,
                    sparse_ops, block_size=4)
    for got, expected in zip(mixed, reference):
        np.testing.assert_allclose(got, expected, rtol=3e-7, atol=3e-8)


def test_bsr_and_configurable_accumulation(system):
    inputs = kwargs(system, dtype=np.float32)
    inputs["harmonic_operator"] = tuple(sparse.bsr_matrix(op, blocksize=(3, 3))
                                         for op in system[-1])
    inputs["accumulator_dtype"] = np.float32
    conductivity, diffusivity = blocked(**inputs, block_size=4)
    assert conductivity.dtype == diffusivity.dtype == np.float32
    assert np.isfinite(conductivity).all() and np.isfinite(diffusivity).all()


def test_matrix_panel_protocol_and_no_input_mutation(system):
    class PanelOperator:
        def __init__(self, array):
            self.array = array
            self.shape = array.shape

        def __matmul__(self, panel):
            return self.array @ panel

    omega, bandwidth, physical, _, eigenvectors, operators = system
    before = [item.copy() for item in (omega, bandwidth, physical, eigenvectors, *operators)]
    inputs = kwargs(system)
    inputs["harmonic_operator"] = tuple(PanelOperator(op) for op in operators)
    actual = blocked(**inputs, block_size=4)
    reference = dense(**kwargs(system))
    for got, expected in zip(actual, reference):
        np.testing.assert_allclose(got, expected, rtol=2e-14, atol=1e-14)
    for item, old in zip((omega, bandwidth, physical, eigenvectors, *operators), before):
        np.testing.assert_array_equal(item, old)


def test_zero_bandwidth_finite(system):
    for shape in ("gauss", "triangle", "lorentz"):
        conductivity, diffusivity = blocked(**kwargs(system, shape=shape, bandwidth=0.0),
                                              block_size=4)
        assert np.all(conductivity == 0) and np.all(diffusivity == 0)


def test_tracked_amorphous_second_order_fixture():
    root = Path(__file__).resolve().parent / "data/input/amorphous-si-tersoff-32"
    second = np.load(root / "second.npy").reshape(96, 96)
    dynamical = (second + second.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(dynamical)
    omega = np.sqrt(eigenvalues - eigenvalues.min() + 0.1)
    physical = np.ones(96, dtype=bool)
    physical[:3] = False
    positions = np.repeat(read(root / "amorphous_si_32.extxyz").positions, 3, axis=0)
    operators = tuple(sparse.csr_matrix(
        dynamical * (positions[:, None, axis] - positions[None, :, axis]))
        for axis in range(3))
    inputs = dict(omega=omega, diffusivity_bandwidth=0.05,
                  physical_mode=physical, temperature=300., is_classic=False,
                  volume=1000., eigenvectors=eigenvectors,
                  harmonic_operator=operators, broadening_shape="triangle",
                  compute_dtype=np.float64)
    reference = dense(**inputs)[0]
    actual = blocked(**inputs, block_size=17)[0]
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-13)


def test_amorphous_reducer_does_not_import_tensorflow():
    # Other tests in this session import TensorFlow, so check a fresh interpreter.
    script = (
        "import sys\n"
        "import numpy as np\n"
        "from kaldo.amorphous import calculate_qhgk_isotropic_blocked, SparseAmorphousHarmonic\n"
        "e = np.eye(6)\n"
        "operators = tuple(np.zeros((6, 6)) for _ in range(3))\n"
        "calculate_qhgk_isotropic_blocked(np.linspace(1, 2, 6), 0.1, np.ones(6, bool), 300., False,\n"
        "                                 10., e, operators, block_size=2)\n"
        "assert 'tensorflow' not in sys.modules, 'kaldo.amorphous imported tensorflow'\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_no_global_pair_field(system, monkeypatch):
    import kaldo.amorphous.qhgk as qhgk

    original = qhgk.qhgk_pair_factors
    shapes = []

    def instrument(*args, **kwargs):
        pair = original(*args, **kwargs)
        shapes.append(pair[0].shape)
        return pair

    monkeypatch.setattr(qhgk, "qhgk_pair_factors", instrument)
    blocked(**kwargs(system), block_size=9)
    assert shapes and (9, 9) not in shapes
    assert max(a * b for a, b in shapes) <= 8 * 8


@pytest.mark.parametrize("bad, message", [
    (dict(omega=np.array([1., 0.])), "sorted"),
    (dict(diffusivity_bandwidth=np.ones(3)), "entries"),
    (dict(block_size=0), "positive integer"),
])
def test_input_errors(system, bad, message):
    inputs = kwargs(system)
    inputs.update(bad)
    with pytest.raises(ValueError, match=message):
        blocked(**inputs)

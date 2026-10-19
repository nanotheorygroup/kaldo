"""Contracts for the dormant pair-first Gamma projection implementation."""

import importlib
from types import SimpleNamespace

from ase import Atoms
import ase.units as units
import numpy as np
import pytest
from sparse import COO

from kaldo.controllers import anharmonic
from kaldo.grid import SupercellGrid, TranslationSupport
from kaldo.observables.harmonic_with_q_temp import CLASSICAL_HBAR_SCALE, HarmonicWithQTemp
from kaldo.observables.thirdorder import ThirdOrder
from kaldo.tests.test_amorphous_projection import _dense_projection_reference, _projection_input


@pytest.fixture(params=["numpy", "numba"])
def projection_backend(request, monkeypatch):
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    if request.param == "numpy":
        reducer = projection._run_tiles_numpy
    else:
        kernels = importlib.import_module("kaldo.controllers._projection_kernels")
        reducer = kernels._run_tiles
    monkeypatch.setattr(projection, "_load_tile_reducer", lambda: reducer)
    return request.param


def _third(value: np.ndarray | COO, translations: int = 1) -> ThirdOrder:
    """Build IFC3 in eV/Angstrom^3 with literal translation axes."""
    n_atoms = value.shape[0] if value.ndim == 8 else value.shape[0] // 3
    atoms = Atoms("H" * n_atoms, positions=np.zeros((n_atoms, 3)), cell=np.eye(3), pbc=True)
    grid = SupercellGrid(np.eye(3, dtype=int))
    support = TranslationSupport(
        np.column_stack((np.arange(translations), np.zeros((translations, 2), dtype=int))),
        grid,
        provenance="file",
    )
    return ThirdOrder(
        atoms=atoms, replicated_positions=atoms.positions[None], supercell=(1, 1, 1),
        value=value, supercell_grid=grid, translation_support=support, folder="",
    )


def _unpack(layout) -> np.ndarray:
    """Expand the private layout only for small, dimensionless test tensors."""
    dense = np.zeros((layout.n_modes,) * 3)
    partner_j = np.repeat(np.arange(layout.n_modes), np.diff(layout.j_ptr))
    for start, stop, matrix in layout.chunks:
        dense[layout.pair_i[start:stop], partner_j[start:stop]] = matrix.toarray()
    return dense


def test_pair_first_private_entry_points_exist():
    assert callable(getattr(ThirdOrder, "_gamma_pair_csr", None))
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    assert callable(projection._project)


@pytest.mark.parametrize("rank", [3, 8])
@pytest.mark.parametrize("is_dense", [False, True])
@pytest.mark.parametrize("translations", [1, 2])
def test_pair_layout_preserves_source_and_folds_literal_translations(rank, is_dense, translations):
    random = np.random.default_rng(601)
    values = random.integers(-4, 5, size=(2, 3, translations, 2, 3, translations, 2, 3)).astype(float)
    values[random.random(values.shape) < 0.7] = 0
    expected = values.sum(axis=(2, 5)).reshape((6,) * 3)
    if rank == 3:
        values = values.reshape((6, 6 * translations, 6 * translations))
    source = values if is_dense else COO.from_numpy(values)
    before = source.copy()
    third = _third(source, translations)

    layout = third._gamma_pair_csr()

    assert layout.n_modes == 6
    assert layout.pair_i.dtype == np.int32
    assert layout.j_ptr.dtype == np.int64
    np.testing.assert_array_equal(_unpack(layout), expected)
    if is_dense:
        np.testing.assert_array_equal(source, before)
    else:
        np.testing.assert_array_equal(source.coords, before.coords)
        np.testing.assert_array_equal(source.data, before.data)
    assert third.value is source
    assert all(matrix.dtype == np.float64 for _, _, matrix in layout.chunks)
    if source.size and (is_dense or source.nnz):
        if is_dense:
            source.flat[0] += 7
        else:
            source.data[0] += 7
        updated = third._gamma_pair_csr()
        assert updated is not layout
        assert not np.array_equal(_unpack(updated), _unpack(layout))
        np.testing.assert_array_equal(_unpack(layout), expected)


def test_pair_layout_sparse_batches_and_chunks_do_not_split_pair_rows(monkeypatch):
    module = importlib.import_module("kaldo.observables.thirdorder")
    monkeypatch.setattr(module, "_PAIR_BATCH_ENTRIES", 7)
    monkeypatch.setattr(module, "_PAIR_CHUNK_ENTRIES", 9)
    values = np.arange(1, 217, dtype=float).reshape((6,) * 3)
    source = COO.from_numpy(values)
    source.coords = source.coords.astype(np.uint64)
    monkeypatch.setattr(COO, "reshape", lambda *args: pytest.fail("rank-three COO must not be reshaped"))
    monkeypatch.setattr(COO, "todense", lambda *args: pytest.fail("sparse packing must not densify"))

    layout = _third(source)._gamma_pair_csr()

    np.testing.assert_array_equal(_unpack(layout), values)
    assert len(layout.chunks) > 1
    cursor = 0
    for start, stop, matrix in layout.chunks:
        assert start == cursor < stop
        assert matrix.shape == (stop - start, 6)
        assert matrix.indptr.dtype == matrix.indices.dtype == np.int32
        assert matrix.indptr[0] == 0
        assert matrix.indptr[-1] == matrix.nnz
        cursor = stop
    assert cursor == layout.pair_i.size


def test_pair_layout_promotes_float32_source_without_changing_its_values():
    values = np.random.default_rng(603).normal(size=(3,) * 3).astype(np.float32)
    source = COO.from_numpy(values)
    layout = _third(source)._gamma_pair_csr()
    np.testing.assert_array_equal(_unpack(layout), values.astype(np.float64))
    assert source.dtype == np.float32


@pytest.mark.parametrize("is_dense", [False, True])
def test_pair_layout_sums_float32_literal_values_in_float64(is_dense):
    values = np.zeros((1, 3, 2, 1, 3, 2, 1, 3), dtype=np.float32)
    values[0, 0, :, 0, 0, :, 0, 0] = [[2 ** 24, 1], [-2 ** 24, 1]]
    source = values if is_dense else COO.from_numpy(values)
    layout = _third(source, translations=2)._gamma_pair_csr()
    assert _unpack(layout)[0, 0, 0] == 2


@pytest.mark.parametrize("is_dense", [False, True])
def test_pair_layout_rejects_nonfinite_fold_of_finite_literal_values(is_dense):
    values = np.zeros((1, 3, 2, 1, 3, 2, 1, 3))
    values[0, 0, :, 0, 0, 0, 0, 0] = 1e308
    source = values if is_dense else COO.from_numpy(values)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite IFC3"):
        _third(source, translations=2)._gamma_pair_csr()


@pytest.mark.parametrize("case", ["shape", "complex", "fill"])
def test_pair_layout_rejects_unsupported_source_without_mutation(case):
    source = COO.from_numpy(np.ones((3,) * 3))
    third = _third(source)
    if case == "shape":
        third.value = np.zeros((3, 4, 3))
    elif case == "complex":
        third.value = source.astype(complex)
    else:
        third.value = COO(np.empty((3, 0), dtype=int), np.empty(0), shape=(3,) * 3, fill_value=1)
    with pytest.raises((ValueError, TypeError)):
        third._gamma_pair_csr()


@pytest.mark.parametrize("temperature", [50.0, 600.0])
@pytest.mark.parametrize("is_classic", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_pair_first_matches_dense_reference_with_tail_blocks(
    monkeypatch, temperature, is_classic, workers, projection_backend,
):
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    for name, value in (("_COLUMN_BLOCK", 4), ("_MODE_BLOCK", 2), ("_ROW_BLOCK", 2), ("_STAGE1_COLUMNS", 3)):
        monkeypatch.setattr(projection, name, value)
    random = np.random.default_rng(602)
    values = random.normal(size=(9,) * 3)
    values[random.random(values.shape) < 0.7] = 0
    phonons = _projection_input(values, bandwidth=0.19)
    phonons.frequency[0] = [0, 1, 1, 2, 3, 3, 4, 6, 7]
    phonons.physical_mode[0, [0, 4]] = False
    hbar_factor = CLASSICAL_HBAR_SCALE if is_classic else 1.0
    phonons.temperature = temperature
    phonons.hbar = units._hbar * hbar_factor
    population = HarmonicWithQTemp._calculate_population(phonons).flatten()
    with np.errstate(divide="ignore"):
        phases, potentials = _dense_projection_reference(values, phonons)
    expected = anharmonic.calculate_ps_and_gamma(
        phases, potentials, population, False, 9, True, hbar_factor=hbar_factor
    )
    expected_counts = np.array([
        [0 if phase is None else len(phase.values) for phase in channels[:2]] for channels in phases
    ])

    actual, counts = projection._project(
        _third(COO.from_numpy(values))._gamma_pair_csr(), phonons._rescaled_eigenvectors[0],
        2 * np.pi * phonons.frequency[0], phonons.physical_mode[0], phonons.third_bandwidth,
        population, hbar_factor, workers,
    )

    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("case", ["zero_width", "no_physical", "no_pairs", "zero_ifc"])
def test_pair_first_empty_cases_write_defined_outputs(case, projection_backend):
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    values = np.zeros((3,) * 3) if case == "zero_ifc" else np.ones((3,) * 3)
    omega = np.array([1.0, 2.0, 3.0])
    physical = np.zeros(3, dtype=bool) if case == "no_physical" else np.ones(3, dtype=bool)
    sigma = 0.0 if case == "zero_width" else 0.01
    if case == "no_pairs":
        omega = np.array([5.0, 7.0, 9.0])

    actual, counts = projection._project(
        _third(COO.from_numpy(values))._gamma_pair_csr(), np.eye(3), omega, physical,
        sigma, np.ones(3), 1.0, None,
    )

    assert np.all(np.isfinite(actual))
    if case == "zero_ifc":
        assert np.any(actual[:, 0] > 0)
        assert np.any(counts > 0)
        np.testing.assert_array_equal(actual[:, 1], 0)
    else:
        np.testing.assert_array_equal(actual, 0)
        np.testing.assert_array_equal(counts, 0)


def test_pair_first_float32_width_keeps_float64_semantics_under_weak_promotion(projection_backend):
    if not hasattr(np, "_set_promotion_state") or not hasattr(np, "_get_promotion_state"):
        pytest.skip("NumPy promotion-state helpers are unavailable")
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    values = np.random.default_rng(606).normal(size=(3,) * 3)
    layout = _third(COO.from_numpy(values))._gamma_pair_csr()
    arguments = (layout, np.eye(3), np.arange(1., 4.), np.ones(3, dtype=bool))
    width = np.float32(.12)
    population = np.array([1.4, .8, .2])
    previous = np._get_promotion_state()
    try:
        np._set_promotion_state("weak")
        actual, counts = projection._project(*arguments, width, population)
        expected, expected_counts = projection._project(*arguments, float(width), population)
    finally:
        np._set_promotion_state(previous)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=1e-12, atol=1e-13)


def test_pair_first_cutoff_selects_inside_but_excludes_boundary_and_outside(monkeypatch, projection_backend):
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    monkeypatch.setattr(projection, "_COLUMN_BLOCK", 4)
    monkeypatch.setattr(projection, "_MODE_BLOCK", 4)
    monkeypatch.setattr(projection, "_ROW_BLOCK", 2)
    values = np.random.default_rng(604).normal(size=(6,) * 3)
    omega = np.arange(1.0, 7.0)
    population = np.linspace(1.3, 0.2, 6)
    expected = np.zeros((6, 2))
    expected_counts = np.zeros((6, 2), dtype=np.int64)
    threshold = 2.0
    distances = omega[3] - omega[:, None] - omega[None, :]
    assert all(np.any(np.abs(distances) == distance) for distance in (1, 2, 3))
    for initial in range(6):
        for channel in (0, 1):
            difference = omega[initial] + (2 * channel - 1) * omega[:, None] - omega[None, :]
            row, column = np.nonzero(np.abs(difference) < threshold)
            phase = np.exp(-difference[row, column] ** 2) / np.sqrt(np.pi)
            if channel:
                population_delta = population[row] - population[column]
            else:
                population_delta = 0.5 * (1 + population[row] + population[column])
            squared = values[initial, row, column] ** 2
            squared /= omega[initial] * omega[row] * omega[column]
            squared *= np.pi * units._hbar / 4 * anharmonic.GAMMA_TO_THZ
            expected[initial] += [np.sum(phase * population_delta), np.sum(squared * phase * population_delta)]
            expected_counts[initial, channel] = row.size

    actual, counts = projection._project(
        _third(COO.from_numpy(values))._gamma_pair_csr(), np.eye(6), omega,
        np.ones(6, dtype=bool), 1 / (2 * np.pi), population,
    )

    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("omega,physical,sigma,expected", [
    ([0, 1, 2], [False, True, True], 0.1, True),
    ([0, 1, 2], [True, True, True], 0.1, False),
    ([-1, 1, 2], [True, True, True], 0.1, False),
    ([2, 1, 3], [True, True, True], 0.1, False),
    ([1, np.nan, 3], [True, True, True], 0.1, False),
    ([1, 2, np.inf], [True, True, True], 0.1, False),
    ([1, 2, 3], [True, True, True], -0.1, False),
    ([1, 2, 3], [True, True, True], 0.0, True),
    ([1, 2, 3], [True, True, True], None, False),
])
def test_pair_first_frequency_eligibility(omega, physical, sigma, expected):
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    assert projection._is_eligible(np.array(omega), np.array(physical), sigma) == expected


def test_pair_first_workspace_budget_reduces_blocks_and_rejects_impossible_budget(monkeypatch):
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    layout = SimpleNamespace(n_modes=9, pair_i=np.arange(60), chunks=())
    monkeypatch.setattr(projection, "virtual_memory", lambda: SimpleNamespace(available=20000))
    columns, modes = projection._block_sizes(layout, 1)
    assert columns < min(9, projection._COLUMN_BLOCK)
    assert modes <= min(9, projection._MODE_BLOCK)
    monkeypatch.setattr(projection, "virtual_memory", lambda: SimpleNamespace(available=1))
    with pytest.raises(MemoryError):
        projection._block_sizes(layout, 1)


@pytest.mark.parametrize("workers", [0, -1, 1.5])
def test_pair_first_rejects_invalid_worker_counts(workers):
    projection = importlib.import_module("kaldo.controllers._amorphous_projection")
    with pytest.raises(ValueError, match="worker count"):
        projection._project(
            _third(COO.from_numpy(np.ones((3,) * 3)))._gamma_pair_csr(), np.eye(3),
            np.arange(1.0, 4.0), np.ones(3, dtype=bool), 0.1, np.ones(3), n_workers=workers,
        )

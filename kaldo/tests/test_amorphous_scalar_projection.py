"""Public scalar dispatch and persistence contracts for Gamma projection."""

from pathlib import Path
from copy import copy
from unittest.mock import Mock

import numpy as np
import pytest
from sparse import COO
import tensorflow as tf

from kaldo.controllers import _amorphous_projection, anharmonic
from kaldo.forceconstants import ForceConstants
from kaldo.grid import TranslationSupport
from kaldo.observables.harmonic_with_q_temp import CLASSICAL_HBAR_SCALE
from kaldo.phonons import Phonons


_FIXTURE = Path(__file__).resolve().parent / "data" / "input" / "amorphous-si-tersoff-32"
_REDUCED_LABEL = (
    "<temperature>/<statistics>/<third_bandwidth>/<broadening_shape>/"
    "<broadening_kernel>/<is_balanced>/<use_q_symmetry>"
)
_SPARSE_LABEL = "<third_bandwidth>/<broadening_shape>/<broadening_kernel>/<use_q_symmetry>"


@pytest.fixture(scope="module")
def forceconstants() -> ForceConstants:
    """Load the 32-atom Si IFC2/IFC3 fixture in eV/Angstrom powers."""
    result = ForceConstants.from_folder(
        folder=str(_FIXTURE), format="numpy", supercell=(1, 1, 1),
        third_supercell=(1, 1, 1), is_acoustic_sum=True,
    )
    result.third
    return result


def _phonons(forceconstants: ForceConstants, **options) -> Phonons:
    """Configure Gamma phonons at 300 K with a Gaussian width in THz."""
    parameters = dict(
        temperature=300.0, third_bandwidth=0.05 / 4.135,
        broadening_shape="gauss", storage="memory",
    )
    parameters.update(options)
    return Phonons(forceconstants=forceconstants, **parameters)


def _legacy_result(phonons: Phonons, tensor: bool = False) -> np.ndarray:
    """Return old ``(n,2[+n])`` phase space and bandwidth in THz."""
    phases, potentials = phonons._project_amorphous()
    return anharmonic.calculate_ps_and_gamma(
        phases, potentials, phonons.population.flatten(), phonons.is_balanced,
        phonons.n_phonons, True, tensor,
        CLASSICAL_HBAR_SCALE if phonons.is_classic else 1,
    )


def _assert_reduced(actual: np.ndarray, expected: np.ndarray) -> None:
    """Compare ``(n,2)`` phase space and THz linewidth at float64 accuracy."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[:, 1:], expected[:, 1:], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("temperature", [0.0, 100.0, 900.0])
@pytest.mark.parametrize("is_classic", [False, True])
def test_scalar_uses_pair_projection_without_mutating_inputs(
    forceconstants, monkeypatch, temperature, is_classic,
):
    phonons = _phonons(forceconstants, temperature=temperature, is_classic=is_classic)
    expected = _legacy_result(phonons)
    third = forceconstants.third.value
    arrays = (
        third.coords, third.data, phonons._eigensystem, phonons.frequency,
        phonons.physical_mode, phonons.population,
    )
    snapshots = tuple(array.copy() for array in arrays)
    identity = phonons.get_folder_from_label(_REDUCED_LABEL)
    projection = Mock(wraps=_amorphous_projection._project)
    monkeypatch.setattr(_amorphous_projection, "_project", projection)
    monkeypatch.setattr(Phonons, "_project_amorphous", Mock(side_effect=AssertionError("legacy projection")))

    actual = np.column_stack((phonons.phase_space.ravel(), phonons.anharmonic_bandwidth.ravel()))

    assert projection.call_count == 1
    _assert_reduced(actual, expected)
    assert phonons.get_folder_from_label(_REDUCED_LABEL) == identity
    assert forceconstants.third.value is third
    assert not hasattr(phonons, "_lazy___sparse_phase_and_potential")
    for array, snapshot in zip(arrays, snapshots):
        np.testing.assert_array_equal(array, snapshot)


@pytest.mark.parametrize("third_bandwidth", [0.0, 0.05 / 4.135])
def test_scalar_empty_selection_stays_on_fast_path(forceconstants, monkeypatch, third_bandwidth):
    phonons = _phonons(forceconstants, third_bandwidth=third_bandwidth)
    if third_bandwidth:
        phonons._lazy__physical_mode = np.zeros(phonons.n_phonons, dtype=bool)
    expected = _legacy_result(phonons)
    projection = Mock(wraps=_amorphous_projection._project)
    monkeypatch.setattr(_amorphous_projection, "_project", projection)

    _assert_reduced(phonons._ps_and_gamma, expected)
    assert projection.call_count == 1
    np.testing.assert_array_equal(phonons._ps_and_gamma, np.zeros((phonons.n_modes, 2)))


@pytest.mark.parametrize("storage", ["numpy", "formatted", "hdf5"])
@pytest.mark.parametrize("cached_name", ["_ps_and_gamma", "_ps_gamma_and_gamma_tensor"])
def test_existing_reduced_or_collision_cache_loads_before_projection(
    forceconstants, monkeypatch, tmp_path, storage, cached_name,
):
    monkeypatch.chdir(tmp_path)
    phonons = _phonons(forceconstants, storage=storage, folder="cache")
    columns = 2 if cached_name == "_ps_and_gamma" else 2 + phonons.n_modes
    cached = np.arange(phonons.n_modes * columns, dtype=np.float64).reshape(phonons.n_modes, columns)
    folder = phonons.get_folder_from_label(_REDUCED_LABEL)
    saved_format = phonons._store_formats[cached_name] if storage == "formatted" else storage
    phonons._save_property(cached_name, folder, cached, format=saved_format)
    projection = Mock(side_effect=AssertionError("cache miss"))
    monkeypatch.setattr(_amorphous_projection, "_project", projection)
    monkeypatch.setattr(Phonons, "_project_amorphous", projection)

    np.testing.assert_array_equal(phonons._ps_and_gamma, cached[:, :2])
    assert not projection.called
    assert phonons.get_folder_from_label(_REDUCED_LABEL) == folder


@pytest.mark.parametrize("storage", ["numpy", "formatted", "hdf5"])
def test_fast_scalar_persists_only_existing_reduced_keys(forceconstants, monkeypatch, tmp_path, storage):
    monkeypatch.chdir(tmp_path)
    phonons = _phonons(forceconstants, storage=storage, folder="cache")
    expected = _legacy_result(phonons)
    folder = phonons.get_folder_from_label(_REDUCED_LABEL)
    projection = Mock(wraps=_amorphous_projection._project)
    monkeypatch.setattr(_amorphous_projection, "_project", projection)

    _assert_reduced(phonons._ps_and_gamma, expected)

    assert projection.call_count == 1
    assert not list(tmp_path.rglob("*sparse*"))
    assert not list(tmp_path.rglob("*csr*"))
    saved_format = "numpy" if storage == "formatted" else storage
    _assert_reduced(phonons._load_property("_ps_and_gamma", folder, format=saved_format), expected)
    fresh = _phonons(forceconstants, storage=storage, folder="cache")
    projection.side_effect = AssertionError("cache miss")
    _assert_reduced(fresh._ps_and_gamma, expected)


def test_explicit_sparse_outputs_and_cache_survive_fast_scalar(forceconstants, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    phonons = _phonons(forceconstants, storage="numpy", folder="cache")
    scalar = phonons._ps_and_gamma
    assert not list(tmp_path.rglob("*sparse*"))
    phases, potentials = phonons.sparse_phase, phonons.sparse_potential
    folder = phonons.get_folder_from_label(_SPARSE_LABEL)
    base = Path(folder) / "_sparse_phase_and_potential"
    mode_list = np.load(str(base) + "_mu_list.npy")
    assert mode_list.dtype == np.int32
    assert len(phases) == len(potentials) == phonons.n_modes
    for mode in mode_list:
        record = np.load(str(base) + f"_mu_{mode}.npy", allow_pickle=True).item()
        assert set(record) == {"exists", "tensors"}
        for tensor in record["tensors"]:
            assert set(tensor) == {"is_plus", "indices", "phase_values", "potential_values", "dense_shape"}
            assert tensor["indices"].dtype == np.int64
            assert tensor["phase_values"].dtype == tensor["potential_values"].dtype == np.float64
    fresh = _phonons(forceconstants, storage="numpy", folder="cache")
    projection = Mock(side_effect=AssertionError("existing sparse cache"))
    monkeypatch.setattr(Phonons, "_project_amorphous", projection)
    for original, reloaded in ((phases, fresh.sparse_phase), (potentials, fresh.sparse_potential)):
        for original_channels, reloaded_channels in zip(original, reloaded):
            for original_tensor, reloaded_tensor in zip(original_channels, reloaded_channels):
                if original_tensor is None:
                    assert reloaded_tensor is None
                else:
                    assert isinstance(reloaded_tensor, tf.SparseTensor)
                    np.testing.assert_array_equal(original_tensor.indices, reloaded_tensor.indices)
                    np.testing.assert_array_equal(original_tensor.values, reloaded_tensor.values)
                    np.testing.assert_array_equal(original_tensor.dense_shape, reloaded_tensor.dense_shape)
    assert not projection.called
    _assert_reduced(fresh._ps_and_gamma, scalar)


@pytest.mark.parametrize("case", [
    "triangle", "lorentz", "balanced", "tensor", "none_width", "negative_width", "array_width", "huge_width",
    "unsorted", "nan_frequency", "infinite_frequency", "zero_physical", "negative_physical",
    "float32_frequency", "float32_eigenvectors", "complex_eigenvectors", "nan_eigenvectors",
    "nan_population", "float32_population", "float32_source", "complex_source", "integer_source",
    "nan_source", "masked_source",
])
def test_unsupported_scalar_cases_keep_legacy_values_or_errors(forceconstants, monkeypatch, case):
    phonons = _phonons(forceconstants)
    phonons.population
    phonons.physical_mode
    tensor = case == "tensor"
    if case in ("triangle", "lorentz"):
        phonons.broadening_shape = case
    elif case == "balanced":
        phonons.is_balanced = True
    elif case.endswith("width"):
        phonons.third_bandwidth = {
            "none_width": None, "negative_width": -0.01, "array_width": np.array([0.01]), "huge_width": 1e308,
        }[case]
    elif case in ("unsorted", "nan_frequency", "infinite_frequency", "zero_physical", "negative_physical"):
        frequency = phonons.frequency.copy()
        if case == "unsorted":
            frequency[:, 10:12] = frequency[:, 10:12][:, ::-1]
        elif case == "nan_frequency":
            frequency[0, 10] = np.nan
        elif case == "infinite_frequency":
            frequency[0, 10] = np.inf
        else:
            frequency[0] = np.arange(phonons.n_modes) - (3 if case == "zero_physical" else 5)
        phonons._lazy__frequency = frequency
    elif case == "float32_frequency":
        phonons._lazy__frequency = phonons.frequency.astype(np.float32)
    elif case == "float32_eigenvectors":
        eigenvectors = phonons._rescaled_eigenvectors.astype(np.float32)
        monkeypatch.setattr(Phonons, "_rescaled_eigenvectors", property(lambda self: eigenvectors))
    elif case in ("complex_eigenvectors", "nan_eigenvectors"):
        eigensystem = phonons._eigensystem.astype(complex if case == "complex_eigenvectors" else float)
        eigensystem[0, 2, 10] = 1j if case == "complex_eigenvectors" else np.nan
        phonons._lazy___eigensystem = eigensystem
    elif case in ("nan_population", "float32_population"):
        population = phonons.population.astype(np.float32 if case == "float32_population" else float)
        if case == "nan_population":
            population[0, 10] = np.nan
        phonons._lazy__population = population
    else:
        source = forceconstants.third.value
        if case in ("float32_source", "complex_source", "integer_source"):
            source = source.astype({
                "float32_source": np.float32, "complex_source": complex, "integer_source": np.int64,
            }[case])
        elif case == "nan_source":
            source = source.copy()
            source.data[0] = np.nan
        else:
            source = np.ma.array(source.todense(), mask=False)
        monkeypatch.setattr(forceconstants.third, "value", source)
    reference_error = None
    try:
        expected = _legacy_result(phonons, tensor)
    except (TypeError, ValueError, tf.errors.OpError) as error:
        reference_error = type(error)
    projection = Mock(side_effect=AssertionError("unsupported pair projection"))
    legacy = Mock(wraps=phonons._project_amorphous)
    packing = Mock(wraps=forceconstants.third._gamma_pair_csr)
    monkeypatch.setattr(_amorphous_projection, "_project", projection)
    monkeypatch.setattr(forceconstants.third, "_gamma_pair_csr", packing)
    monkeypatch.setattr(phonons, "_project_amorphous", legacy)

    if reference_error is not None:
        with pytest.raises(reference_error):
            phonons._select_algorithm_for_phase_space_and_gamma(tensor)
    else:
        _assert_reduced(phonons._select_algorithm_for_phase_space_and_gamma(tensor), expected)
    assert legacy.call_count == 1
    assert not projection.called
    if case != "nan_source":
        assert not packing.called


@pytest.mark.parametrize("is_dense", [False, True])
def test_float64_source_representations_stay_eligible(forceconstants, monkeypatch, is_dense):
    source = forceconstants.third.value
    source = source.todense() if is_dense else source.copy()
    monkeypatch.setattr(forceconstants.third, "value", source)
    phonons = _phonons(forceconstants)
    expected = _legacy_result(phonons)
    projection = Mock(wraps=_amorphous_projection._project)
    monkeypatch.setattr(_amorphous_projection, "_project", projection)

    _assert_reduced(phonons._ps_and_gamma, expected)
    assert projection.call_count == 1


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_unsupported_packing_falls_back(forceconstants, monkeypatch, error_type):
    phonons = _phonons(forceconstants)
    expected = _legacy_result(phonons)
    monkeypatch.setattr(forceconstants.third, "_gamma_pair_csr", Mock(side_effect=error_type("packing")))
    projection = Mock(side_effect=AssertionError("unsupported pair projection"))
    monkeypatch.setattr(_amorphous_projection, "_project", projection)

    _assert_reduced(phonons._ps_and_gamma, expected)
    assert not projection.called


@pytest.mark.parametrize("stage", ["packing", "projection"])
def test_fast_execution_failures_are_not_hidden(forceconstants, monkeypatch, stage):
    phonons = _phonons(forceconstants)
    error_type = MemoryError if stage == "packing" else RuntimeError
    target = forceconstants.third if stage == "packing" else _amorphous_projection
    name = "_gamma_pair_csr" if stage == "packing" else "_project"
    monkeypatch.setattr(target, name, Mock(side_effect=error_type("real execution failure")))

    with pytest.raises(error_type, match="real execution failure"):
        phonons._ps_and_gamma


def test_float32_literal_folding_keeps_legacy_rounding(forceconstants, monkeypatch):
    third = copy(forceconstants.third)
    values = np.zeros((32, 3, 2, 32, 3, 2, 32, 3), dtype=np.float32)
    values[0, 0, 0, 0, 0, 0, 0, 0] = 2 ** 24
    values[0, 0, 0, 0, 0, 1, 0, 0] = 1
    values[0, 0, 1, 0, 0, 0, 0, 0] = -(2 ** 24)
    third.value = values
    third.translation_support = TranslationSupport(
        [[0, 0, 0], [1, 0, 0]], third.supercell_grid, provenance="file",
    )
    third.n_translations = 2
    monkeypatch.setattr(forceconstants, "_third", third)
    phonons = _phonons(forceconstants, third_bandwidth=0.0)
    before = third.gamma_contracted_value()
    assert before.dtype == np.float32
    assert before[0, 0, 0] == 0
    assert values.astype(np.float64).sum() == 1
    projection = Mock(side_effect=AssertionError("float32 promotion"))
    legacy = Mock(wraps=phonons._project_amorphous)
    monkeypatch.setattr(_amorphous_projection, "_project", projection)
    monkeypatch.setattr(phonons, "_project_amorphous", legacy)

    np.testing.assert_array_equal(phonons._ps_and_gamma, np.zeros((phonons.n_modes, 2)))
    np.testing.assert_array_equal(third.gamma_contracted_value(), before)
    assert legacy.call_count == 1
    assert not projection.called


def test_public_projection_keeps_phase_space_for_zero_ifc(forceconstants, monkeypatch):
    source = forceconstants.third.value
    zero = COO(np.empty((source.ndim, 0), dtype=np.int64), np.empty(0), shape=source.shape)
    monkeypatch.setattr(forceconstants.third, "value", zero)
    phonons = _phonons(forceconstants)
    expected = _legacy_result(phonons)
    projection = Mock(wraps=_amorphous_projection._project)
    monkeypatch.setattr(_amorphous_projection, "_project", projection)

    actual = phonons._ps_and_gamma

    _assert_reduced(actual, expected)
    assert actual[:, 0].sum() > 0
    np.testing.assert_array_equal(actual[:, 1], np.zeros(phonons.n_modes))
    assert projection.call_count == 1


@pytest.mark.parametrize("n_workers", [None, 2])
def test_existing_worker_option_reaches_cpu_projection(forceconstants, monkeypatch, n_workers):
    phonons = _phonons(forceconstants, n_workers=n_workers)
    expected = _legacy_result(phonons)
    projection = Mock(return_value=(expected.copy(), np.zeros((phonons.n_modes, 2), dtype=np.int64)))
    monkeypatch.setattr(_amorphous_projection, "_project", projection)

    _assert_reduced(phonons._ps_and_gamma, expected)
    assert projection.call_count == 1
    assert projection.call_args.args[-1] == n_workers


def test_public_projection_has_working_numpy_fallback(forceconstants, monkeypatch):
    phonons = _phonons(forceconstants)
    expected = _legacy_result(phonons)
    monkeypatch.setattr(_amorphous_projection, "_load_tile_reducer", lambda: _amorphous_projection._run_tiles_numpy)
    projection = Mock(wraps=_amorphous_projection._project)
    monkeypatch.setattr(_amorphous_projection, "_project", projection)

    _assert_reduced(phonons._ps_and_gamma, expected)
    assert projection.call_count == 1

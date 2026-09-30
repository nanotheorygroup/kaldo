import inspect
import json

import numpy as np

from kaldo.conductivity import Conductivity
from kaldo.controllers.interpolator import BandwidthInterpolator
from kaldo.phonons import Phonons


def test_low_frequency_quadratic_limit():
    frequency = np.linspace(0.25, 8.0, 80).reshape(1, -1)
    coefficient = 0.37
    bandwidth = coefficient * frequency**2

    interpolator = BandwidthInterpolator(
        frequency=frequency,
        bandwidth=bandwidth,
        sigma=0.05,
        low_frequency_cutoff=1.0,
        n_grid_points=160,
    )

    target_frequency = np.array([[0.10, 0.20, 0.50, 0.90]])
    predicted = interpolator.predict_bandwidth(target_frequency)
    ratio = predicted / target_frequency**2

    np.testing.assert_allclose(ratio, ratio[0, 0], rtol=1e-12, atol=1e-12)


def test_interpolator_reconstructs_smoothed_source_frequencies():
    frequency = np.linspace(0.5, 6.0, 40)
    bandwidth = 0.2 + 0.03 * frequency + 0.01 * frequency**2

    interpolator = BandwidthInterpolator(
        frequency=frequency,
        bandwidth=bandwidth,
        sigma=0.08,
        low_frequency_cutoff=0.2,
        n_grid_points=frequency.size,
    )

    predicted = interpolator.predict_bandwidth(frequency)

    np.testing.assert_allclose(predicted, interpolator.smoothed_bandwidth, rtol=1e-10, atol=1e-10)


def test_persistence_writes_fit_and_labeled_predictions(tmp_path):
    frequency = np.linspace(0.5, 4.0, 12).reshape(1, -1)
    bandwidth = 0.1 * frequency**2

    interpolator = BandwidthInterpolator(
        frequency=frequency,
        bandwidth=bandwidth,
        sigma=0.1,
        folder=tmp_path,
        low_frequency_cutoff=0.5,
        n_grid_points=40,
    )
    predicted = interpolator.predict_bandwidth(frequency, label="large_model")

    fit_folder = tmp_path / "fit"
    prediction_folder = tmp_path / "predictions" / "large_model"
    assert (fit_folder / "source_frequency.npy").exists()
    assert (fit_folder / "source_bandwidth.npy").exists()
    assert (fit_folder / "smoothed_frequency.npy").exists()
    assert (fit_folder / "smoothed_bandwidth.npy").exists()
    assert (fit_folder / "interpolation_parameters.json").exists()
    assert (prediction_folder / "frequency.npy").exists()
    assert (prediction_folder / "bandwidth.npy").exists()

    with open(fit_folder / "interpolation_parameters.json") as handle:
        parameters = json.load(handle)
    assert parameters["sigma"] == 0.1
    assert parameters["frequency_units"] == "THz"
    assert parameters["bandwidth_units"] == "THz"
    np.testing.assert_allclose(np.load(prediction_folder / "bandwidth.npy"), predicted)


def test_max_bandwidth_excludes_unphysical_source_outlier():
    frequency = np.linspace(0.5, 6.0, 60)
    bandwidth = 0.2 + 0.05 * frequency
    bandwidth[12] = 120.0

    interpolator = BandwidthInterpolator(
        frequency=frequency,
        bandwidth=bandwidth,
        sigma=0.1,
        low_frequency_cutoff=0.2,
        max_bandwidth=5.0,
        n_grid_points=120,
    )

    predicted = interpolator.predict_bandwidth(np.array([frequency[12]]))

    assert interpolator.excluded_source_counts["above_max_bandwidth"] == 1
    assert predicted[0] < 1.0


def test_qhgk_uses_interpolator_without_evaluating_phonons_bandwidth(monkeypatch):
    captured = {}

    class DummyInterpolator:
        def __init__(self, predicted):
            self.predicted = predicted
            self.calls = []

        def predict_bandwidth(self, frequency, label=None):
            self.calls.append((frequency.copy(), label))
            return self.predicted

    class DummyPhonons:
        n_k_points = 1
        n_modes = 2
        n_phonons = 2
        omega = np.array([[1.0, 2.0]])
        frequency = np.array([[1.0 / (2 * np.pi), 2.0 / (2 * np.pi)]])
        physical_mode = np.array([[True, True]])
        atoms = type("Atoms", (), {"cell": np.eye(3)})()
        _reciprocal_grid = type("Grid", (), {"fractional_points": np.array([[0.0, 0.0, 0.0]])})()
        forceconstants = type("ForceConstants", (), {"second": None, "distance_threshold": None})()
        folder = "unused"
        storage = "memory"
        kpts = np.array([1, 1, 1])
        temperature = 300
        is_classic = False
        is_nw = False
        ifc_interpolation = "auto"
        _is_amorphous = True
        is_nac = False
        nac_bvk_supercell_matrix = None
        third_bandwidth = None
        include_isotopes = False
        ifc_cache_key = "dummy"

        def __init__(self):
            self.interpolator = DummyInterpolator(np.array([[0.4, 0.8]]))

        @property
        def bandwidth(self):
            raise AssertionError("Phonons.bandwidth should not be evaluated when an interpolator is attached")

    class DummyHarmonicWithQTemp:
        heat_capacity_2d = np.eye(2)
        _sij_x = np.eye(2)
        _sij_y = np.eye(2)
        _sij_z = np.eye(2)

        def __init__(self, **kwargs):
            pass

    def fake_calculate_diffusivity(omega, sij_left, sij_right, diffusivity_bandwidth, physical_mode, curve,
                                   is_diffusivity_including_antiresonant=False, diffusivity_threshold=None):
        captured["diffusivity_bandwidth"] = diffusivity_bandwidth.copy()
        return np.eye(2)

    monkeypatch.setattr("kaldo.conductivity.hwqwt.HarmonicWithQTemp", DummyHarmonicWithQTemp)
    monkeypatch.setattr("kaldo.conductivity.calculate_diffusivity", fake_calculate_diffusivity)

    phonons = DummyPhonons()
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory")
    cond.calculate_conductivity_and_diffusivity_qhgk()

    np.testing.assert_allclose(captured["diffusivity_bandwidth"], np.array([0.2, 0.4]))
    assert len(phonons.interpolator.calls) == 1
    np.testing.assert_allclose(phonons.interpolator.calls[0][0], phonons.frequency)
    assert phonons.interpolator.calls[0][1] == "qhgk"

    phonons_with_override = DummyPhonons()
    cond = Conductivity(
        phonons=phonons_with_override,
        method="qhgk",
        storage="memory",
        diffusivity_bandwidth=1.5,
    )
    cond.calculate_conductivity_and_diffusivity_qhgk()

    np.testing.assert_allclose(captured["diffusivity_bandwidth"], np.array([1.5, 1.5]))
    assert phonons_with_override.interpolator.calls == []


def test_phonons_accepts_interpolator_keyword():
    assert "interpolator" in inspect.signature(Phonons).parameters

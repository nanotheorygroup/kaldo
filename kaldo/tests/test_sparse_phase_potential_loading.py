from types import MethodType, SimpleNamespace

import numpy as np

from kaldo.phonons import Phonons
import kaldo.phonons as phonons_module


def _write_sparse_mu(cache_dir, mu, *, is_plus=0):
    data = {
        "exists": True,
        "tensors": [
            {
                "is_plus": is_plus,
                "indices": np.array([[0, 1], [1, 0]], dtype=np.int64),
                "phase_values": np.array([0.25, 0.5], dtype=np.float64),
                "potential_values": np.array([1.25, 1.5], dtype=np.float64),
                "dense_shape": (2, 2),
            }
        ],
    }
    np.save(cache_dir / f"_sparse_phase_and_potential_mu_{mu}.npy", data, allow_pickle=True)


def test_ps_and_gamma_loads_sparse_phase_and_potential_pair_once(tmp_path, monkeypatch):
    cache_dir = tmp_path / "sparse-cache"
    cache_dir.mkdir()
    np.save(cache_dir / "_sparse_phase_and_potential_mu_list.npy", np.array([0, 1], dtype=np.int32))
    _write_sparse_mu(cache_dir, 0, is_plus=0)
    _write_sparse_mu(cache_dir, 1, is_plus=1)

    phonons = object.__new__(Phonons)
    phonons.storage = "numpy"
    phonons.n_k_points = 1
    phonons.n_modes = 2
    phonons.n_phonons = 2
    phonons.kpts = np.array([1, 1, 1])
    phonons.forceconstants = SimpleNamespace(supercell_grid=SimpleNamespace(size=1))
    phonons.is_classic = False
    phonons.is_balanced = False
    phonons.use_q_symmetry = False
    phonons.get_folder_from_label = lambda *args, **kwargs: str(cache_dir)
    monkeypatch.setattr(Phonons, "population", property(lambda self: np.array([1.0, 2.0])))

    load_calls = []
    original_load_property = Phonons._load_property

    def counting_load_property(self, property_name, folder, format="formatted"):
        if property_name == "_sparse_phase_and_potential":
            load_calls.append((property_name, folder, format))
        return original_load_property(self, property_name, folder, format)

    phonons._load_property = MethodType(counting_load_property, phonons)

    def fake_calculate_ps_and_gamma(
        sparse_phase,
        sparse_potential,
        population,
        is_balanced,
        n_phonons,
        is_amorphous,
        is_gamma_tensor_enabled=False,
        hbar_factor=1,
    ):
        assert sparse_phase[0][0] is not None
        assert sparse_potential[0][0] is not None
        assert sparse_phase[1][1] is not None
        assert sparse_potential[1][1] is not None
        return np.zeros((n_phonons, 2))

    monkeypatch.setattr(phonons_module.aha, "calculate_ps_and_gamma", fake_calculate_ps_and_gamma)

    phonons._select_algorithm_for_phase_space_and_gamma(is_gamma_tensor_enabled=False)

    assert len(load_calls) == 1


def test_sparse_phase_and_potential_loader_converts_mu_files_incrementally(tmp_path):
    cache_dir = tmp_path / "sparse-cache"
    cache_dir.mkdir()
    np.save(cache_dir / "_sparse_phase_and_potential_mu_list.npy", np.array([0, 1], dtype=np.int32))
    _write_sparse_mu(cache_dir, 0, is_plus=0)
    _write_sparse_mu(cache_dir, 1, is_plus=1)

    phonons = object.__new__(Phonons)
    phonons.n_phonons = 2

    def fail_bulk_conversion(self, per_mu_data):
        raise AssertionError("loader should not retain all raw per-mu records before conversion")

    phonons._convert_per_mu_arrays_to_sparse_tensors = MethodType(fail_bulk_conversion, phonons)

    sparse_phase, sparse_potential = phonons._load_property(
        "_sparse_phase_and_potential",
        str(cache_dir),
        format="numpy",
    )

    assert sparse_phase[0][0] is not None
    assert sparse_potential[0][0] is not None
    assert sparse_phase[1][1] is not None
    assert sparse_potential[1][1] is not None

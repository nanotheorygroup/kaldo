import numpy as np
import pytest
from ase import units as ase_units

from kaldo.forceconstants import ForceConstants
from kaldo.controllers.nac import bvk_supercell_matrix_key


def nacl_phonopy_debug_supercell_matrix():
    return np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]], dtype=int)


def nacl_phonopy_debug_supercell_matrix_att3():
    return np.diag([8, 8, 8]).astype(int)
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.phonons import Phonons


@pytest.fixture(scope="module")
def nac_second_order(tmp_path_factory):
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    forceconstants.second.folder = str(tmp_path_factory.mktemp("nac_runtime_cache"))
    # The machinery tests attach charges to the file constants and exercise the
    # non-analytic correction pipeline as if they were total force constants.
    forceconstants.second.atoms.info.pop("dipole_subtracted_fc", None)
    return forceconstants.second


def test_second_order_nac_short_range_force_constants_use_lazy_numpy_cache(
    nac_second_order, tmp_path, monkeypatch
):
    original_folder = nac_second_order.folder
    nac_second_order.folder = str(tmp_path)
    try:
        expected = np.arange(2 * 4 * 3 * 3, dtype=float).reshape(2, 4, 3, 3)
        calls = {"count": 0}

        def calculate_once():
            calls["count"] += 1
            return expected

        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            calculate_once,
        )
        np.testing.assert_allclose(
            nac_second_order.nac_short_range_force_constants, expected
        )
        assert calls["count"] == 1
        assert (tmp_path / "nac_short_range_force_constants.npy").exists()

        def fail_if_recomputed():
            raise AssertionError(
                "cached NAC short-range force constants were not reused"
            )

        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            fail_if_recomputed,
        )
        np.testing.assert_allclose(
            nac_second_order.nac_short_range_force_constants, expected
        )
    finally:
        nac_second_order.folder = original_folder


def test_second_order_nac_short_range_force_constants_use_matrix_specific_cache(
    nac_second_order, tmp_path, monkeypatch
):
    original_folder = nac_second_order.folder
    nac_second_order.folder = str(tmp_path)
    try:
        matrix = nacl_phonopy_debug_supercell_matrix()
        expected = np.arange(2 * 64 * 3 * 3, dtype=float).reshape(2, 64, 3, 3)
        calls = {"count": 0}

        def calculate_once(nac_bvk_supercell_matrix=None):
            calls["count"] += 1
            np.testing.assert_array_equal(nac_bvk_supercell_matrix, matrix)
            return expected

        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            calculate_once,
        )
        actual = nac_second_order.get_nac_short_range_force_constants(matrix)
        np.testing.assert_allclose(actual, expected)
        assert calls["count"] == 1

        property_name = (
            "nac_short_range_force_constants_" + bvk_supercell_matrix_key(matrix)
        )
        assert (tmp_path / f"{property_name}.npy").exists()

        def fail_if_recomputed(nac_bvk_supercell_matrix=None):
            raise AssertionError("matrix-specific NAC cache was not reused")

        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            fail_if_recomputed,
        )
        actual = nac_second_order.get_nac_short_range_force_constants(matrix)
        np.testing.assert_allclose(actual, expected)
    finally:
        nac_second_order.folder = original_folder


def test_second_order_nac_short_range_force_constants_reuse_in_memory_matrix_cache(
    nac_second_order, tmp_path, monkeypatch
):
    original_folder = nac_second_order.folder
    nac_second_order.folder = str(tmp_path)
    try:
        matrix = nacl_phonopy_debug_supercell_matrix()
        expected = np.arange(2 * 64 * 3 * 3, dtype=float).reshape(2, 64, 3, 3)

        def calculate_once(nac_bvk_supercell_matrix=None):
            np.testing.assert_array_equal(nac_bvk_supercell_matrix, matrix)
            return expected

        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            calculate_once,
        )
        first = nac_second_order.get_nac_short_range_force_constants(matrix)
        np.testing.assert_allclose(first, expected)

        def fail_if_loaded(*args, **kwargs):
            raise AssertionError("matrix-specific NAC array was not reused from memory")

        monkeypatch.setattr(nac_second_order, "_load_property", fail_if_loaded)
        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            fail_if_loaded,
        )
        second = nac_second_order.get_nac_short_range_force_constants(matrix)
        np.testing.assert_allclose(second, expected)
    finally:
        nac_second_order.folder = original_folder


def test_nac_velocity_shape_and_finite(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.0]),
        second=nac_second_order,
        storage="memory",
        nac_bvk_supercell_matrix=nacl_phonopy_debug_supercell_matrix_att3(),
    )
    velocity = phonon.velocity
    assert velocity.shape == (1, 6, 3)
    assert np.isfinite(velocity).all()


def test_dielectric_without_charges_raises():
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    second = forceconstants.second
    # These fixtures attach reference charges by hand and exercise the
    # machinery on the file constants as if they were totals.
    second.atoms.info.pop("dipole_subtracted_fc", None)
    if "charges" in second.atoms.arrays:
        del second.atoms.arrays["charges"]
    assert "dielectric" in second.atoms.info
    with pytest.raises(ValueError, match="charges"):
        HarmonicWithQ(q_point=np.zeros(3), second=second, storage="memory")

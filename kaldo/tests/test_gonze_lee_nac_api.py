import numpy as np
import pytest
from ase import units as ase_units

from kaldo.forceconstants import ForceConstants
from kaldo.observables.secondorder import bvk_supercell_matrix_key


def nacl_phonopy_debug_supercell_matrix():
    return np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]], dtype=int)


def nacl_phonopy_debug_supercell_matrix_att3():
    return np.diag([8, 8, 8]).astype(int)
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.phonons import Phonons
from kaldo.tests.test_gonze_lee_nac_helpers import require_nacl_debug


@pytest.fixture(scope="module")
def nac_second_order(tmp_path_factory):
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    forceconstants.second.folder = str(tmp_path_factory.mktemp("gonze_runtime_cache"))
    return forceconstants.second


def test_harmonic_with_q_accepts_nac_options(nac_second_order):
    matrix = nacl_phonopy_debug_supercell_matrix()
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
        nac_bvk_supercell_matrix=matrix,
        nac_q_direction=[1, 0, 0],
        q_index=7,
    )
    assert phonon.nac_method == "gonze"
    assert phonon.nac_debug is True
    assert phonon.nac_debug_folder == "debug"
    np.testing.assert_array_equal(phonon.nac_bvk_supercell_matrix, matrix)
    np.testing.assert_array_equal(phonon.nac_q_direction, np.array([1.0, 0.0, 0.0]))
    assert phonon.q_index == 7


def test_unknown_nac_method_raises_value_error(nac_second_order):
    with pytest.raises(ValueError, match="Unknown nac_method"):
        HarmonicWithQ(
            q_point=np.array([0.1, 0.0, 0.1]),
            second=nac_second_order,
            storage="memory",
            nac_method="bad-method",
        )


def test_gonze_velocity_returns_finite_array(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
    )
    velocity = phonon.velocity
    assert velocity.shape == (1, 6, 3)
    assert np.isfinite(velocity).all()


def test_phonons_stores_nac_options():
    matrix = nacl_phonopy_debug_supercell_matrix()
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[1, 1, 1],
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
        nac_bvk_supercell_matrix=matrix,
    )
    assert phonons.nac_method == "gonze"
    assert phonons.nac_debug is True
    assert phonons.nac_debug_folder == "debug"
    np.testing.assert_array_equal(phonons.nac_bvk_supercell_matrix, matrix)


def test_gonze_debug_folder_for_index(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
        q_index=3,
    )
    assert phonon._gonze_debug_q_folder().as_posix() == "debug/q-00003"


def test_gonze_debug_folder_for_single_q(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
    )
    assert phonon._gonze_debug_q_folder().as_posix() == "debug/q_0p1_0p0_0p1"


def test_gonze_static_data_contains_expected_nacl_shapes(nac_second_order, tmp_path):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder=str(tmp_path / "debug"),
        q_index=0,
    )
    data = phonon._build_gonze_static_data()
    assert data["born"].shape == (2, 3, 3)
    assert data["dielectric"].shape == (3, 3)
    assert data["primitive_cell"].shape == (3, 3)
    assert data["primitive_positions"].shape == (2, 3)
    assert data["reciprocal_lattice"].shape == (3, 3)
    assert data["masses"].shape == (2,)
    assert data["G_list"].ndim == 2
    assert data["G_list"].shape[1] == 3
    assert data["dd_q0"].shape == (2, 3, 3)
    assert data["dd_limiting"].shape == (3, 3)


def test_gonze_full_dynamical_matrix_returns_hermitian_matrix(nac_second_order, tmp_path):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder=str(tmp_path / "debug"),
        q_index=4,
    )
    dm = phonon._calculate_gonze_dynamical_matrix()
    assert dm.shape == (6, 6)
    np.testing.assert_allclose(dm, dm.conj().T, atol=1e-8, rtol=0.0)
    assert (tmp_path / "debug" / "q-00004" / "dm_final.npy").exists()


def test_gonze_frequency_calculation_returns_real_frequencies(nac_second_order, tmp_path):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder=str(tmp_path / "debug"),
        q_index=5,
    )
    frequency = phonon.frequency.flatten()
    assert frequency.shape == (6,)
    assert np.isfinite(frequency).all()
    assert (tmp_path / "debug" / "q-00005" / "frequencies.npy").exists()


def test_second_order_gonze_short_range_force_constants_use_lazy_numpy_cache(
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
            "calculate_gonze_short_range_force_constants",
            calculate_once,
        )
        np.testing.assert_allclose(
            nac_second_order.gonze_short_range_force_constants, expected
        )
        assert calls["count"] == 1
        assert (tmp_path / "gonze_short_range_force_constants.npy").exists()

        def fail_if_recomputed():
            raise AssertionError(
                "cached Gonze-Lee short-range force constants were not reused"
            )

        monkeypatch.setattr(
            nac_second_order,
            "calculate_gonze_short_range_force_constants",
            fail_if_recomputed,
        )
        np.testing.assert_allclose(
            nac_second_order.gonze_short_range_force_constants, expected
        )
    finally:
        nac_second_order.folder = original_folder


def test_second_order_gonze_short_range_force_constants_use_matrix_specific_cache(
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
            "calculate_gonze_short_range_force_constants",
            calculate_once,
        )
        actual = nac_second_order.get_gonze_short_range_force_constants(matrix)
        np.testing.assert_allclose(actual, expected)
        assert calls["count"] == 1

        property_name = (
            "gonze_short_range_force_constants_" + bvk_supercell_matrix_key(matrix)
        )
        assert (tmp_path / f"{property_name}.npy").exists()

        def fail_if_recomputed(nac_bvk_supercell_matrix=None):
            raise AssertionError("matrix-specific Gonze-Lee cache was not reused")

        monkeypatch.setattr(
            nac_second_order,
            "calculate_gonze_short_range_force_constants",
            fail_if_recomputed,
        )
        actual = nac_second_order.get_gonze_short_range_force_constants(matrix)
        np.testing.assert_allclose(actual, expected)
    finally:
        nac_second_order.folder = original_folder


@pytest.mark.parametrize("q_name", [
    "q-00000",
    "q-00013",
    "q-00011",
    "q-00020",
    "q-00023",
    "q-00025",
    "q-00030",
])
def test_gonze_nacl_frequencies_match_reference_with_reference_short_range(
    nac_second_order, tmp_path, monkeypatch, q_name
):
    debug_dir = require_nacl_debug()
    q_dir = debug_dir / q_name
    q_point = np.load(q_dir / "q_red.npy")
    q_index = int(q_name.split("-")[1])
    short_range_force_constants_ry = np.load(
        debug_dir / "static" / "short_range_force_constants.npy"
    )
    # Reference FCs are in Ry/Bohr²; KALDO expects eV/Å²
    short_range_force_constants = short_range_force_constants_ry * (
        ase_units.Rydberg / ase_units.Bohr ** 2
    )

    def use_reference_short_range(nac_bvk_supercell_matrix=None):
        return short_range_force_constants

    monkeypatch.setattr(
        nac_second_order,
        "get_gonze_short_range_force_constants",
        use_reference_short_range,
    )
    phonon = HarmonicWithQ(
        q_point=q_point,
        second=nac_second_order,
        storage="memory",
        is_unfolding=True,
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder=str(tmp_path / "debug"),
        nac_bvk_supercell_matrix=nacl_phonopy_debug_supercell_matrix_att3(),
        nac_q_direction=[1, 0, 0],
        q_index=q_index,
    )
    actual = phonon.frequency.flatten()
    expected = np.load(q_dir / "frequencies.npy")
    if q_name == "q-00000":
        np.testing.assert_allclose(actual[3:], expected[3:], rtol=0.02, atol=0.05)
    else:
        np.testing.assert_allclose(actual[:], expected[:], rtol=0.02, atol=0.05)

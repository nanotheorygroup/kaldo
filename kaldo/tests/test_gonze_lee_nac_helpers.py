import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.interfaces import shengbte_io
from kaldo.observables import harmonic_with_q as hwq
from kaldo.observables import secondorder as so
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.observables.secondorder import _gonze_get_commensurate_qpoints_diagonal
from kaldo.tests.gonze_debug_reference import (
    compare_tensors,
    load_q_tensor,
    load_static_tensor,
    require_nacl_att3_debug,
)


def load_v2_second_order_with_reference_nac():
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    _, _, charges = shengbte_io.read_second_order_qe_matrix("examples/nacl_phonopy/espresso.ifc2")
    forceconstants.second.atoms.info["dielectric"] = charges[0, :, :]
    forceconstants.second.atoms.set_array("charges", charges[1:, :, :], shape=(3, 3))
    return forceconstants.second


def test_att3_debug_tree_is_loadable():
    root = require_nacl_att3_debug()
    assert (root / "static" / "metadata.json").exists()
    np.testing.assert_array_equal(load_static_tensor(root, "supercell_matrix"), np.diag([8, 8, 8]))


def test_compare_tensors_reports_zero_diff_for_identical_arrays():
    arr = np.eye(3)
    diff = compare_tensors("eye", arr, arr)
    assert diff.name == "eye"
    assert diff.shape == (3, 3)
    assert diff.dtype == str(arr.dtype)
    assert diff.max_abs_diff == 0.0
    assert diff.rel_diff == 0.0


def test_compare_tensors_reports_nonzero_diff_for_different_arrays():
    actual = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[1.0, 2.0], [3.0, 5.0]])
    diff = compare_tensors("two_by_two", actual, expected)
    assert diff.name == "two_by_two"
    assert diff.shape == (2, 2)
    assert diff.dtype == str(actual.dtype)
    assert diff.max_abs_diff == pytest.approx(1.0)
    assert diff.rel_diff > 0.0


def test_compare_tensors_raises_for_shape_mismatch():
    actual = np.array([1.0, 2.0, 3.0])
    expected = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_tensors("bad_shapes", actual, expected)


def test_gonze_dielectric_part_matches_quadratic_form():
    vector = np.array([1.0, 2.0, -1.0])
    dielectric = np.diag([2.0, 3.0, 4.0])
    assert hwq._gonze_dielectric_part(vector, dielectric) == pytest.approx(18.0)


def test_gonze_multiply_borns_contracts_cartesian_axes():
    dd_in = np.zeros((1, 3, 1, 3), dtype=np.complex128)
    dd_in[0, :, 0, :] = np.arange(9, dtype=float).reshape(3, 3)
    born = np.zeros((1, 3, 3), dtype=float)
    born[0] = np.diag([2.0, 3.0, 5.0])
    actual = hwq._gonze_multiply_borns(dd_in, born)
    expected = np.zeros_like(actual)
    expected[0, :, 0, :] = born[0].T @ dd_in[0, :, 0, :] @ born[0]
    np.testing.assert_allclose(actual, expected)


def test_gonze_get_g_list_matches_nacl_debug_reference():
    debug_dir = require_nacl_att3_debug()
    reciprocal_lattice = load_static_tensor(debug_dir, "reciprocal_lattice")
    g_cutoff = float(load_static_tensor(debug_dir, "G_cutoff"))
    expected = load_static_tensor(debug_dir, "G_list")
    actual = hwq._gonze_get_g_list(reciprocal_lattice, g_cutoff)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.0)


def test_gonze_q0_and_limiting_terms_match_nacl_debug_reference():
    debug_dir = require_nacl_att3_debug()
    g_list = load_static_tensor(debug_dir, "G_list")
    born = load_static_tensor(debug_dir, "born")
    dielectric = load_static_tensor(debug_dir, "dielectric")
    positions = load_static_tensor(debug_dir, "primitive_positions")
    lambda_ = float(load_static_tensor(debug_dir, "Lambda"))
    tolerance = 1e-5
    actual_q0 = hwq._gonze_recip_dipole_dipole_q0(
        g_list, born, dielectric, positions, lambda_, tolerance
    )
    actual_limiting = hwq._gonze_limiting_dipole_dipole(dielectric, lambda_)
    np.testing.assert_allclose(actual_q0, load_static_tensor(debug_dir, "dd_q0"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        actual_limiting, load_static_tensor(debug_dir, "dd_limiting"), atol=1e-14, rtol=0.0
    )


def test_gonze_recip_real_and_mass_weight_terms_match_nacl_debug_reference():
    debug_dir = require_nacl_att3_debug()
    q_name = "q-00013"
    g_list = load_static_tensor(debug_dir, "G_list")
    born = load_static_tensor(debug_dir, "born")
    dielectric = load_static_tensor(debug_dir, "dielectric")
    positions = load_static_tensor(debug_dir, "primitive_positions")
    lambda_ = float(load_static_tensor(debug_dir, "Lambda"))
    tolerance = 1e-5
    nac_factor = float(load_static_tensor(debug_dir, "nac_factor"))
    q_red = load_q_tensor(debug_dir, q_name, "q_red")
    q_cart = load_q_tensor(debug_dir, q_name, "q_cart")
    q_direction_cart = load_q_tensor(debug_dir, q_name, "q_direction_cart")
    masses = load_static_tensor(debug_dir, "masses")
    svecs = load_static_tensor(debug_dir, "svecs")
    multi = load_static_tensor(debug_dir, "multi")
    s2pp_map = load_static_tensor(debug_dir, "s2pp_map")
    supercell_cell = load_static_tensor(debug_dir, "supercell_cell")

    recip_dd_q0 = np.zeros((len(masses), 3, 3), dtype=np.complex128)
    dd_recip = hwq._gonze_recip_dipole_dipole(
        recip_dd_q0,
        g_list,
        q_cart,
        q_direction_cart,
        born,
        dielectric,
        positions,
        nac_factor,
        lambda_,
        tolerance,
    )
    dd_real = hwq._gonze_real_dipole_dipole(
        q_red, svecs, multi, s2pp_map, dielectric, lambda_, supercell_cell
    )
    dd_total = load_q_tensor(debug_dir, q_name, "dd_total_mass_weighted").reshape(2, 3, 2, 3).copy()
    for i in range(len(masses)):
        for j in range(len(masses)):
            dd_total[i, :, j, :] *= np.sqrt(masses[i] * masses[j])
    mass_weighted = hwq._gonze_mass_weight(dd_total, masses)

    np.testing.assert_allclose(dd_recip, load_q_tensor(debug_dir, q_name, "dd_recip"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(dd_real, load_q_tensor(debug_dir, q_name, "dd_real"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        mass_weighted, load_q_tensor(debug_dir, q_name, "dd_total_mass_weighted"), atol=1e-12, rtol=0.0
    )


def test_gonze_short_range_dynamical_matrix_matches_debug_reference():
    debug_dir = require_nacl_att3_debug()
    q_name = "q-00013"
    actual = hwq._gonze_short_range_dynamical_matrix(
        load_static_tensor(debug_dir, "short_range_force_constants"),
        load_q_tensor(debug_dir, q_name, "q_red"),
        load_static_tensor(debug_dir, "svecs"),
        load_static_tensor(debug_dir, "multi"),
        load_static_tensor(debug_dir, "masses"),
        load_static_tensor(debug_dir, "s2p_map"),
        load_static_tensor(debug_dir, "p2s_map"),
    )
    np.testing.assert_allclose(actual, load_q_tensor(debug_dir, q_name, "dm_short"), atol=1e-10, rtol=0.0)


def test_att3_static_tensors_match_reference():
    debug_dir = require_nacl_att3_debug()
    second_order = load_v2_second_order_with_reference_nac()
    phonon = HarmonicWithQ(
        q_point=np.array([0.0, 0.0, 0.0]),
        second=second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=False,
    )
    static_data = phonon._build_gonze_static_data()

    np.testing.assert_allclose(static_data["primitive_cell"], load_static_tensor(debug_dir, "primitive_cell"))
    np.testing.assert_allclose(static_data["supercell_cell"], load_static_tensor(debug_dir, "supercell_cell"))
    np.testing.assert_allclose(
        static_data["reciprocal_lattice"], load_static_tensor(debug_dir, "reciprocal_lattice")
    )
    np.testing.assert_allclose(static_data["born"], load_static_tensor(debug_dir, "born"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        static_data["dielectric"], load_static_tensor(debug_dir, "dielectric"), atol=1e-9, rtol=0.0
    )
    np.testing.assert_allclose(static_data["masses"], load_static_tensor(debug_dir, "masses"), atol=1e-2, rtol=0.0)
    np.testing.assert_allclose(static_data["G_list"], load_static_tensor(debug_dir, "G_list"), atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(static_data["dd_q0"], load_static_tensor(debug_dir, "dd_q0"), atol=1e-9, rtol=0.0)


def test_att3_commensurate_qpoints_diagonal_match_expected_count_and_range():
    qpoints = _gonze_get_commensurate_qpoints_diagonal((8, 8, 8))
    assert qpoints.shape == (512, 3)
    assert np.all(qpoints <= 0.5 + 1e-12)
    assert np.all(qpoints >= -0.5 - 1e-12)
    assert np.any(np.all(np.isclose(qpoints, [0.0, 0.0, 0.0]), axis=1))
    assert np.any(np.all(np.isclose(qpoints, [0.5, 0.0, 0.5]), axis=1))


def test_att3_gonze_short_range_force_constants_shape_and_storage(tmp_path):
    second_order = load_v2_second_order_with_reference_nac()
    second_order.folder = str(tmp_path)
    actual = second_order.gonze_short_range_force_constants
    assert actual.shape == (2, 1024, 3, 3)
    assert np.isfinite(actual).all()
    full_fc = so._gonze_build_full_fc_compact(second_order)
    assert not np.allclose(actual, full_fc)
    assert (tmp_path / "gonze_short_range_force_constants.npy").exists()


def test_att3_gonze_short_range_force_constants_cache_round_trip(tmp_path, monkeypatch):
    second_order = load_v2_second_order_with_reference_nac()
    second_order.folder = str(tmp_path)
    expected = second_order.gonze_short_range_force_constants
    monkeypatch.setattr(
        so,
        "_calculate_gonze_short_range_force_constants",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should load from cache")),
    )
    loaded = second_order.gonze_short_range_force_constants
    np.testing.assert_allclose(loaded, expected, atol=0.0, rtol=0.0)
    assert (tmp_path / "gonze_short_range_force_constants.npy").exists()


def test_gonze_short_range_force_constants_rejects_nondiagonal_supercell(tmp_path):
    second_order = load_v2_second_order_with_reference_nac()
    second_order.folder = str(tmp_path)
    second_order.atoms.info["gonze_nac_supercell_matrix"] = np.array(
        [[8, 1, 0], [0, 8, 0], [0, 0, 8]],
        dtype=int,
    )
    with pytest.raises(NotImplementedError, match="diagonal supercell matrices only"):
        _ = second_order.gonze_short_range_force_constants

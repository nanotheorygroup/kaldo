from pathlib import Path

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.interfaces import shengbte_io
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.tests.gonze_debug_reference import (
    diagnostic_q_names_att3,
    format_tensor_diff,
    load_velocity_direction_tensor,
    load_velocity_json,
    load_velocity_q_tensor,
    require_nacl_att3_velocity_debug,
)


def attach_reference_nac(second_order, nac_file="examples/nacl_phonopy_v2/espresso.ifc2"):
    _, _, charges = shengbte_io.read_second_order_qe_matrix(nac_file)
    if charges is None:
        static_dir = Path(nac_file).parent / "debug" / "static"
        dielectric = np.load(static_dir / "dielectric.npy")
        born = np.load(static_dir / "born.npy")
    else:
        dielectric = charges[0, :, :]
        born = charges[1:, :, :]
    second_order.atoms.info["dielectric"] = dielectric
    second_order.atoms.set_array("charges", born, shape=(3, 3))
    return second_order


@pytest.fixture(scope="module")
def nac_second_order(tmp_path_factory):
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    forceconstants.second.folder = str(tmp_path_factory.mktemp("gonze_velocity_cache"))
    return attach_reference_nac(forceconstants.second)


TOP_LEVEL_TENSOR_NAMES = [
    "q_red",
    "q_cart",
    "dm_q",
    "eigenvalues",
    "frequencies",
]

# Known failures due to upstream QE force-constant mismatch vs att3 reference.
_UPSTREAM_FC_XFAIL = {
    ("dm_q", "q-00013"),
    ("dm_q", "q-00020"),
    ("eigenvalues", "q-00013"),
    ("eigenvalues", "q-00020"),
    ("eigenvalues", "q-00030"),
}


def _run_gonze_velocity_debug_for_q_name(nac_second_order, q_name, debug_root, out_root):
    q_point = load_velocity_q_tensor(debug_root, q_name, "q_red")
    q_index = int(q_name.split("-")[1])
    phonon = HarmonicWithQ(
        q_point=q_point,
        second=nac_second_order,
        storage="memory",
        is_unfolding=True,
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder=str(out_root),
        q_index=q_index,
    )
    _ = phonon._calculate_gonze_velocity_debug_data()
    return out_root / q_name


def test_gonze_velocity_debug_data_contains_top_level_fields(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        is_unfolding=True,
        nac_method="gonze",
    )
    data = phonon._calculate_gonze_velocity_debug_data()
    assert data["dm_q"].shape == (6, 6)
    assert data["eigenvectors"].shape == (6, 6)
    assert sorted(data["directions"]) == ["d0", "d1", "d2", "d3"]


@pytest.mark.parametrize("q_name", diagnostic_q_names_att3())
@pytest.mark.parametrize("tensor_name", TOP_LEVEL_TENSOR_NAMES)
def test_gonze_velocity_top_level_tensors_match_phonopy_debug(
    nac_second_order, tmp_path, q_name, tensor_name
):
    debug_dir = require_nacl_att3_velocity_debug()
    actual_root = _run_gonze_velocity_debug_for_q_name(
        nac_second_order, q_name, debug_dir, tmp_path / "debug-velocity"
    )
    actual = np.load(actual_root / f"{tensor_name}.npy", allow_pickle=False)
    expected = load_velocity_q_tensor(debug_dir, q_name, tensor_name)
    if (tensor_name, q_name) in _UPSTREAM_FC_XFAIL:
        pytest.xfail("Upstream QE force-constant mismatch vs att3 reference.")
    rtol = 0.02
    if tensor_name == "frequencies":
        if q_name == "q-00000":
            pytest.xfail("Gamma frequency parity remains unresolved in the Gonze velocity path.")
        rtol = 0.03
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=1e-8,
        err_msg=format_tensor_diff(tensor_name, q_name, actual, expected),
    )


@pytest.mark.parametrize("q_name", diagnostic_q_names_att3())
def test_gonze_velocity_json_payloads_match_phonopy_debug(nac_second_order, tmp_path, q_name):
    if q_name in {"q-00013", "q-00020"}:
        pytest.xfail("Upstream FC mismatch causes wrong eigenvalues and wrong degenerate_sets grouping.")
    debug_dir = require_nacl_att3_velocity_debug()
    actual_root = _run_gonze_velocity_debug_for_q_name(
        nac_second_order, q_name, debug_dir, tmp_path / "debug-velocity"
    )
    assert load_velocity_json(actual_root.parent, q_name, "degenerate_sets") == load_velocity_json(
        debug_dir, q_name, "degenerate_sets"
    )
    actual_nac = load_velocity_json(actual_root.parent, q_name, "nac_branch")
    expected_nac = load_velocity_json(debug_dir, q_name, "nac_branch")
    assert actual_nac["nac_applied"] == expected_nac["nac_applied"]
    assert actual_nac["q_direction_red"] == expected_nac["q_direction_red"]
    assert actual_nac["tolerance"] == expected_nac["tolerance"]
    np.testing.assert_allclose(actual_nac["q_norm"], expected_nac["q_norm"], rtol=1e-6)
    if expected_nac["q_red"] is not None:
        np.testing.assert_allclose(actual_nac["q_red"], expected_nac["q_red"], rtol=1e-6, atol=1e-15)

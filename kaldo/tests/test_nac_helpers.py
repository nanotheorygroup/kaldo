from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase import units as ase_units

from kaldo.forceconstants import ForceConstants
from kaldo.grid import SupercellGrid, TranslationSupport
from kaldo.interfaces import qe_io
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import kaldo.controllers.nac as so
from kaldo.controllers.nac import (
    _short_range_dynamical_matrix,
    _build_interleaved_fc,
)


def format_tensor_diff(name, label, actual, expected):
    return f"{name} mismatch at {label}"


def test_interleaved_fc_maps_c_storage_to_compact_translation_order():
    """Exact quotient ids must not be confused with Phonopy compact slots."""
    shape = (2, 3, 4)
    grid = SupercellGrid(np.diag(shape), order="C")
    support = TranslationSupport.periodic(grid, order="C")
    n_replicas = grid.size
    value = np.zeros((1, 1, 3, n_replicas, 1, 3), dtype=np.float64)
    for replica_id in range(n_replicas):
        value[0, 0, :, replica_id, 0, :] = np.eye(3) * (replica_id + 1)
    second = SimpleNamespace(
        value=value,
        atoms=Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True),
        supercell_grid=grid,
        translation_support=support,
    )

    actual = _build_interleaved_fc(second)[0, :, 0, 0]
    expected = np.empty(n_replicas, dtype=np.float64)
    for compact_id in range(n_replicas):
        compact_translation = np.array(
            np.unravel_index(compact_id, shape, order="F")
        )
        source_translation = np.mod(-compact_translation, shape)
        source_id = np.ravel_multi_index(source_translation, shape, order="C")
        expected[compact_id] = source_id + 1

    np.testing.assert_array_equal(actual, expected)


def _native_pair_gauge_dynamical_matrix(second_order, q_red):
    """Return the ordinary IFC transform in the WS atom-pair phase gauge."""
    harmonic = HarmonicWithQ(q_point=q_red, second=second_order, storage="memory")
    native = np.asarray(harmonic.calculate_dynmat_fourier())
    atom_count = len(second_order.atoms)
    blocks = native.reshape(atom_count, 3, atom_count, 3)
    scaled_positions = second_order.atoms.get_scaled_positions(wrap=False)
    pair_displacements = (
        scaled_positions[np.newaxis, :, :] - scaled_positions[:, np.newaxis, :]
    )
    phases = np.exp(2j * np.pi * np.einsum("a,ija->ij", q_red, pair_displacements))
    return (blocks * phases[:, np.newaxis, :, np.newaxis]).reshape(
        3 * atom_count, 3 * atom_count
    )


def nacl_phonopy_debug_supercell_matrix_att3():
    return np.diag([8, 8, 8]).astype(int)


def load_att3_v2_second_order_with_reference_nac(storage_folder) -> object:
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    _, _, charges = qe_io.read_second_order_qe_matrix(
        "kaldo/tests/nacl_phonopy/espresso.ifc2"
    )
    forceconstants.second.atoms.info["dielectric"] = charges[0, :, :]
    forceconstants.second.atoms.set_array("charges", charges[1:, :, :], shape=(3, 3))
    forceconstants.second.folder = str(storage_folder)
    return forceconstants.second


def test_nac_dielectric_part_matches_quadratic_form():
    vector = np.array([1.0, 2.0, -1.0])
    dielectric = np.diag([2.0, 3.0, 4.0])
    assert so._dielectric_part(vector, dielectric) == pytest.approx(18.0)


def test_nac_multiply_borns_contracts_cartesian_axes():
    dd_in = np.zeros((1, 3, 1, 3), dtype=np.complex128)
    dd_in[0, :, 0, :] = np.arange(9, dtype=float).reshape(3, 3)
    born = np.zeros((1, 3, 3), dtype=float)
    born[0] = np.diag([2.0, 3.0, 5.0])
    actual = so._multiply_borns(dd_in, born)
    expected = np.zeros_like(actual)
    expected[0, :, 0, :] = born[0].T @ dd_in[0, :, 0, :] @ born[0]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)


def test_nac_controller_rejects_unknown_data_convention():
    static_data = {"convention": "unknown"}

    with pytest.raises(ValueError, match="unknown NAC data convention 'unknown'"):
        so.ensure_kernel_cache(static_data, {})

    with pytest.raises(ValueError, match="unknown NAC data convention 'unknown'"):
        so.dynamical_matrices(
            q_reds=[[0.0, 0.0, 0.0]],
            static_data=static_data,
            mapping={},
            q_direction_carts=[[1.0, 0.0, 0.0]],
        )


@pytest.mark.parametrize(
    "q_red",
    [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.125, 0.0, 0.0]),
        np.array([0.125, 0.125, 0.125]),
    ],
)
def test_matrix_specific_total_dynamical_matrix_matches_input_force_constants_forward_transform(
    q_red, tmp_path
):
    second_order = load_att3_v2_second_order_with_reference_nac(tmp_path)
    matrix = nacl_phonopy_debug_supercell_matrix_att3()
    mapping = second_order._build_nac_mapping(matrix)
    actual = _native_pair_gauge_dynamical_matrix(second_order, q_red)
    expected = _short_range_dynamical_matrix(
        _build_interleaved_fc(second_order)
        * (ase_units.mol / (10 * ase_units.J)),
        q_red,
        mapping["phase_svecs"],
        mapping["multi"],
        second_order.atoms.get_masses(),
        mapping["s2p_map"],
        mapping["p2s_map"],
    )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-10,
        rtol=1e-10,
        err_msg=format_tensor_diff("dm_total_from_input_fc", str(q_red.tolist()), actual, expected),
    )

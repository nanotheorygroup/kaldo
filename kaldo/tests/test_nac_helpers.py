from pathlib import Path
import os

import h5py
import numpy as np
import pytest
import ase.io
from ase import units as ase_units

from kaldo.forceconstants import ForceConstants
from kaldo.interfaces import shengbte_io
from kaldo.observables import harmonic_with_q as hwq
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import kaldo.controllers.nac as so
from kaldo.observables.secondorder import _dynamical_matrix_from_second_order
from kaldo.controllers.nac import (
    _short_range_dynamical_matrix,
    _mass_weight,
    _build_supercell_matrix_mapping as build_supercell_matrix_mapping,
    _commensurate_points as commensurate_points,
    _build_interleaved_fc,
)


def format_tensor_diff(name, label, actual, expected):
    return f"{name} mismatch at {label}"



def nacl_phonopy_debug_supercell_matrix():
    return np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]], dtype=int)


def nacl_phonopy_debug_supercell_matrix_att3():
    return np.diag([8, 8, 8]).astype(int)


def summarize_att3_fc_deltas(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = actual - expected
    n_atom = expected.shape[0]
    n_replicas = expected.shape[1] // n_atom
    atom_index = np.arange(expected.shape[1]) // n_replicas
    replica_index = np.arange(expected.shape[1]) % n_replicas
    onsite_mask = replica_index == 0
    same_type_masks = [atom_index == atom for atom in range(n_atom)]

    return {
        "onsite_na": float(np.max(np.abs(delta[0, onsite_mask & same_type_masks[0]]))),
        "onsite_cl": float(np.max(np.abs(delta[1, onsite_mask & same_type_masks[1]]))),
        "offsite_same_type": float(
            max(
                np.max(np.abs(delta[0, ~onsite_mask & same_type_masks[0]])),
                np.max(np.abs(delta[1, ~onsite_mask & same_type_masks[1]])),
            )
        ),
        "cross_type": float(
            max(
                np.max(np.abs(delta[0, same_type_masks[1]])),
                np.max(np.abs(delta[1, same_type_masks[0]])),
            )
        ),
        "overall": float(np.max(np.abs(delta))),
    }


def load_att3_second_order(storage_folder) -> object:
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    forceconstants.second.folder = str(storage_folder)
    return forceconstants.second


def load_att3_v2_second_order_with_reference_nac(storage_folder) -> object:
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    _, _, charges = shengbte_io.read_second_order_qe_matrix(
        "kaldo/tests/nacl_phonopy/espresso.ifc2"
    )
    forceconstants.second.atoms.info["dielectric"] = charges[0, :, :]
    forceconstants.second.atoms.set_array("charges", charges[1:, :, :], shape=(3, 3))
    forceconstants.second.folder = str(storage_folder)
    return forceconstants.second


def load_att4_v2_second_order_with_reference_nac(storage_folder) -> object:
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    _, _, charges = shengbte_io.read_second_order_qe_matrix(
        "kaldo/tests/nacl_phonopy/espresso.ifc2"
    )
    forceconstants.second.atoms.info["dielectric"] = charges[0, :, :]
    forceconstants.second.atoms.set_array("charges", charges[1:, :, :], shape=(3, 3))
    forceconstants.second.folder = str(storage_folder)
    return forceconstants.second


def first_meaningful_stage_difference(
    actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> tuple[str | None, float]:
    order = [
        "q_red",
        "q_cart",
        "q_direction_cart",
        "dm_short",
        "dd_recip",
        "dd_real",
        "dd_total_mass_weighted",
        "dm_final",
        "eigenvalues",
        "frequencies",
    ]
    tolerances = {
        "q_red": (0.0, 1e-14),
        "q_cart": (0.0, 1e-12),
        "q_direction_cart": (0.0, 1e-12),
        "dm_short": (0.0, 1e-10),
        "dd_recip": (0.0, 1e-10),
        "dd_real": (0.0, 1e-10),
        "dd_total_mass_weighted": (0.0, 1e-10),
        "dm_final": (0.0, 1e-10),
        "eigenvalues": (0.0, 1e-10),
        "frequencies": (0.0, 1e-8),
    }
    for name in order:
        rtol, atol = tolerances[name]
        if name == "q_direction_cart":
            actual_norm = np.linalg.norm(actual[name])
            expected_norm = np.linalg.norm(expected[name])
            if actual_norm > 0.0 and expected_norm > 0.0:
                actual_dir = actual[name] / actual_norm
                expected_dir = expected[name] / expected_norm
                if np.allclose(actual_dir, expected_dir, rtol=0.0, atol=1e-12):
                    continue
        if not np.allclose(actual[name], expected[name], rtol=rtol, atol=atol):
            return name, float(np.max(np.abs(actual[name] - expected[name])))
    return None, 0.0


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
    np.testing.assert_allclose(actual, expected)


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
    actual = _dynamical_matrix_from_second_order(second_order, q_red)
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

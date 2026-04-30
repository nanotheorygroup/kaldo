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
from kaldo.observables import gonze_lee_nac as gln
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.observables.gonze_lee_nac import (
    build_supercell_matrix_mapping,
    commensurate_points,
    nacl_phonopy_debug_supercell_matrix,
    nacl_phonopy_debug_supercell_matrix_att3,
)
from kaldo.tests.gonze_debug_reference import (
    compare_tensors,
    format_tensor_diff,
    load_velocity_direction_tensor,
    load_velocity_json,
    load_velocity_q_tensor,
    require_nacl_att3_velocity_debug,
)


DEFAULT_NACL_DEBUG = Path(
    "/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replay/"
    "example/nacl-att3/debug"
)
DEFAULT_NACL_ATT4_DEBUG = Path(
    "/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replace/"
    "example/nacl-att4/debug"
)
DEFAULT_NACL_ATT4_DEBUG_FALLBACK = Path(
    "/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replay/"
    "example/nacl-att4/debug"
)
LOCAL_V2_DEBUG = Path("examples/nacl_phonopy_v2/debug")


def nacl_debug_dir() -> Path:
    return Path(os.environ.get("NACL_ATT3_DEBUG_DIR", DEFAULT_NACL_DEBUG))


def require_nacl_debug() -> Path:
    path = nacl_debug_dir()
    if not (path / "static" / "metadata.json").exists():
        pytest.skip(f"NaCl Gonze-Lee debug tree not found at {path}")
    return path


def nacl_att4_debug_dir() -> Path:
    env_override = os.environ.get("NACL_ATT4_DEBUG_DIR")
    if env_override:
        return Path(env_override)
    if DEFAULT_NACL_ATT4_DEBUG.exists():
        return DEFAULT_NACL_ATT4_DEBUG
    return DEFAULT_NACL_ATT4_DEBUG_FALLBACK


def require_nacl_att4_debug() -> Path:
    path = nacl_att4_debug_dir()
    if not (path / "static" / "metadata.json").exists():
        pytest.skip(f"NaCl att4 debug tree not found at {path}")
    return path


def gather_att4_q_dirs(debug_root: Path) -> list[Path]:
    expected = [debug_root / f"q-{index:05d}" for index in range(71)]
    missing = [path.name for path in expected if not path.is_dir()]
    if missing:
        pytest.skip(f"NaCl att4 debug tree is incomplete at {debug_root}: {missing}")
    return expected


def classify_first_frequency_failure(
    actual_path: np.ndarray, expected_path: np.ndarray
) -> dict[str, object]:
    for index, (actual, expected) in enumerate(zip(actual_path, expected_path)):
        if not np.allclose(actual, expected, rtol=1e-8, atol=1e-8):
            abs_diff = np.abs(actual - expected)
            rel_diff = abs_diff / np.maximum(np.abs(expected), 1e-30)
            return {
                "index": index,
                "max_abs": float(np.max(abs_diff)),
                "max_rel": float(np.max(rel_diff)),
                "actual": actual,
                "expected": expected,
            }
    return {
        "index": None,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "actual": None,
        "expected": None,
    }


def require_nacl_debug_worktree() -> Path:
    path = nacl_debug_dir().parent / "debug_worktree"
    required = [
        "primitive_matrix.npy",
        "supercell_matrix.npy",
        "primitive_scaled_positions.npy",
        "supercell_scaled_positions.npy",
        "p2p_map.npy",
        "svecs_cart.npy",
        "multi_counts.npy",
        "multi_offsets.npy",
    ]
    if not all((path / "static" / name).exists() for name in required):
        pytest.skip(f"NaCl Gonze-Lee mapping debug tree not found at {path}")
    return path


def require_local_v2_debug() -> Path:
    path = LOCAL_V2_DEBUG
    if not (path / "static" / "short_range_force_constants.npy").exists():
        pytest.skip(f"Local NaCl att3 debug tree not found at {path}")
    return path


def load_att3_phonopy_force_constants() -> np.ndarray:
    path = nacl_debug_dir().parent / "force_constants.hdf5"
    if not path.exists():
        pytest.skip(f"Phonopy compact FC reference not found at {path}")
    with h5py.File(path, "r") as handle:
        return np.array(handle["force_constants"], dtype=float)


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
        folder="examples/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    forceconstants.second.folder = str(storage_folder)
    return forceconstants.second


def load_att3_v2_second_order_with_reference_nac(storage_folder) -> object:
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    _, _, charges = shengbte_io.read_second_order_qe_matrix(
        "examples/nacl_phonopy/espresso.ifc2"
    )
    forceconstants.second.atoms.info["dielectric"] = charges[0, :, :]
    forceconstants.second.atoms.set_array("charges", charges[1:, :, :], shape=(3, 3))
    forceconstants.second.folder = str(storage_folder)
    return forceconstants.second


def load_att4_v2_second_order_with_reference_nac(storage_folder) -> object:
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    _, _, charges = shengbte_io.read_second_order_qe_matrix(
        "examples/nacl_phonopy/espresso.ifc2"
    )
    forceconstants.second.atoms.info["dielectric"] = charges[0, :, :]
    forceconstants.second.atoms.set_array("charges", charges[1:, :, :], shape=(3, 3))
    forceconstants.second.folder = str(storage_folder)
    return forceconstants.second


def collect_att4_gx_frequencies(
    second_order, debug_root: Path
) -> tuple[np.ndarray, np.ndarray]:
    actual = []
    expected = []
    for q_index, q_dir in enumerate(gather_att4_q_dirs(debug_root)):
        q_red = np.load(q_dir / "q_red.npy", allow_pickle=False)
        q_direction_red = np.load(q_dir / "q_direction_red.npy", allow_pickle=False)
        expected_freqs = np.load(q_dir / "frequencies.npy", allow_pickle=False)
        phonon = HarmonicWithQ(
            q_point=q_red,
            second=second_order,
            storage="memory",
            is_unfolding=True,
            nac_method="gonze",
            nac_bvk_supercell_matrix=nacl_phonopy_debug_supercell_matrix_att3(),
            nac_q_direction=q_direction_red,
            q_index=q_index,
        )
        actual.append(phonon.frequency.flatten())
        expected.append(expected_freqs)
    return np.array(actual), np.array(expected)


def collect_kaldo_stage_data(
    second_order, q_red: np.ndarray, q_index: int, nac_q_direction: np.ndarray
) -> dict[str, np.ndarray]:
    phonon = HarmonicWithQ(
        q_point=q_red,
        second=second_order,
        storage="memory",
        is_unfolding=True,
        nac_method="gonze",
        nac_debug=False,
        nac_bvk_supercell_matrix=nacl_phonopy_debug_supercell_matrix_att3(),
        nac_q_direction=nac_q_direction,
        q_index=q_index,
    )
    static_data = phonon._build_gonze_static_data()
    mapping = phonon._build_gonze_short_range_inputs(static_data)
    masses = static_data["masses"]
    q_red = np.array(q_red, dtype=float, copy=True)
    q_cart = static_data["reciprocal_lattice"] @ q_red
    if np.linalg.norm(q_cart) >= static_data["q_direction_tolerance"]:
        q_direction_cart = q_cart
    else:
        q_direction_cart = static_data["reciprocal_lattice"] @ phonon.nac_q_direction

    recip_dd_q0 = np.zeros_like(static_data["dd_q0"])
    dd_recip = hwq._gonze_recip_dipole_dipole(
        recip_dd_q0,
        static_data["G_list"],
        q_cart,
        q_direction_cart,
        static_data["born"],
        static_data["dielectric"],
        static_data["primitive_positions"],
        float(static_data["nac_factor"]),
        float(static_data["Lambda"]),
        float(static_data["q_direction_tolerance"]),
    )
    dd_real = hwq._gonze_real_dipole_dipole(
        q_red,
        mapping["svecs"],
        mapping["multi"],
        mapping["s2pp_map"],
        static_data["dielectric"],
        float(static_data["Lambda"]),
        mapping.get("svecs_cell", static_data["supercell_cell"]),
    )
    dd_limiting_expanded = np.zeros_like(dd_recip)
    for i in range(len(masses)):
        dd_limiting_expanded[i, :, i, :] = static_data["dd_limiting"]
    dd_real_q0_full = hwq._gonze_real_dipole_dipole(
        np.zeros(3, dtype=float),
        mapping["svecs"],
        mapping["multi"],
        mapping["s2pp_map"],
        static_data["dielectric"],
        float(static_data["Lambda"]),
        mapping.get("svecs_cell", static_data["supercell_cell"]),
    )
    dd_real_q0 = dd_real_q0_full.sum(axis=2)
    dd_drift = (
        static_data["dd_q0"] * float(ase_units.Rydberg / ase_units.Bohr ** 2)
        + static_data["dd_limiting"] * len(masses)
        + dd_real_q0
    )
    dd_total = dd_recip + dd_limiting_expanded + dd_real
    for i in range(len(masses)):
        dd_total[i, :, i, :] -= dd_drift[i]
    conversion = ase_units.mol / (10 * ase_units.J)
    dd_total_mass_weighted = hwq._gonze_mass_weight(dd_total * conversion, masses)

    fc_short = second_order.get_gonze_short_range_force_constants(
        nacl_phonopy_debug_supercell_matrix_att3()
    )
    dm_short = hwq._gonze_short_range_dynamical_matrix(
        fc_short * conversion,
        q_red,
        mapping.get("phase_svecs", mapping["svecs"]),
        mapping["multi"],
        masses,
        mapping["s2p_map"],
        mapping["p2s_map"],
    )
    dm_final = dm_short + dd_total_mass_weighted
    dm_final = (dm_final + dm_final.conj().T) / 2
    eigenvalues = np.linalg.eigvalsh(dm_final).real
    frequencies = np.abs(eigenvalues) ** 0.5 * np.sign(eigenvalues) / (2 * np.pi)
    return {
        "q_red": q_red,
        "q_cart": q_cart,
        "q_direction_cart": q_direction_cart,
        "dm_short": dm_short,
        "dd_recip": dd_recip,
        "dd_real": dd_real,
        "dd_total_mass_weighted": dd_total_mass_weighted,
        "dm_final": dm_final,
        "eigenvalues": eigenvalues,
        "frequencies": frequencies,
    }


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


def input_force_constants_compact(second_order) -> np.ndarray:
    n_atom = len(second_order.atoms)
    n_replicas = int(np.prod(second_order.supercell))
    force_constants = np.array(second_order.value[0], dtype=float)
    force_constants = force_constants.transpose(0, 2, 3, 1, 4)
    force_constants = force_constants.reshape(n_atom, n_replicas * n_atom, 3, 3)
    permutation = np.concatenate(
        [
            np.arange(atom_j, n_replicas * n_atom, n_atom, dtype=int)
            for atom_j in range(n_atom)
        ]
    )
    return force_constants[:, permutation]


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
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    reciprocal_lattice = np.load(static / "reciprocal_lattice.npy")
    g_cutoff = float(np.load(static / "G_cutoff.npy"))
    expected = np.load(static / "G_list.npy")
    actual = hwq._gonze_get_g_list(reciprocal_lattice, g_cutoff)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.0)


def test_gonze_q0_and_limiting_terms_match_nacl_debug_reference():
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    g_list = np.load(static / "G_list.npy")
    born = np.load(static / "born.npy")
    dielectric = np.load(static / "dielectric.npy")
    positions = np.load(static / "primitive_positions.npy")
    lambda_ = float(np.load(static / "Lambda.npy"))
    tolerance = 1e-5
    actual_q0 = hwq._gonze_recip_dipole_dipole_q0(
        g_list, born, dielectric, positions, lambda_, tolerance
    )
    actual_limiting = hwq._gonze_limiting_dipole_dipole(dielectric, lambda_)
    np.testing.assert_allclose(actual_q0, np.load(static / "dd_q0.npy"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        actual_limiting, np.load(static / "dd_limiting.npy"), atol=1e-14, rtol=0.0
    )


def test_gonze_recip_real_and_mass_weight_terms_match_nacl_debug_reference():
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    q_dir = debug_dir / "q-00013"
    g_list = np.load(static / "G_list.npy")
    born = np.load(static / "born.npy")
    dielectric = np.load(static / "dielectric.npy")
    positions = np.load(static / "primitive_positions.npy")
    lambda_ = float(np.load(static / "Lambda.npy"))
    tolerance = 1e-5
    nac_factor = float(np.load(static / "nac_factor.npy"))
    q_red = np.load(q_dir / "q_red.npy")
    q_cart = np.load(q_dir / "q_cart.npy")
    q_direction_cart = np.load(q_dir / "q_direction_cart.npy")
    masses = np.load(static / "masses.npy")
    svecs = np.load(static / "svecs.npy")
    multi = np.load(static / "multi.npy")
    s2pp_map = np.load(static / "s2pp_map.npy")
    supercell_cell = np.load(static / "supercell_cell.npy")

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
    dd_total = np.load(q_dir / "dd_total_mass_weighted.npy").reshape(2, 3, 2, 3).copy()
    for i in range(len(masses)):
        for j in range(len(masses)):
            dd_total[i, :, j, :] *= np.sqrt(masses[i] * masses[j])
    mass_weighted = hwq._gonze_mass_weight(dd_total, masses)

    np.testing.assert_allclose(dd_recip, np.load(q_dir / "dd_recip.npy"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(dd_real, np.load(q_dir / "dd_real.npy"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        mass_weighted, np.load(q_dir / "dd_total_mass_weighted.npy"), atol=1e-12, rtol=0.0
    )


def test_gonze_short_range_dynamical_matrix_matches_debug_reference():
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    q_dir = debug_dir / "q-00013"
    actual = hwq._gonze_short_range_dynamical_matrix(
        np.load(static / "short_range_force_constants.npy"),
        np.load(q_dir / "q_red.npy"),
        np.load(static / "svecs.npy"),
        np.load(static / "multi.npy"),
        np.load(static / "masses.npy"),
        np.load(static / "s2p_map.npy"),
        np.load(static / "p2s_map.npy"),
    )
    np.testing.assert_allclose(actual, np.load(q_dir / "dm_short.npy"), atol=1e-10, rtol=0.0)


def test_v2_force_constants_are_closer_to_att3_short_range_reference_than_legacy_input(tmp_path):
    debug_dir = require_nacl_debug()
    matrix = nacl_phonopy_debug_supercell_matrix_att3()
    expected = np.load(debug_dir / "static" / "short_range_force_constants.npy") * (
        ase_units.Rydberg / ase_units.Bohr ** 2
    )

    legacy_second = load_att3_second_order(tmp_path / "legacy")
    legacy_actual = legacy_second.get_gonze_short_range_force_constants(matrix)
    legacy_diff = compare_tensors(
        "legacy_short_range_force_constants", legacy_actual, expected
    )

    v2_second = load_att3_v2_second_order_with_reference_nac(tmp_path / "v2")
    v2_actual = v2_second.get_gonze_short_range_force_constants(matrix)
    v2_diff = compare_tensors("v2_short_range_force_constants", v2_actual, expected)

    assert v2_diff.rel_diff < legacy_diff.rel_diff, (
        f"legacy rel_diff={legacy_diff.rel_diff:.8e}, "
        f"v2 rel_diff={v2_diff.rel_diff:.8e}"
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
    mapping = gln.build_short_range_inputs(second_order, matrix)
    actual = gln.dynamical_matrix_from_second_order(second_order, q_red)
    expected = hwq._gonze_short_range_dynamical_matrix(
        input_force_constants_compact(second_order)
        * (ase_units.mol / (10 * ase_units.J)),
        q_red,
        mapping["svecs"],
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


@pytest.mark.parametrize("q_name", ["q-00000", "q-00013", "q-00020", "q-00030"])
def test_reconstructed_short_range_force_constants_reproduce_att3_dm_short(q_name, tmp_path):
    debug_dir = require_nacl_debug()
    second_order = load_att3_v2_second_order_with_reference_nac(tmp_path)
    fc_short = second_order.get_gonze_short_range_force_constants(
        nacl_phonopy_debug_supercell_matrix_att3()
    )
    fc_short_phonopy_units = fc_short / (ase_units.Rydberg / ase_units.Bohr ** 2)
    q_dir = debug_dir / q_name
    actual = hwq._gonze_short_range_dynamical_matrix(
        fc_short_phonopy_units,
        np.load(q_dir / "q_red.npy"),
        np.load(debug_dir / "static" / "svecs.npy"),
        np.load(debug_dir / "static" / "multi.npy"),
        np.load(debug_dir / "static" / "masses.npy"),
        np.load(debug_dir / "static" / "s2p_map.npy"),
        np.load(debug_dir / "static" / "p2s_map.npy"),
    )
    expected = np.load(q_dir / "dm_short.npy")
    # The att3 roundtrip applies kALDo's ASR-adjusted FCs, which injects small
    # symmetry-breaking noise at non-commensurate q that is not present in raw
    # phonopy FCs.
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.02,
        atol=3e-5,
        err_msg=format_tensor_diff("dm_short", q_name, actual, expected),
    )


def test_att3_interleaved_fc_diagnostic_matches_phonopy_reference(tmp_path):
    second_order = load_att3_v2_second_order_with_reference_nac(tmp_path)
    actual = gln._build_interleaved_fc(second_order) / (
        ase_units.Rydberg / ase_units.Bohr ** 2
    )
    expected = load_att3_phonopy_force_constants()
    summary = summarize_att3_fc_deltas(actual, expected)
    q_red = np.array([0.21666666666666667, 0.0, 0.21666666666666667])
    mapping = gln.build_short_range_inputs(
        second_order, nacl_phonopy_debug_supercell_matrix_att3()
    )
    dm_from_delta = hwq._gonze_short_range_dynamical_matrix(
        actual - expected,
        q_red,
        mapping.get("phase_svecs", mapping["svecs"]),
        mapping["multi"],
        second_order.atoms.get_masses(),
        mapping["s2p_map"],
        mapping["p2s_map"],
    )
    assert summary["onsite_na"] > summary["offsite_same_type"]
    assert summary["onsite_cl"] > summary["cross_type"]
    assert np.max(np.abs(dm_from_delta)) < 2e-5


def test_att4_gx_path_reports_first_frequency_failure(tmp_path):
    debug_root = require_nacl_att4_debug()
    second_order = load_att4_v2_second_order_with_reference_nac(tmp_path)
    actual, expected = collect_att4_gx_frequencies(second_order, debug_root)
    first = classify_first_frequency_failure(actual, expected)
    assert first["index"] is not None
    assert 0 <= first["index"] < 71
    assert first["max_abs"] > 1e-5
    assert first["max_rel"] > 1e-4


def test_att4_first_failure_reports_first_meaningful_stage_difference(tmp_path):
    debug_root = require_nacl_att4_debug()
    second_order = load_att4_v2_second_order_with_reference_nac(tmp_path)
    path_actual, path_expected = collect_att4_gx_frequencies(second_order, debug_root)
    first = classify_first_frequency_failure(path_actual, path_expected)
    assert first["index"] is not None
    q_index = int(first["index"])
    q_dir = debug_root / f"q-{q_index:05d}"
    actual = collect_kaldo_stage_data(
        second_order,
        np.load(q_dir / "q_red.npy", allow_pickle=False),
        q_index,
        np.load(q_dir / "q_direction_red.npy", allow_pickle=False),
    )
    expected = {
        "q_red": np.load(q_dir / "q_red.npy", allow_pickle=False),
        "q_cart": np.load(q_dir / "q_cart.npy", allow_pickle=False),
        "q_direction_cart": np.load(q_dir / "q_direction_cart.npy", allow_pickle=False),
        "dm_short": np.load(q_dir / "dm_short.npy", allow_pickle=False),
        "dd_recip": np.load(q_dir / "dd_recip.npy", allow_pickle=False),
        "dd_real": np.load(q_dir / "dd_real.npy", allow_pickle=False),
        "dd_total_mass_weighted": np.load(
            q_dir / "dd_total_mass_weighted.npy", allow_pickle=False
        ),
        "dm_final": np.load(q_dir / "dm_final.npy", allow_pickle=False),
        "eigenvalues": np.load(q_dir / "eigenvalues.npy", allow_pickle=False),
        "frequencies": np.load(q_dir / "frequencies.npy", allow_pickle=False),
    }
    stage_name, max_abs = first_meaningful_stage_difference(actual, expected)
    pytest.fail(
        "FIRST_STAGE_DIFFERENCE\n"
        f"q_index={q_index}\n"
        f"stage={stage_name}\n"
        f"max_abs={max_abs:.8e}"
    )


def test_att3_diagonal_mapping_matches_local_debug_reference(tmp_path):
    debug_dir = require_local_v2_debug()
    second_order = load_att3_v2_second_order_with_reference_nac(tmp_path)
    mapping = gln.build_short_range_inputs(
        second_order, nacl_phonopy_debug_supercell_matrix_att3()
    )

    np.testing.assert_array_equal(mapping["p2s_map"], np.load(debug_dir / "static" / "p2s_map.npy"))
    np.testing.assert_array_equal(mapping["s2p_map"], np.load(debug_dir / "static" / "s2p_map.npy"))
    np.testing.assert_array_equal(mapping["s2pp_map"], np.load(debug_dir / "static" / "s2pp_map.npy"))
    np.testing.assert_array_equal(mapping["multi"], np.load(debug_dir / "static" / "multi.npy"))
    np.testing.assert_allclose(mapping["svecs"], np.load(debug_dir / "static" / "svecs.npy"))
    np.testing.assert_allclose(
        mapping.get("phase_svecs", mapping["svecs"]),
        np.load(debug_dir / "static" / "phase_svecs.npy"),
    )


def test_gonze_real_dipole_q0_matches_debug_reference():
    debug_dir = require_nacl_debug_worktree()
    static = debug_dir / "static"
    dd_real_q0_full = hwq._gonze_real_dipole_dipole(
        np.zeros(3, dtype=float),
        np.load(static / "svecs.npy"),
        np.load(static / "multi.npy"),
        np.load(static / "s2pp_map.npy"),
        np.load(static / "dielectric.npy"),
        float(np.load(static / "Lambda.npy")),
        np.load(static / "supercell_cell.npy"),
    )
    actual = dd_real_q0_full.sum(axis=2)
    np.testing.assert_allclose(
        actual, np.load(static / "dd_real_q0.npy"), atol=1e-12, rtol=0.0
    )


def test_nacl_phonopy_equivalent_mapping_matches_debug_reference():
    debug_dir = require_nacl_debug_worktree()
    static = debug_dir / "static"
    atoms = ase.io.read("examples/nacl_phonopy/POSCAR")
    mapping = build_supercell_matrix_mapping(
        atoms, nacl_phonopy_debug_supercell_matrix()
    )

    np.testing.assert_array_equal(
        mapping["supercell_matrix"], nacl_phonopy_debug_supercell_matrix()
    )
    np.testing.assert_allclose(
        mapping["primitive_matrix"], np.load(static / "primitive_matrix.npy")
    )
    np.testing.assert_allclose(
        mapping["supercell_cell"], np.load(static / "supercell_cell.npy")
    )
    np.testing.assert_allclose(
        mapping["primitive_scaled_positions"],
        np.load(static / "primitive_scaled_positions.npy"),
    )
    np.testing.assert_allclose(
        mapping["supercell_scaled_positions"],
        np.load(static / "supercell_scaled_positions.npy"),
    )
    np.testing.assert_array_equal(mapping["p2s_map"], np.load(static / "p2s_map.npy"))
    np.testing.assert_array_equal(mapping["s2p_map"], np.load(static / "s2p_map.npy"))
    np.testing.assert_array_equal(mapping["s2pp_map"], np.load(static / "s2pp_map.npy"))
    np.testing.assert_array_equal(mapping["multi"], np.load(static / "multi.npy"))
    np.testing.assert_allclose(mapping["svecs"], np.load(static / "svecs.npy"))

    p2p_items = np.array(sorted(mapping["p2p_map"].items()), dtype=np.int64)
    np.testing.assert_array_equal(p2p_items, np.load(static / "p2p_map.npy"))
    np.testing.assert_allclose(
        mapping["svecs"] @ mapping["supercell_cell"], np.load(static / "svecs_cart.npy")
    )
    np.testing.assert_array_equal(
        mapping["multi"][:, :, 0], np.load(static / "multi_counts.npy")
    )
    np.testing.assert_array_equal(
        mapping["multi"][:, :, 1], np.load(static / "multi_offsets.npy")
    )


def test_nacl_phonopy_equivalent_mapping_reproduces_dd_real_q0():
    debug_dir = require_nacl_debug_worktree()
    static = debug_dir / "static"
    atoms = ase.io.read("examples/nacl_phonopy/POSCAR")
    mapping = build_supercell_matrix_mapping(
        atoms, nacl_phonopy_debug_supercell_matrix()
    )
    dd_real_q0_full = hwq._gonze_real_dipole_dipole(
        np.zeros(3, dtype=float),
        mapping["svecs"],
        mapping["multi"],
        mapping["s2pp_map"],
        np.load(static / "dielectric.npy"),
        float(np.load(static / "Lambda.npy")),
        mapping["supercell_cell"],
    )
    actual = dd_real_q0_full.sum(axis=2)
    np.testing.assert_allclose(
        actual, np.load(static / "dd_real_q0.npy"), atol=1e-12, rtol=0.0
    )


def test_nacl_commensurate_points_fold_to_phonopy_bz_order():
    debug_dir = require_nacl_debug()
    reciprocal_lattice = np.load(debug_dir / "static" / "reciprocal_lattice.npy")
    actual = commensurate_points(
        nacl_phonopy_debug_supercell_matrix(), reciprocal_lattice
    )
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.25, 0.25],
            [0.0, 0.5, 0.5],
            [0.0, -0.25, -0.25],
            [0.25, 0.0, 0.25],
            [0.25, 0.25, 0.5],
            [0.25, 0.5, -0.25],
            [0.25, -0.25, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.25, -0.25],
            [0.5, 0.5, 0.0],
            [0.5, -0.25, 0.25],
            [-0.25, 0.0, -0.25],
            [-0.25, 0.25, 0.0],
            [-0.25, 0.5, 0.25],
            [-0.25, -0.25, -0.5],
            [0.25, 0.25, 0.0],
            [0.25, 0.5, 0.25],
            [0.25, -0.25, 0.5],
            [0.25, 0.0, -0.25],
            [0.5, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [-0.5, -0.25, -0.25],
            [0.5, 0.0, 0.0],
            [-0.25, 0.25, 0.5],
            [-0.25, -0.5, -0.25],
            [-0.25, -0.25, 0.0],
            [-0.25, 0.0, 0.25],
            [0.0, 0.25, -0.25],
            [0.0, 0.5, 0.0],
            [0.0, -0.25, 0.25],
            [0.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    canonical_actual = np.array(sorted(map(tuple, np.round(actual, 12))))
    canonical_expected = np.array(sorted(map(tuple, np.round(expected, 12))))
    np.testing.assert_allclose(
        canonical_actual, canonical_expected, atol=1e-12, rtol=0.0
    )


def test_att3_velocity_debug_tree_is_loadable():
    root = require_nacl_att3_velocity_debug()
    assert (root / "q-00013" / "gv_scaled.npy").exists()


def test_load_velocity_direction_tensor_reads_per_direction_matrix():
    root = require_nacl_att3_velocity_debug()
    actual = load_velocity_direction_tensor(root, "q-00013", "d0", "ddm_fd")
    assert actual.shape == (6, 6)
    assert np.iscomplexobj(actual)


def test_load_velocity_json_reads_named_payload():
    root = require_nacl_att3_velocity_debug()
    payload = load_velocity_json(root, "q-00013", "degenerate_sets")
    assert payload["sets"] == [[0, 1], [2], [3, 4], [5]]
    np.testing.assert_allclose(
        load_velocity_q_tensor(root, "q-00013", "gv_cutoff_mask"),
        np.array([1, 1, 1, 1, 1, 1]),
        atol=0,
        rtol=0,
    )

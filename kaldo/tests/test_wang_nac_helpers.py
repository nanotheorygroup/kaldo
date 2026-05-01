import json
import re
from pathlib import Path

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.interfaces import shengbte_io
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.tests.wang_debug_reference import (
    compare_tensors,
    diagnostic_q_names_wang_att3,
    format_tensor_diff,
    load_q_tensor,
    require_wang_att3_debug,
    wang_att3_debug_dir,
)


def _load_wang_binary_tensor(root: Path, q_name: str, stem: str) -> np.ndarray:
    metadata_path = root / q_name / f"{stem}.json"
    data_path = root / q_name / f"{stem}.bin"
    raw_metadata = metadata_path.read_text()
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError:
        metadata = {}
        dtype_match = re.search(r'"dtype"\s*:\s*"([^"]+)"', raw_metadata)
        if dtype_match:
            metadata["dtype"] = dtype_match.group(1)
        shape_match = re.search(r'"shape"\s*:\s*\[(.*?)\]', raw_metadata, re.S)
        if shape_match:
            metadata["shape"] = shape_match.group(1)
        for key in ("num_patom", "num_satom", "q_index", "line"):
            match = re.search(rf'"{key}"\s*:\s*([0-9]+)', raw_metadata)
            if match:
                metadata[key] = int(match.group(1))

    shape = metadata.get("shape") or metadata.get("dims") or metadata.get("array_shape")
    if shape is None:
        raise KeyError(f"{metadata_path} is missing a shape field")
    if isinstance(shape, str):
        locals_map = {
            key: value
            for key, value in metadata.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
        shape_parts = [part for part in re.split(r"\s*,\s*", shape.strip().strip("[]()")) if part.strip()]
        shape = []
        unresolved_index = None
        for index, part in enumerate(shape_parts):
            try:
                shape.append(int(eval(part.strip(), {"__builtins__": {}}, locals_map)))
            except NameError:
                shape.append(part.strip())
                unresolved_index = index
    else:
        locals_map = {}
        unresolved_index = None
    dtype = np.dtype(
        metadata.get("dtype")
        or metadata.get("data_type")
        or metadata.get("type")
        or "float64"
    )
    array = np.fromfile(data_path, dtype=dtype)
    if unresolved_index is not None:
        known_product = 1
        for value in shape:
            if isinstance(value, int):
                known_product *= value
        unresolved_value = array.size // known_product
        if "num_patom*num_patom" in str(shape[unresolved_index]):
            unresolved_value = int(round(np.sqrt(unresolved_value))) ** 2
        shape[unresolved_index] = unresolved_value
    return array.reshape(tuple(int(value) for value in shape))


def _load_wang_second_order():
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


def _load_wang_q_point(root: Path, q_name: str) -> np.ndarray:
    q_points = np.asarray(load_q_tensor(root, q_name, "py_qpoints"), dtype=float)
    return q_points.reshape(-1, 3)[0]


def test_wang_att3_debug_tree_is_loadable():
    root = require_wang_att3_debug()
    q_points = load_q_tensor(root, "q-00000", "py_qpoints")
    assert q_points.size > 0
    assert q_points.shape[-1] == 3
    assert np.isfinite(q_points).all()


def test_wang_diagnostic_q_names_are_the_expected_sequence():
    assert diagnostic_q_names_wang_att3() == [
        "q-00000",
        "q-00010",
        "q-00020",
        "q-00030",
    ]


def test_wang_diagnostic_q_names_match_expected_points():
    root = require_wang_att3_debug()
    q_names = diagnostic_q_names_wang_att3()
    assert len(q_names) == 4
    assert q_names[0] == "q-00000"
    assert q_names[-1] == "q-00030"
    assert q_names == sorted(q_names, key=lambda name: int(name.split("-")[1]))
    for q_name in q_names:
        assert (root / q_name).is_dir()
        q_points = load_q_tensor(root, q_name, "py_qpoints")
        assert q_points.size > 0
        assert np.isfinite(q_points).all()


def test_wang_att3_debug_dir_env_override_takes_precedence(monkeypatch, tmp_path):
    env_root = tmp_path / "env"
    default_root = tmp_path / "default"
    fallback_root = tmp_path / "fallback"
    env_root.mkdir()
    default_root.mkdir()
    fallback_root.mkdir()
    monkeypatch.setenv("WANG_ATT3_DEBUG_DIR", str(env_root))
    monkeypatch.setattr(
        "kaldo.tests.wang_debug_reference.DEFAULT_WANG_ATT3_DEBUG",
        default_root,
    )
    monkeypatch.setattr(
        "kaldo.tests.wang_debug_reference.DEFAULT_WANG_ATT3_DEBUG_FALLBACK",
        fallback_root,
    )
    assert wang_att3_debug_dir() == env_root


def test_wang_att3_debug_dir_uses_valid_fallback_when_primary_is_stale(
    monkeypatch, tmp_path
):
    primary_root = tmp_path / "primary"
    fallback_root = tmp_path / "fallback"
    (primary_root / "q-00000").mkdir(parents=True)
    for q_name in ["q-00000", "q-00010", "q-00020", "q-00030"]:
        q_root = fallback_root / q_name
        q_root.mkdir(parents=True)
        np.save(q_root / "py_qpoints.npy", np.array([0.0, 0.0, 0.0]))
    monkeypatch.delenv("WANG_ATT3_DEBUG_DIR", raising=False)
    monkeypatch.setattr(
        "kaldo.tests.wang_debug_reference.DEFAULT_WANG_ATT3_DEBUG",
        primary_root,
    )
    monkeypatch.setattr(
        "kaldo.tests.wang_debug_reference.DEFAULT_WANG_ATT3_DEBUG_FALLBACK",
        fallback_root,
    )
    assert wang_att3_debug_dir() == fallback_root


def test_compare_tensors_reports_expected_diff():
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


def test_format_tensor_diff_includes_summary_fields():
    actual = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[1.0, 2.0], [3.0, 5.0]])
    text = format_tensor_diff("two_by_two", "q-00000", actual, expected)
    assert "q-00000 two_by_two:" in text
    assert "shape=(2, 2)" in text
    assert "dtype=float64" in text
    assert "max_abs_diff=1.00000000e+00" in text
    assert "rel_diff=" in text


def test_wang_q_cart_matches_debug_reference():
    root = require_wang_att3_debug()
    q_name = "q-00010"
    phonon = HarmonicWithQ(
        q_point=_load_wang_q_point(root, q_name),
        second=_load_wang_second_order(),
        storage="memory",
        nac_method="wang",
        q_index=int(q_name.split("-")[1]),
    )

    actual = phonon._build_wang_static_data()["q_cart"]
    expected = _load_wang_binary_tensor(root, q_name, "wang_q_cart")
    diff = compare_tensors("q_cart", actual, expected)
    assert diff.max_abs_diff < 1e-8, format_tensor_diff("q_cart", q_name, actual, expected)


def test_wang_charge_sum_matches_debug_reference():
    root = require_wang_att3_debug()
    q_name = "q-00020"
    phonon = HarmonicWithQ(
        q_point=_load_wang_q_point(root, q_name),
        second=_load_wang_second_order(),
        storage="memory",
        nac_method="wang",
        q_index=int(q_name.split("-")[1]),
    )

    actual = phonon._calculate_wang_dynamical_matrix(return_debug_data=True)["charge_sum"]
    expected = _load_wang_binary_tensor(root, q_name, "wang_charge_sum")
    diff = compare_tensors("charge_sum", actual, expected)
    assert diff.max_abs_diff < 1e-8, format_tensor_diff("charge_sum", q_name, actual, expected)

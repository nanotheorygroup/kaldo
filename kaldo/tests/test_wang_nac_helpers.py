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
        for key in ("num_patom", "num_satom", "num_band", "q_index", "line"):
            match = re.search(rf'"{key}"\s*:\s*([0-9]+)', raw_metadata)
            if match:
                metadata[key] = int(match.group(1))
        size_match = re.search(r'"size_(?:complex_)?elements"\s*:\s*([0-9]+)', raw_metadata)
        if size_match and ("num_patom" not in metadata or "num_band" not in metadata):
            shape_str_match = re.search(r'"shape"\s*:\s*\[(.*?)\]', raw_metadata, re.S)
            if shape_str_match:
                raw_tokens = [t.strip() for t in shape_str_match.group(1).split(",") if t.strip()]
                symbol_counts = {
                    "num_patom": sum(1 for t in raw_tokens if t == "num_patom"),
                    "num_band": sum(1 for t in raw_tokens if t == "num_band"),
                }
                known = 1
                for t in raw_tokens:
                    if t.isdigit():
                        known *= int(t)
                total = int(size_match.group(1))
                for symbol, count in symbol_counts.items():
                    if count <= 0 or symbol in metadata or known <= 0:
                        continue
                    inferred = round((total / known) ** (1.0 / count))
                    if inferred ** count * known == total:
                        metadata[symbol] = inferred

    shape = metadata.get("shape")
    if shape is None:
        raise KeyError(f"{metadata_path} is missing a shape field")
    if isinstance(shape, str):
        shape_tokens = [part.strip() for part in shape.strip("[]()").split(",") if part.strip()]
        resolved_shape = []
        square_index = None
        for token in shape_tokens:
            if token.isdigit():
                resolved_shape.append(int(token))
            elif token == "num_patom":
                if "num_patom" not in metadata:
                    raise KeyError(f"{metadata_path} is missing num_patom metadata")
                resolved_shape.append(int(metadata["num_patom"]))
            elif token == "num_band":
                if "num_band" not in metadata:
                    raise KeyError(f"{metadata_path} is missing num_band metadata")
                resolved_shape.append(int(metadata["num_band"]))
            elif token == "num_patom*num_patom":
                if "num_patom" in metadata:
                    resolved_shape.append(int(metadata["num_patom"]) * int(metadata["num_patom"]))
                else:
                    resolved_shape.append(token)
                    square_index = len(resolved_shape) - 1
            else:
                raise ValueError(f"Unsupported Wang shape token {token!r} in {metadata_path}")
        shape = resolved_shape
    elif not isinstance(shape, (list, tuple)):
        raise TypeError(f"Unsupported Wang shape metadata in {metadata_path}: {shape!r}")
    dtype = np.dtype(metadata.get("dtype") or "float64")
    array = np.fromfile(data_path, dtype=dtype)
    if shape and any(isinstance(value, str) for value in shape):
        known_product = 1
        for value in shape:
            if isinstance(value, int):
                known_product *= value
        unresolved = [index for index, value in enumerate(shape) if isinstance(value, str)]
        if unresolved != [square_index] or shape[square_index] != "num_patom*num_patom":
            raise ValueError(f"Unsupported unresolved Wang shape in {metadata_path}: {shape!r}")
        square_product = array.size // known_product
        square_root = int(round(np.sqrt(square_product)))
        if square_root * square_root != square_product:
            raise ValueError(f"Cannot infer num_patom from {metadata_path}: {shape!r}")
        shape[square_index] = square_root * square_root
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

    actual = phonon._calculate_wang_dynamical_matrix(return_debug_data=True)["q_cart"]
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
    assert diff.max_abs_diff < 1e-10, format_tensor_diff("charge_sum", q_name, actual, expected)


def test_wang_charge_sum_axis_ordering_is_atom_major():
    root = require_wang_att3_debug()
    q_name = "q-00020"
    phonon = HarmonicWithQ(
        q_point=_load_wang_q_point(root, q_name),
        second=_load_wang_second_order(),
        storage="memory",
        nac_method="wang",
        q_index=int(q_name.split("-")[1]),
    )

    debug_data = phonon._calculate_wang_dynamical_matrix(return_debug_data=True)
    charge_sum = debug_data["charge_sum"]
    charge_terms = debug_data["charge_terms"]
    masses = debug_data["masses"]
    n_atom = len(masses)
    # charge_sum[i*n+j, alpha, beta] should equal scale * A_i_alpha * A_j_beta
    # where A = charge_terms.  Verify for (i=0, j=0) and (i=0, j=1).
    dielectric_contraction = debug_data["dielectric_contraction"]
    nac_factor = float(debug_data["nac_factor"])
    supercell_multiplicity = int(debug_data["supercell_multiplicity"])
    scale = nac_factor / supercell_multiplicity / dielectric_contraction
    for i in range(n_atom):
        for j in range(n_atom):
            expected_block = scale * np.outer(charge_terms[i], charge_terms[j])
            actual_block = charge_sum[i * n_atom + j]
            diff = compare_tensors(f"charge_sum[{i},{j}]", actual_block, expected_block)
            assert diff.max_abs_diff < 1e-12, format_tensor_diff(
                f"charge_sum[{i},{j}]", q_name, actual_block, expected_block
            )


def test_wang_dnac_matches_debug_reference():
    root = require_wang_att3_debug()
    second_order = _load_wang_second_order()
    q_name = "q-00010"
    # Reference wang_dnac/ddnac are computed at py_derivative_q, not py_qpoints
    q_red = load_q_tensor(root, q_name, "py_derivative_q").flatten()
    phonon = HarmonicWithQ(
        q_point=q_red,
        second=second_order,
        storage="memory",
        nac_method="wang",
        q_index=int(q_name.split("-")[1]),
    )
    actual = phonon._calculate_wang_derivative_dynamical_matrix(return_debug_data=True)["wang_dnac"]
    expected = _load_wang_binary_tensor(root, q_name, "wang_dnac")
    diff = compare_tensors("wang_dnac", actual, expected)
    assert diff.max_abs_diff < 1e-8, format_tensor_diff("wang_dnac", q_name, actual, expected)


def test_wang_ddnac_matches_debug_reference():
    root = require_wang_att3_debug()
    second_order = _load_wang_second_order()
    q_name = "q-00020"
    # Reference wang_dnac/ddnac are computed at py_derivative_q, not py_qpoints
    q_red = load_q_tensor(root, q_name, "py_derivative_q").flatten()
    phonon = HarmonicWithQ(
        q_point=q_red,
        second=second_order,
        storage="memory",
        nac_method="wang",
        q_index=int(q_name.split("-")[1]),
    )
    actual = phonon._calculate_wang_derivative_dynamical_matrix(return_debug_data=True)["wang_ddnac"]
    expected = _load_wang_binary_tensor(root, q_name, "wang_ddnac")
    diff = compare_tensors("wang_ddnac", actual, expected)
    assert diff.max_abs_diff < 1e-8, format_tensor_diff("wang_ddnac", q_name, actual, expected)


def test_wang_derivative_dynmat_matches_debug_reference():
    root = require_wang_att3_debug()
    second_order = _load_wang_second_order()
    q_name = "q-00010"
    q_red = load_q_tensor(root, q_name, "py_derivative_q").flatten()
    phonon = HarmonicWithQ(
        q_point=q_red,
        second=second_order,
        storage="memory",
        nac_method="wang",
        q_index=int(q_name.split("-")[1]),
    )
    actual = phonon._calculate_wang_derivative_dynamical_matrix(return_debug_data=True)[
        "wang_derivative_dynmat"
    ]
    expected = _load_wang_binary_tensor(root, q_name, "wang_derivative_dynmat_after_hermitian")
    diff = compare_tensors("wang_derivative_dynmat", actual, expected)
    assert diff.max_abs_diff < 1e-8, format_tensor_diff(
        "wang_derivative_dynmat", q_name, actual, expected
    )

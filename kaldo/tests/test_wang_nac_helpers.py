import numpy as np
import pytest

from kaldo.tests.wang_debug_reference import (
    compare_tensors,
    diagnostic_q_names_wang_att3,
    format_tensor_diff,
    load_q_tensor,
    require_wang_att3_debug,
    wang_att3_debug_dir,
)


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
    fallback_q_root = fallback_root / "q-00000"
    fallback_q_root.mkdir(parents=True)
    np.save(fallback_q_root / "py_qpoints.npy", np.array([0.0, 0.0, 0.0]))
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

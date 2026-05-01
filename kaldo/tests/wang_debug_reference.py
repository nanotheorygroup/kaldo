from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import pytest


DEFAULT_WANG_ATT3_DEBUG = (
    Path.home() / "data/rephonopy/.worktrees/wang-nac-trace/example/nacl-att3/debug-wang-band"
)
DEFAULT_WANG_ATT3_DEBUG_FALLBACK = Path(
    "/data/nwlundgren/rephonopy/.worktrees/wang-nac-trace/example/nacl-att3/debug-wang-band"
)


def wang_att3_debug_dir() -> Path:
    env_override = os.environ.get("WANG_ATT3_DEBUG_DIR")
    if env_override:
        return Path(env_override)
    for candidate in (DEFAULT_WANG_ATT3_DEBUG, DEFAULT_WANG_ATT3_DEBUG_FALLBACK):
        if (candidate / "q-00000" / "py_qpoints.npy").exists():
            return candidate
    return DEFAULT_WANG_ATT3_DEBUG


def require_wang_att3_debug() -> Path:
    path = wang_att3_debug_dir()
    if not (path / "q-00000" / "py_qpoints.npy").exists():
        pytest.skip(f"Wang att3 debug tree not found at {path}")
    return path


def diagnostic_q_names_wang_att3() -> list[str]:
    return [
        "q-00000",
        "q-00010",
        "q-00020",
        "q-00030",
    ]


def load_q_tensor(root: Path, q_name: str, name: str) -> np.ndarray:
    return np.load(root / q_name / f"{name}.npy", allow_pickle=False)


@dataclass(frozen=True)
class TensorDiff:
    name: str
    shape: tuple[int, ...]
    dtype: str
    max_abs_diff: float
    rel_diff: float


def compare_tensors(name: str, actual: np.ndarray, expected: np.ndarray) -> TensorDiff:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(
            f"{name} shape mismatch: actual {actual.shape} != expected {expected.shape}"
        )
    diff = np.abs(actual - expected)
    denom = max(float(np.linalg.norm(expected)), 1e-30)
    return TensorDiff(
        name=name,
        shape=actual.shape,
        dtype=str(actual.dtype),
        max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
        rel_diff=float(np.linalg.norm(actual - expected) / denom),
    )


def format_tensor_diff(name: str, q_name: str, actual: np.ndarray, expected: np.ndarray) -> str:
    diff = compare_tensors(name, actual, expected)
    return (
        f"{q_name} {name}: "
        f"shape={diff.shape} dtype={diff.dtype} "
        f"max_abs_diff={diff.max_abs_diff:.8e} "
        f"rel_diff={diff.rel_diff:.8e}"
    )

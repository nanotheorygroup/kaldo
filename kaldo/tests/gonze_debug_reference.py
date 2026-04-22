from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import pytest


DEFAULT_NACL_ATT3_DEBUG = Path(
    "/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replay/"
    "example/nacl-att3/debug"
)


def nacl_att3_debug_dir() -> Path:
    return Path(os.environ.get("NACL_ATT3_DEBUG_DIR", DEFAULT_NACL_ATT3_DEBUG))


def require_nacl_att3_debug() -> Path:
    path = nacl_att3_debug_dir()
    if not (path / "static" / "metadata.json").exists():
        pytest.skip(f"NaCl att3 debug tree not found at {path}")
    return path


def load_static_tensor(root: Path, name: str) -> np.ndarray:
    return np.load(root / "static" / f"{name}.npy", allow_pickle=False)


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

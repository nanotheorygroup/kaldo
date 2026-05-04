from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
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


def diagnostic_q_names_att3() -> list[str]:
    return [
        "q-00000",
        "q-00013",
        "q-00020",
        "q-00030",
    ]


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


@dataclass(frozen=True)
class AcousticResidualReport:
    q_name: str
    subspace_size: int
    full_norm: float
    projected_norm: float
    projected_fraction: float
    reference_eigenvalues: np.ndarray
    actual_eigenvalues: np.ndarray
    overlap_matrix: np.ndarray


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


DEFAULT_NACL_ATT3_VELOCITY_DEBUG = Path(
    "/home/nwlundgren/data/rephonopy/.worktrees/gonze-velocity-debug-dump/"
    "example/nacl-att3/debug-velocity"
)


def nacl_att3_velocity_debug_dir() -> Path:
    return Path(
        os.environ.get(
            "NACL_ATT3_VELOCITY_DEBUG_DIR",
            DEFAULT_NACL_ATT3_VELOCITY_DEBUG,
        )
    )


def require_nacl_att3_velocity_debug() -> Path:
    path = nacl_att3_velocity_debug_dir()
    if not (path / "q-00013" / "gv_scaled.npy").exists():
        pytest.skip(f"NaCl att3 velocity debug tree not found at {path}")
    return path


def load_velocity_q_tensor(root: Path, q_name: str, name: str) -> np.ndarray:
    return np.load(root / q_name / f"{name}.npy", allow_pickle=False)


def load_velocity_direction_tensor(
    root: Path, q_name: str, direction_name: str, name: str
) -> np.ndarray:
    return np.load(root / q_name / direction_name / f"{name}.npy", allow_pickle=False)


def load_velocity_json(root: Path, q_name: str, name: str) -> dict:
    return json.loads((root / q_name / f"{name}.json").read_text())


def acoustic_subspace_residual_report(
    q_name: str,
    actual_dm_final: np.ndarray,
    reference_dm_final: np.ndarray,
    subspace_size: int = 3,
) -> AcousticResidualReport:
    reference_eigenvalues, reference_eigenvectors = np.linalg.eigh(reference_dm_final)
    actual_eigenvalues, actual_eigenvectors = np.linalg.eigh(actual_dm_final)
    subspace = min(subspace_size, len(reference_eigenvalues))
    basis = reference_eigenvectors[:, :subspace]
    delta_dm = actual_dm_final - reference_dm_final
    projected = basis.conj().T @ delta_dm @ basis
    full_norm = float(np.linalg.norm(delta_dm))
    projected_norm = float(np.linalg.norm(projected))
    overlap = np.abs(basis.conj().T @ actual_eigenvectors[:, :subspace])
    return AcousticResidualReport(
        q_name=q_name,
        subspace_size=subspace,
        full_norm=full_norm,
        projected_norm=projected_norm,
        projected_fraction=projected_norm / max(full_norm, 1e-30),
        reference_eigenvalues=reference_eigenvalues[:subspace],
        actual_eigenvalues=actual_eigenvalues[:subspace],
        overlap_matrix=overlap,
    )

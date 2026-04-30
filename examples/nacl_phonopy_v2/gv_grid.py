#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kaldo.forceconstants import ForceConstants
from kaldo.interfaces import shengbte_io
from kaldo.observables.harmonic_with_q import HarmonicWithQ


DEFAULT_ATT3_DEBUG = Path(
    "/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replace/example/nacl-att3/debug"
)
DEFAULT_ATT3_DEBUG_FALLBACK = Path(
    "/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replay/example/nacl-att3/debug"
)
DEFAULT_ATT3_VELOCITY_DEBUG = Path(
    "/data/nwlundgren/rephonopy/.worktrees/gonze-velocity-debug-dump/example/nacl-att3/debug-velocity"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare and plot kALDo vs phonopy frequencies along G->X with 31 points."
    )
    parser.add_argument(
        "--debug-root",
        type=Path,
        default=None,
        help="Path to phonopy debug root with q-00000 ... q-00030 directories.",
    )
    parser.add_argument(
        "--folder",
        default="examples/nacl_phonopy_v2",
        help="Folder containing v2 inputs (POSCAR, espresso.ifc2).",
    )
    parser.add_argument(
        "--nac-source",
        default="examples/nacl_phonopy/espresso.ifc2",
        help="QE IFC file used to read dielectric/Born charges for NAC.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/nacl_phonopy_v2/plots/gx_compare_kaldo_vs_phonopy.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--rtol-all", type=float, default=0.02, help="Relative tolerance for all-mode match.")
    parser.add_argument("--atol-all", type=float, default=0.05, help="Absolute tolerance for all-mode match.")
    parser.add_argument(
        "--rtol-optical", type=float, default=0.02, help="Relative tolerance for optical-mode match."
    )
    parser.add_argument("--atol-optical", type=float, default=0.05, help="Absolute tolerance for optical-mode match.")
    return parser.parse_args()


def resolve_debug_root(debug_root: Path | None) -> Path:
    if debug_root is not None:
        return debug_root
    env_override = os.environ.get("NACL_ATT3_DEBUG_DIR")
    if env_override:
        return Path(env_override)
    if DEFAULT_ATT3_DEBUG.exists():
        return DEFAULT_ATT3_DEBUG
    return DEFAULT_ATT3_DEBUG_FALLBACK


def gather_q_dirs(debug_root: Path) -> list[Path]:
    q_dirs = sorted(
        [path for path in debug_root.glob("q-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[1]),
    )
    expected = [f"q-{i:05d}" for i in range(31)]
    selected = [debug_root / name for name in expected]
    missing = [path.name for path in selected if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing expected q directories in {debug_root}: {missing}")
    return selected


def load_second_order(folder: str, nac_source: str):
    forceconstants = ForceConstants.from_folder(
        folder=folder,
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    _, _, charges = shengbte_io.read_second_order_qe_matrix(nac_source)
    if charges is None:
        raise ValueError(f"No NAC dielectric/Born data found in {nac_source}")
    forceconstants.second.atoms.info["dielectric"] = charges[0, :, :]
    forceconstants.second.atoms.set_array("charges", charges[1:, :, :], shape=(3, 3))
    return forceconstants.second


def point_match_status(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol_all: float,
    atol_all: float,
    rtol_optical: float,
    atol_optical: float,
) -> tuple[str, float, float]:
    all_match = np.allclose(actual, expected, rtol=rtol_all, atol=atol_all)
    optical_match = np.allclose(actual[3:], expected[3:], rtol=rtol_optical, atol=atol_optical)
    max_abs = float(np.max(np.abs(actual - expected)))
    max_rel = float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-30)))
    if all_match:
        return "MATCH_ALL", max_abs, max_rel
    if optical_match:
        return "MATCH_OPTICAL_ONLY", max_abs, max_rel
    return "FAIL", max_abs, max_rel


def evaluate_path(second_order, q_dirs: list[Path], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_vel_dir = DEFAULT_ATT3_VELOCITY_DEBUG
    kaldo_freqs = []
    ref_freqs = []
    kaldo_vels = []
    ref_vels = []
    print(
        "index q_red_x q_red_y q_red_z freq_status vel_max_rel "
        "freq_kaldo freq_phonopy vel_kaldo vel_phonopy"
    )
    for idx, q_dir in enumerate(q_dirs):
        q_red = np.load(q_dir / "q_red.npy", allow_pickle=False)
        expected = np.load(q_dir / "frequencies.npy", allow_pickle=False)
        phonon = HarmonicWithQ(
            q_point=q_red,
            second=second_order,
            storage="memory",
            is_unfolding=True,
            nac_method="gonze",
            nac_q_direction=[1, 0, 0],
            q_index=idx,
        )
        actual = phonon.frequency.flatten()
        vels = np.linalg.norm(phonon.velocity[0], axis=-1)
        expected_vels = np.linalg.norm(np.load(q_vel_dir / f"q-{str(idx).zfill(5)}" / "gv_scaled.npy", allow_pickle=False), axis=-1)
        status, max_abs, max_rel = point_match_status(
            actual, expected, args.rtol_all, args.atol_all, args.rtol_optical, args.atol_optical
        )
        vel_max_rel = float(np.max(
            np.abs(vels - expected_vels) / np.maximum(np.abs(expected_vels), 1e-6)
        ))
        print(
            f"{idx:02d} "
            f"{q_red[0]:.8f} {q_red[1]:.8f} {q_red[2]:.8f} "
            f"{status} vel_rel={vel_max_rel:.4f} "
            f"[{', '.join(f'{v:.4f}' for v in actual)}] "
            f"[{', '.join(f'{v:.4f}' for v in expected)}] "
            f"vel_k=[{', '.join(f'{v:.2f}' for v in vels)}] "
            f"vel_p=[{', '.join(f'{v:.2f}' for v in expected_vels)}]"
        )
        kaldo_freqs.append(actual)
        ref_freqs.append(expected)
        kaldo_vels.append(vels)
        ref_vels.append(expected_vels)
    return np.array(kaldo_freqs), np.array(ref_freqs), np.array(kaldo_vels), np.array(ref_vels)


def save_plot(kaldo_freqs: np.ndarray, ref_freqs: np.ndarray, kaldo_vels: np.ndarray, ref_vels: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(kaldo_freqs.shape[0], dtype=float)
    #fig, ax = plt.subplots(figsize=(10, 6))
    fig = plt.figure(figsize=(12,14))
    grid = plt.GridSpec(1,2)
    ax = fig.add_subplot(grid[:,0])
    ax1 = fig.add_subplot(grid[:,1])
    for mode in range(kaldo_freqs.shape[1]):
        ax.plot(x, ref_freqs[:, mode], color=f"C{mode}", linewidth=1.8, linestyle="-", label=f"Phonopy m{mode + 1}")
        ax.plot(
            x,
            kaldo_freqs[:, mode],
            color=f"C{mode}",
            linewidth=1.2,
            linestyle="--",
            label=f"kALDo m{mode + 1}",
        )
    for mode in range(kaldo_freqs.shape[1]):
        ax1.scatter(x, ref_vels[:, mode], s=25, color='k')
        ax1.scatter(x, kaldo_vels[:, mode], s=25, color=f"C{mode}")
    ax.set_xlim(0, len(x) - 1)
    ax1.set_xlim(0, len(x) - 1)
    ax1.set_xlabel("Path index (G->X, 31 points)")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("NaCl G->X Dispersion")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    debug_root = resolve_debug_root(args.debug_root)
    if not (debug_root / "static" / "metadata.json").exists():
        raise FileNotFoundError(f"Debug root does not look valid: {debug_root}")
    q_dirs = gather_q_dirs(debug_root)
    second_order = load_second_order(args.folder, args.nac_source)
    kaldo_freqs, ref_freqs, kaldo_vels, ref_vels = evaluate_path(second_order, q_dirs, args)
    save_plot(kaldo_freqs, ref_freqs, kaldo_vels, ref_vels, args.output)
    max_abs = float(np.max(np.abs(kaldo_freqs - ref_freqs)))
    max_rel = float(np.max(np.abs(kaldo_freqs - ref_freqs) / np.maximum(np.abs(ref_freqs), 1e-30)))
    vel_max_rel = float(np.max(
        np.abs(kaldo_vels - ref_vels) / np.maximum(np.abs(ref_vels), 1e-6)
    ))
    print()
    print(f"saved_plot={args.output}")
    print(f"path_max_abs_diff={max_abs:.8e}")
    print(f"path_max_rel_diff={max_rel:.8e}")
    print(f"path_vel_max_rel_diff={vel_max_rel:.8e}")


if __name__ == "__main__":
    main()

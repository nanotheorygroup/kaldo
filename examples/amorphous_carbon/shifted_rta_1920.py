"""Shifted 2x2x2 three-phonon RTA conductivity of the 1920-atom amorphous carbon cell.

Loads the single-cell IFCs of ``../kaldo3/examples/18GPa-3`` (``second.npy``, ``third.npz``,
``replicated_atoms.xyz``), computes harmonic properties on the half-step shifted 2x2x2 grid, the exact three-phonon
linewidths and the RTA conductivity, and prints stage wall times, the process peak RSS and the kappa tensor.

Example::

    CUDA_VISIBLE_DEVICES="" python shifted_rta_1920.py --run-folder shifted_rta_1920_run --workers 64

All outputs (kALDo cache folder, arrays) go under ``--run-folder``; the source folder is only read.
"""

import argparse
import logging
import resource
import time
from pathlib import Path

import numpy as np

from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons

DEFAULT_SOURCE = Path(__file__).resolve().parents[3] / "kaldo3" / "examples" / "18GPa-3"
THIRD_BANDWIDTH_THZ = 0.5 / 4.136  # 0.5 meV


def peak_rss_gib():
    """Process peak resident set size in GiB (``ru_maxrss`` is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def report(label, started):
    elapsed = time.perf_counter() - started
    print(f"[timing] {label}: {elapsed:.1f} s, peak RSS so far {peak_rss_gib():.1f} GiB", flush=True)
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-folder", type=Path, required=True, help="kALDo storage folder for this run")
    parser.add_argument("--workers", type=int, default=64, help="worker threads (default 64)")
    parser.add_argument("--temperature", type=float, default=300, help="temperature in K (default 300)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="folder with the IFC files (read only)")
    parser.add_argument("--classic", action="store_true", help="classical instead of quantum occupations")
    parser.add_argument("--harmonic-only", action="store_true", help="stop after the shifted harmonics")
    args = parser.parse_args()
    logging.getLogger("kaldo").setLevel(logging.INFO)
    args.run_folder.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tick = time.perf_counter()
    forceconstants = ForceConstants.from_folder(
        folder=str(args.source), supercell=(1, 1, 1), format="numpy", is_acoustic_sum=True, chunk_size=500000,
        only_second=args.harmonic_only,
    )
    report("force constants loaded", tick)

    phonons = Phonons(
        forceconstants=forceconstants, kpts=(2, 2, 2), kpts_shift=(0.5, 0.5, 0.5), temperature=args.temperature,
        is_classic=args.classic, third_bandwidth=THIRD_BANDWIDTH_THZ, n_workers=args.workers,
        folder=str(args.run_folder), storage="numpy",
    )
    tick = time.perf_counter()
    harmonics = phonons._shifted_harmonics
    harmonics.partner()
    report("harmonics (4 complex + 8 real eigensolves, velocities)", tick)
    frequency, physical = phonons.frequency, phonons.physical_mode
    speed = np.linalg.norm(phonons.velocity, axis=-1).reshape(physical.shape)
    print(f"frequency range {frequency.min():.4f} to {frequency.max():.4f} THz; "
          f"median physical |v| {np.median(speed[physical]):.4f} (kALDo velocity unit, 100 m/s)", flush=True)
    if args.harmonic_only:
        report("total", started)
        return

    tick = time.perf_counter()
    bandwidth = phonons.bandwidth
    report("three-phonon projection (bandwidth)", tick)
    print(f"median physical bandwidth {np.median(bandwidth.reshape(physical.shape)[physical]):.6g} THz", flush=True)

    tick = time.perf_counter()
    kappa = Conductivity(phonons=phonons, method="rta", storage="numpy").conductivity.sum(axis=0)
    report("RTA conductivity", tick)
    np.set_printoptions(precision=6, suppress=True)
    print(f"RTA kappa (W/m/K):\n{kappa}\ntrace/3 = {np.trace(kappa) / 3:.6g} W/m/K", flush=True)
    report("total", started)


if __name__ == "__main__":
    main()

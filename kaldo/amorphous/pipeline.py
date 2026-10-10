"""Restartable spectrum-window solve and distributed QHGK reduction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .checkpoint import RowShardedEigenvectorSource
from .distributed import run_distributed_qhgk
from .slepc_solver import SlepcRowEigenvectorSource
from .windows import eigenvalue_window_intervals


def run_interval_pipeline(harmonic, eigenvalue_boundaries, max_bandwidth,
                          bandwidth_provider, temperature, is_classic, volume,
                          output_directory, *, solver=SlepcRowEigenvectorSource,
                          solver_options=None, block_size=256,
                          compute_dtype=np.float32,
                          accumulator_dtype=np.float64,
                          max_windows=None):
    """Solve halo intervals and accumulate their disjoint core modes.

    ``eigenvalue_boundaries`` cover the entire dynamical spectrum, including
    negative/zero modes. They can come from approximate DOS quantiles or a
    full eigenvalue planning pass. ``max_bandwidth`` is a verified global upper
    bound in rad/ps; it fixes each triangular halo before solving. The final
    core count must equal ``3*N`` or the pipeline fails rather than returning
    a silently incomplete conductivity.
    """
    comm = harmonic.comm
    rank = comm.Get_rank()
    options = {} if solver_options is None else dict(solver_options)
    if max_windows is not None and (not isinstance(max_windows, int) or max_windows < 0):
        raise ValueError("max_windows must be a nonnegative integer")
    intervals = eigenvalue_window_intervals(eigenvalue_boundaries, max_bandwidth)
    directory = Path(output_directory)
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    config = {"format": "kaldo-amorphous-interval-pipeline-v1",
              "harmonic_arrays_sha256": harmonic.input_hashes,
              "rank_count": comm.Get_size(),
              "eigenvalue_boundaries_rad2_ps2": list(map(float, eigenvalue_boundaries)),
              "maximum_bandwidth_rad_ps": float(max_bandwidth),
              "bandwidth_source": getattr(bandwidth_provider, "metadata", None),
              "temperature_K": float(temperature), "is_classic": bool(is_classic),
              "volume_angstrom3": float(volume),
              "solver": f"{solver.__module__}.{solver.__name__}",
              "solver_options": {key: str(value) for key, value in options.items()},
              "block_size": block_size,
              "dtypes": {"compute": str(np.dtype(compute_dtype)),
                         "accumulator": str(np.dtype(accumulator_dtype))},
              "broadening_policy": "triangle exact support, exact zero self flux"}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("config_sha256") != config_hash:
        raise ValueError("interval pipeline configuration changed")
    completed = []
    core_data = []
    new_windows = 0
    for index, (core_lower, core_upper, halo_lower, halo_upper) in enumerate(intervals):
        window_dir = directory / f"window_{index:05d}"
        eigen_dir = window_dir / "eigenvectors"
        if (eigen_dir / "manifest.json").exists():
            source = RowShardedEigenvectorSource(eigen_dir, comm)
            omega = np.load(eigen_dir / "omega.npy")
            physical = np.load(eigen_dir / "physical_mode.npy")
        elif max_windows is None or new_windows < max_windows:
            source, omega, physical = solver.solve_interval(
                harmonic, eigen_dir, interval_lambda=(halo_lower, halo_upper),
                **options)
        else:
            continue
        eigenvalues = np.load(eigen_dir / "eigenvalues.npy")
        if (eigenvalues.shape != omega.shape or physical.shape != omega.shape or
                np.any(np.diff(eigenvalues) < 0) or
                np.any(eigenvalues < halo_lower) or np.any(eigenvalues > halo_upper)):
            raise ValueError("solver returned invalid halo eigenvalues")
        if len(omega):
            measured = bandwidth_provider.evaluate(omega, temperature)
            if np.any(measured > max_bandwidth*(1+1e-12)):
                raise ValueError("declared maximum bandwidth is too small for solved modes")
        first = int(np.searchsorted(eigenvalues, core_lower, side="left"))
        last = int(np.searchsorted(eigenvalues, core_upper,
                                   side="right" if index == len(intervals)-1 else "left"))
        if first < last:
            result = run_distributed_qhgk(
                omega, bandwidth_provider, physical, temperature, is_classic,
                volume, source, harmonic, window_dir / "qhgk",
                block_size=block_size, broadening_shape="triangle",
                compute_dtype=compute_dtype,
                accumulator_dtype=accumulator_dtype,
                core_start=first, core_stop=last)
            if rank == 0:
                qhgk_manifest = json.loads((window_dir / "qhgk/manifest.json").read_text())
                if not qhgk_manifest["complete"]:
                    raise RuntimeError("incomplete core reduction in interval pipeline")
                core_data.append((eigenvalues[first:last], omega[first:last],
                                  physical[first:last], result[0][first:last],
                                  result[1][first:last]))
                pair_count = qhgk_manifest["evaluated_pairs"]
        else:
            pair_count = 0
        if rank == 0:
            completed.append({"window": index,
                              "core_eigenvalue_interval": [core_lower, core_upper],
                              "halo_eigenvalue_interval": [halo_lower, halo_upper],
                              "halo_modes": len(eigenvalues),
                              "core_modes": last-first,
                              "eigenvector_manifest_sha256": hashlib.sha256(
                                  (eigen_dir / "manifest.json").read_bytes()).hexdigest(),
                              "evaluated_pairs": pair_count})
        new_windows += 1
    if rank != 0:
        return None
    complete_windows = len(completed) == len(intervals)
    core_count = sum(item["core_modes"] for item in completed)
    if complete_windows and core_count != 3*harmonic.n_atoms:
        raise RuntimeError(f"spectrum windows returned {core_count} of {3*harmonic.n_atoms} modes")
    manifest = {**config, "config_sha256": config_hash,
                "completed_windows": completed,
                "complete": complete_windows,
                "core_mode_count": core_count,
                "evaluated_pairs": sum(item["evaluated_pairs"] for item in completed)}
    temporary = directory / ".manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, manifest_path)
    if not complete_windows:
        return None
    arrays = [np.concatenate([item[column] for item in core_data])
              for column in range(5)]
    ordering = np.argsort(arrays[0], kind="stable")
    for filename, array in zip(("eigenvalues", "omega", "physical_mode",
                                "conductivity", "diffusivity"), arrays):
        np.save(directory / (filename + ".npy"), array[ordering])
    return arrays[3][ordering], arrays[4][ordering]

"""Spectrum slicing for distributed Gamma QHGK: window planning and the restartable interval pipeline.

Window boundaries and conservative triangular halos are in eigenvalue units
(rad/ps squared). The pipeline solves each halo interval and reduces only its
disjoint core modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kaldo.helpers.atomic_io import sha256_file
from kaldo.helpers.checkpoint_store import CheckpointStore
from kaldo.observables.eigenvector_source import RowShardedEigenvectorSource
from kaldo.parallel.distributed_qhgk import run_distributed_qhgk
from kaldo.parallel.slepc_eigensolver import SlepcRowEigenvectorSource


def approximate_dos_boundaries(bin_edges, density, n_windows):
    """Choose spectral boundaries from a supplied approximate DOS histogram."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    if (edges.ndim != 1 or density.shape != (len(edges)-1,) or
            len(density) < 1 or np.any(np.diff(edges) <= 0) or
            not np.isfinite(edges).all() or not np.isfinite(density).all() or
            np.any(density < 0) or np.sum(density*np.diff(edges)) <= 0 or
            n_windows < 1):
        raise ValueError("invalid approximate DOS")
    cumulative = np.r_[0., np.cumsum(density*np.diff(edges))]
    fractions = np.linspace(0., cumulative[-1], n_windows+1)
    boundaries = np.interp(fractions, cumulative, edges)
    boundaries[0], boundaries[-1] = edges[0], edges[-1]
    return boundaries


def eigenvalue_boundaries_from_full_spectrum(eigenvalues, n_windows):
    """Optional exact mode-count planning pass; avoid splitting degeneracies."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or
            not np.isfinite(values).all() or np.any(np.diff(values) < 0) or
            not isinstance(n_windows, int) or n_windows < 1):
        raise ValueError("invalid full eigenvalue spectrum or window count")
    cuts = []
    for index in range(1, n_windows):
        target = len(values)*index//n_windows
        while target < len(values) and target > 0 and values[target] == values[target-1]:
            target += 1
        if 0 < target < len(values):
            cuts.append((values[target-1]+values[target])/2)
    scale = max(1., float(np.max(np.abs(values))))
    padding = max(1e-8, 1e-6*scale)
    return np.array([values[0]-padding, *sorted(set(cuts)), values[-1]+padding])


def eigenvalue_window_intervals(boundaries, max_bandwidth):
    """Core and halo intervals in eigenvalue units (rad/ps squared).

    Signed frequencies are monotone in eigenvalue, including unstable modes.
    A 4*maximum-bandwidth frequency halo is conservative for triangular QHGK.
    """
    edges = np.asarray(boundaries, dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 2 or
            np.any(np.diff(edges) <= 0) or not np.isfinite(edges).all() or
            not np.isfinite(max_bandwidth) or max_bandwidth < 0):
        raise ValueError("invalid eigenvalue boundaries or maximum bandwidth")
    frequency = np.sign(edges)*np.sqrt(np.abs(edges))
    reach = 4*max_bandwidth
    lower = frequency[:-1]-reach
    upper = frequency[1:]+reach
    # Interval eigensolvers may treat an endpoint as open. Leave a small
    # outward numerical margin so a mode exactly on a DOS/core endpoint is
    # present in at least one halo, even when the linewidth bound is zero.
    margin = 1e-8*max(1., float(np.max(np.abs(edges))))
    halo_lower = np.sign(lower)*lower**2 - margin
    halo_upper = np.sign(upper)*upper**2 + margin
    return [(float(a), float(b), float(c), float(d))
            for a,b,c,d in zip(edges[:-1], edges[1:], halo_lower, halo_upper)]


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
    store = CheckpointStore(directory, config, comm=comm, label="interval pipeline")
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
                              "eigenvector_manifest_sha256": sha256_file(eigen_dir / "manifest.json"),
                              "evaluated_pairs": pair_count})
        new_windows += 1
    if rank != 0:
        return None
    complete_windows = len(completed) == len(intervals)
    core_count = sum(item["core_modes"] for item in completed)
    if complete_windows and core_count != 3*harmonic.n_atoms:
        raise RuntimeError(f"spectrum windows returned {core_count} of {3*harmonic.n_atoms} modes")
    store.write_manifest(completed_windows=completed, complete=complete_windows, core_mode_count=core_count,
                         evaluated_pairs=sum(item["evaluated_pairs"] for item in completed))
    if not complete_windows:
        return None
    arrays = [np.concatenate([item[column] for item in core_data])
              for column in range(5)]
    ordering = np.argsort(arrays[0], kind="stable")
    for filename, array in zip(("eigenvalues", "omega", "physical_mode",
                                "conductivity", "diffusivity"), arrays):
        store.save_array(filename, array[ordering])
    return arrays[3][ordering], arrays[4][ordering]

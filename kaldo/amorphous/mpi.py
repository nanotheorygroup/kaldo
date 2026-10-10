"""Restartable local-MPI QHGK prototype with replicated sparse operators.

Run this through ``mpiexec -n R``. Only final per-mode vectors are gathered;
each rank opens eigenvector panels from its own source and writes disjoint
core-mode checkpoint files. MPI-enabled HDF5 is not needed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource

import numpy as np

from .checkpoint import ShardedEigenvectorSource, sha256_file
from .qhgk import NpyEigensystemSource, calculate_qhgk_isotropic_rows
from .sparse_harmonic import SparseAmorphousHarmonic


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _result_hash(conductivity, diffusivity):
    digest = hashlib.sha256()
    for array in (conductivity, diffusivity):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _source_hash(source):
    if isinstance(source, NpyEigensystemSource):
        return sha256_file(source.path)
    if isinstance(source, ShardedEigenvectorSource):
        for shard in source.shards:
            if sha256_file(source.directory / shard["file"]) != shard["sha256"]:
                raise ValueError(f"eigenvector shard hash mismatch: {shard['file']}")
        return hashlib.sha256(json.dumps(source.manifest,
                                        sort_keys=True).encode()).hexdigest()
    raise ValueError("MPI requires an on-disk NpyEigensystemSource or ShardedEigenvectorSource")


def run_mpi_qhgk(omega, bandwidth_provider, physical_mode, temperature,
                 is_classic, volume, eigenvectors, harmonic, output_directory, *,
                 block_size=256, broadening_shape="triangle",
                 diffusivity_threshold=None, compute_dtype=np.float32,
                 accumulator_dtype=np.float64, comm=None, max_blocks=None):
    """Compute or resume owned core blocks; return final vectors on rank zero.

    ``max_blocks`` limits newly calculated blocks per rank for restart testing.
    A partial run returns ``None`` and writes an incomplete manifest.
    """
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if broadening_shape != "triangle" and diffusivity_threshold is None:
        # Exact Gaussian/Lorentzian all-pairs calculations remain available,
        # but require an explicit policy choice for a large MPI job.
        pass
    if (not isinstance(block_size, int) or block_size < 1 or
            max_blocks is not None and (not isinstance(max_blocks, int) or max_blocks < 0)):
        raise ValueError("invalid block count or block size")
    if not isinstance(harmonic, SparseAmorphousHarmonic):
        raise ValueError("MPI prototype requires SparseAmorphousHarmonic")
    w = np.asarray(omega, dtype=np.float64)
    physical = np.asarray(physical_mode, dtype=bool)
    bw = bandwidth_provider.evaluate(w, temperature) if hasattr(bandwidth_provider, "evaluate") else np.asarray(bandwidth_provider, dtype=np.float64)
    if bw.ndim == 0:
        bw = np.full(len(w), bw)
    if bw.shape != w.shape or physical.shape != w.shape:
        raise ValueError("bandwidth and physical mode arrays must match omega")
    if eigenvectors.n_modes != len(w) or harmonic.n_atoms * 3 != len(w):
        raise ValueError("eigenvectors and sparse harmonic must match omega")
    source_hash = _source_hash(eigenvectors) if rank == 0 else None
    source_hash = comm.bcast(source_hash, root=0)
    harmonic_hash = (_array_hash(harmonic.row_ptr), _array_hash(harmonic.column_atom),
                     _array_hash(harmonic.blocks), _array_hash(harmonic.displacement),
                     None if harmonic.masses is None else _array_hash(harmonic.masses),
                     None if harmonic.positions is None else _array_hash(harmonic.positions),
                     None if harmonic.cell is None else _array_hash(harmonic.cell))
    config = {"format": "kaldo-amorphous-mpi-qhgk-v1", "mode_count": len(w),
              "omega_sha256": _array_hash(w), "bandwidth_sha256": _array_hash(bw),
              "physical_sha256": _array_hash(physical),
              "eigenvector_source_sha256": source_hash,
              "sparse_arrays_sha256": harmonic_hash,
              "harmonic_source_sha256": harmonic.metadata.get("source_sha256"),
              "bandwidth_source": getattr(bandwidth_provider, "metadata", None),
              "units": {"omega": "rad/ps", "bandwidth": "rad/ps",
                        "volume": "angstrom^3", "conductivity": "W/(m K)",
                        "diffusivity": "mm^2/s"},
              "temperature_K": float(temperature), "is_classic": bool(is_classic),
              "volume_angstrom3": float(volume),
              "broadening_policy": {"shape": broadening_shape,
                                    "triangular_support": "exact" if broadening_shape == "triangle" else None,
                                    "self_pair_flux": "exact_zero",
                                    "legacy_asymmetric_threshold": diffusivity_threshold},
              "dtypes": {"compute": str(np.dtype(compute_dtype)),
                         "accumulator": str(np.dtype(accumulator_dtype))},
              "block_size": block_size}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output_directory = Path(output_directory)
    if rank == 0:
        output_directory.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    old = output_directory / "manifest.json"
    if old.exists():
        previous = json.loads(old.read_text())
        if previous.get("config_sha256") != config_hash:
            raise ValueError("checkpoint configuration changed")
    ownership = [(start, min(start + block_size, len(w)))
                 for start in range(0, len(w), block_size)]
    local = []
    stats = {"evaluated_pairs": 0, "skipped_blocks": 0,
             "panel_io_seconds": 0., "panel_io_bytes": 0,
             "operator_seconds": 0.,
             "contraction_seconds": 0., "pair_seconds": 0.,
             "reduction_seconds": 0.}
    completed_stats = []
    new_blocks = 0
    for index, (start, stop) in enumerate(ownership):
        if index % size != rank:
            continue
        path = output_directory / f"rows_{start:09d}_{stop:09d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if (str(saved["config_sha256"]) != config_hash or
                        int(saved["start"]) != start or int(saved["stop"]) != stop or
                        saved["conductivity"].shape != (stop-start,) or
                        saved["diffusivity"].shape != (stop-start,)):
                    raise ValueError(f"invalid checkpoint {path}")
                conductivity = saved["conductivity"].copy()
                diffusivity = saved["diffusivity"].copy()
                result_hash = str(saved["result_sha256"])
                if _result_hash(conductivity, diffusivity) != result_hash:
                    raise ValueError(f"checkpoint checksum mismatch: {path}")
                block_stats = json.loads(str(saved["stats_json"])) if "stats_json" in saved else None
        elif max_blocks is None or new_blocks < max_blocks:
            measured = {}
            conductivity, diffusivity = calculate_qhgk_isotropic_rows(
                w, bw, physical, temperature, is_classic, volume, eigenvectors,
                harmonic.operators, core_start=start, core_stop=stop,
                block_size=block_size, broadening_shape=broadening_shape,
                diffusivity_threshold=diffusivity_threshold,
                compute_dtype=compute_dtype, accumulator_dtype=accumulator_dtype,
                stats=measured)
            for key in stats:
                stats[key] += measured[key]
            block_stats = measured
            result_hash = _result_hash(conductivity, diffusivity)
            tmp = output_directory / f".{path.name}.rank{rank}.tmp"
            with tmp.open("wb") as stream:
                np.savez(stream, config_sha256=config_hash, start=start, stop=stop,
                         conductivity=conductivity, diffusivity=diffusivity,
                         result_sha256=result_hash,
                         stats_json=json.dumps(block_stats))
            os.replace(tmp, path)
            new_blocks += 1
        else:
            continue
        local.append((start, stop, conductivity, diffusivity, result_hash))
        if block_stats is not None:
            completed_stats.append(block_stats)
    gathered = comm.gather(local, root=0)
    gathered_completed_stats = comm.gather(completed_stats, root=0)
    sparse_bytes = harmonic.storage_bytes + sum(
        cast.storage_bytes for cast in harmonic._casts.values())
    metrics = comm.gather({**stats, "rank": rank,
                           "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "sparse_storage_bytes": sparse_bytes}, root=0)
    if rank != 0:
        return None
    result_c = np.zeros(len(w), dtype=accumulator_dtype)
    result_d = np.zeros(len(w), dtype=accumulator_dtype)
    completed = []
    for rank_id, rows in enumerate(gathered):
        for start, stop, conductivity, diffusivity, result_hash in rows:
            result_c[start:stop] = conductivity
            result_d[start:stop] = diffusivity
            completed.append({"start": start, "stop": stop, "rank": rank_id,
                              "result_sha256": result_hash})
    completed.sort(key=lambda item: item["start"])
    complete = len(completed) == len(ownership)
    all_block_stats = [item for rank_stats in gathered_completed_stats for item in rank_stats]
    manifest = {**config, "config_sha256": config_hash, "rank_count": size,
                "mode_ownership": [{"start": a, "stop": b, "rank": i % size}
                                   for i, (a, b) in enumerate(ownership)],
                "completed_core_ranges": completed, "complete": complete,
                "metrics_per_rank": metrics,
                "evaluated_pairs": sum(item["evaluated_pairs"] for item in all_block_stats),
                "executed_pairs_this_run": sum(item["evaluated_pairs"] for item in metrics),
                "panel_io_bytes": sum(item["panel_io_bytes"] for item in all_block_stats)}
    tmp = output_directory / ".manifest.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, output_directory / "manifest.json")
    if complete:
        np.save(output_directory / "conductivity.npy", result_c)
        np.save(output_directory / "diffusivity.npy", result_d)
        return result_c, result_d
    return None

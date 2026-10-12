"""Coordinate-row distributed sparse harmonic action and QHGK contraction.

Each rank owns atom rows of the mass-weighted dynamical matrix and the same
coordinate rows of every eigenvector. For a partner mode panel, ranks exchange
only eigenvector rows for off-rank neighbor atoms, apply their local derivative
rows, and reduce bounded flux blocks to the owner of the core mode rows.
"""

from __future__ import annotations

import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from kaldo.helpers.atomic_io import (atomic_save_npy, atomic_savez, atomic_write_json,
                                     sha256_arrays, sha256_file, sha256_json)
from kaldo.observables.eigenvector_source import RowShardedEigenvectorSource, atom_row_bounds
from kaldo.controllers.qhgk import (_finish, _triangle_block_outside_support, _zero_self_pairs,
                                   qhgk_pair_factors)
from kaldo.observables.sparse_harmonic import apply_derivative_panel


class DistributedSparseHarmonic:
    """Local atom rows of the direct sparse input contract.

    This adapter keeps the on-disk format and derivative action of
    ``SparseHarmonic`` while replacing global storage with local rows.
    It does not change the original harmonic or legacy kALDo object APIs.
    """

    def __init__(self, directory, comm):
        self.directory = Path(directory)
        self.comm = comm
        self.rank, self.size = comm.Get_rank(), comm.Get_size()
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata.get("format") != "kaldo-amorphous-atom-block-v1":
            raise ValueError("unsupported sparse harmonic input")
        self.n_atoms = int(self.metadata["n_atoms"])
        if self.rank == 0:
            actual_hashes = {path.name: sha256_file(path)
                             for path in self.directory.glob("*.npy")}
            expected_hashes = self.metadata.get("array_sha256")
            valid_hashes = expected_hashes is None or actual_hashes == expected_hashes
        else:
            actual_hashes, valid_hashes = None, None
        valid_hashes = comm.bcast(valid_hashes, root=0)
        self.input_hashes = comm.bcast(actual_hashes, root=0)
        if not valid_hashes:
            raise ValueError("sparse input array checksum mismatch")
        if self.size > self.n_atoms:
            raise ValueError("MPI ranks cannot exceed atom count")
        self.bounds = atom_row_bounds(self.n_atoms, self.size)
        self.atom_start, self.atom_stop = self.bounds[self.rank]
        self.n_local_atoms = self.atom_stop - self.atom_start
        def read(name):
            return np.load(self.directory / (name + ".npy"), mmap_mode="r",
                           allow_pickle=False)
        global_ptr = read("row_ptr")
        if global_ptr.shape != (self.n_atoms + 1,):
            raise ValueError("invalid sparse row pointer shape")
        first, last = int(global_ptr[self.atom_start]), int(global_ptr[self.atom_stop])
        self.row_ptr = np.asarray(global_ptr[self.atom_start:self.atom_stop+1] - first,
                                  dtype=np.int64)
        self.column_atom = read("column_atom")[first:last]
        self.blocks = read("dynamical_blocks")[first:last]
        self.displacement = read("displacement")[first:last]
        if (self.blocks.shape != (last-first, 3, 3) or
                self.displacement.shape != (last-first, 3) or
                np.any(self.column_atom >= self.n_atoms)):
            raise ValueError("invalid local sparse harmonic slab")
        self._work_data = {}
        self._prepare_exchange()

    @property
    def storage_bytes(self):
        return sum(item.nbytes for item in (self.row_ptr, self.column_atom,
                                            self.blocks, self.displacement,
                                            self.remapped_columns)) + sum(
            blocks.nbytes + displacement.nbytes
            for blocks, displacement in self._work_data.values())

    def _prepare_exchange(self):
        remote = self.column_atom[(self.column_atom < self.atom_start) |
                                  (self.column_atom >= self.atom_stop)]
        unique_remote = np.unique(remote)
        boundaries = np.array([start for start, _ in self.bounds] + [self.n_atoms])
        owners = np.searchsorted(boundaries, unique_remote, side="right") - 1
        self.requests = [unique_remote[owners == rank].astype(np.int64)
                         for rank in range(self.size)]
        self.send_ids = [np.asarray(item, dtype=np.int64)
                         for item in self.comm.alltoall(self.requests)]
        self.ghost_ids = np.concatenate(self.requests) if len(unique_remote) else np.empty(0, dtype=np.int64)
        columns = np.asarray(self.column_atom, dtype=np.int64)
        local = (columns >= self.atom_start) & (columns < self.atom_stop)
        remapped = np.empty(len(columns), dtype=np.int32)
        remapped[local] = columns[local] - self.atom_start
        if np.any(~local):
            positions = np.searchsorted(self.ghost_ids, columns[~local])
            if np.any(positions >= len(self.ghost_ids)) or np.any(self.ghost_ids[positions] != columns[~local]):
                raise ValueError("incomplete ghost schedule")
            remapped[~local] = self.n_local_atoms + positions
        self.remapped_columns = remapped

    def exchange_panel(self, local_panel):
        """Return local plus needed remote atom rows for one mode panel."""
        local_panel = np.ascontiguousarray(local_panel)
        width = local_panel.shape[1]
        if local_panel.shape[0] != 3*self.n_local_atoms:
            raise ValueError("local eigenvector panel has wrong row count")
        atom_panel = local_panel.reshape(self.n_local_atoms, 3, width)
        chunks = [np.ascontiguousarray(atom_panel[ids-self.atom_start]).reshape(-1)
                  for ids in self.send_ids]
        send_counts = np.array([len(chunk) for chunk in chunks], dtype=np.int32)
        recv_counts = np.array([len(ids)*3*width for ids in self.requests], dtype=np.int32)
        send_offsets = np.r_[0, np.cumsum(send_counts[:-1])].astype(np.int32)
        recv_offsets = np.r_[0, np.cumsum(recv_counts[:-1])].astype(np.int32)
        send = np.concatenate(chunks) if chunks else np.empty(0, dtype=local_panel.dtype)
        received = np.empty(int(np.sum(recv_counts)), dtype=local_panel.dtype)
        MPI = _mpi()
        mpi_dtype = MPI._typedict[local_panel.dtype.char]
        self.comm.Alltoallv([send, send_counts, send_offsets, mpi_dtype],
                            [received, recv_counts, recv_offsets, mpi_dtype])
        return np.concatenate((local_panel, received.reshape(-1, width)), axis=0)

    def directional_action(self, axis, extended_panel):
        dtype = extended_panel.dtype
        if dtype not in self._work_data:
            self._work_data[dtype] = (self.blocks.astype(dtype, copy=False),
                                      self.displacement.astype(dtype, copy=False))
        blocks, displacement = self._work_data[dtype]
        return apply_derivative_panel(
            self.row_ptr, self.remapped_columns, blocks, displacement, axis,
            extended_panel,
            (3*self.n_local_atoms, 3*(self.n_local_atoms+len(self.ghost_ids))))


def _mpi():
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError("distributed amorphous QHGK requires mpi4py. "
                          "Install with: pip install kaldo[mpi]") from exc
    return MPI


def _checkpoint_path(directory, start, stop):
    return directory / f"rows_{start:09d}_{stop:09d}.npz"


def run_distributed_qhgk(omega, bandwidth_provider, physical_mode, temperature,
                         is_classic, volume, eigenvectors, harmonic,
                         output_directory, *, block_size=256,
                         broadening_shape="triangle", diffusivity_threshold=None,
                         compute_dtype=np.float32,
                         accumulator_dtype=np.float64, max_blocks=None,
                         core_start=0, core_stop=None):
    """Collectively reduce QHGK with distributed coordinate rows.

    Every rank participates in each non-skipped flux block. The core owner
    alone forms pair factors, accumulates its modes, and writes its checkpoint.
    The returned pair is available on rank zero after completed runs.
    """
    comm = harmonic.comm
    rank, size = comm.Get_rank(), comm.Get_size()
    if (not isinstance(eigenvectors, RowShardedEigenvectorSource) or
            eigenvectors.rank != rank or eigenvectors.size != size or
            (eigenvectors.atom_start, eigenvectors.atom_stop) !=
            (harmonic.atom_start, harmonic.atom_stop)):
        raise ValueError("eigenvector and harmonic atom-row ownership must match")
    if (not isinstance(block_size, (int, np.integer)) or block_size < 1 or
            max_blocks is not None and
            (not isinstance(max_blocks, (int, np.integer)) or max_blocks < 0)):
        raise ValueError("block_size and max_blocks must be nonnegative integers")
    if broadening_shape not in ("triangle", "gauss", "lorentz"):
        raise ValueError("unsupported broadening shape")
    m = eigenvectors.n_modes
    if core_stop is None:
        core_stop = m
    if not 0 <= core_start <= core_stop <= m:
        raise ValueError("invalid window core mode range")
    w = np.asarray(omega, dtype=np.float64)
    physical = np.asarray(physical_mode, dtype=bool)
    bw = bandwidth_provider.evaluate(w, temperature) if hasattr(bandwidth_provider, "evaluate") else np.asarray(bandwidth_provider, dtype=np.float64)
    if bw.ndim == 0:
        bw = np.full(m, bw)
    if (w.shape != (m,) or physical.shape != (m,) or bw.shape != (m,) or
            np.any(np.diff(w) < 0) or not np.isfinite(w).all() or
            not np.isfinite(bw).all() or np.any(bw < 0) or
            not np.isfinite(volume) or volume <= 0 or
            eigenvectors.global_n_coordinates != 3*harmonic.n_atoms or
            not np.isfinite(temperature)):
        raise ValueError("invalid distributed QHGK inputs")
    work, accum = np.dtype(compute_dtype), np.dtype(accumulator_dtype)
    if work not in (np.dtype("float32"), np.dtype("float64")) or accum not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("unsupported work or accumulation dtype")
    directory = Path(output_directory)
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    # Row shards fix the rank count; their hashes also bind restarts to exact
    # eigenvector data. Each rank checks its own file and all ranks agree.
    local_hash = sha256_file(eigenvectors.directory / eigenvectors.shard["file"])
    local_hash_error = (f"rank {rank} eigenvector row shard hash mismatch"
                        if local_hash != eigenvectors.shard["sha256"] else None)
    hash_errors = comm.allgather(local_hash_error)
    if any(hash_errors):
        raise ValueError("; ".join(error for error in hash_errors if error))
    source_hash = sha256_json(eigenvectors.manifest)
    config = {"format": "kaldo-distributed-amorphous-qhgk-v1",
              "rank_count": size, "mode_count": m,
              "eigenvector_source_sha256": source_hash,
              "harmonic_source_sha256": harmonic.metadata.get("source_sha256"),
              "harmonic_metadata_sha256": sha256_json(harmonic.metadata),
              "harmonic_arrays_sha256": harmonic.input_hashes,
              "omega_sha256": sha256_arrays(w), "bandwidth_sha256": sha256_arrays(bw),
              "physical_sha256": sha256_arrays(physical),
              "bandwidth_source": getattr(bandwidth_provider, "metadata", None),
              "units": {"omega": "rad/ps", "bandwidth": "rad/ps",
                        "volume": "angstrom^3", "conductivity": "W/(m K)",
                        "diffusivity": "mm^2/s"},
              "temperature_K": float(temperature), "is_classic": bool(is_classic),
              "volume_angstrom3": float(volume), "block_size": int(block_size),
              "core_mode_range": [core_start, core_stop],
              "dtypes": {"compute": str(work), "accumulator": str(accum)},
              "broadening_policy": {"shape": broadening_shape,
                                    "triangular_support": "exact" if broadening_shape == "triangle" else None,
                                    "self_pair_flux": "exact_zero",
                                    "legacy_asymmetric_threshold": diffusivity_threshold}}
    config_hash = sha256_json(config)
    old = directory / "manifest.json"
    if old.exists() and json.loads(old.read_text()).get("config_sha256") != config_hash:
        raise ValueError("distributed checkpoint configuration changed")
    block = min(block_size, max(1, m-1))
    ownership = [(first, min(first+block, core_stop))
                 for first in range(core_start, core_stop, block)]
    local_results = []
    local_stats = {"evaluated_pairs": 0, "skipped_blocks": 0,
                   "panel_io_bytes": 0, "panel_io_seconds": 0.,
                   "exchange_seconds": 0., "operator_seconds": 0.,
                   "contraction_seconds": 0., "reduction_seconds": 0.}
    completed_new = 0
    for index, (i0, i1) in enumerate(ownership):
        owner = index % size
        path = _checkpoint_path(directory, i0, i1)
        exists = path.exists() if rank == owner else None
        exists = comm.bcast(exists, root=owner)
        if exists:
            checkpoint_error = None
            if rank == owner:
                try:
                    with np.load(path, allow_pickle=False) as saved:
                        c, d = saved["conductivity"].copy(), saved["diffusivity"].copy()
                        digest = str(saved["result_sha256"])
                        pair_count = int(saved["evaluated_pairs"])
                        if (str(saved["config_sha256"]) != config_hash or
                                int(saved["start"]) != i0 or int(saved["stop"]) != i1 or
                                c.shape != (i1-i0,) or d.shape != (i1-i0,) or
                                digest != sha256_arrays(c, d)):
                            raise ValueError("checkpoint fields or result hash differ")
                except Exception as exc:
                    checkpoint_error = f"invalid distributed checkpoint {path}: {exc}"
            checkpoint_error = comm.bcast(checkpoint_error, root=owner)
            if checkpoint_error is not None:
                raise ValueError(checkpoint_error)
            if rank == owner:
                local_results.append((i0, i1, c, d, digest, pair_count))
            continue
        compute = max_blocks is None or completed_new < max_blocks
        # Every rank follows the same block schedule. A cap is global rather
        # than per-owner so a partial run cannot strand a collective.
        if not compute:
            continue
        t = perf_counter()
        ei = eigenvectors.read_modes(i0, i1, dtype=work)
        local_stats["panel_io_seconds"] += perf_counter()-t
        local_stats["panel_io_bytes"] += ei.nbytes
        if rank == owner:
            local_c = np.zeros(i1-i0, dtype=accum)
            local_d = np.zeros(i1-i0, dtype=accum)
            pair_count = 0
        for j0 in range(0, m, block):
            j1 = min(j0+block, m)
            if broadening_shape == "triangle" and _triangle_block_outside_support(
                    w[i0:i1], w[j0:j1], bw[i0:i1], bw[j0:j1]):
                if rank == owner:
                    local_stats["skipped_blocks"] += 1
                continue
            t = perf_counter()
            ej = eigenvectors.read_modes(j0, j1, dtype=work)
            local_stats["panel_io_seconds"] += perf_counter()-t
            local_stats["panel_io_bytes"] += ej.nbytes
            t = perf_counter()
            extended = harmonic.exchange_panel(ej)
            local_stats["exchange_seconds"] += perf_counter()-t
            flux2 = np.zeros((i1-i0, j1-j0), dtype=work) if rank == owner else None
            for axis in range(3):
                t = perf_counter()
                y = harmonic.directional_action(axis, extended)
                local_stats["operator_seconds"] += perf_counter()-t
                t = perf_counter()
                local_flux = np.ascontiguousarray(ei.T @ y)
                if rank == owner:
                    flux = np.empty_like(local_flux)
                else:
                    flux = None
                comm.Reduce(local_flux, flux, root=owner)
                local_stats["contraction_seconds"] += perf_counter()-t
                if rank == owner:
                    flux2 += flux*flux
            if rank == owner:
                t = perf_counter()
                flux2 /= 3
                _zero_self_pairs(flux2, i0, j0)
                kernel, heat = qhgk_pair_factors(
                    w[i0:i1], w[j0:j1], bw[i0:i1], bw[j0:j1],
                    physical[i0:i1], physical[j0:j1], temperature=temperature,
                    is_classic=is_classic, broadening_shape=broadening_shape,
                    diffusivity_threshold=diffusivity_threshold, dtype=work)
                kernel *= flux2
                local_d += np.sum(kernel, axis=1, dtype=accum)
                kernel *= heat
                local_c += np.sum(kernel, axis=1, dtype=accum)
                local_stats["reduction_seconds"] += perf_counter()-t
                local_stats["evaluated_pairs"] += (i1-i0)*(j1-j0)
                pair_count += (i1-i0)*(j1-j0)
        if rank == owner:
            c, d = _finish(local_c, local_d, volume)
            digest = sha256_arrays(c, d)
            atomic_savez(path, config_sha256=config_hash, start=i0, stop=i1,
                         conductivity=c, diffusivity=d, result_sha256=digest,
                         evaluated_pairs=pair_count)
            local_results.append((i0, i1, c, d, digest, pair_count))
        completed_new += 1
    gathered = comm.gather(local_results, root=0)
    metrics = comm.gather({**local_stats, "rank": rank,
                           "sparse_storage_bytes": harmonic.storage_bytes,
                           "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
                          root=0)
    if rank != 0:
        return None
    conductivity = np.zeros(m, dtype=accum)
    diffusivity = np.zeros(m, dtype=accum)
    completed = []
    for owner, items in enumerate(gathered):
        for i0, i1, c, d, digest, pair_count in items:
            conductivity[i0:i1], diffusivity[i0:i1] = c, d
            completed.append({"start": i0, "stop": i1, "rank": owner,
                              "result_sha256": digest,
                              "evaluated_pairs": pair_count})
    completed.sort(key=lambda item: item["start"])
    complete = len(completed) == len(ownership)
    manifest = {**config, "config_sha256": config_hash,
                "mode_ownership": [{"start": a, "stop": b, "rank": index % size}
                                   for index, (a,b) in enumerate(ownership)],
                "coordinate_ownership": [{"rank": index, "atom_start": a,
                                          "atom_stop": b} for index, (a,b) in
                                         enumerate(harmonic.bounds)],
                "completed_core_ranges": completed, "complete": complete,
                "metrics_per_rank": metrics,
                "evaluated_pairs": sum(item["evaluated_pairs"] for item in completed),
                "executed_pairs_this_run": sum(item["evaluated_pairs"] for item in metrics)}
    atomic_write_json(directory / "manifest.json", manifest)
    if complete:
        atomic_save_npy(directory / "conductivity.npy", conductivity)
        atomic_save_npy(directory / "diffusivity.npy", diffusivity)
        return conductivity, diffusivity
    return None

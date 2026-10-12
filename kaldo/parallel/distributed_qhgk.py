"""Coordinate-row distributed sparse harmonic action and QHGK flux projection.

Each rank owns atom rows of the mass-weighted dynamical matrix and the same
coordinate rows of every eigenvector. For a partner mode panel, ranks exchange
only eigenvector rows for off-rank neighbor atoms, apply their local derivative
rows, and reduce bounded flux blocks to the owner of the core mode rows. The
reduction loop itself is :func:`kaldo.controllers.qhgk.reduce_qhgk_blocks`.
"""

from __future__ import annotations

import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from kaldo.controllers.qhgk import check_dtypes, check_mode_inputs, read_panel, reduce_qhgk_blocks
from kaldo.helpers.atomic_io import sha256_arrays, sha256_file, sha256_json
from kaldo.helpers.checkpoint_store import CheckpointStore
from kaldo.observables.eigenvector_source import RowShardedEigenvectorSource, atom_row_bounds
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


class DistributedFluxProjector:
    """Flux blocks whose coordinate rows are split across MPI ranks.

    Core-mode block ``k`` belongs to rank ``k % size``. Every rank contracts
    its local coordinate rows, and the partial flux blocks are summed onto the
    owner, so ``flux`` is collective and returns the block only on the owner.
    """

    def __init__(self, eigenvectors, harmonic, *, dtype):
        if (not isinstance(eigenvectors, RowShardedEigenvectorSource) or
                eigenvectors.rank != harmonic.rank or eigenvectors.size != harmonic.size or
                (eigenvectors.atom_start, eigenvectors.atom_stop) != (harmonic.atom_start, harmonic.atom_stop)):
            raise ValueError("eigenvector and harmonic atom-row ownership must match")
        if eigenvectors.global_n_coordinates != 3 * harmonic.n_atoms:
            raise ValueError("eigenvectors and sparse harmonic describe different systems")
        self.source = eigenvectors
        self.harmonic = harmonic
        self.comm = harmonic.comm
        self.rank, self.size = harmonic.rank, harmonic.size
        self.n_modes = eigenvectors.n_modes
        self.dtype = np.dtype(dtype)
        self.stats = {"panel_io_seconds": 0., "panel_io_bytes": 0, "exchange_seconds": 0.,
                      "operator_seconds": 0., "contraction_seconds": 0.}

    def owner(self, block_index):
        return block_index % self.size

    def core_panel(self, start, stop):
        t = perf_counter()
        panel = read_panel(self.source, start, stop, self.dtype)
        self.stats["panel_io_seconds"] += perf_counter() - t
        self.stats["panel_io_bytes"] += panel.nbytes
        return panel

    def flux(self, core_panel, start, stop, owner):
        partner = self.core_panel(start, stop)
        t = perf_counter()
        extended = self.harmonic.exchange_panel(partner)
        self.stats["exchange_seconds"] += perf_counter() - t
        flux = np.empty((3, core_panel.shape[1], stop - start), dtype=self.dtype) if self.rank == owner else None
        for axis in range(3):
            t = perf_counter()
            y = self.harmonic.directional_action(axis, extended)
            self.stats["operator_seconds"] += perf_counter() - t
            t = perf_counter()
            local = np.ascontiguousarray(core_panel.T @ y)
            self.comm.Reduce(local, flux[axis] if flux is not None else None, root=owner)
            self.stats["contraction_seconds"] += perf_counter() - t
        return flux


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
    work, accum = check_dtypes(compute_dtype, accumulator_dtype)
    projector = DistributedFluxProjector(eigenvectors, harmonic, dtype=work)
    m = projector.n_modes
    core_stop = m if core_stop is None else core_stop
    w = np.asarray(omega, dtype=np.float64)
    bw = (bandwidth_provider.evaluate(w, temperature) if hasattr(bandwidth_provider, "evaluate")
          else bandwidth_provider)
    w, bw, physical = check_mode_inputs(w, bw, physical_mode, volume, temperature)
    if w.shape != (m,):
        raise ValueError("omega must have one entry per eigenvector mode")
    # Row shards fix the rank count; their hashes also bind restarts to exact
    # eigenvector data. Each rank checks its own file and all ranks agree.
    local_hash = sha256_file(eigenvectors.directory / eigenvectors.shard["file"])
    local_hash_error = (f"rank {rank} eigenvector row shard hash mismatch"
                        if local_hash != eigenvectors.shard["sha256"] else None)
    hash_errors = comm.allgather(local_hash_error)
    if any(hash_errors):
        raise ValueError("; ".join(error for error in hash_errors if error))
    config = {"format": "kaldo-distributed-amorphous-qhgk-v1",
              "rank_count": size, "mode_count": m,
              "eigenvector_source_sha256": sha256_json(eigenvectors.manifest),
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
    store = CheckpointStore(output_directory, config, comm=comm, label="distributed QHGK")
    blocks, stats = reduce_qhgk_blocks(
        projector, w, bw, physical, temperature=temperature, is_classic=is_classic, volume=volume,
        core_start=core_start, core_stop=core_stop, block_size=block_size,
        broadening_shape=broadening_shape, diffusivity_threshold=diffusivity_threshold,
        accumulator_dtype=accum, checkpoint=store, max_blocks=max_blocks)
    gathered = comm.gather(blocks, root=0)
    metrics = comm.gather({**stats, "rank": rank,
                           "sparse_storage_bytes": harmonic.storage_bytes,
                           "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
                          root=0)
    if rank != 0:
        return None
    block = min(block_size, max(1, m - 1))
    ownership = [(first, min(first + block, core_stop)) for first in range(core_start, core_stop, block)]
    conductivity = np.zeros(m, dtype=accum)
    diffusivity = np.zeros(m, dtype=accum)
    completed = []
    for owner, items in enumerate(gathered):
        for i0, i1, c, d, pairs in items:
            conductivity[i0:i1], diffusivity[i0:i1] = c, d
            completed.append({"start": i0, "stop": i1, "rank": owner,
                              "result_sha256": sha256_arrays(c, d), "evaluated_pairs": pairs})
    completed.sort(key=lambda item: item["start"])
    complete = len(completed) == len(ownership)
    store.write_manifest(
        mode_ownership=[{"start": a, "stop": b, "rank": projector.owner(index)}
                        for index, (a, b) in enumerate(ownership)],
        coordinate_ownership=[{"rank": index, "atom_start": a, "atom_stop": b}
                              for index, (a, b) in enumerate(harmonic.bounds)],
        completed_core_ranges=completed, complete=complete, metrics_per_rank=metrics,
        evaluated_pairs=sum(item["evaluated_pairs"] for item in completed),
        executed_pairs_this_run=sum(item["evaluated_pairs"] for item in metrics))
    if not complete:
        return None
    store.save_array("conductivity", conductivity)
    store.save_array("diffusivity", diffusivity)
    return conductivity, diffusivity

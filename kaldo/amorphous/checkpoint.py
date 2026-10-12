"""Coordinate-row eigenvector shards read without loading the full eigensystem."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kaldo.helpers.atomic_io import atomic_write_json, sha256_file
from .qhgk import NumpyEigenvectorSource


def atom_row_bounds(n_atoms, size):
    """Contiguous balanced atom ownership; coordinates follow atom ownership."""
    return [(n_atoms * rank // size, n_atoms * (rank + 1) // size)
            for rank in range(size)]


class RowShardedEigenvectorSource(NumpyEigenvectorSource):
    """A rank's coordinate rows of every mode, stored as a memory-mapped .npy.

    This is a child of the in-memory source so panel reads retain its API;
    ``read_modes`` returns only locally owned coordinate rows. Distributed
    reducers use ``global_n_coordinates`` to distinguish local from global.
    """

    def __init__(self, directory, comm):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("format") != "kaldo-eigenvector-row-shards-v1":
            raise ValueError("unsupported row-sharded eigenvector format")
        self.rank, self.size = comm.Get_rank(), comm.Get_size()
        if self.manifest["rank_count"] != self.size:
            raise ValueError("row shard rank count does not match communicator")
        self.n_atoms = self.manifest["n_atoms"]
        self.global_n_coordinates = 3 * self.n_atoms
        shards = self.manifest["shards"]
        if len(shards) != self.size:
            raise ValueError("missing row shards")
        cursor = 0
        for expected_rank, shard in enumerate(shards):
            if (shard["rank"] != expected_rank or shard["atom_start"] != cursor or
                    shard["atom_stop"] <= cursor):
                raise ValueError("row shards have missing or overlapping atom rows")
            cursor = shard["atom_stop"]
        if cursor != self.n_atoms:
            raise ValueError("row shards do not cover all atoms")
        self.shard = shards[self.rank]
        self.atom_start, self.atom_stop = self.shard["atom_start"], self.shard["atom_stop"]
        array = np.load(self.directory / self.shard["file"], mmap_mode="r",
                        allow_pickle=False)
        super().__init__(array)
        if (self.n_coordinates != 3 * (self.atom_stop - self.atom_start) or
                self.n_modes != self.manifest["n_modes"] or
                str(array.dtype) != self.manifest["dtype"]):
            raise ValueError("row shard shape does not match manifest")

    @classmethod
    def create(cls, source, directory, comm, *, panel_size=256,
               dtype=np.float32, source_sha256=None):
        """Collectively stream a full source into rank-owned coordinate rows."""
        rank, size = comm.Get_rank(), comm.Get_size()
        if (source.n_coordinates % 3 or source.n_modes < 1 or
                panel_size < 1 or size > source.n_coordinates // 3):
            raise ValueError("source must have atom-coordinate rows and at least one mode")
        n_atoms = source.n_coordinates // 3
        start, stop = atom_row_bounds(n_atoms, size)[rank]
        directory = Path(directory)
        if rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
        filename = f"rank_{rank:05d}_atoms_{start:09d}_{stop:09d}.npy"
        target = directory / filename
        mapped = np.lib.format.open_memmap(target, mode="w+", dtype=dtype,
                                           shape=(3 * (stop-start), source.n_modes))
        for first in range(0, source.n_modes, panel_size):
            last = min(first + panel_size, source.n_modes)
            if hasattr(source, "read_rows_modes"):
                mapped[:, first:last] = source.read_rows_modes(
                    3*start, 3*stop, first, last, dtype=dtype)
            else:
                mapped[:, first:last] = source.read_modes(
                    first, last, dtype=dtype)[3*start:3*stop]
        mapped.flush()
        del mapped
        local = {"rank": rank, "atom_start": start, "atom_stop": stop,
                 "file": filename, "sha256": sha256_file(target)}
        shards = comm.gather(local, root=0)
        if rank == 0:
            manifest = {"format": "kaldo-eigenvector-row-shards-v1",
                        "n_atoms": n_atoms, "n_modes": source.n_modes,
                        "rank_count": size, "dtype": str(np.dtype(dtype)),
                        "source_sha256": source_sha256, "shards": shards}
            atomic_write_json(directory / "manifest.json", manifest)
        comm.Barrier()
        return cls(directory, comm)

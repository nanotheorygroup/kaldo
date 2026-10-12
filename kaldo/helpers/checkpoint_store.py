"""Hashed, restartable block checkpoints shared by serial and MPI reductions."""

import json
from pathlib import Path

import numpy as np

from kaldo.helpers.atomic_io import atomic_save_npy, atomic_savez, atomic_write_json, sha256_arrays, sha256_json

_RESERVED = ("config_sha256", "start", "stop", "result_sha256")


class CheckpointStore:
    """A directory of per-block ``.npz`` results bound to one calculation configuration.

    ``config`` is a JSON-compatible description of the calculation. Its hash is
    stored in every block file and in ``manifest.json``; an existing manifest
    with a different hash is rejected, so a restart can never mix results from
    two configurations. Each block also records the hash of its arrays.

    With an MPI ``comm``, rank zero creates the directory and writes the
    manifest and summary arrays, and only a block's owner rank reads or writes
    that block. Whether a block exists, and whether it is valid, is agreed
    collectively, so every rank follows the same schedule and none waits on a
    collective another rank skipped.
    """

    def __init__(self, directory, config, *, comm=None, label="checkpoint"):
        self.directory = Path(directory)
        self.config = config
        self.config_hash = sha256_json(config)
        self.comm = comm
        self.rank = 0 if comm is None else comm.Get_rank()
        self.label = label
        if self.rank == 0:
            self.directory.mkdir(parents=True, exist_ok=True)
        self._barrier()
        manifest = self.directory / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("config_sha256") != self.config_hash:
            raise ValueError(f"{label} configuration changed")

    def _barrier(self):
        if self.comm is not None:
            self.comm.Barrier()

    def _bcast(self, value, root):
        return value if self.comm is None else self.comm.bcast(value, root=root)

    def block_path(self, start, stop):
        return self.directory / f"rows_{start:09d}_{stop:09d}.npz"

    def load_block(self, start, stop, *, owner=0, shapes=None):
        """Return the saved arrays for ``[start, stop)``, or ``None`` if the block is absent.

        Non-owner ranks receive an empty dict for a present block. A block with a
        foreign configuration, wrong bounds, unexpected ``shapes``, or a result
        hash mismatch raises ``ValueError`` on every rank.
        """
        path = self.block_path(start, stop)
        exists = self._bcast(path.exists() if self.rank == owner else None, owner)
        if not exists:
            return None
        arrays, error = None, None
        if self.rank == owner:
            try:
                arrays = self._read(path, start, stop, shapes or {})
            except Exception as exc:  # Any unreadable or inconsistent file must stop every rank.
                error = f"invalid checkpoint {path}: {exc}"
        error = self._bcast(error, owner)
        if error is not None:
            raise ValueError(error)
        return arrays if self.rank == owner else {}

    def _read(self, path, start, stop, shapes):
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files if name not in _RESERVED}
            if (str(saved["config_sha256"]) != self.config_hash or int(saved["start"]) != start or
                    int(saved["stop"]) != stop):
                raise ValueError("block configuration or bounds differ")
            digest = str(saved["result_sha256"])
        if digest != sha256_arrays(*(arrays[name] for name in sorted(arrays))):
            raise ValueError("block result hash differs")
        for name, shape in shapes.items():
            if arrays.get(name, np.empty(0)).shape != tuple(shape):
                raise ValueError(f"block array {name!r} has the wrong shape")
        return arrays

    def save_block(self, start, stop, **arrays):
        """Write one block atomically; call only on the block's owner rank."""
        digest = sha256_arrays(*(np.asarray(arrays[name]) for name in sorted(arrays)))
        atomic_savez(self.block_path(start, stop), config_sha256=self.config_hash, start=start, stop=stop,
                     result_sha256=digest, **arrays)

    def write_manifest(self, **fields):
        """Write ``manifest.json`` with the configuration, its hash, and ``fields`` (rank zero only)."""
        if self.rank == 0:
            atomic_write_json(self.directory / "manifest.json",
                              {**self.config, "config_sha256": self.config_hash, **fields})

    def save_array(self, name, array):
        """Write ``name.npy`` atomically (rank zero only)."""
        if self.rank == 0:
            atomic_save_npy(self.directory / f"{name}.npy", array)

"""Gamma-only ordinary IFC2 blocks from colored central differences."""

import json
from pathlib import Path

import ase.units as units
import numpy as np
from scipy.sparse import bsr_array
from scipy.sparse.linalg import LinearOperator

from kaldo.helpers.atomic_io import atomic_savez_compressed, atomic_write_json, sha256_file


_FORMAT = "kaldo-sparse-ifc2-v1"


class SparseSecondOrder:
    """Unweighted eV/Angstrom squared IFC2 in response-row BSR order.

    The stored matrix retains ordinary force constants; dynamical-matrix
    accessors apply masses and kALDo's unit conversion without changing it.
    """

    def __init__(self, matrix, *, metadata=None):
        if not isinstance(matrix, bsr_array) or matrix.blocksize != (3, 3):
            raise ValueError("sparse IFC2 must be a scipy bsr_array with 3x3 blocks")
        if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 3:
            raise ValueError("sparse IFC2 must be square with three coordinates per atom")
        if matrix.data.dtype != np.float64 or not np.isfinite(matrix.data).all():
            raise ValueError("sparse IFC2 blocks must be finite float64 values")
        self.matrix = matrix
        self.metadata = {} if metadata is None else dict(metadata)

    @property
    def n_atoms(self):
        return self.matrix.shape[0] // 3

    @property
    def nnz_blocks(self):
        return len(self.matrix.indices)

    @property
    def storage_bytes(self):
        return self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes

    def to_dense(self, *, max_bytes=512 * 1024**2):
        """Return kALDo's `(1,b,alpha,1,a,beta)` tensor for small systems."""
        required = (3 * self.n_atoms)**2 * 8
        if max_bytes is not None and required > max_bytes:
            raise MemoryError(f"dense IFC2 would need {required / 1e9:.3f} GB; use sparse IFC2 accessors")
        dense = self.matrix.toarray().reshape(self.n_atoms, 3, self.n_atoms, 3)
        return dense.transpose(2, 3, 0, 1)[np.newaxis, :, :, np.newaxis, :, :]

    def _mass_scale(self, masses):
        masses = np.asarray(masses, dtype=np.float64)
        if masses.shape != (self.n_atoms,) or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("masses must be one positive finite value per atom")
        return 1 / np.sqrt(masses)

    def mass_weighted_bsr(self, masses):
        """Return kALDo-unit dynamical blocks as a separate 3x3 BSR matrix."""
        scale = self._mass_scale(masses)
        rows = np.repeat(np.arange(self.n_atoms), np.diff(self.matrix.indptr))
        factor = units.mol / (10 * units.J)
        data = self.matrix.data * (factor * scale[rows] * scale[self.matrix.indices])[:, None, None]
        return bsr_array((data, self.matrix.indices.copy(), self.matrix.indptr.copy()),
                         shape=self.matrix.shape)

    def mass_weighted_operator(self, masses):
        """Apply the dynamical matrix without duplicating its BSR block data."""
        scale = np.repeat(self._mass_scale(masses), 3)
        factor = units.mol / (10 * units.J)
        matrix = self.matrix
        row_ids = None

        def matvec(vector):
            return factor * scale * (matrix @ (scale * vector))

        def rmatvec(vector):
            nonlocal row_ids
            if row_ids is None:
                row_ids = np.repeat(np.arange(self.n_atoms, dtype=np.int32), np.diff(matrix.indptr))
            input_blocks = (scale * vector).reshape(self.n_atoms, 3)
            output = np.zeros((self.n_atoms, 3), dtype=np.float64)
            for start in range(0, len(matrix.indices), 65536):
                stop = min(start + 65536, len(matrix.indices))
                contribution = np.einsum("kij,ki->kj", matrix.data[start:stop],
                                         input_blocks[row_ids[start:stop]])
                np.add.at(output, matrix.indices[start:stop], contribution)
            return factor * scale * output.ravel()

        def matmat(panel):
            return factor * scale[:, None] * (matrix @ (scale[:, None] * panel))

        return LinearOperator(matrix.shape, matvec=matvec, rmatvec=rmatvec,
                              matmat=matmat, dtype=np.float64)

    def save(self, directory):
        """Write BSR buffers and metadata atomically to a standalone directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "ifc2.npz"
        atomic_savez_compressed(path, data=self.matrix.data, indices=self.matrix.indices,
                                indptr=self.matrix.indptr, shape=np.asarray(self.matrix.shape, dtype=np.int64))
        metadata = {**self.metadata, "format": _FORMAT, "units": "eV/angstrom^2", "n_atoms": self.n_atoms,
                    "arrays_sha256": sha256_file(path)}
        atomic_write_json(directory / "metadata.json", metadata)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("format") != _FORMAT or metadata.get("units") != "eV/angstrom^2":
            raise ValueError("unsupported sparse IFC2 metadata")
        archive = directory / "ifc2.npz"
        if sha256_file(archive) != metadata.get("arrays_sha256"):
            raise ValueError("sparse IFC2 array hash mismatch")
        with np.load(archive, allow_pickle=False) as source:
            if set(source.files) != {"data", "indices", "indptr", "shape"}:
                raise ValueError("sparse IFC2 archive is incomplete")
            shape = tuple(map(int, source["shape"]))
            matrix = bsr_array((source["data"], source["indices"], source["indptr"]), shape=shape)
        if metadata.get("n_atoms") != shape[0] // 3:
            raise ValueError("sparse IFC2 atom count mismatch")
        return cls(matrix, metadata=metadata)

"""Gamma-only ordinary IFC2 as sparse 3x3 blocks, with the structure it belongs to.

``SparseSecondOrder`` is the sparse counterpart of :class:`SecondOrder` for
one periodic or nonperiodic cell sampled at Gamma. It stores ordinary force
constants (eV/Å²) and derives everything else on demand: the mass-weighted
dynamical matrix, a matrix-free dynamical operator, the three real derivative
operators used by QHGK, the :class:`SparseHarmonic` atom-block contract for
distributed runs, and, for small cells, a dense :class:`SecondOrder`.
"""

import json
from pathlib import Path

from ase import Atoms
from ase.geometry import find_mic
import numpy as np
from scipy.sparse import bsr_array
from scipy.sparse.linalg import LinearOperator

from kaldo.helpers.atomic_io import atomic_save_npy, atomic_savez_compressed, atomic_write_json, sha256_file
from kaldo.helpers.sparse_blocks import PAIR_BATCH
from kaldo.helpers.units import EV_TO_10J_PER_MOL
from kaldo.observables.sparse_harmonic import SparseHarmonic


_FORMAT = "kaldo-sparse-ifc2-v1"
# Minimum-image vectors must be shorter than this fraction of the narrowest
# periodic cell width to identify one image per stored pair.
_UNIQUE_IMAGE_FRACTION = 0.5


def _perpendicular_widths(atoms):
    periodic = np.flatnonzero(atoms.pbc)
    if not len(periodic):
        return np.empty(0)
    inverse = np.linalg.inv(np.asarray(atoms.cell.complete(), dtype=np.float64))
    return 1.0 / np.linalg.norm(inverse[:, periodic], axis=0)


class SparseSecondOrder:
    """Unweighted eV/Angstrom squared IFC2 in response-row BSR order.

    Block ``(a, b)`` holds the force response of atom ``a`` to displacing atom
    ``b``. ``atoms`` supplies the positions, cell, and masses that the
    dynamical-matrix and derivative accessors need. ``displacement`` optionally
    gives each stored block's pair vector ``r_a - r_b``; without it the
    minimum-image vector is used, which requires every stored pair to be
    shorter than half the narrowest periodic cell width. The stored matrix is
    never modified by the accessors.
    """

    def __init__(self, matrix, *, atoms=None, displacement=None, metadata=None):
        if not isinstance(matrix, bsr_array) or matrix.blocksize != (3, 3):
            raise ValueError("sparse IFC2 must be a scipy bsr_array with 3x3 blocks")
        if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 3:
            raise ValueError("sparse IFC2 must be square with three coordinates per atom")
        if matrix.data.dtype != np.float64 or not np.isfinite(matrix.data).all():
            raise ValueError("sparse IFC2 blocks must be finite float64 values")
        if atoms is not None and len(atoms) != matrix.shape[0] // 3:
            raise ValueError("atoms must have one entry per IFC2 block row")
        if displacement is not None:
            displacement = np.asarray(displacement, dtype=np.float64)
            if displacement.shape != (len(matrix.indices), 3) or not np.isfinite(displacement).all():
                raise ValueError("displacement must be finite with one vector per stored block")
        self.matrix = matrix
        self.atoms = atoms
        self.displacement = displacement
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

    def _require_atoms(self):
        if self.atoms is None:
            raise ValueError("this accessor needs the structure; construct SparseSecondOrder with atoms")
        return self.atoms

    def _rows(self):
        return np.repeat(np.arange(self.n_atoms), np.diff(self.matrix.indptr))

    def to_dense(self, *, max_bytes=512 * 1024**2):
        """Return kALDo's `(1,b,alpha,1,a,beta)` tensor for small systems."""
        required = (3 * self.n_atoms)**2 * 8
        if max_bytes is not None and required > max_bytes:
            raise MemoryError(f"dense IFC2 would need {required / 1e9:.3f} GB; use sparse IFC2 accessors")
        dense = self.matrix.toarray().reshape(self.n_atoms, 3, self.n_atoms, 3)
        return dense.transpose(2, 3, 0, 1)[np.newaxis, :, :, np.newaxis, :, :]

    def _mass_scale(self, masses):
        masses = self._require_atoms().get_masses() if masses is None else masses
        masses = np.asarray(masses, dtype=np.float64)
        if masses.shape != (self.n_atoms,) or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("masses must be one positive finite value per atom")
        return 1 / np.sqrt(masses)

    def mass_weighted_bsr(self, masses=None):
        """Return kALDo-unit dynamical blocks as a separate 3x3 BSR matrix.

        ``masses`` default to ``atoms.get_masses()``.
        """
        scale = self._mass_scale(masses)
        rows = self._rows()
        data = self.matrix.data * (EV_TO_10J_PER_MOL * scale[rows] * scale[self.matrix.indices])[:, None, None]
        return bsr_array((data, self.matrix.indices.copy(), self.matrix.indptr.copy()),
                         shape=self.matrix.shape)

    @property
    def dynmat(self):
        """Mass-weighted dynamical matrix in (rad/ps)^2, as a 3x3 BSR matrix."""
        return self.mass_weighted_bsr()

    def mass_weighted_operator(self, masses=None):
        """Apply the dynamical matrix without duplicating its BSR block data."""
        scale = np.repeat(self._mass_scale(masses), 3)
        factor = EV_TO_10J_PER_MOL
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
            for start in range(0, len(matrix.indices), PAIR_BATCH):
                stop = min(start + PAIR_BATCH, len(matrix.indices))
                contribution = np.einsum("kij,ki->kj", matrix.data[start:stop],
                                         input_blocks[row_ids[start:stop]])
                np.add.at(output, matrix.indices[start:stop], contribution)
            return factor * scale * output.ravel()

        def matmat(panel):
            return factor * scale[:, None] * (matrix @ (scale[:, None] * panel))

        return LinearOperator(matrix.shape, matvec=matvec, rmatvec=rmatvec,
                              matmat=matmat, dtype=np.float64)

    def block_displacement(self):
        """Pair vector ``r_a - r_b`` in Å for every stored block ``(a, b)``.

        Uses the explicit ``displacement`` when given; otherwise the
        minimum-image vector, which raises if any stored pair is too long for
        its image to be unique.
        """
        if self.displacement is not None:
            return self.displacement
        atoms = self._require_atoms()
        rows, columns = self._rows(), self.matrix.indices
        positions = np.asarray(atoms.positions, dtype=np.float64)
        widths = _perpendicular_widths(atoms)
        limit = _UNIQUE_IMAGE_FRACTION * float(widths.min()) if len(widths) else np.inf
        vectors = np.empty((self.nnz_blocks, 3), dtype=np.float64)
        for start in range(0, self.nnz_blocks, PAIR_BATCH):
            stop = min(start + PAIR_BATCH, self.nnz_blocks)
            raw = positions[rows[start:stop]] - positions[columns[start:stop]]
            vectors[start:stop], lengths = find_mic(raw, atoms.cell, atoms.pbc)
            if np.any(lengths >= limit):
                raise ValueError("a stored pair is at least half the narrowest periodic cell width, so its "
                                 "image is ambiguous; construct SparseSecondOrder with explicit displacement")
        return vectors

    def to_sparse_harmonic(self, *, dtype=np.float64):
        """Return the mass-weighted :class:`SparseHarmonic` atom-block contract.

        Its derivative operators follow the amorphous Gamma convention:
        ``(r_a - r_b) * D_ab`` for each stored block.
        """
        atoms = self._require_atoms()
        masses = np.asarray(atoms.get_masses(), dtype=np.float64)
        rows, columns = self._rows(), self.matrix.indices
        weights = EV_TO_10J_PER_MOL / np.sqrt(masses[rows] * masses[columns])
        return SparseHarmonic(
            self.matrix.indptr, self.matrix.indices,
            (self.matrix.data * weights[:, None, None]).astype(dtype),
            self.block_displacement().astype(dtype), masses=masses,
            positions=atoms.positions, cell=np.asarray(atoms.cell),
            metadata={"source_format": "sparse-ifc2-bsr",
                      "periodic_wrapping": "explicit" if self.displacement is not None else "minimum_image",
                      **{key: self.metadata[key] for key in ("plan_sha256", "processing") if key in self.metadata}})

    def derivative_operators(self, *, dtype=np.float64):
        """Three real ``(3N, 3N)`` derivative operators supporting ``@ panel``."""
        return self.to_sparse_harmonic(dtype=dtype).operators

    def to_second_order(self, folder, *, is_acoustic_sum=False, max_bytes=512 * 1024**2):
        """Return a dense one-cell :class:`SecondOrder` for the existing harmonic and QHGK paths."""
        from kaldo.observables.secondorder import SecondOrder

        return SecondOrder.from_supercell(atoms=self._require_atoms().copy(), grid_type="C", supercell=(1, 1, 1),
                                          value=self.to_dense(max_bytes=max_bytes),
                                          is_acoustic_sum=is_acoustic_sum, folder=str(folder))

    def save(self, directory):
        """Write BSR buffers, structure, and metadata atomically to a standalone directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "ifc2.npz"
        atomic_savez_compressed(path, data=self.matrix.data, indices=self.matrix.indices,
                                indptr=self.matrix.indptr, shape=np.asarray(self.matrix.shape, dtype=np.int64))
        if self.displacement is not None:
            atomic_save_npy(directory / "displacement.npy", self.displacement)
        else:
            (directory / "displacement.npy").unlink(missing_ok=True)
        self.save_atoms(directory)
        metadata = {**self.metadata, "format": _FORMAT, "units": "eV/angstrom^2", "n_atoms": self.n_atoms,
                    "arrays_sha256": sha256_file(path)}
        atomic_write_json(directory / "metadata.json", metadata)

    def save_atoms(self, directory):
        """Write ``atoms.npz`` (numbers, positions, cell, pbc, masses), or remove a stale one."""
        path = Path(directory) / "atoms.npz"
        if self.atoms is None:
            path.unlink(missing_ok=True)
            return
        atomic_savez_compressed(path, numbers=np.asarray(self.atoms.numbers, dtype=np.int32),
                                positions=np.asarray(self.atoms.positions, dtype=np.float64),
                                cell=np.asarray(self.atoms.cell, dtype=np.float64),
                                pbc=np.asarray(self.atoms.pbc, dtype=np.bool_),
                                masses=np.asarray(self.atoms.get_masses(), dtype=np.float64))

    @classmethod
    def load(cls, directory):
        """Load a saved IFC2; ``atoms`` and ``displacement`` are restored when present."""
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
        atoms = None
        if (directory / "atoms.npz").exists():
            with np.load(directory / "atoms.npz", allow_pickle=False) as saved:
                atoms = Atoms(numbers=saved["numbers"], positions=saved["positions"], cell=saved["cell"],
                              pbc=saved["pbc"], masses=saved["masses"])
        displacement = None
        if (directory / "displacement.npy").exists():
            displacement = np.load(directory / "displacement.npy", allow_pickle=False)
        return cls(matrix, atoms=atoms, displacement=displacement, metadata=metadata)

"""Atom-block sparse real Gamma harmonic data and derivative panel actions.

The displacement stored for (i,j) multiplies its mass-weighted dynamical
block to give the kALDo real amorphous derivative. For a nonperiodic legacy
matrix this is ``positions[i] - positions[j]`` in Å. Periodic producers must
provide their actual pair/image displacement; a single aggregate block cannot
represent several images with different displacements.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import argparse

import ase.units as units
import numpy as np
from scipy.sparse import bsr_matrix


def apply_derivative_panel(row_ptr, column_atom, blocks, displacement,
                           axis, panel, shape):
    """Apply one block sparse derivative without persisting derivative values."""
    data = blocks * displacement[:, axis, None, None]
    matrix = bsr_matrix((data, column_atom, row_ptr), shape=shape)
    return matrix @ panel


class DerivativeView:
    """One Cartesian derivative with the graph shared by all three views."""

    def __init__(self, harmonic, axis):
        self.harmonic = harmonic
        self.axis = axis
        self.shape = (3 * harmonic.n_atoms, 3 * harmonic.n_atoms)

    def astype(self, dtype, copy=False):
        return DerivativeView(self.harmonic.astype(dtype), self.axis)

    def __matmul__(self, panel):
        panel = np.asarray(panel)
        if panel.ndim != 2 or panel.shape[0] != self.shape[1]:
            raise ValueError("panel has the wrong coordinate dimension")
        # Only one direction's sparse values exist during this product. The
        # row/column graph and base dynamical blocks are never duplicated.
        return apply_derivative_panel(
            self.harmonic.row_ptr, self.harmonic.column_atom,
            self.harmonic.blocks, self.harmonic.displacement,
            self.axis, panel, self.shape)


class SparseAmorphousHarmonic:
    """Directed atom-pair BSR graph with mass-weighted 3x3 blocks."""

    def __init__(self, row_ptr, column_atom, blocks, displacement, *, masses=None,
                 positions=None, cell=None, metadata=None):
        self.row_ptr = np.asarray(row_ptr)
        self.column_atom = np.asarray(column_atom)
        self.blocks = np.asarray(blocks)
        self.displacement = np.asarray(displacement)
        self.masses = None if masses is None else np.asarray(masses, dtype=np.float64)
        self.positions = None if positions is None else np.asarray(positions, dtype=np.float64)
        self.cell = None if cell is None else np.asarray(cell, dtype=np.float64)
        self.metadata = {} if metadata is None else dict(metadata)
        self.n_atoms = len(self.row_ptr) - 1
        nb = len(self.column_atom)
        if (self.n_atoms < 1 or self.row_ptr.ndim != 1 or
                self.row_ptr.dtype.kind not in "iu" or
                self.column_atom.dtype.kind not in "iu" or
                self.row_ptr[0] != 0 or self.row_ptr[-1] != nb or
                np.any(np.diff(self.row_ptr) < 0) or
                np.any(self.column_atom >= self.n_atoms) or
                self.blocks.shape != (nb, 3, 3) or
                self.displacement.shape != (nb, 3) or
                self.blocks.dtype.kind != "f" or
                self.displacement.dtype.kind != "f" or
                not np.isfinite(self.blocks).all() or
                not np.isfinite(self.displacement).all()):
            raise ValueError("invalid atom-block harmonic arrays")
        for row in range(self.n_atoms):
            columns = self.column_atom[self.row_ptr[row]:self.row_ptr[row + 1]]
            if len(columns) and (np.any(np.diff(columns) < 0) or columns[0] < 0):
                raise ValueError("column atoms must be sorted within each row")
        if self.masses is not None and (self.masses.shape != (self.n_atoms,) or
                                        not np.isfinite(self.masses).all() or
                                        np.any(self.masses <= 0)):
            raise ValueError("masses must be positive finite values per atom")
        if self.positions is not None and (self.positions.shape != (self.n_atoms, 3) or
                                           not np.isfinite(self.positions).all()):
            raise ValueError("positions must have shape (N,3) and be finite")
        if self.cell is not None and (self.cell.shape != (3, 3) or
                                      not np.isfinite(self.cell).all()):
            raise ValueError("cell must have shape (3,3) and be finite")
        self._casts = {}

    @property
    def operators(self):
        return tuple(DerivativeView(self, axis) for axis in range(3))

    @property
    def storage_bytes(self):
        return sum(a.nbytes for a in (self.row_ptr, self.column_atom,
                                       self.blocks, self.displacement,
                                       self.masses, self.positions, self.cell) if a is not None)

    def astype(self, dtype):
        dtype = np.dtype(dtype)
        if self.blocks.dtype == dtype and self.displacement.dtype == dtype:
            return self
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("sparse harmonic work dtype must be float32 or float64")
        if dtype not in self._casts:
            self._casts[dtype] = SparseAmorphousHarmonic(
                self.row_ptr, self.column_atom, self.blocks.astype(dtype),
                self.displacement.astype(dtype), masses=self.masses,
                positions=self.positions, cell=self.cell,
                metadata=self.metadata)
        return self._casts[dtype]

    def validation_report(self, *, tolerance=1e-8, require_acoustic_sum=False):
        """Check reverse pairs, onsite terms, skew derivatives, and ASR."""
        def find_group(row, column):
            start, stop = int(self.row_ptr[row]), int(self.row_ptr[row + 1])
            columns = self.column_atom[start:stop]
            left = start + int(np.searchsorted(columns, column, side="left"))
            right = start + int(np.searchsorted(columns, column, side="right"))
            return range(left, right)

        missing_reverse = 0
        missing_onsite = 0
        onsite_displacement = 0.
        symmetry = 0.
        reverse_displacement = 0.
        skew = 0.
        asr = 0.
        position_displacement = 0.
        for i in range(self.n_atoms):
            if not find_group(i, i):
                missing_onsite += 1
            row_sum = np.zeros((3, 3), dtype=np.float64)
            for p in range(self.row_ptr[i], self.row_ptr[i + 1]):
                j = int(self.column_atom[p])
                d = self.blocks[p].astype(np.float64)
                r = self.displacement[p].astype(np.float64)
                if j == i:
                    onsite_displacement = max(onsite_displacement, float(np.max(np.abs(r))))
                    symmetry = max(symmetry, float(np.max(np.abs(d - d.T))))
                if (self.positions is not None and
                        self.metadata.get("periodic_wrapping") == "none"):
                    position_displacement = max(position_displacement,
                        float(np.max(np.abs(r - (self.positions[i] - self.positions[j])))))
                if self.masses is not None:
                    row_sum += d * np.sqrt(self.masses[i] * self.masses[j])
            if self.masses is not None:
                asr = max(asr, float(np.max(np.abs(row_sum))))
            columns = self.column_atom[self.row_ptr[i]:self.row_ptr[i+1]]
            for j in np.unique(columns):
                j = int(j)
                if j < i:
                    if not find_group(j, i):
                        missing_reverse += len(find_group(i, j))
                    continue
                if j == i:
                    continue
                forward = list(find_group(i, j))
                reverse = list(find_group(j, i))
                if len(forward) == len(reverse) == 1:
                    p, q = forward[0], reverse[0]
                    d, r = self.blocks[p].astype(np.float64), self.displacement[p].astype(np.float64)
                    dr, rr = self.blocks[q].astype(np.float64), self.displacement[q].astype(np.float64)
                    symmetry = max(symmetry, float(np.max(np.abs(d-dr.T))))
                    reverse_displacement = max(reverse_displacement,
                                               float(np.max(np.abs(r+rr))))
                    for axis in range(3):
                        skew = max(skew, float(np.max(np.abs(r[axis]*d + rr[axis]*dr.T))))
                    continue
                candidates = {}
                for p in forward:
                    d, r = self.blocks[p], self.displacement[p]
                    candidates[p] = sorted(
                        (q for q in reverse if
                         max(float(np.max(np.abs(d-self.blocks[q].T))),
                             float(np.max(np.abs(r+self.displacement[q])))) <= tolerance),
                        key=lambda q: max(float(np.max(np.abs(d-self.blocks[q].T))),
                                          float(np.max(np.abs(r+self.displacement[q])))))
                matched = {}

                def augment(p, seen):
                    for q in candidates[p]:
                        if q in seen:
                            continue
                        seen.add(q)
                        if q not in matched or augment(matched[q], seen):
                            matched[q] = p
                            return True
                    return False

                for p in forward:
                    if not augment(p, set()):
                        missing_reverse += 1
                for q, p in matched.items():
                    d = self.blocks[p].astype(np.float64)
                    r = self.displacement[p].astype(np.float64)
                    dr = self.blocks[q].astype(np.float64)
                    rr = self.displacement[q].astype(np.float64)
                    symmetry = max(symmetry, float(np.max(np.abs(d-dr.T))))
                    reverse_displacement = max(reverse_displacement,
                                               float(np.max(np.abs(r+rr))))
                    for axis in range(3):
                        skew = max(skew, float(np.max(np.abs(r[axis]*d + rr[axis]*dr.T))))
                missing_reverse += len(reverse)-len(matched)
        report = dict(missing_reverse=missing_reverse, missing_onsite=missing_onsite,
                      onsite_displacement=onsite_displacement,
                      symmetry_residual=symmetry,
                      reverse_displacement_residual=reverse_displacement,
                      skew_derivative_residual=skew,
                      position_displacement_residual=(position_displacement
                                                       if self.positions is not None and
                                                       self.metadata.get("periodic_wrapping") == "none"
                                                       else None),
                      acoustic_sum_residual=asr if self.masses is not None else None)
        if (missing_reverse or missing_onsite or
                max(onsite_displacement, symmetry, reverse_displacement,
                    skew, position_displacement) > tolerance or
                (require_acoustic_sum and (self.masses is None or asr > tolerance))):
            raise ValueError(f"sparse harmonic validation failed: {report}")
        return report

    def save(self, directory, *, validation_tolerance=1e-5):
        """Write the direct sparse input contract as separate NumPy arrays."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        report = self.validation_report(tolerance=validation_tolerance)
        for name, array in (("row_ptr", self.row_ptr),
                            ("column_atom", self.column_atom),
                            ("dynamical_blocks", self.blocks),
                            ("displacement", self.displacement)):
            np.save(directory / (name + ".npy"), array)
        for name, array in (("masses", self.masses),
                            ("positions", self.positions), ("cell", self.cell)):
            path = directory / (name + ".npy")
            if array is None:
                path.unlink(missing_ok=True)
            else:
                np.save(path, array)
        hashes = {}
        for name in ("row_ptr", "column_atom", "dynamical_blocks",
                     "displacement", "masses", "positions", "cell"):
            path = directory / (name + ".npy")
            if not path.exists():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4*1024*1024), b""):
                    digest.update(chunk)
            hashes[path.name] = digest.hexdigest()
        meta = {**self.metadata, "format": "kaldo-amorphous-atom-block-v1",
                "n_atoms": self.n_atoms, "n_blocks": len(self.column_atom),
                "mass_weighted": True, "length_unit": "angstrom",
                "block_dtype": str(self.blocks.dtype),
                "displacement_dtype": str(self.displacement.dtype),
                "array_sha256": hashes,
                "validation": report,
                "validation_tolerance": validation_tolerance,
                "acoustic_sum_status": "measured" if self.masses is not None else "not_measured"}
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("format") != "kaldo-amorphous-atom-block-v1":
            raise ValueError("unsupported sparse harmonic format")
        if metadata.get("mass_weighted") is not True or metadata.get("length_unit") != "angstrom":
            raise ValueError("sparse harmonic requires mass-weighted blocks and angstrom displacement")
        def read(name):
            return np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        masses = read("masses") if (directory / "masses.npy").exists() else None
        positions = read("positions") if (directory / "positions.npy").exists() else None
        cell = read("cell") if (directory / "cell.npy").exists() else None
        result = cls(read("row_ptr"), read("column_atom"), read("dynamical_blocks"),
                     read("displacement"), masses=masses, positions=positions,
                     cell=cell, metadata=metadata)
        if result.n_atoms != metadata["n_atoms"] or len(result.column_atom) != metadata["n_blocks"]:
            raise ValueError("sparse harmonic metadata count mismatch")
        result.validation_report(tolerance=metadata.get("validation_tolerance", 1e-5))
        return result


def from_sparse_force_constants(row_ptr, column_atom, force_blocks,
                                displacement, masses, *, positions=None,
                                cell=None, apply_acoustic_sum=True,
                                symmetrize=False,
                                force_constant_unit="eV/angstrom^2",
                                dtype=np.float32):
    """Mass-weight atom-block IFC2 and optionally enforce the kALDo ASR.

    The ASR follows ``SecondOrder.acoustic_sum_rule``: subtract the complete
    IFC row sum from its onsite block before mass weighting. The eV/Å²
    conversion uses the same ``units.mol/(10*units.J)`` factor as
    ``SecondOrder.calculate_dynmat``. No dense coordinate matrix is formed.
    """
    masses = np.asarray(masses, dtype=np.float64)
    ptr = np.asarray(row_ptr)
    columns = np.asarray(column_atom)
    displacement = np.asarray(displacement, dtype=np.float64)
    force = np.array(force_blocks, dtype=np.float64, copy=True)
    if (masses.ndim != 1 or not len(masses) or
            not np.isfinite(masses).all() or np.any(masses <= 0) or
            ptr.ndim != 1 or ptr.dtype.kind not in "iu" or
            columns.ndim != 1 or columns.dtype.kind not in "iu" or
            len(ptr) != len(masses)+1 or ptr[0] != 0 or
            ptr[-1] != len(columns) or np.any(np.diff(ptr) < 0) or
            np.any(columns >= len(masses)) or
            displacement.shape != (len(columns), 3) or
            not np.isfinite(displacement).all() or
            force.shape != (len(columns), 3, 3) or
            not np.isfinite(force).all()):
        raise ValueError("invalid sparse force-constant arrays")
    for atom in range(len(masses)):
        row_columns = columns[ptr[atom]:ptr[atom+1]]
        if (np.any(row_columns < 0) or np.any(np.diff(row_columns) < 0)):
            raise ValueError("sparse force-constant columns must be sorted and valid")
    if force_constant_unit == "eV/angstrom^2":
        factor = units.mol/(10*units.J)
    elif force_constant_unit == "kaldo_10J_per_mol":
        factor = 1.
    else:
        raise ValueError("unsupported force-constant unit")
    if symmetrize:
        for atom in range(len(masses)):
            start, stop = int(ptr[atom]), int(ptr[atom+1])
            row_columns = columns[start:stop]
            for neighbor in np.unique(row_columns):
                neighbor = int(neighbor)
                forward = range(start+int(np.searchsorted(row_columns, neighbor, side="left")),
                                start+int(np.searchsorted(row_columns, neighbor, side="right")))
                if neighbor == atom:
                    for p in forward:
                        force[p] = (force[p]+force[p].T)/2
                    continue
                if neighbor < atom:
                    continue
                other_start, other_stop = int(ptr[neighbor]), int(ptr[neighbor+1])
                other_columns = columns[other_start:other_stop]
                reverse = list(range(
                    other_start+int(np.searchsorted(other_columns, atom, side="left")),
                    other_start+int(np.searchsorted(other_columns, atom, side="right"))))
                if len(forward) != len(reverse):
                    raise ValueError("cannot symmetrize unmatched reverse atom blocks")
                used = set()
                for p in forward:
                    choices = [q for q in reverse if q not in used]
                    if not choices:
                        raise ValueError("cannot symmetrize unmatched reverse atom blocks")
                    q = min(choices, key=lambda index: np.max(
                        np.abs(displacement[p]+displacement[index])))
                    if np.max(np.abs(displacement[p]+displacement[q])) > 1e-5:
                        raise ValueError("cannot symmetrize unmatched reverse displacements")
                    used.add(q)
                    average = (force[p]+force[q].T)/2
                    force[p], force[q] = average, average.T
    for atom in range(len(masses)):
        start, stop = int(ptr[atom]), int(ptr[atom+1])
        onsite = np.flatnonzero(columns[start:stop] == atom)
        if len(onsite) != 1:
            raise ValueError("one onsite force-constant block is required per atom")
        if apply_acoustic_sum:
            force[start+onsite[0]] -= np.sum(force[start:stop], axis=0)
        force[start:stop] *= factor / np.sqrt(masses[atom]*masses[columns[start:stop]])[:,None,None]
    result = SparseAmorphousHarmonic(
        ptr, columns, force.astype(dtype), displacement, masses=masses,
        positions=positions, cell=cell,
        metadata={"source_format": "sparse-ifc2",
                  "force_constant_unit": force_constant_unit,
                  "acoustic_sum_applied": bool(apply_acoustic_sum),
                  "symmetrized": bool(symmetrize)})
    result.validation_report(tolerance=1e-5,
                             require_acoustic_sum=apply_acoustic_sum)
    return result


def convert_legacy_dense(dynamical_path, positions_path, output_directory, *,
                         masses_path=None, derivative_paths=None, tolerance=1e-8,
                         dtype=np.float32, apply_acoustic_sum=False):
    """Stream a nonperiodic legacy dense .npy dynamical matrix by atom row.

    The source is already mass weighted. Its zero 3x3 blocks are omitted;
    onsite blocks are retained. Optional derivative paths are checked against
    the displacement convention in bounded 3-row slabs before saving.
    """
    dynamical_path = Path(dynamical_path)
    dense = np.load(dynamical_path, mmap_mode="r", allow_pickle=False)
    positions = np.load(positions_path, mmap_mode="r", allow_pickle=False)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("positions must have shape (N,3) in angstrom")
    n = len(positions)
    if dense.shape != (3*n, 3*n) or np.iscomplexobj(dense):
        raise ValueError("legacy dynamical matrix must be real (3N,3N)")
    slabs = (dense[3*i:3*i+3, :] for i in range(n))
    return _convert_legacy_slabs(
        slabs, dynamical_path, "legacy-dense-npy", positions_path,
        output_directory, masses_path=masses_path,
        derivative_paths=derivative_paths, tolerance=tolerance,
        dtype=dtype, apply_acoustic_sum=apply_acoustic_sum)


def _convert_legacy_slabs(slabs, source_path, source_format, positions_path,
                          output_directory, *, masses_path, derivative_paths,
                          tolerance, dtype, apply_acoustic_sum):
    positions = np.load(positions_path, mmap_mode="r", allow_pickle=False)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("positions must have shape (N,3) in angstrom")
    n = len(positions)
    masses = None if masses_path is None else np.load(masses_path, mmap_mode="r")
    if apply_acoustic_sum and masses is None:
        raise ValueError("masses are required to apply the acoustic sum rule")
    derivatives = None if derivative_paths is None else [
        np.load(path, mmap_mode="r", allow_pickle=False) for path in derivative_paths]
    if derivatives is not None and (len(derivatives) != 3 or
                                    any(a.shape != (3*n, 3*n) for a in derivatives)):
        raise ValueError("provide three derivative matrices of matching shape")
    ptr = [0]
    columns, blocks, displacement = [], [], []
    max_derivative_error = 0.
    for i, source_slab in enumerate(slabs):
        if i >= n:
            raise ValueError("legacy source has too many coordinate rows")
        slab = np.array(source_slab, dtype=np.float64, copy=True)
        if slab.shape != (3, 3*n) or not np.isfinite(slab).all():
            raise ValueError("legacy source has an invalid coordinate slab")
        if apply_acoustic_sum:
            row_blocks = slab.reshape(3,n,3).transpose(1,0,2)
            force_sum = np.einsum("jab,j->ab", row_blocks,
                                  np.sqrt(masses[i]*masses))
            row_blocks[i] -= force_sum/masses[i]
        for j in range(n):
            block = slab[:, 3*j:3*j+3]
            if i != j and not np.any(block):
                continue
            shift = np.asarray(positions[i] - positions[j])
            if derivatives is not None:
                for axis in range(3):
                    saved = derivatives[axis][3*i:3*i+3, 3*j:3*j+3]
                    max_derivative_error = max(max_derivative_error,
                        float(np.max(np.abs(saved - block * shift[axis]))))
            columns.append(j)
            blocks.append(np.array(block, dtype=dtype))
            displacement.append(np.array(shift, dtype=dtype))
        ptr.append(len(columns))
    if len(ptr) != n+1:
        raise ValueError("legacy source has too few coordinate rows")
    if max_derivative_error > tolerance:
        raise ValueError(f"legacy derivatives require pair/image data; residual={max_derivative_error}")
    def file_hash(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    harmonic = SparseAmorphousHarmonic(
        np.asarray(ptr, dtype=np.int64), np.asarray(columns, dtype=np.int32),
        np.asarray(blocks, dtype=dtype).reshape(-1, 3, 3),
        np.asarray(displacement, dtype=dtype).reshape(-1, 3), masses=masses,
        positions=positions,
        metadata={"source_sha256": file_hash(source_path),
                  "positions_sha256": file_hash(positions_path),
                  "masses_sha256": None if masses_path is None else file_hash(masses_path),
                  "source_format": source_format, "periodic_wrapping": "none",
                  "acoustic_sum_applied": bool(apply_acoustic_sum),
                  "derivative_check_max_abs": max_derivative_error})
    harmonic.save(output_directory)
    return harmonic


def convert_legacy_dynform(dynform_path, positions_path, output_directory, *,
                           masses_path=None, derivative_paths=None,
                           tolerance=1e-8, dtype=np.float32,
                           apply_acoustic_sum=False):
    """Stream an amorphous one-replica ``Dyn.form`` without pandas or dense D.

    kALDo's ``eskm_io.import_second`` first multiplies this file by the
    ten-J/mol conversion and square-root masses; ``SecondOrder.dynmat``
    divides by those same factors. Thus the raw file values are already the
    mass-weighted dynamical matrix in kALDo's rad/ps squared convention.
    Text tokens may be arranged as a matrix or a flattened vector.
    """
    positions = np.load(positions_path, mmap_mode="r", allow_pickle=False)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N,3)")
    n = len(positions)
    dynform_path = Path(dynform_path)
    def slabs():
        with dynform_path.open("rt") as stream:
            def tokens():
                for line in stream:
                    for value in np.fromstring(line, sep=" "):
                        yield value
            values = tokens()
            for _ in range(n):
                slab = np.fromiter(values, dtype=np.float64, count=9*n)
                if slab.size != 9*n:
                    raise ValueError("Dyn.form has too few matrix entries")
                yield slab.reshape(3, 3*n)
            if next(values, None) is not None:
                raise ValueError("Dyn.form has too many matrix entries")
    return _convert_legacy_slabs(
        slabs(), dynform_path, "legacy-dynform-one-replica",
        positions_path, output_directory, masses_path=masses_path,
        derivative_paths=derivative_paths, tolerance=tolerance,
        dtype=dtype, apply_acoustic_sum=apply_acoustic_sum)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert nonperiodic .npy or one-replica Dyn.form harmonic input")
    parser.add_argument("dynamical", type=Path)
    parser.add_argument("positions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--masses", type=Path)
    parser.add_argument("--derivative", type=Path, action="append", default=[],
                        help="one each for x, y, z; verifies the displacement convention")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--apply-acoustic-sum", action="store_true")
    args = parser.parse_args(argv)
    if args.derivative and len(args.derivative) != 3:
        parser.error("provide exactly three --derivative paths")
    converter = convert_legacy_dense if args.dynamical.suffix == ".npy" else convert_legacy_dynform
    harmonic = converter(
        args.dynamical, args.positions, args.output, masses_path=args.masses,
        derivative_paths=args.derivative or None, dtype=args.dtype,
        apply_acoustic_sum=args.apply_acoustic_sum)
    print(json.dumps({"n_atoms": harmonic.n_atoms,
                      "n_blocks": len(harmonic.column_atom),
                      "storage_bytes": harmonic.storage_bytes,
                      "validation": harmonic.validation_report()}, indent=2))


if __name__ == "__main__":
    main()

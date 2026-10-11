"""Portable, verified input plan for disconnected colored IFC2 workers."""

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path

import ase
from ase import Atoms
import numpy as np
import scipy
from scipy.sparse import csr_array

import kaldo
from kaldo.helpers.atomic_io import atomic_savez_compressed, atomic_write_json, sha256_file
from ._common import PLAN_FORMAT, atoms_extra_arrays_hash, atoms_hash, atoms_info_hash
from .structure import _DEFAULT_PAIR_LIMIT, build_neighbor_structure, color_structure, validate_coloring


def _metadata_hash(metadata):
    payload = {key: value for key, value in metadata.items() if key != "plan_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _row_ids(indptr):
    return np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))


def _segment_positions(indptr, items):
    """Flat positions of the CSR segments ``indptr[i]:indptr[i + 1]`` for ``items``, and their lengths."""
    starts = indptr[items].astype(np.int64)
    counts = indptr[items + 1].astype(np.int64) - starts
    offsets = np.repeat(starts - (np.cumsum(counts) - counts), counts)
    return offsets + np.arange(len(offsets), dtype=np.int64), counts


def _csr_from_arrays(indptr, indices, n_atoms):
    indptr = np.asarray(indptr, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int32)
    if (indptr.shape != (n_atoms + 1,) or indptr[0] != 0 or indptr[-1] != len(indices) or
            np.any(np.diff(indptr) < 0) or np.any(indices < 0) or np.any(indices >= n_atoms)):
        raise ValueError("invalid neighbor CSR arrays in colored IFC plan")
    rows = _row_ids(indptr)
    same_row = rows[1:] == rows[:-1]
    if np.any(np.diff(indptr) == 0) or np.any(np.diff(indices)[same_row] <= 0):
        raise ValueError("neighbor rows must be nonempty, sorted, and unique")
    onsite = np.zeros(n_atoms, dtype=np.bool_)
    onsite[rows[indices == rows]] = True
    if not onsite.all():
        raise ValueError("neighbor graph must include every onsite block")
    matrix = csr_array((np.ones(len(indices), dtype=np.bool_), indices, indptr), shape=(n_atoms, n_atoms))
    matrix.has_sorted_indices = True
    return matrix


def _contains_pattern(outer, inner):
    """True when every stored entry of canonical CSR ``inner`` is stored in ``outer``."""
    n_columns = outer.shape[1]
    outer_keys = _row_ids(outer.indptr) * n_columns + outer.indices
    inner_keys = _row_ids(inner.indptr) * n_columns + inner.indices
    found = np.searchsorted(outer_keys, inner_keys)
    return bool(np.all(found < len(outer_keys)) and
                np.array_equal(outer_keys[np.minimum(found, len(outer_keys) - 1)], inner_keys))


@dataclass(frozen=True)
class ColoredSecondOrderPlan:
    """One-cell force input, two cached atom graphs, and a fixed task table."""

    directory: Path
    atoms: Atoms
    color: csr_array
    store: csr_array
    labels: np.ndarray
    tasks: np.ndarray
    metadata: dict

    @property
    def plan_hash(self):
        return self.metadata["plan_sha256"]

    @property
    def n_atoms(self):
        return len(self.atoms)

    @property
    def n_groups(self):
        return int(self.labels.max()) + 1

    @property
    def delta_shift(self):
        return float(self.metadata["delta_shift_angstrom"])

    @cached_property
    def _group_index(self):
        order = np.argsort(self.labels, kind="stable").astype(np.int32)
        bounds = np.searchsorted(self.labels[order], np.arange(self.n_groups + 1))
        return order, bounds

    @cached_property
    def _store_by_column(self):
        """Storage graph in column-major order, with each entry's row-major BSR block slot."""
        slots = np.argsort(self.store.indices, kind="stable")
        rows = _row_ids(self.store.indptr)[slots].astype(np.int32)
        indptr = np.zeros(self.n_atoms + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(np.bincount(self.store.indices, minlength=self.n_atoms))
        return indptr, rows, slots

    def group_members(self, group):
        """Atoms displaced together in ``group``, in ascending index order."""
        order, bounds = self._group_index
        return order[bounds[group]:bounds[group + 1]]

    def group_pattern(self, group):
        """Stored ``(rows, columns, slots)`` records for one displaced group.

        ``rows`` are responding atoms, ``columns`` the displaced atom of each
        record, and ``slots`` the record's block index in the BSR data. Records
        are ordered by displaced atom, then responding atom.
        """
        members = self.group_members(group)
        indptr, rows, slots = self._store_by_column
        positions, counts = _segment_positions(indptr, members)
        return rows[positions], np.repeat(members, counts), slots[positions]

    def group_color_rows(self, group):
        """Responding atoms structurally coupled to ``group`` at the coloring radius."""
        positions, _ = _segment_positions(self.color.indptr, self.group_members(group))
        return self.color.indices[positions]


def load_colored_second_order_plan(plan_dir):
    """Load a plan only after checking its file, structure, and task hashes."""
    directory = Path(plan_dir)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("format") != PLAN_FORMAT or metadata.get("plan_sha256") != _metadata_hash(metadata):
        raise ValueError("colored IFC plan metadata or hash mismatch")
    archive = directory / "plan.npz"
    if sha256_file(archive) != metadata.get("arrays_sha256"):
        raise ValueError("colored IFC plan array hash mismatch")
    with np.load(archive, allow_pickle=False) as source:
        expected = {"numbers", "positions", "cell", "pbc", "color_indptr", "color_indices",
                    "store_indptr", "store_indices", "labels", "tasks"}
        if set(source.files) != expected:
            raise ValueError("colored IFC plan arrays are incomplete")
        arrays = {key: source[key] for key in expected}
    atoms = Atoms(numbers=arrays["numbers"], positions=arrays["positions"],
                  cell=arrays["cell"], pbc=arrays["pbc"])
    n_atoms = len(atoms)
    if atoms_hash(atoms) != metadata.get("atoms_sha256"):
        raise ValueError("colored IFC plan atoms hash mismatch")
    color = _csr_from_arrays(arrays["color_indptr"], arrays["color_indices"], n_atoms)
    store = _csr_from_arrays(arrays["store_indptr"], arrays["store_indices"], n_atoms)
    if (color != color.T).nnz or (store != store.T).nnz:
        raise ValueError("colored IFC neighbor graphs must be symmetric")
    labels = np.asarray(arrays["labels"], dtype=np.int32)
    validate_coloring(color, labels)
    n_groups = int(labels.max()) + 1
    if not np.array_equal(np.unique(labels), np.arange(n_groups)):
        raise ValueError("colored IFC plan contains an empty color group")
    tasks = np.asarray(arrays["tasks"], dtype=np.int32)
    expected_tasks = np.array([(group, axis) for group in range(n_groups) for axis in range(3)],
                              dtype=np.int32)
    if (tasks.shape != expected_tasks.shape or not np.array_equal(tasks, expected_tasks) or
            metadata.get("n_groups") != n_groups or metadata.get("n_tasks") != len(tasks)):
        raise ValueError("colored IFC plan task table mismatch")
    if not _contains_pattern(color, store):
        raise ValueError("storage graph is not contained in coloring graph")
    return ColoredSecondOrderPlan(directory, atoms, color, store, labels, tasks, metadata)


def plan_colored_second_order(atoms, color_radius, store_radius, *, delta_shift=1e-3,
                              plan_dir, seeds=(0, 1, 2), calculator_description,
                              pair_memory_limit=_DEFAULT_PAIR_LIMIT):
    """Create or reuse the immutable coloring and task checkpoint.

    The input is interpreted as one Gamma-only physical cell.
    The storage graph is a distance filter of the coloring graph, so only one
    neighbor search runs.
    """
    if not isinstance(calculator_description, str) or not calculator_description.strip():
        raise ValueError("calculator_description must be a nonempty string")
    if atoms.constraints:
        raise ValueError("colored IFC plan requires unconstrained atoms for force derivatives")
    extra_hash, extra_names = atoms_extra_arrays_hash(atoms)
    info_hash = atoms_info_hash(atoms)
    color_radius = float(color_radius)
    store_radius = float(store_radius)
    if (not np.isfinite(color_radius) or not np.isfinite(store_radius) or
            not 0 < store_radius <= color_radius):
        raise ValueError("require 0 < store_radius <= color_radius")
    if not np.isfinite(delta_shift) or delta_shift <= 0:
        raise ValueError("delta_shift must be positive")
    seeds = tuple(int(seed) for seed in seeds)
    directory = Path(plan_dir)
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        plan = load_colored_second_order_plan(directory)
        if (plan.metadata["atoms_sha256"] != atoms_hash(atoms) or
                plan.metadata["extra_arrays_sha256"] != extra_hash or
                plan.metadata["atoms_info_sha256"] != info_hash or
                plan.metadata["color_radius_angstrom"] != color_radius or
                plan.metadata["store_radius_angstrom"] != store_radius or
                plan.delta_shift != delta_shift or plan.metadata["seeds"] != list(seeds) or
                plan.metadata["calculator_description"] != calculator_description):
            raise ValueError("existing colored IFC plan does not match requested inputs")
        return plan
    if directory.exists():
        unexpected = {path.name for path in directory.iterdir()} - {"plan.npz", "plan.npz.tmp", "metadata.json.tmp"}
        if unexpected:
            raise ValueError("plan directory contains files but has no valid metadata")
    directory.mkdir(parents=True, exist_ok=True)
    color = build_neighbor_structure(atoms, color_radius, pair_memory_limit=pair_memory_limit)
    store = color.restrict(store_radius)
    coloring = color_structure(color, seeds=seeds)
    tasks = np.array([(group, axis) for group in range(coloring.n_groups) for axis in range(3)], dtype=np.int32)
    archive = directory / "plan.npz"
    atomic_savez_compressed(archive,
                            numbers=np.asarray(atoms.numbers, dtype=np.int32),
                            positions=np.asarray(atoms.positions, dtype=np.float64),
                            cell=np.asarray(atoms.cell, dtype=np.float64),
                            pbc=np.asarray(atoms.pbc, dtype=np.bool_),
                            color_indptr=np.asarray(color.matrix.indptr, dtype=np.int32),
                            color_indices=np.asarray(color.matrix.indices, dtype=np.int32),
                            store_indptr=np.asarray(store.matrix.indptr, dtype=np.int32),
                            store_indices=np.asarray(store.matrix.indices, dtype=np.int32),
                            labels=np.asarray(coloring.labels, dtype=np.int32), tasks=tasks)
    metadata = {"format": PLAN_FORMAT, "atoms_sha256": atoms_hash(atoms),
                "extra_arrays_sha256": extra_hash, "extra_array_names": extra_names,
                "atoms_info_sha256": info_hash, "has_atoms_info": bool(atoms.info),
                "arrays_sha256": sha256_file(archive),
                "color_radius_angstrom": color_radius, "store_radius_angstrom": store_radius,
                "delta_shift_angstrom": float(delta_shift), "difference_scheme": "central-3point",
                "n_atoms": len(atoms), "n_groups": coloring.n_groups, "n_tasks": len(tasks),
                "seeds": list(seeds), "calculator_description": calculator_description,
                "versions": {"kaldo": kaldo.__version__, "scipy": scipy.__version__, "ase": ase.__version__}}
    metadata["plan_sha256"] = _metadata_hash(metadata)
    atomic_write_json(metadata_path, metadata)
    return load_colored_second_order_plan(directory)

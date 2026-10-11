"""Verify IFC2 chunks, assemble sparse blocks, and enforce symmetry and ASR."""

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse import bsr_array

from kaldo.helpers.atomic_io import atomic_write_json, sha256_file
from kaldo.helpers.logger import get_logger
from kaldo.observables.sparse_secondorder import SparseSecondOrder
from ._common import MERGE_FORMAT, SHARD_FORMAT
from .plan import load_colored_second_order_plan
from .processing import ALGORITHM_VERSION, process_sparse_ifc


_LOG = get_logger()


def _load_shards(plan, shard_dirs):
    shards = []
    all_tasks = []
    descriptions = set()
    for directory in map(Path, shard_dirs):
        manifest = json.loads((directory / "manifest.json").read_text())
        shard = manifest.get("shard")
        if (manifest.get("format") != SHARD_FORMAT or manifest.get("plan_sha256") != plan.plan_hash or
                manifest.get("atoms_sha256") != plan.metadata["atoms_sha256"] or
                not isinstance(shard, list) or len(shard) != 2 or shard[1] < 1 or
                not 0 <= shard[0] < shard[1]):
            raise ValueError(f"shard manifest plan, atoms, or shard mismatch: {directory}")
        descriptions.add(manifest["calculator_description"])
        listed = []
        for chunk in manifest["chunks"]:
            if Path(chunk["file"]).name != chunk["file"]:
                raise ValueError("shard chunk filename must be local to workdir")
            path = directory / chunk["file"]
            if not path.is_file() or sha256_file(path) != chunk["sha256"]:
                raise ValueError(f"shard chunk missing or corrupt: {path}")
            listed.extend(chunk["task_ids"])
        if sorted(listed) != manifest["completed_task_ids"]:
            raise ValueError(f"shard completed task list differs from chunks: {directory}")
        if any(task < 0 or task >= len(plan.tasks) or task % shard[1] != shard[0] for task in listed):
            raise ValueError(f"shard contains an unassigned task: {directory}")
        all_tasks.extend(listed)
        shards.append((directory, manifest))
    if len(descriptions) != 1:
        raise ValueError("shards used different calculator descriptions")
    counts = np.bincount(all_tasks, minlength=len(plan.tasks))
    missing = np.flatnonzero(counts == 0).tolist()
    duplicate = np.flatnonzero(counts > 1).tolist()
    if missing or duplicate:
        raise ValueError(f"colored IFC merge has missing task ids {missing} and duplicated task ids {duplicate}")
    return shards, descriptions.pop()


def _read_chunk(path, listed_tasks):
    with np.load(path, allow_pickle=False) as source:
        expected = {"task_ids", "offsets", "rows", "columns", "values", "outside_max", "truncation_max"}
        if set(source.files) != expected:
            raise ValueError(f"chunk record arrays are incomplete: {path}")
        arrays = {name: source[name] for name in expected}
    tasks = arrays["task_ids"]
    offsets = arrays["offsets"]
    n_records = len(arrays["rows"])
    if (tasks.ndim != 1 or not np.array_equal(tasks, listed_tasks) or
            offsets.shape != (len(tasks) + 1,) or offsets[0] != 0 or offsets[-1] != n_records or
            np.any(np.diff(offsets) < 0) or arrays["columns"].shape != (n_records,) or
            arrays["values"].shape != (n_records, 3) or
            arrays["outside_max"].shape != tasks.shape or arrays["truncation_max"].shape != tasks.shape or
            not np.isfinite(arrays["values"]).all() or
            not np.isfinite(arrays["outside_max"]).all() or
            not np.isfinite(arrays["truncation_max"]).all()):
        raise ValueError(f"invalid colored IFC chunk records: {path}")
    return arrays


def _log_processing(options, report, *, cached=False):
    _LOG.info("Colored IFC processing: version=%s cached=%s symmetrize=%s apply_acoustic_sum=%s "
              "asr_mode=%s asr_tolerance=%g asr_max_iter=%d",
              report["algorithm_version"], cached, options["symmetrize"],
              options["apply_acoustic_sum"], options["asr_mode"],
              options["asr_tolerance"], options["asr_max_iter"])
    for stage in ("before", "after"):
        residual = report[stage]
        _LOG.info("Colored IFC %s residuals: asymmetry=%.6e row_sum=%.6e column_sum=%.6e",
                  stage, residual["relative_asymmetry"], residual["relative_row_sum"],
                  residual["relative_column_sum"])
    counts = report["solver_iterations"]
    _LOG.info("Colored IFC ASR solver: symmetric_iterations=%s skew_iterations=%s status=%s",
              counts["symmetric"], counts["skew"], report["solver_status"])


def merge_colored_second_order(plan_dir, shard_dirs, *, output_dir=None, warning_threshold=1e-3,
                               apply_acoustic_sum=True, asr_mode="projection", symmetrize=True,
                               asr_tolerance=1e-10, asr_max_iter=1000):
    """Merge a complete task partition into a processed 3x3 BSR IFC2.

    The precomputed sorted storage graph is the canonical lexicographic block
    order. Chunks are read one at a time and scattered into that pattern, so
    merge memory is proportional to the output BSR, not all task records.
    Projection ASR preserves the fixed block pattern, transpose symmetry,
    and zero atom-row sums. ``match_dense`` reproduces kALDo's legacy onsite
    correction, which sums BSR columns and may break transpose symmetry.
    """
    options = {"apply_acoustic_sum": bool(apply_acoustic_sum), "asr_mode": asr_mode,
               "symmetrize": bool(symmetrize), "asr_tolerance": float(asr_tolerance),
               "asr_max_iter": asr_max_iter, "algorithm_version": ALGORITHM_VERSION}
    if (asr_mode not in ("projection", "match_dense") or
            (apply_acoustic_sum and asr_mode == "projection" and not symmetrize) or
            not np.isfinite(asr_tolerance) or not 0 < asr_tolerance < 1 or
            not isinstance(asr_max_iter, int) or asr_max_iter < 1):
        raise ValueError("invalid sparse IFC processing options")
    plan = load_colored_second_order_plan(plan_dir)
    shards, description = _load_shards(plan, shard_dirs)
    source_entries = [(manifest["shard"], [(chunk["file"], chunk["sha256"], chunk["task_ids"])
                                          for chunk in manifest["chunks"]])
                      for _, manifest in shards]
    source_entries.sort(key=lambda item: item[0])
    source_hash = hashlib.sha256(json.dumps([description, source_entries], sort_keys=True).encode()).hexdigest()
    processing_hash = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()
    output_dir = Path(output_dir) if output_dir is not None else plan.directory / "merged"
    if (output_dir / "metadata.json").exists() and (output_dir / "diagnostics.json").exists():
        existing_metadata = json.loads((output_dir / "metadata.json").read_text())
        if existing_metadata.get("plan_sha256") != plan.plan_hash:
            raise ValueError("existing merged IFC belongs to a different plan")
        if (existing_metadata.get("source_sha256") == source_hash and
                existing_metadata.get("processing_sha256") == processing_hash):
            try:
                result = SparseSecondOrder.load(output_dir)
                diagnostics = json.loads((output_dir / "diagnostics.json").read_text())
                _log_processing(options, diagnostics["processing_report"], cached=True)
                return result
            except (OSError, ValueError, KeyError, TypeError):
                pass  # Interrupted output write: reconstruct from verified chunks.
    data = np.zeros((plan.store.nnz, 3, 3), dtype=np.float64)
    max_outside = max_truncation = 0.0
    for directory, manifest in shards:
        for chunk in manifest["chunks"]:
            arrays = _read_chunk(directory / chunk["file"], chunk["task_ids"])
            for index, task_id in enumerate(arrays["task_ids"]):
                group, axis = map(int, plan.tasks[int(task_id)])
                first, last = map(int, arrays["offsets"][index:index + 2])
                rows = arrays["rows"][first:last]
                columns = arrays["columns"][first:last]
                values = arrays["values"][first:last]
                expected_rows, expected_columns, slots = plan.group_pattern(group)
                if not np.array_equal(rows, expected_rows) or not np.array_equal(columns, expected_columns):
                    raise ValueError(f"task {task_id} has missing, duplicated, or reordered IFC records")
                data[slots, :, axis] = values
                max_outside = max(max_outside, float(arrays["outside_max"][index]))
                max_truncation = max(max_truncation, float(arrays["truncation_max"][index]))
    matrix = bsr_array((data, plan.store.indices.copy(), plan.store.indptr.copy()),
                       shape=(3 * plan.n_atoms, 3 * plan.n_atoms))
    row_ids = np.repeat(np.arange(plan.n_atoms), np.diff(plan.store.indptr))
    onsite = np.flatnonzero(plan.store.indices == row_ids)
    median_self = float(np.median(np.linalg.norm(data[onsite], axis=(1, 2))))
    denominator = max(median_self, np.finfo(float).tiny)
    _LOG.info("Colored IFC processing start: version=%s options=%s", ALGORITHM_VERSION, options)
    try:
        report = process_sparse_ifc(matrix, apply_acoustic_sum=apply_acoustic_sum,
                                    asr_mode=asr_mode, symmetrize=symmetrize,
                                    asr_tolerance=asr_tolerance, asr_max_iter=asr_max_iter)
    except RuntimeError as exc:
        _LOG.error("Colored IFC ASR solver: status=failed version=%s error=%s", ALGORITHM_VERSION, exc)
        raise
    diagnostics = {"format": MERGE_FORMAT, "plan_sha256": plan.plan_hash,
                   "source_sha256": source_hash,
                   "processing_sha256": processing_hash,
                   "atoms_sha256": plan.metadata["atoms_sha256"], "calculator_description": description,
                   "n_shards": len(shards), "n_tasks": len(plan.tasks), "n_groups": plan.n_groups,
                   "n_blocks": plan.store.nnz, "maximum_outside_response": max_outside,
                   "maximum_truncation_response": max_truncation,
                   "median_self_block_norm": median_self,
                   "relative_outside_response": max_outside / denominator,
                   "relative_truncation_response": max_truncation / denominator,
                   "processing_options": options, "processing_report": report}
    for label in ("outside", "truncation"):
        if diagnostics[f"relative_{label}_response"] > warning_threshold:
            warnings.warn(f"colored IFC {label} response exceeds {warning_threshold:g} of median onsite block",
                          RuntimeWarning, stacklevel=2)
    result = SparseSecondOrder(matrix, metadata={"plan_sha256": plan.plan_hash,
                                                 "atoms_sha256": plan.metadata["atoms_sha256"],
                                                 "source_sha256": source_hash,
                                                 "processing_sha256": processing_hash,
                                                 "plan_metadata": plan.metadata,
                                                 "processing": ("projection" if apply_acoustic_sum and
                                                                asr_mode == "projection" else
                                                                "match_dense" if apply_acoustic_sum else
                                                                "symmetric" if symmetrize else "raw")})
    result.save(output_dir)
    atomic_write_json(output_dir / "diagnostics.json", diagnostics)
    _LOG.info("Merged colored IFC: %d blocks, %d tasks, %d shards", plan.store.nnz,
              len(plan.tasks), len(shards))
    _log_processing(options, report)
    return result

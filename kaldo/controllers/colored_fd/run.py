"""Independent, restartable central-difference force tasks for one shard."""

import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from kaldo.helpers.atomic_io import atomic_savez_compressed, atomic_write_json, sha256_file
from kaldo.helpers.logger import get_logger
from ._common import SHARD_FORMAT, atoms_extra_arrays_hash, atoms_hash, atoms_info_hash, calculator_instance
from .plan import load_colored_second_order_plan


_LOG = get_logger()


def _force_pair(work, initial_positions, members, axis, delta_shift):
    forces = []
    for sign in (1, -1):
        work.positions[:] = initial_positions
        work.positions[members, axis] += sign * delta_shift
        measured = np.asarray(work.get_forces(), dtype=np.float64)
        if measured.shape != initial_positions.shape or not np.isfinite(measured).all():
            raise ValueError("calculator returned nonfinite or incorrectly shaped forces")
        forces.append(measured)
    work.positions[:] = initial_positions
    return (forces[0] - forces[1]) / (2 * delta_shift)


def _decompress_task(plan, group, derivative):
    rows, columns, _ = plan.group_pattern(group)
    covered_color = np.zeros(plan.n_atoms, dtype=np.bool_)
    covered_color[plan.group_color_rows(group)] = True
    covered_store = np.zeros(plan.n_atoms, dtype=np.bool_)
    covered_store[rows] = True
    values = np.asarray(-derivative[rows], dtype=np.float64)
    outside = float(np.max(np.abs(derivative[~covered_color]), initial=0.0))
    truncated = float(np.max(np.abs(derivative[covered_color & ~covered_store]), initial=0.0))
    return rows, columns, values, outside, truncated


def _flush_chunk(workdir, manifest, buffered):
    if not buffered:
        return
    task_ids = np.array([task[0] for task in buffered], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum([len(task[1]) for task in buffered], dtype=np.int64)))
    rows = np.concatenate([task[1] for task in buffered]).astype(np.int32, copy=False)
    columns = np.concatenate([task[2] for task in buffered]).astype(np.int32, copy=False)
    values = np.concatenate([task[3] for task in buffered]).astype(np.float64, copy=False)
    outside = np.array([task[4] for task in buffered], dtype=np.float64)
    truncated = np.array([task[5] for task in buffered], dtype=np.float64)
    filename = f"chunk_{task_ids[0]:09d}_{task_ids[-1]:09d}.npz"
    path = workdir / filename
    atomic_savez_compressed(path, task_ids=task_ids, offsets=offsets, rows=rows, columns=columns,
                            values=values, outside_max=outside, truncation_max=truncated)
    manifest["chunks"].append({"file": filename, "sha256": sha256_file(path),
                               "task_ids": task_ids.tolist()})
    completed = set(manifest["completed_task_ids"])
    completed.update(task_ids.tolist())
    manifest["completed_task_ids"] = sorted(completed)
    atomic_write_json(workdir / "manifest.json", manifest)


def _local_chunk_path(workdir, chunk):
    if Path(chunk["file"]).name != chunk["file"]:
        raise ValueError("shard chunk filename must be local to workdir")
    return workdir / chunk["file"]


def _load_manifest(workdir, plan, shard, description, restart):
    path = workdir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else None
    discarded = []
    if manifest is not None:
        if (manifest.get("format") != SHARD_FORMAT or manifest.get("plan_sha256") != plan.plan_hash or
                manifest.get("atoms_sha256") != plan.metadata["atoms_sha256"] or
                manifest.get("shard") != list(shard)):
            raise ValueError("shard manifest plan, atoms, or shard hash mismatch")
        if restart:
            discarded = [_local_chunk_path(workdir, chunk) for chunk in manifest["chunks"]]
            manifest = None
        elif manifest.get("calculator_description") != description:
            raise ValueError("shard calculator description changed; pass restart=True to discard this "
                             "shard's progress")
    if manifest is None:
        manifest = {"format": SHARD_FORMAT, "plan_sha256": plan.plan_hash,
                    "atoms_sha256": plan.metadata["atoms_sha256"],
                    "calculator_description": description, "shard": list(shard),
                    "completed_task_ids": [], "chunks": []}
        atomic_write_json(path, manifest)
        # Remove superseded chunks only after the empty manifest no longer lists them.
        for chunk_path in discarded:
            chunk_path.unlink(missing_ok=True)
        return manifest
    listed = []
    for chunk in manifest["chunks"]:
        chunk_path = _local_chunk_path(workdir, chunk)
        if not chunk_path.is_file() or sha256_file(chunk_path) != chunk["sha256"]:
            raise ValueError(f"shard chunk is missing or corrupt: {chunk_path}")
        with np.load(chunk_path, allow_pickle=False) as source:
            if not np.array_equal(source["task_ids"], chunk["task_ids"]):
                raise ValueError(f"shard chunk task IDs differ from manifest: {chunk_path}")
        listed.extend(chunk["task_ids"])
    if (len(listed) != len(set(listed)) or sorted(listed) != manifest["completed_task_ids"] or
            any(task < 0 or task >= len(plan.tasks) or task % shard[1] != shard[0] for task in listed)):
        raise ValueError("shard manifest completed task table is inconsistent")
    return manifest


def run_colored_second_order(plan_dir, calculator, *, shard=(0, 1), workdir,
                             calculator_description, atoms=None, restart=False,
                             allow_description_mismatch=False, flush_tasks=8, flush_seconds=120):
    """Run every assigned group/direction task and checkpoint decompressed IFCs.

    Task ``k`` belongs to shard ``k % n_shards``. All atoms in the selected
    group move together by ``+delta_shift`` and ``-delta_shift`` along one
    Cartesian direction. The two complete force arrays live only in memory;
    chunk files contain the derived, sparse IFC column records.

    ``restart=True`` discards this shard's completed tasks and chunk files,
    which is required to resume under a different ``calculator_description``.
    ``allow_description_mismatch=True`` permits a description that differs
    from the one recorded in the plan; merge still requires all shards to
    agree.
    """
    if sys.platform != "linux":
        _LOG.warning("Colored IFC progress logs report peak memory from resource.ru_maxrss, which Linux "
                     "reports in KiB and macOS in bytes; the logged MB value is only correct on Linux")
    plan = load_colored_second_order_plan(plan_dir)
    if atoms is not None and atoms.constraints:
        raise ValueError("colored IFC worker requires unconstrained atoms")
    if atoms is not None and atoms_hash(atoms) != plan.metadata["atoms_sha256"]:
        raise ValueError("worker atoms hash does not match colored IFC plan")
    if (plan.metadata["extra_array_names"] or plan.metadata["has_atoms_info"]) and atoms is None:
        raise ValueError("worker must supply atoms with the plan's extra ASE arrays or info")
    if atoms is not None and atoms_extra_arrays_hash(atoms)[0] != plan.metadata["extra_arrays_sha256"]:
        raise ValueError("worker extra ASE arrays do not match colored IFC plan")
    if atoms is not None and atoms_info_hash(atoms) != plan.metadata["atoms_info_sha256"]:
        raise ValueError("worker ASE info does not match colored IFC plan")
    shard = tuple(shard)
    if (len(shard) != 2 or not all(isinstance(value, (int, np.integer)) for value in shard) or
            shard[1] < 1 or not 0 <= shard[0] < shard[1]):
        raise ValueError("shard must be (index, count) with 0 <= index < count")
    if not isinstance(calculator_description, str) or not calculator_description.strip():
        raise ValueError("calculator_description must be a nonempty string")
    if calculator_description != plan.metadata["calculator_description"] and not allow_description_mismatch:
        raise ValueError("calculator description does not match colored IFC plan; pass "
                         "allow_description_mismatch=True to run it anyway")
    if not isinstance(flush_tasks, int) or flush_tasks < 1 or flush_seconds <= 0:
        raise ValueError("flush_tasks and flush_seconds must be positive")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(workdir, plan, shard, calculator_description, restart)
    assigned = np.arange(shard[0], len(plan.tasks), shard[1], dtype=np.int32)
    completed = set(manifest["completed_task_ids"])
    remaining = [int(task) for task in assigned if int(task) not in completed]
    if not remaining:
        return manifest
    # Use the caller's verified ASE object when supplied; some calculators
    # read extra per-atom arrays that the portable geometric plan does not
    # serialize. The four force-defining geometric arrays are hash-checked.
    work = atoms.copy() if atoms is not None else plan.atoms.copy()
    work.calc = calculator_instance(calculator)
    initial_positions = work.positions.copy()
    buffered = []
    started = last_flush = time.monotonic()
    for count, task_id in enumerate(remaining, 1):
        group, axis = map(int, plan.tasks[task_id])
        derivative = _force_pair(work, initial_positions, plan.group_members(group), axis, plan.delta_shift)
        rows, columns, values, outside, truncated = _decompress_task(plan, group, derivative)
        buffered.append((task_id, rows, columns, values, outside, truncated))
        now = time.monotonic()
        if len(buffered) >= flush_tasks or now - last_flush >= flush_seconds or count == len(remaining):
            _flush_chunk(workdir, manifest, buffered)
            buffered.clear()
            last_flush = now
            elapsed = max(now - started, np.finfo(float).tiny)
            rate = 2 * count / elapsed
            eta = 2 * (len(remaining) - count) / rate
            peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            _LOG.info("Colored IFC shard %d/%d: %d/%d tasks, %.2f calls/s, ETA %.1fs, peak %.1f MB",
                      shard[0], shard[1], count, len(remaining), rate, eta, peak_mb)
    return manifest

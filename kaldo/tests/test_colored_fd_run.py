"""Checkpointed group displacement, shard merge, and dense sign/layout checks."""

from functools import partial
import json
import logging
import sys

import numpy as np
import pytest

from kaldo.controllers.colored_fd import (load_colored_second_order_plan,
                                          merge_colored_second_order, plan_colored_second_order,
                                          run_colored_second_order)
from kaldo.controllers.colored_fd.plan import _csr_from_arrays
from kaldo.controllers.displacement import calculate_second
from kaldo.observables.sparse_secondorder import SparseSecondOrder
from kaldo.tests.colored_fd_helpers import ToyFiniteRange, chain as _chain


def _plan(atoms, root, *, radius=2.5):
    return plan_colored_second_order(atoms, radius, radius, plan_dir=root / "plan",
                                     calculator_description="toy-finite-range-v1",
                                     delta_shift=1e-3)


class RecordingFiniteRange(ToyFiniteRange):

    def __init__(self, positions):
        super().__init__(many_body=True)
        self.original = positions.copy()
        self.displaced_counts = []

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
        self.displaced_counts.append(int(np.count_nonzero(
            np.linalg.norm(atoms.positions - self.original, axis=1) > 1e-12)))
        super().calculate(atoms, properties, system_changes)


class FailingFiniteRange(ToyFiniteRange):

    def __init__(self, max_calls):
        super().__init__(many_body=True)
        self.max_calls = max_calls

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
        if self.calls >= self.max_calls:
            raise RuntimeError("injected calculator interruption")
        super().calculate(atoms, properties, system_changes)


def test_mover_displaces_every_member_of_large_groups(tmp_path):
    atoms = _chain(15)
    plan = _plan(atoms, tmp_path)
    assert max(np.bincount(plan.labels)) > 2
    recorder = RecordingFiniteRange(atoms.positions)
    manifest = run_colored_second_order(plan.directory, recorder, workdir=tmp_path / "shard",
                                         calculator_description="toy-finite-range-v1",
                                         flush_tasks=2)
    assert len(manifest["completed_task_ids"]) == len(plan.tasks)
    assert recorder.calls == 2 * len(plan.tasks)
    assert max(recorder.displaced_counts) > 2
    assert sorted(set(recorder.displaced_counts)) == sorted(set(np.bincount(plan.labels)))


def test_raw_merge_matches_dense_many_body_ifc(tmp_path):
    atoms = _chain(15)
    plan = _plan(atoms, tmp_path)
    run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True),
                             workdir=tmp_path / "shard", calculator_description="toy-finite-range-v1")
    merged = merge_colored_second_order(plan.directory, [tmp_path / "shard"],
                                        output_dir=tmp_path / "merged", symmetrize=False,
                                        apply_acoustic_sum=False)
    diagnostics = json.loads((tmp_path / "merged" / "diagnostics.json").read_text())
    assert diagnostics["maximum_outside_response"] < 1e-12
    dense = calculate_second(atoms, atoms.copy(), 1e-3, calculator=partial(ToyFiniteRange, many_body=True))
    np.testing.assert_allclose(merged.to_dense(), dense, rtol=1e-10, atol=1e-12)


def test_sparse_save_load_and_merge_cache(tmp_path):
    plan = _plan(_chain(9), tmp_path)
    run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True),
                             workdir=tmp_path / "shard", calculator_description="toy-finite-range-v1")
    merged = merge_colored_second_order(plan.directory, [tmp_path / "shard"], output_dir=tmp_path / "merged")
    loaded = SparseSecondOrder.load(tmp_path / "merged")
    np.testing.assert_array_equal(loaded.matrix.data, merged.matrix.data)
    saved_time = (tmp_path / "merged" / "ifc2.npz").stat().st_mtime_ns
    merge_colored_second_order(plan.directory, [tmp_path / "shard"], output_dir=tmp_path / "merged")
    assert (tmp_path / "merged" / "ifc2.npz").stat().st_mtime_ns == saved_time


def test_chunk_files_store_derived_ifc_not_raw_forces(tmp_path):
    plan = _plan(_chain(9), tmp_path)
    run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True),
                             workdir=tmp_path / "shard", calculator_description="toy-finite-range-v1")
    with np.load(next((tmp_path / "shard").glob("chunk_*.npz")), allow_pickle=False) as source:
        assert "forces" not in source.files and "values" in source.files


def test_shard_counts_merge_bitwise_identically(tmp_path):
    atoms = _chain(12)
    plan = _plan(atoms, tmp_path)
    outputs = []
    for n_shards in (1, 2, 3):
        directories = []
        for index in range(n_shards):
            directory = tmp_path / f"run_{n_shards}" / f"shard_{index}"
            run_colored_second_order(plan.directory, partial(ToyFiniteRange, many_body=True),
                                     shard=(index, n_shards), workdir=directory,
                                     calculator_description="toy-finite-range-v1", flush_tasks=2)
            directories.append(directory)
        result = merge_colored_second_order(plan.directory, directories,
                                            output_dir=tmp_path / f"merged_{n_shards}")
        outputs.append(result.matrix.data)
    for output in outputs[1:]:
        np.testing.assert_array_equal(output, outputs[0])


def test_checkpoint_resume_and_input_guards(tmp_path):
    atoms = _chain(12)
    plan = _plan(atoms, tmp_path)
    directory = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="injected"):
        run_colored_second_order(plan.directory, FailingFiniteRange(5), workdir=directory,
                                 calculator_description="toy-finite-range-v1", flush_tasks=2)
    saved = load_colored_second_order_plan(plan.directory)
    assert saved.plan_hash == plan.plan_hash
    with pytest.raises(ValueError, match="atoms hash"):
        moved = atoms.copy()
        moved.positions[0, 0] += 0.01
        run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True), atoms=moved,
                                 workdir=directory, calculator_description="toy-finite-range-v1")
    with pytest.raises(ValueError, match="description"):
        run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True),
                                 workdir=directory, calculator_description="different")
    resumed = ToyFiniteRange(many_body=True)
    manifest = run_colored_second_order(plan.directory, resumed, workdir=directory,
                                         calculator_description="toy-finite-range-v1", flush_tasks=2)
    assert len(manifest["completed_task_ids"]) == len(plan.tasks)
    assert resumed.calls == 2 * (len(plan.tasks) - 2)
    result = merge_colored_second_order(plan.directory, [directory], output_dir=tmp_path / "resumed")
    fresh_dir = tmp_path / "fresh"
    run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True), workdir=fresh_dir,
                             calculator_description="toy-finite-range-v1")
    fresh = merge_colored_second_order(plan.directory, [fresh_dir], output_dir=tmp_path / "fresh_result")
    np.testing.assert_array_equal(result.matrix.data, fresh.matrix.data)


def test_merge_rejects_missing_duplicates_and_modified_plan(tmp_path):
    atoms = _chain(10)
    plan = _plan(atoms, tmp_path)
    directories = []
    for index in (0, 1):
        directory = tmp_path / f"shard_{index}"
        run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True),
                                 shard=(index, 2), workdir=directory,
                                 calculator_description="toy-finite-range-v1")
        directories.append(directory)
    with pytest.raises(ValueError, match="missing task ids"):
        merge_colored_second_order(plan.directory, directories[:1])
    with pytest.raises(ValueError, match="duplicated task ids"):
        merge_colored_second_order(plan.directory, directories + directories[:1])
    with (plan.directory / "plan.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="array hash mismatch"):
        run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True),
                                 workdir=tmp_path / "other", calculator_description="toy-finite-range-v1")


def test_worker_checks_extra_ase_arrays(tmp_path):
    atoms = _chain(5)
    atoms.set_initial_charges(np.arange(len(atoms), dtype=float))
    atoms.info["model"] = "charged-toy"
    plan = _plan(atoms, tmp_path)
    with pytest.raises(ValueError, match="extra ASE arrays"):
        run_colored_second_order(plan.directory, ToyFiniteRange(), workdir=tmp_path / "missing_arrays",
                                 calculator_description="toy-finite-range-v1")
    changed = atoms.copy()
    changed.set_initial_charges(np.zeros(len(atoms)))
    with pytest.raises(ValueError, match="extra ASE arrays"):
        run_colored_second_order(plan.directory, ToyFiniteRange(), atoms=changed,
                                 workdir=tmp_path / "different_arrays",
                                 calculator_description="toy-finite-range-v1")
    changed.set_initial_charges(atoms.get_initial_charges())
    changed.info["model"] = "different"
    with pytest.raises(ValueError, match="ASE info"):
        run_colored_second_order(plan.directory, ToyFiniteRange(), atoms=changed,
                                 workdir=tmp_path / "different_info",
                                 calculator_description="toy-finite-range-v1")
    run_colored_second_order(plan.directory, ToyFiniteRange(), atoms=atoms,
                             workdir=tmp_path / "same_arrays",
                             calculator_description="toy-finite-range-v1")


def test_short_coloring_radius_reports_outside_response(tmp_path):
    plan = _plan(_chain(9), tmp_path, radius=1.5)
    run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True),
                             workdir=tmp_path / "shard", calculator_description="toy-finite-range-v1")
    with pytest.warns(RuntimeWarning, match="outside response"):
        merge_colored_second_order(plan.directory, [tmp_path / "shard"],
                                   output_dir=tmp_path / "merged")
    diagnostics = json.loads((tmp_path / "merged" / "diagnostics.json").read_text())
    assert diagnostics["relative_outside_response"] > 1e-3


def test_restart_discards_shard_after_calculator_change(tmp_path):
    plan = _plan(_chain(5), tmp_path)
    directory = tmp_path / "shard"
    with pytest.raises(RuntimeError, match="injected"):
        run_colored_second_order(plan.directory, FailingFiniteRange(3), workdir=directory,
                                 calculator_description="toy-finite-range-v1", flush_tasks=1)
    with pytest.raises(ValueError, match="allow_description_mismatch"):
        run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True), workdir=directory,
                                 calculator_description="changed-model")
    with pytest.raises(ValueError, match="restart=True"):
        run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True), workdir=directory,
                                 calculator_description="changed-model", allow_description_mismatch=True)
    calculator = ToyFiniteRange(many_body=True)
    manifest = run_colored_second_order(plan.directory, calculator, workdir=directory,
                                         calculator_description="changed-model",
                                         allow_description_mismatch=True, restart=True)
    assert calculator.calls == 2 * len(plan.tasks)
    assert manifest["calculator_description"] == "changed-model"
    listed = {chunk["file"] for chunk in manifest["chunks"]}
    assert {path.name for path in directory.glob("chunk_*.npz")} == listed


def test_run_warns_that_peak_memory_units_are_linux_only(tmp_path, monkeypatch, caplog):
    plan = _plan(_chain(3), tmp_path)
    monkeypatch.setattr(sys, "platform", "darwin")
    with caplog.at_level(logging.WARNING, logger="kaldo"):
        run_colored_second_order(plan.directory, ToyFiniteRange(), workdir=tmp_path / "shard",
                                 calculator_description="toy-finite-range-v1")
    assert "ru_maxrss" in caplog.text


@pytest.mark.parametrize("indptr,indices,message", [
    ([0, 2, 4], [1, 0, 0, 1], "sorted"),
    ([0, 2, 4], [0, 0, 0, 1], "sorted"),
    ([0, 0, 2], [0, 1], "nonempty"),
    ([0, 1, 2], [1, 0], "onsite"),
])
def test_plan_graph_loader_rejects_malformed_rows(indptr, indices, message):
    with pytest.raises(ValueError, match=message):
        _csr_from_arrays(indptr, indices, 2)

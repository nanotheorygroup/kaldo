"""Unit tests for kaldo.parallel.dispatch_with_resume.

These tests pin the dispatcher's contract: resume detection, sentinel
atomicity, error propagation, and the serial/parallel parity that lets
callers ignore which backend the executor picked.
"""
import os
import time

import pytest

from kaldo.parallel import dispatch_with_resume


def _identity(uid):
    return uid * 2


def test_empty_work_ids_yields_nothing(tmp_path):
    results = list(dispatch_with_resume(
        [], _identity, n_workers=1, output_dir=str(tmp_path),
    ))
    assert results == []


def test_resume_skips_units_with_existing_sentinel(tmp_path):
    """Units whose .done sentinel already exists must be skipped."""
    for uid in (1, 3):
        open(tmp_path / f"unit_{uid:05d}.done", "w").close()

    out = sorted(dispatch_with_resume(
        [0, 1, 2, 3, 4], _identity,
        n_workers=1, output_dir=str(tmp_path),
    ))

    assert out == [(0, 0), (2, 4), (4, 8)]


def test_resume_writes_sentinel_after_worker_returns(tmp_path):
    """Sentinel must be created by the helper after the worker returns."""
    list(dispatch_with_resume(
        [0, 1], _identity, n_workers=1, output_dir=str(tmp_path),
    ))

    assert (tmp_path / "unit_00000.done").exists()
    assert (tmp_path / "unit_00001.done").exists()


def test_no_output_dir_means_no_sentinel(tmp_path):
    """When output_dir is None the helper writes nothing to disk."""
    list(dispatch_with_resume(
        [0, 1], _identity, n_workers=1, output_dir=None,
    ))

    assert list(tmp_path.iterdir()) == []


def _raises_for_unit_one(uid):
    if uid == 1:
        raise RuntimeError("worker exploded")
    return uid


def test_failed_worker_leaves_no_sentinel(tmp_path):
    """If the worker raises, no sentinel exists for that unit."""
    with pytest.raises(RuntimeError, match="worker exploded"):
        list(dispatch_with_resume(
            [0, 1, 2], _raises_for_unit_one,
            n_workers=1, output_dir=str(tmp_path),
        ))

    assert not (tmp_path / "unit_00001.done").exists()


def test_parallel_backend_matches_serial(tmp_path):
    """Parallel and serial backends produce the same set of results."""
    serial = dict(dispatch_with_resume(
        list(range(8)), _identity, n_workers=1,
    ))
    parallel = dict(dispatch_with_resume(
        list(range(8)), _identity, n_workers=2,
    ))
    assert serial == parallel

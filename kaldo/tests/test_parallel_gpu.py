"""Tests for GPU-pinned parallel execution in kaldo.parallel.executor.

All tests are mock-only and require no CUDA hardware. The race-freedom test
in Task 5 actually spawns worker processes and inspects os.environ, but still
needs no GPU.
"""

import os
import sys
import warnings

import pytest

from kaldo.parallel import get_executor


def test_gpu_ids_validates_type():
    with pytest.raises(TypeError, match="gpu_ids must be a list"):
        get_executor(backend='process', gpu_ids='0,1')


def test_gpu_ids_validates_element_type():
    with pytest.raises(TypeError, match="gpu_ids must be a list of int"):
        get_executor(backend='process', gpu_ids=['0', '1'])


def test_gpu_ids_validates_duplicates():
    with pytest.raises(ValueError, match="unique"):
        get_executor(backend='process', gpu_ids=[0, 0])


def test_gpu_ids_validates_negative():
    with pytest.raises(ValueError, match="non-negative"):
        get_executor(backend='process', gpu_ids=[-1])


def test_process_empty_gpu_ids_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        get_executor(backend='process', gpu_ids=[])


def test_mpi_rejects_gpu_ids():
    with pytest.raises(NotImplementedError, match="gpu_ids is not supported"):
        get_executor(backend='mpi', gpu_ids=[0])


def test_mpi_rejects_empty_gpu_ids():
    with pytest.raises(NotImplementedError, match="gpu_ids is not supported"):
        get_executor(backend='mpi', gpu_ids=[])


def test_serial_multi_gpu_rejected():
    with pytest.raises(ValueError, match="at most one GPU"):
        get_executor(backend='serial', gpu_ids=[0, 1])


def test_serial_empty_gpu_ids_is_noop(monkeypatch):
    # Remove CUDA_VISIBLE_DEVICES if present, assert it stays removed.
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning becomes an exception
        get_executor(backend='serial', gpu_ids=[])
    assert 'CUDA_VISIBLE_DEVICES' not in os.environ


def test_serial_single_gpu_sets_env_var(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    # Ensure torch/tensorflow aren't in sys.modules so we don't warn.
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    monkeypatch.delitem(sys.modules, 'tensorflow', raising=False)
    get_executor(backend='serial', gpu_ids=[3])
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'


def test_serial_single_gpu_warns_when_torch_imported(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setitem(sys.modules, 'torch', object())
    with pytest.warns(RuntimeWarning, match="may be too late"):
        get_executor(backend='serial', gpu_ids=[2])


def test_serial_single_gpu_warns_when_tensorflow_imported(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setitem(sys.modules, 'tensorflow', object())
    with pytest.warns(RuntimeWarning, match="may be too late"):
        get_executor(backend='serial', gpu_ids=[2])


def test_nworkers_defaults_to_gpu_count():
    exe = get_executor(backend='process', gpu_ids=[0, 1, 2])
    try:
        assert exe._max_workers == 3
    finally:
        exe.shutdown(wait=True)


def test_nworkers_matches_gpu_count_explicit():
    exe = get_executor(backend='process', n_workers=2, gpu_ids=[0, 1])
    try:
        assert exe._max_workers == 2
    finally:
        exe.shutdown(wait=True)


def test_nworkers_mismatch_rejected():
    with pytest.raises(ValueError, match="n_workers .* must equal len"):
        get_executor(backend='process', n_workers=4, gpu_ids=[0, 1])


# Module-level function so ProcessPoolExecutor can pickle it.
def _return_cuda_visible_devices(_):
    # Small sleep forces the pool to spawn all workers rather than reusing
    # a single fast worker for every task.
    import time
    time.sleep(0.05)
    return os.environ.get('CUDA_VISIBLE_DEVICES')


def test_worker_receives_cuda_visible_devices():
    """Each worker must see a distinct CUDA_VISIBLE_DEVICES.

    Submits many tasks to ensure every worker processes at least one. The
    returned set must equal the provided gpu_ids (as strings). Repeated to
    catch queue-ordering flakes.
    """
    for _ in range(5):
        with get_executor(backend='process', gpu_ids=[7, 42]) as exe:
            results = list(exe.map(_return_cuda_visible_devices, range(20)))
        assert set(results) == {'7', '42'}, (
            f"Expected {{'7', '42'}}, got {set(results)} (raw: {results})"
        )


def test_worker_four_gpus_distinct():
    """Four workers must each pin a distinct GPU ID."""
    with get_executor(backend='process', gpu_ids=[0, 1, 2, 3]) as exe:
        results = list(exe.map(_return_cuda_visible_devices, range(40)))
    assert set(results) == {'0', '1', '2', '3'}

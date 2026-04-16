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
# A Barrier blocks each worker until all n_workers have arrived, guaranteeing
# every worker spawns (and thus runs the GPU-pinning initializer) before any
# result comes back. Deterministic — no sleeps, no queue-ordering flakes.
def _barrier_wait_and_return_cuda(barrier):
    barrier.wait(timeout=10)
    return os.environ.get('CUDA_VISIBLE_DEVICES')


def test_worker_receives_cuda_visible_devices():
    """Each worker must see a distinct CUDA_VISIBLE_DEVICES.

    Submits exactly n_workers tasks, each blocking on a Barrier(n_workers).
    All workers must run concurrently (otherwise the barrier deadlocks), so
    every worker is guaranteed to execute the GPU-pinning initializer.
    """
    import multiprocessing
    with multiprocessing.Manager() as manager:
        barrier = manager.Barrier(2)
        with get_executor(backend='process', gpu_ids=[7, 42]) as exe:
            futures = [exe.submit(_barrier_wait_and_return_cuda, barrier)
                       for _ in range(2)]
            results = [f.result() for f in futures]
    assert set(results) == {'7', '42'}, (
        f"Expected {{'7', '42'}}, got {set(results)} (raw: {results})"
    )


def test_worker_four_gpus_distinct():
    """Four workers must each pin a distinct GPU ID."""
    import multiprocessing
    with multiprocessing.Manager() as manager:
        barrier = manager.Barrier(4)
        with get_executor(backend='process', gpu_ids=[0, 1, 2, 3]) as exe:
            futures = [exe.submit(_barrier_wait_and_return_cuda, barrier)
                       for _ in range(4)]
            results = [f.result() for f in futures]
    assert set(results) == {'0', '1', '2', '3'}


def test_gpu_pool_logs_info(caplog):
    import logging as std_logging
    with caplog.at_level(std_logging.INFO):
        exe = get_executor(backend='process', gpu_ids=[0, 1])
        exe.shutdown(wait=True)
    assert any(
        'GPU-pinned process pool' in rec.message and 'gpu_ids=[0, 1]' in rec.message
        for rec in caplog.records
    ), f"Expected log line not found. Records: {[r.message for r in caplog.records]}"


# Fixture and helper live at module scope so ProcessPoolExecutor can pickle.
@pytest.fixture(scope="module")
def _al_fixture():
    from ase.build import bulk
    from ase.calculators.emt import EMT
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    replicated_atoms = atoms.repeat((1, 1, 2))
    replicated_atoms.calc = EMT()
    return atoms, replicated_atoms


def test_calculate_second_accepts_gpu_ids(_al_fixture):
    """calculate_second with gpu_ids=[0, 1] + EMT runs and matches serial.

    EMT doesn't care about CUDA_VISIBLE_DEVICES, but the plumbing path must
    still work end-to-end: the gpu_ids parameter is accepted, forwarded to
    get_executor, and workers spawn with the env var set.
    """
    import numpy as np
    from ase.calculators.emt import EMT
    from kaldo.controllers.displacement import calculate_second

    atoms, replicated_atoms = _al_fixture
    serial = calculate_second(atoms, replicated_atoms, 1e-5,
                              n_workers=1, calculator=EMT())
    parallel = calculate_second(atoms, replicated_atoms, 1e-5,
                                n_workers=2, calculator=EMT,
                                gpu_ids=[0, 1])
    np.testing.assert_allclose(parallel, serial, rtol=1e-7, atol=1e-9)


def test_calculate_third_accepts_gpu_ids(_al_fixture, tmp_path):
    import numpy as np
    from ase.calculators.emt import EMT
    from kaldo.controllers.displacement import calculate_third

    atoms, replicated_atoms = _al_fixture
    serial = calculate_third(atoms, replicated_atoms, 1e-5,
                             n_workers=1, calculator=EMT())
    parallel = calculate_third(atoms, replicated_atoms, 1e-5,
                               n_workers=2, calculator=EMT,
                               gpu_ids=[0, 1])
    np.testing.assert_allclose(parallel.todense(), serial.todense(), rtol=1e-7, atol=1e-9)


def test_secondorder_calculate_forwards_gpu_ids(tmp_path):
    """SecondOrder.calculate must pass gpu_ids through to calculate_second.

    A pure signature check is insufficient: a copy/paste slip at either of
    the two call sites in the try/except would go undetected. Patch the
    imported callable and assert on the recorded kwargs.
    """
    import numpy as np
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from unittest.mock import patch
    from kaldo.observables.secondorder import SecondOrder

    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    ifc = SecondOrder.from_supercell(
        atoms=atoms, supercell=(1, 1, 2), grid_type='C',
        folder=str(tmp_path / 'kALDo'),
    )
    n_atoms = len(atoms.numbers)
    n_replicated = n_atoms * 2
    fake_second = np.zeros((n_atoms, 3, n_replicated * 3))

    with patch('kaldo.observables.secondorder.calculate_second',
               return_value=fake_second) as mock_cs:
        ifc.calculate(calculator=EMT, n_workers=2, gpu_ids=[0, 1],
                      is_storing=False)

    mock_cs.assert_called_once()
    assert mock_cs.call_args.kwargs.get('gpu_ids') == [0, 1]


def test_thirdorder_calculate_forwards_gpu_ids(tmp_path):
    """ThirdOrder.calculate must pass gpu_ids through to calculate_third."""
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from unittest.mock import patch, MagicMock
    from kaldo.observables.thirdorder import ThirdOrder

    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    ifc = ThirdOrder.from_supercell(
        atoms=atoms, supercell=(1, 1, 2), grid_type='C',
        folder=str(tmp_path / 'kALDo'),
    )

    with patch('kaldo.observables.thirdorder.calculate_third',
               return_value=MagicMock()) as mock_ct:
        ifc.calculate(calculator=EMT, n_workers=2, gpu_ids=[0, 1],
                      is_storing=False)

    mock_ct.assert_called_once()
    assert mock_ct.call_args.kwargs.get('gpu_ids') == [0, 1]

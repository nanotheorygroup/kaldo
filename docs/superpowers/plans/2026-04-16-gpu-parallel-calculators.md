# GPU-Parallel Force Constant Calculators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `gpu_ids` parameter end-to-end (executor → `calculate_second` / `calculate_third` → `SecondOrder.calculate` / `ThirdOrder.calculate`) that pins each worker process to exactly one visible CUDA device via `CUDA_VISIBLE_DEVICES`.

**Architecture:** All GPU-awareness lives in `kaldo/parallel/executor.py`. `get_executor` gains a validated `gpu_ids` kwarg; when set with `backend='process'`, it creates a `ProcessPoolExecutor` with `mp_context='spawn'` and an `initializer` that atomically pops a GPU ID from a pre-filled `multiprocessing.Queue` and writes `CUDA_VISIBLE_DEVICES` in each worker's environment. `CalculatorFactory`, `_compute_iat_second`, and `_compute_iat_third` are untouched — the factory lazily builds the GPU calculator inside the worker after the initializer has already constrained what CUDA sees.

**Tech Stack:** Python 3.11+, `concurrent.futures.ProcessPoolExecutor`, `multiprocessing` (spawn context + Queue), `pytest`.

**Spec:** See `docs/superpowers/specs/2026-04-16-gpu-parallel-calculators-design.md`.

---

## Task 1: Create test module and validation for `gpu_ids` (type, duplicate, negative, empty-for-process)

**Files:**
- Create: `kaldo/tests/test_parallel_gpu.py`
- Modify: `kaldo/parallel/executor.py`

- [ ] **Step 1: Write the failing tests**

Create `kaldo/tests/test_parallel_gpu.py` with this content:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest kaldo/tests/test_parallel_gpu.py -v`

Expected: All 5 tests fail with `TypeError: get_executor() got an unexpected keyword argument 'gpu_ids'`.

- [ ] **Step 3: Add `gpu_ids` parameter and validation helper**

Modify `kaldo/parallel/executor.py`. Replace the entire file with:

```python
"""Pluggable executor backends for kaldo parallel computation.

All executors conform to the ``concurrent.futures.Executor`` interface,
so computation code is identical regardless of backend.
"""

import multiprocessing
import os
import sys
import warnings
from concurrent.futures import Future, ProcessPoolExecutor

from kaldo.helpers.logger import get_logger

logging = get_logger()


class SerialExecutor:
    """Executor that runs tasks synchronously in the calling process.

    Implements the ``concurrent.futures.Executor`` interface with zero
    overhead. Useful for debugging, small systems, and environments
    where multiprocessing is unavailable.
    """

    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return map(fn, *iterables)

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


def _validate_gpu_ids(gpu_ids):
    """Type-check gpu_ids. Backend-specific policy is enforced by callers."""
    if gpu_ids is None:
        return
    if not isinstance(gpu_ids, (list, tuple)):
        raise TypeError(
            f"gpu_ids must be a list of int, got {type(gpu_ids).__name__}"
        )
    if not all(isinstance(g, int) and not isinstance(g, bool) for g in gpu_ids):
        raise TypeError(
            f"gpu_ids must be a list of int, got elements "
            f"{[type(g).__name__ for g in gpu_ids]}"
        )
    if any(g < 0 for g in gpu_ids):
        raise ValueError(
            f"gpu_ids must be non-negative, got {list(gpu_ids)}"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(
            f"gpu_ids must be unique (oversubscription is disallowed); "
            f"got {list(gpu_ids)}"
        )


def get_executor(backend='process', n_workers=None, gpu_ids=None, **kwargs):
    """Create a parallel executor.

    Parameters
    ----------
    backend : str
        One of ``'serial'``, ``'process'``, or ``'mpi'``.
    n_workers : int or None
        Number of worker processes. ``None`` uses all available CPUs (or
        ``len(gpu_ids)`` when ``gpu_ids`` is set with ``backend='process'``).
    gpu_ids : list of int or None
        If provided, each worker is pinned to exactly one GPU ID via the
        ``CUDA_VISIBLE_DEVICES`` environment variable. Behavior by backend:

        - ``'process'``: ``len(gpu_ids)`` must equal ``n_workers``. If
          ``n_workers`` is ``None``, it defaults to ``len(gpu_ids)``. The
          pool uses spawn context so workers start with a pristine Python
          interpreter, guaranteeing the env var takes effect before any
          CUDA library is imported.
        - ``'serial'``: ``len(gpu_ids)`` must be 0 or 1. A single ID sets
          ``CUDA_VISIBLE_DEVICES`` in the main process before returning.
          Emits ``RuntimeWarning`` if torch or tensorflow are already
          imported (the env var may be too late).
        - ``'mpi'``: raises ``NotImplementedError``. MPI launchers (srun,
          mpirun) own GPU binding on HPC clusters.
    **kwargs
        Passed to the underlying executor constructor.

    Returns
    -------
    executor : concurrent.futures.Executor
    """
    _validate_gpu_ids(gpu_ids)

    if backend == 'serial':
        return SerialExecutor()

    elif backend == 'process':
        if gpu_ids is not None:
            if len(gpu_ids) == 0:
                raise ValueError(
                    "gpu_ids must be non-empty when specified for process backend"
                )
        return ProcessPoolExecutor(max_workers=n_workers, **kwargs)

    elif backend == 'mpi':
        try:
            from mpi4py.futures import MPIPoolExecutor
        except ImportError:
            raise ImportError(
                "mpi4py is required for backend='mpi'. "
                "Install with: MPICC=cc pip install mpi4py"
            )
        return MPIPoolExecutor(max_workers=n_workers, **kwargs)

    else:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Choose from 'serial', 'process', 'mpi'."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest kaldo/tests/test_parallel_gpu.py -v`

Expected: All 5 tests pass.

- [ ] **Step 5: Verify existing tests still pass**

Run: `pytest kaldo/tests/test_parallel_second.py kaldo/tests/test_parallel_third.py kaldo/tests/test_calculator_factory.py -v`

Expected: All pass (we haven't changed any existing behavior).

- [ ] **Step 6: Commit**

```bash
git add kaldo/parallel/executor.py kaldo/tests/test_parallel_gpu.py
git commit -m "Add gpu_ids validation to get_executor"
```

---

## Task 2: Reject `gpu_ids` for MPI backend

**Files:**
- Modify: `kaldo/parallel/executor.py`
- Modify: `kaldo/tests/test_parallel_gpu.py`

- [ ] **Step 1: Write the failing test**

Append to `kaldo/tests/test_parallel_gpu.py`:

```python
def test_mpi_rejects_gpu_ids():
    with pytest.raises(NotImplementedError, match="gpu_ids is not supported"):
        get_executor(backend='mpi', gpu_ids=[0])


def test_mpi_rejects_empty_gpu_ids():
    with pytest.raises(NotImplementedError, match="gpu_ids is not supported"):
        get_executor(backend='mpi', gpu_ids=[])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_mpi_rejects_gpu_ids kaldo/tests/test_parallel_gpu.py::test_mpi_rejects_empty_gpu_ids -v`

Expected: Both fail. Either mpi4py isn't installed (ImportError) or the executor doesn't yet raise `NotImplementedError`.

- [ ] **Step 3: Add MPI rejection before the mpi4py import**

In `kaldo/parallel/executor.py`, edit the `elif backend == 'mpi':` branch to check `gpu_ids` BEFORE the mpi4py import (so the test passes even when mpi4py is absent):

```python
    elif backend == 'mpi':
        if gpu_ids is not None:
            raise NotImplementedError(
                "gpu_ids is not supported with backend='mpi'. MPI launchers "
                "own GPU binding on HPC clusters; use your launcher's flags "
                "(e.g. `srun --gpus-per-task=1` or `mpirun --map-by ppr:1:gpu`) "
                "instead."
            )
        try:
            from mpi4py.futures import MPIPoolExecutor
        except ImportError:
            raise ImportError(
                "mpi4py is required for backend='mpi'. "
                "Install with: MPICC=cc pip install mpi4py"
            )
        return MPIPoolExecutor(max_workers=n_workers, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest kaldo/tests/test_parallel_gpu.py -v`

Expected: All 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add kaldo/parallel/executor.py kaldo/tests/test_parallel_gpu.py
git commit -m "Reject gpu_ids for MPI backend with actionable error"
```

---

## Task 3: Serial backend `gpu_ids` handling (multi-reject, single sets env, warn, empty no-op)

**Files:**
- Modify: `kaldo/parallel/executor.py`
- Modify: `kaldo/tests/test_parallel_gpu.py`

- [ ] **Step 1: Write the failing tests**

Append to `kaldo/tests/test_parallel_gpu.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest kaldo/tests/test_parallel_gpu.py -v -k serial`

Expected: All 5 new tests fail (serial backend currently ignores `gpu_ids`).

- [ ] **Step 3: Implement serial handling**

In `kaldo/parallel/executor.py`, replace the `if backend == 'serial':` branch with:

```python
    if backend == 'serial':
        if gpu_ids is not None:
            if len(gpu_ids) > 1:
                raise ValueError(
                    f"serial backend can use at most one GPU; got "
                    f"{len(gpu_ids)}. Use backend='process' for multi-GPU "
                    "parallelism."
                )
            if len(gpu_ids) == 1:
                gpu_id = gpu_ids[0]
                if 'torch' in sys.modules or 'tensorflow' in sys.modules:
                    warnings.warn(
                        "Setting CUDA_VISIBLE_DEVICES after torch/tensorflow "
                        "has been imported may be too late: the CUDA runtime "
                        "is already initialized and env-var changes are "
                        "silently ignored. For reliable serial GPU pinning, "
                        "export CUDA_VISIBLE_DEVICES in your shell before "
                        "starting Python.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # len(gpu_ids) == 0: explicit "no GPU"; no-op
        return SerialExecutor()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest kaldo/tests/test_parallel_gpu.py -v`

Expected: All 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add kaldo/parallel/executor.py kaldo/tests/test_parallel_gpu.py
git commit -m "Handle gpu_ids in serial backend with init-timing warning"
```

---

## Task 4: Process backend `n_workers` defaulting and mismatch validation

**Files:**
- Modify: `kaldo/parallel/executor.py`
- Modify: `kaldo/tests/test_parallel_gpu.py`

- [ ] **Step 1: Write the failing tests**

Append to `kaldo/tests/test_parallel_gpu.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest kaldo/tests/test_parallel_gpu.py -v -k nworkers`

Expected: `test_nworkers_defaults_to_gpu_count` fails (`_max_workers` is whatever Python's CPU count returns, not 3). `test_nworkers_mismatch_rejected` fails (no mismatch check).

- [ ] **Step 3: Implement defaulting and mismatch check**

In `kaldo/parallel/executor.py`, replace the `elif backend == 'process':` branch with:

```python
    elif backend == 'process':
        if gpu_ids is not None:
            if len(gpu_ids) == 0:
                raise ValueError(
                    "gpu_ids must be non-empty when specified for process backend"
                )
            if n_workers is None:
                n_workers = len(gpu_ids)
            elif n_workers != len(gpu_ids):
                raise ValueError(
                    f"n_workers ({n_workers}) must equal len(gpu_ids) "
                    f"({len(gpu_ids)}) when both are given"
                )
        return ProcessPoolExecutor(max_workers=n_workers, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest kaldo/tests/test_parallel_gpu.py -v`

Expected: All 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add kaldo/parallel/executor.py kaldo/tests/test_parallel_gpu.py
git commit -m "Default n_workers to len(gpu_ids) and reject mismatches"
```

---

## Task 5: Core mechanism — spawn context, initializer, race-free queue handout

This is the critical task. After this, GPU pinning actually works.

**Files:**
- Modify: `kaldo/parallel/executor.py`
- Modify: `kaldo/tests/test_parallel_gpu.py`

- [ ] **Step 1: Write the failing test (and helpers)**

Append to `kaldo/tests/test_parallel_gpu.py`:

```python
# Module-level function so ProcessPoolExecutor can pickle it.
def _return_cuda_visible_devices(_):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_worker_receives_cuda_visible_devices -v`

Expected: FAIL. Either `CUDA_VISIBLE_DEVICES` is `None` (initializer not wired) or matches the parent process (inherited, not pinned).

- [ ] **Step 3: Add `_gpu_worker_initializer` and `_make_gpu_process_pool`**

In `kaldo/parallel/executor.py`, add these module-level functions above `get_executor`:

```python
def _gpu_worker_initializer(gpu_id_queue):
    """Set CUDA_VISIBLE_DEVICES for this worker from a shared queue.

    Runs exactly once per worker process, before any submitted task. Blocks
    on the queue with a 30s timeout — if the queue is empty (which only
    happens on bugs, since we pre-fill it with exactly n_workers entries),
    the worker raises queue.Empty and ProcessPoolExecutor surfaces a
    BrokenProcessPool on the first task result.
    """
    gpu_id = gpu_id_queue.get(block=True, timeout=30)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def _make_gpu_process_pool(n_workers, gpu_ids, **kwargs):
    """Create a ProcessPoolExecutor that pins each worker to one GPU.

    Uses spawn context so workers start with a pristine Python interpreter.
    This is critical: forked workers inherit parent CUDA state, and setting
    CUDA_VISIBLE_DEVICES after CUDA is initialized is a silent no-op.
    """
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    for gpu_id in gpu_ids:
        queue.put(gpu_id)
    return ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_gpu_worker_initializer,
        initargs=(queue,),
        **kwargs,
    )
```

- [ ] **Step 4: Wire it into the `process` branch of `get_executor`**

In `kaldo/parallel/executor.py`, replace the `elif backend == 'process':` branch with:

```python
    elif backend == 'process':
        if gpu_ids is not None:
            if len(gpu_ids) == 0:
                raise ValueError(
                    "gpu_ids must be non-empty when specified for process backend"
                )
            if n_workers is None:
                n_workers = len(gpu_ids)
            elif n_workers != len(gpu_ids):
                raise ValueError(
                    f"n_workers ({n_workers}) must equal len(gpu_ids) "
                    f"({len(gpu_ids)}) when both are given"
                )
            return _make_gpu_process_pool(n_workers, list(gpu_ids), **kwargs)
        return ProcessPoolExecutor(max_workers=n_workers, **kwargs)
```

- [ ] **Step 5: Run the race-freedom tests**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_worker_receives_cuda_visible_devices kaldo/tests/test_parallel_gpu.py::test_worker_four_gpus_distinct -v`

Expected: Both PASS. `test_worker_receives_cuda_visible_devices` runs 5 iterations internally; all must agree.

- [ ] **Step 6: Run the full executor test suite**

Run: `pytest kaldo/tests/test_parallel_gpu.py -v`

Expected: All 17 tests pass.

- [ ] **Step 7: Verify existing parallel tests still pass**

Run: `pytest kaldo/tests/test_parallel_second.py kaldo/tests/test_parallel_third.py kaldo/tests/test_calculator_factory.py -v`

Expected: All pass (non-GPU process pools still use the default context; we only switch to spawn when `gpu_ids` is set).

- [ ] **Step 8: Commit**

```bash
git add kaldo/parallel/executor.py kaldo/tests/test_parallel_gpu.py
git commit -m "Pin each worker to one GPU via spawn + race-free queue handout"
```

---

## Task 6: Log one INFO line at GPU pool creation

**Files:**
- Modify: `kaldo/parallel/executor.py`
- Modify: `kaldo/tests/test_parallel_gpu.py`

- [ ] **Step 1: Write the failing test**

Append to `kaldo/tests/test_parallel_gpu.py`:

```python
def test_gpu_pool_logs_info(caplog):
    import logging as std_logging
    with caplog.at_level(std_logging.INFO):
        exe = get_executor(backend='process', gpu_ids=[0, 1])
        exe.shutdown(wait=True)
    assert any(
        'GPU-pinned process pool' in rec.message and 'gpu_ids=[0, 1]' in rec.message
        for rec in caplog.records
    ), f"Expected log line not found. Records: {[r.message for r in caplog.records]}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_gpu_pool_logs_info -v`

Expected: FAIL (no such log line exists yet).

- [ ] **Step 3: Add the log line**

In `kaldo/parallel/executor.py`, add the log line at the top of `_make_gpu_process_pool`:

```python
def _make_gpu_process_pool(n_workers, gpu_ids, **kwargs):
    """Create a ProcessPoolExecutor that pins each worker to one GPU.

    Uses spawn context so workers start with a pristine Python interpreter.
    This is critical: forked workers inherit parent CUDA state, and setting
    CUDA_VISIBLE_DEVICES after CUDA is initialized is a silent no-op.
    """
    logging.info(
        f"Created GPU-pinned process pool: {n_workers} workers, "
        f"gpu_ids={list(gpu_ids)}"
    )
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    for gpu_id in gpu_ids:
        queue.put(gpu_id)
    return ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_gpu_worker_initializer,
        initargs=(queue,),
        **kwargs,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_gpu_pool_logs_info -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kaldo/parallel/executor.py kaldo/tests/test_parallel_gpu.py
git commit -m "Log INFO line at GPU-pinned pool creation"
```

---

## Task 7: Plumb `gpu_ids` through `calculate_second`

**Files:**
- Modify: `kaldo/controllers/displacement.py`
- Modify: `kaldo/tests/test_parallel_gpu.py`

- [ ] **Step 1: Write the failing test**

Append to `kaldo/tests/test_parallel_gpu.py`:

```python
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
    np.testing.assert_allclose(parallel, serial, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_calculate_second_accepts_gpu_ids -v`

Expected: FAIL with `TypeError: calculate_second() got an unexpected keyword argument 'gpu_ids'`.

- [ ] **Step 3: Add `gpu_ids` parameter to `calculate_second`**

In `kaldo/controllers/displacement.py`, modify the `calculate_second` signature and the `get_executor` call:

Change line 70-71 from:
```python
def calculate_second(atoms, replicated_atoms, second_order_delta, is_verbose=False, n_workers=1, calculator=None,
                     scratch_dir=None, keep_scratch=False):
```
to:
```python
def calculate_second(atoms, replicated_atoms, second_order_delta, is_verbose=False, n_workers=1, calculator=None,
                     scratch_dir=None, keep_scratch=False, gpu_ids=None):
```

Change the `get_executor` call (currently at line 109) from:
```python
    with get_executor(backend=backend, n_workers=executor_workers) as executor:
```
to:
```python
    with get_executor(backend=backend, n_workers=executor_workers, gpu_ids=gpu_ids) as executor:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_calculate_second_accepts_gpu_ids -v`

Expected: PASS.

- [ ] **Step 5: Verify existing parallel second tests still pass**

Run: `pytest kaldo/tests/test_parallel_second.py -v`

Expected: All pass (we added an optional kwarg, default behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add kaldo/controllers/displacement.py kaldo/tests/test_parallel_gpu.py
git commit -m "Plumb gpu_ids through calculate_second"
```

---

## Task 8: Plumb `gpu_ids` through `calculate_third`

**Files:**
- Modify: `kaldo/controllers/displacement.py`
- Modify: `kaldo/tests/test_parallel_gpu.py`

- [ ] **Step 1: Write the failing test**

Append to `kaldo/tests/test_parallel_gpu.py`:

```python
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
    np.testing.assert_allclose(parallel.todense(), serial.todense(), atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_calculate_third_accepts_gpu_ids -v`

Expected: FAIL with `TypeError: calculate_third() got an unexpected keyword argument 'gpu_ids'`.

- [ ] **Step 3: Add `gpu_ids` parameter to `calculate_third`**

In `kaldo/controllers/displacement.py`, modify the `calculate_third` signature and the `get_executor` call:

Change lines 177-179 from:
```python
def calculate_third(atoms, replicated_atoms, third_order_delta, distance_threshold=None, is_verbose=False,
                    n_workers=1, calculator=None, scratch_dir=None, keep_scratch=False,
                    jat_flush_every=50):
```
to:
```python
def calculate_third(atoms, replicated_atoms, third_order_delta, distance_threshold=None, is_verbose=False,
                    n_workers=1, calculator=None, scratch_dir=None, keep_scratch=False,
                    jat_flush_every=50, gpu_ids=None):
```

Change the `get_executor` call (currently at line 270) from:
```python
    with get_executor(backend=backend, n_workers=executor_workers) as executor:
```
to:
```python
    with get_executor(backend=backend, n_workers=executor_workers, gpu_ids=gpu_ids) as executor:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_calculate_third_accepts_gpu_ids -v`

Expected: PASS.

- [ ] **Step 5: Verify existing parallel third tests still pass**

Run: `pytest kaldo/tests/test_parallel_third.py -v`

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add kaldo/controllers/displacement.py kaldo/tests/test_parallel_gpu.py
git commit -m "Plumb gpu_ids through calculate_third"
```

---

## Task 9: Plumb `gpu_ids` through `SecondOrder.calculate`

**Files:**
- Modify: `kaldo/observables/secondorder.py`
- Modify: `kaldo/tests/test_parallel_gpu.py`

- [ ] **Step 1: Write the failing test**

Append to `kaldo/tests/test_parallel_gpu.py`:

```python
def test_secondorder_calculate_accepts_gpu_ids():
    """SecondOrder.calculate exposes gpu_ids and forwards it to calculate_second."""
    import inspect
    from kaldo.observables.secondorder import SecondOrder
    sig = inspect.signature(SecondOrder.calculate)
    assert 'gpu_ids' in sig.parameters
    assert sig.parameters['gpu_ids'].default is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_secondorder_calculate_accepts_gpu_ids -v`

Expected: FAIL — `gpu_ids` not in the signature.

- [ ] **Step 3: Add `gpu_ids` to `SecondOrder.calculate`**

In `kaldo/observables/secondorder.py`, modify `SecondOrder.calculate`.

Change the signature at line 290-291 from:
```python
    def calculate(self, calculator, delta_shift=1e-3, is_storing=True, is_verbose=False, n_workers=1,
                  scratch_dir=None, keep_scratch=False):
```
to:
```python
    def calculate(self, calculator, delta_shift=1e-3, is_storing=True, is_verbose=False, n_workers=1,
                  scratch_dir=None, keep_scratch=False, gpu_ids=None):
```

Change the TWO `calculate_second(...)` calls in this method (at line 367-376 and line 384-393). Each currently ends with:
```python
                    scratch_dir=scratch_dir,
                    keep_scratch=keep_scratch,
                )
```

Add `gpu_ids=gpu_ids,` to each call so they read:
```python
                    scratch_dir=scratch_dir,
                    keep_scratch=keep_scratch,
                    gpu_ids=gpu_ids,
                )
```

Also add a docstring entry for `gpu_ids`. Insert this after the `keep_scratch` docstring block (after line 354):
```python
        gpu_ids : list of int or None, optional
            If provided, pin each worker process to exactly one GPU ID via
            CUDA_VISIBLE_DEVICES. Requires ``n_workers >= 1`` with a process
            backend; ``len(gpu_ids)`` must equal ``n_workers`` (or
            ``n_workers`` may be omitted to default to ``len(gpu_ids)``).
            Default: None (CPU workers).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_secondorder_calculate_accepts_gpu_ids -v`

Expected: PASS.

- [ ] **Step 5: Verify existing second-order tests still pass**

Run: `pytest kaldo/tests/ -k "second" -v`

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add kaldo/observables/secondorder.py kaldo/tests/test_parallel_gpu.py
git commit -m "Plumb gpu_ids through SecondOrder.calculate"
```

---

## Task 10: Plumb `gpu_ids` through `ThirdOrder.calculate`

**Files:**
- Modify: `kaldo/observables/thirdorder.py`
- Modify: `kaldo/tests/test_parallel_gpu.py`

- [ ] **Step 1: Write the failing test**

Append to `kaldo/tests/test_parallel_gpu.py`:

```python
def test_thirdorder_calculate_accepts_gpu_ids():
    """ThirdOrder.calculate exposes gpu_ids and forwards it to calculate_third."""
    import inspect
    from kaldo.observables.thirdorder import ThirdOrder
    sig = inspect.signature(ThirdOrder.calculate)
    assert 'gpu_ids' in sig.parameters
    assert sig.parameters['gpu_ids'].default is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_thirdorder_calculate_accepts_gpu_ids -v`

Expected: FAIL.

- [ ] **Step 3: Add `gpu_ids` to `ThirdOrder.calculate`**

In `kaldo/observables/thirdorder.py`, modify `ThirdOrder.calculate`.

Change the signature at line 270-271 from:
```python
    def calculate(self, calculator=None, delta_shift=1e-4, distance_threshold=None, is_storing=True, is_verbose=False,
                  n_workers=1, scratch_dir=None, keep_scratch=False, jat_flush_every=50):
```
to:
```python
    def calculate(self, calculator=None, delta_shift=1e-4, distance_threshold=None, is_storing=True, is_verbose=False,
                  n_workers=1, scratch_dir=None, keep_scratch=False, jat_flush_every=50, gpu_ids=None):
```

Change the TWO `calculate_third(...)` calls (at line 344-353 and line 359-368). Each currently ends with:
```python
                                             jat_flush_every=jat_flush_every)
```

Add `gpu_ids=gpu_ids` before the closing paren, preserving the existing keyword layout:
```python
                                             jat_flush_every=jat_flush_every,
                                             gpu_ids=gpu_ids)
```

Add a docstring entry for `gpu_ids` after the `jat_flush_every` block (after line 327):
```python
        gpu_ids : list of int or None
            If provided, pin each worker process to exactly one GPU ID via
            CUDA_VISIBLE_DEVICES. Requires ``n_workers >= 1`` with a process
            backend; ``len(gpu_ids)`` must equal ``n_workers`` (or
            ``n_workers`` may be omitted to default to ``len(gpu_ids)``).
            Default: None (CPU workers).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest kaldo/tests/test_parallel_gpu.py::test_thirdorder_calculate_accepts_gpu_ids -v`

Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest kaldo/tests/ -v`

Expected: All tests pass. No regressions.

- [ ] **Step 6: Commit**

```bash
git add kaldo/observables/thirdorder.py kaldo/tests/test_parallel_gpu.py
git commit -m "Plumb gpu_ids through ThirdOrder.calculate"
```

---

## Task 11: Add a worked example to `CalculatorFactory` docstring

**Files:**
- Modify: `kaldo/parallel/calculator.py`

This is pure documentation so users can see the happy path end-to-end.

- [ ] **Step 1: Add usage example**

In `kaldo/parallel/calculator.py`, extend the module-level docstring (the big triple-quoted block at the top of the file) by adding this block immediately before the closing `"""`:

```python
GPU parallelization
-------------------
Combine with ``gpu_ids`` on the calculate call to pin each worker to a
distinct GPU. ``CUDA_VISIBLE_DEVICES`` is set in each worker before the
calculator is constructed, so ``device='cuda'`` binds correctly::

    from kaldo.parallel import CalculatorFactory
    from mace.calculators import MACECalculator

    calculator = CalculatorFactory(
        MACECalculator,
        kwargs={'model_paths': 'model.pt', 'device': 'cuda'},
        validate=False,  # defer GPU allocation to workers
    )
    second_order.calculate(
        calculator=calculator,
        n_workers=4,
        gpu_ids=[0, 1, 2, 3],
    )
```

- [ ] **Step 2: Verify the module still imports cleanly**

Run: `python -c "from kaldo.parallel import CalculatorFactory; help(CalculatorFactory)" | head -40`

Expected: Module imports, help text shown without error.

- [ ] **Step 3: Commit**

```bash
git add kaldo/parallel/calculator.py
git commit -m "Document gpu_ids example in CalculatorFactory docstring"
```

---

## Done criteria

After all 11 tasks:

- `kaldo/tests/test_parallel_gpu.py` contains 20+ passing tests (validation, serial handling, race-freedom, plumbing).
- `pytest kaldo/tests/ -v` passes without regressions.
- `get_executor(backend='process', gpu_ids=[0,1,2,3])` creates a pool that pins each worker to a distinct GPU.
- `second_order.calculate(calculator=factory, n_workers=4, gpu_ids=[0,1,2,3])` runs with GPU-aware workers.
- `third_order.calculate(calculator=factory, n_workers=4, gpu_ids=[0,1,2,3])` runs with GPU-aware workers.
- `backend='mpi' + gpu_ids` raises `NotImplementedError`. `backend='serial' + len(gpu_ids) > 1` raises `ValueError`. `backend='serial' + gpu_ids=[N]` sets the env var and warns if torch/TF already imported.

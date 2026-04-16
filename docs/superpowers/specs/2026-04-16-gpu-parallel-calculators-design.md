# Design: GPU-parallel force constant calculators

**Date:** 2026-04-16
**Status:** Draft — awaiting review
**Scope:** Enable users to parallelize `calculate_second` / `calculate_third` over multiple GPUs by pinning each worker process to exactly one visible CUDA device.

## Goal

When a user supplies a GPU-capable ASE calculator (MACE, NequIP, SchNet, JAX-based, etc.), allow `n_workers` processes to run in parallel with each worker bound to a distinct GPU. Calculator-library-agnostic: we rely on `CUDA_VISIBLE_DEVICES` rather than any library-specific device API.

## Non-goals

- Auto-detecting available GPUs. Users pass an explicit list (they already know what they exposed via `CUDA_VISIBLE_DEVICES` on the shell).
- GPU oversubscription (multiple workers per GPU). Rejected at validation time.
- MPI-launcher-based GPU pinning. Out of scope; users handle that via `srun` / `mpirun`.
- Any changes to `GPUStrategy` (`kaldo/parallel/gpu.py`). That module is about TF device placement inside a single process and is orthogonal to the executor-level GPU pinning designed here.

## Architecture

All changes live in the executor layer. The calculator layer is untouched.

```
┌──────────────────────────────────────────────────────────────────┐
│  User code                                                       │
│  factory = CalculatorFactory(MACECalculator, kwargs={...})       │
│  second_order.calculate(calculator=factory,                      │
│                         n_workers=4, gpu_ids=[0,1,2,3])          │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  calculate_second / calculate_third (displacement.py)            │
│    forwards gpu_ids → get_executor                               │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  get_executor (executor.py)                                      │
│    validates gpu_ids, backend compatibility, n_workers match     │
│    calls _make_gpu_process_pool for process+gpu_ids path         │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  _make_gpu_process_pool (executor.py, new)                       │
│    pre-fills multiprocessing.Queue with gpu_ids                  │
│    creates ProcessPoolExecutor with:                             │
│      • mp_context='spawn' (never fork)                           │
│      • initializer=_gpu_worker_initializer(queue)                │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  Worker process (clean Python interpreter, no CUDA yet)          │
│    1. _gpu_worker_initializer runs once per worker               │
│       - queue.get() → one GPU ID (atomic)                        │
│       - os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)         │
│    2. _compute_iat_second / _compute_iat_third runs tasks        │
│       - factory() imports torch/TF; CUDA sees exactly 1 GPU      │
└──────────────────────────────────────────────────────────────────┘
```

### Key invariant

`CUDA_VISIBLE_DEVICES` must be set **before** any CUDA library is imported in the worker. The design guarantees this because:

- `spawn` context gives each worker a pristine Python interpreter (no inherited CUDA state).
- `ProcessPoolExecutor` runs the `initializer` before any submitted task.
- `CalculatorFactory.__call__` constructs the calculator (and thus imports torch/TF) inside the task, after the initializer has already set the env var.

### Why spawn, not fork

Even on Linux, the GPU-parallel pool uses `mp_context='spawn'`. Forked workers inherit the parent's imported modules including any CUDA runtime state; setting `CUDA_VISIBLE_DEVICES` in the forked child is a no-op because CUDA is already initialized. Spawn avoids this footgun at the cost of slightly slower worker startup.

Non-GPU process pools (`gpu_ids=None`) continue to use the platform default context to avoid imposing spawn's startup cost on CPU-only users.

## API

### `kaldo.parallel.get_executor`

```python
def get_executor(backend='process', n_workers=None, gpu_ids=None, **kwargs):
    """
    Parameters
    ----------
    backend : str
        'serial', 'process', or 'mpi'.
    n_workers : int or None
        Number of worker processes.
    gpu_ids : list of int or None
        If provided, each worker is pinned to exactly one GPU ID via
        CUDA_VISIBLE_DEVICES. Behavior by backend:
          - 'process': len(gpu_ids) must equal n_workers. If n_workers is
            None, it defaults to len(gpu_ids). Uses spawn context.
          - 'serial': len(gpu_ids) must be 0 or 1. A single ID sets
            CUDA_VISIBLE_DEVICES in the main process before returning.
          - 'mpi': raises NotImplementedError.
    """
```

### `kaldo.controllers.displacement.calculate_second` / `calculate_third`

Both gain `gpu_ids=None` and forward it to `get_executor`. Docstrings document the new parameter with a MACE example.

### `kaldo.observables.secondorder.SecondOrder.calculate` / `thirdorder.ThirdOrder.calculate`

Both gain `gpu_ids=None` and pass it through to the underlying `calculate_second` / `calculate_third` call.

### Validation rules

| Condition | Behavior |
|---|---|
| `gpu_ids` not a list/tuple of ints | `TypeError` |
| Duplicates in `gpu_ids` | `ValueError` (oversubscription disallowed) |
| Negative integer in `gpu_ids` | `ValueError` |
| `backend='mpi'` + `gpu_ids is not None` | `NotImplementedError` with pointer to MPI launcher flags |
| `backend='serial'` + `len(gpu_ids) > 1` | `ValueError` |
| `backend='process'` + explicit `n_workers != len(gpu_ids)` | `ValueError` |
| `backend='process'` + `n_workers is None` + `gpu_ids` set | `n_workers` defaults to `len(gpu_ids)` |
| `backend='process'` + `gpu_ids=[]` | `ValueError`: "gpu_ids must be non-empty when specified for process backend" |
| `backend='serial'` + `gpu_ids=[]` | No-op, no warning |
| `backend='serial'` + `gpu_ids=[N]` + `torch` or `tensorflow` already in `sys.modules` | Set env var, emit `RuntimeWarning` |

## Race-free GPU ID handout

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os


def _gpu_worker_initializer(gpu_id_queue):
    gpu_id = gpu_id_queue.get(block=True, timeout=30)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def _make_gpu_process_pool(n_workers, gpu_ids, **kwargs):
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

### Why this is race-free

- `multiprocessing.Queue.get()` is atomic across processes. Two workers cannot observe the same ID.
- The queue is pre-filled with exactly `n_workers` items before the pool is created, so each initializer call gets exactly one, and the queue is empty afterward.
- The 30-second timeout prevents an indefinite hang if something goes wrong during worker spawn; a timed-out worker raises `queue.Empty`, which `ProcessPoolExecutor` surfaces as `BrokenProcessPool` on the first task result.

## Error handling

- **Worker init timeout:** Natural `BrokenProcessPool` from `ProcessPoolExecutor`. No kaldo-level wrapping.
- **Invalid GPU ID at runtime (e.g., `gpu_ids=[99]`, no such device):** The env var is set successfully; the calculator's CUDA initialization fails later with the library's native error message. Kaldo does not pre-validate against hardware (would require importing a CUDA library in the main process, defeating the lazy-import design).
- **Serial + single GPU, CUDA already loaded:** `RuntimeWarning` via `sys.modules` heuristic. We do not error — the user may have intentionally pre-selected the same device, in which case our env var mutation is harmless.

### Logging

One `INFO` line at pool creation:
```
Created GPU-pinned process pool: 4 workers, gpu_ids=[0, 1, 2, 3]
```

No per-worker log (would require logging from inside the initializer; noisy and low-value).

## Testing

New module: `kaldo/tests/test_parallel_gpu.py`. All tests are mock-only and require no GPU hardware.

1. `test_gpu_ids_validates_duplicates` — `gpu_ids=[0, 0]` raises `ValueError`.
2. `test_gpu_ids_validates_negative` — `gpu_ids=[-1]` raises `ValueError`.
3. `test_gpu_ids_validates_type` — `gpu_ids=['0']` raises `TypeError`.
4. `test_mpi_rejects_gpu_ids` — `backend='mpi', gpu_ids=[0]` raises `NotImplementedError`.
5. `test_serial_multi_gpu_rejected` — `backend='serial', gpu_ids=[0, 1]` raises `ValueError`.
6. `test_nworkers_defaults_to_gpu_count` — `backend='process', gpu_ids=[0,1,2]` yields a pool with `max_workers == 3`.
7. `test_nworkers_mismatch_rejected` — explicit `n_workers=4` + `gpu_ids=[0, 1]` raises `ValueError`.
8. `test_worker_receives_cuda_visible_devices` — spawn a GPU-pinned pool with `gpu_ids=[7, 42]`, submit tasks that return `os.environ.get('CUDA_VISIBLE_DEVICES')`, assert collected values (as a sorted list) equal `['42', '7']`. Run 5 iterations to catch flaky races.
9. `test_serial_single_gpu_sets_env_var` — in a subprocess (to keep `sys.modules` clean), call `get_executor(backend='serial', gpu_ids=[3])`, assert `os.environ['CUDA_VISIBLE_DEVICES'] == '3'`.
10. `test_serial_single_gpu_warns_when_torch_imported` — monkeypatch `sys.modules['torch'] = object()`, assert `RuntimeWarning`.
11. `test_serial_empty_gpu_ids_is_noop` — `gpu_ids=[]` does not mutate environment and does not warn.
12. `test_process_empty_gpu_ids_rejected` — `backend='process', gpu_ids=[]` raises `ValueError`.

Test 8 is the critical race-freedom check. Five iterations per run is sufficient to surface any queue-ordering bug without making the test suite slow.

## Files touched

- `kaldo/parallel/executor.py` — add `gpu_ids` parameter, `_make_gpu_process_pool`, `_gpu_worker_initializer`, validation.
- `kaldo/parallel/__init__.py` — no change (existing exports cover the new path).
- `kaldo/controllers/displacement.py` — plumb `gpu_ids` through `calculate_second` and `calculate_third`.
- `kaldo/observables/secondorder.py` — plumb `gpu_ids` through `SecondOrder.calculate`.
- `kaldo/observables/thirdorder.py` — plumb `gpu_ids` through `ThirdOrder.calculate`.
- `kaldo/tests/test_parallel_gpu.py` — new file, tests 1–11 above.

## Out of scope / deferred

- Integration tests that run MACE on actual GPUs. Removed per review — mock tests fully verify the plumbing, and real-hardware coverage can be added later if regressions appear.
- Auto-detection of available GPUs.
- CPU thread-capping for GPU workers (kaldo intentionally removed `OMP_NUM_THREADS=1` enforcement in commit cf5ed606; we maintain that philosophy).
- `GPUStrategy` integration. Keeping orthogonal: `GPUStrategy` governs TF device placement within a single process; this design governs per-worker GPU assignment across processes.

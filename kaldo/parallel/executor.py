"""Pluggable executor backends for kaldo parallel computation.

All executors conform to the ``concurrent.futures.Executor`` interface,
so computation code is identical regardless of backend.
"""

import multiprocessing
import os
import sys
import warnings
from concurrent.futures import Future, ProcessPoolExecutor


def _default_mp_context():
    """Return a multiprocessing context safe for TF and CUDA calculators.

    Fork (Linux default) is unsafe in two situations kaldo regularly hits:

    1. CUDA initialized in the parent: torch refuses to re-initialize CUDA
       in a forked child (BrokenProcessPool /
       'Cannot re-initialize CUDA in forked subprocess').
    2. TensorFlow imported in the parent: TF eagerly creates a thread pool
       and grpc/cuda runtime state at import time. Forking duplicates the
       file descriptors but not the threads, leaving the child's runtime
       in a half-initialized state. Symptoms range from silent deadlocks
       (TF op blocks waiting for a worker thread that never existed in
       this process) to ``CUDA_ERROR_INVALID_HANDLE`` when the child
       inherits a stale GPU context. xdist + parallel kaldo on Linux
       reliably tickles this — see PR #231 CI hang.

    Spawn avoids both because the worker starts from a clean Python
    interpreter and re-imports modules itself. Cold-start cost is ~0.5s
    per worker, negligible compared to one phonon calculation.

    We anticipate CUDA: ``torch.cuda.is_available()`` returns True even
    before any tensor touches the device. By the time the calculator runs
    in a worker it will initialize CUDA, so picking spawn up front is the
    correct choice. Same logic for TF: if it's already imported, the
    parent is tainted; even if it isn't, kaldo's own modules import TF on
    demand inside ``Phonons``, so the worker will be tainted too.
    """
    if 'tensorflow' in sys.modules:
        return multiprocessing.get_context('spawn')
    try:
        import torch  # lazy: don't force-import torch for non-ML users
        if torch.cuda.is_available():
            return multiprocessing.get_context('spawn')
    except ImportError:
        pass
    return multiprocessing.get_context()  # platform default


def _require_main_process(n_workers):
    """Raise a clear error when a parallel call is fired from a worker.

    Spawn-based multiprocessing re-imports the entry-point module in every
    worker. Top-level kaldo calls re-execute on import and try to spawn
    their own pool, which Python rejects (``RuntimeError: An attempt has
    been made to start a new process before the current process has
    finished its bootstrapping phase``) with a message users find cryptic.

    Catch it earlier and tell users exactly what to fix: wrap their kaldo
    calls in ``if __name__ == '__main__':`` so the worker's re-import
    skips them.

    No-op for serial calls (``n_workers in (None, 1)`` only when the
    parent explicitly opted into a single worker; ``None`` here means the
    auto path which is parallel).
    """
    if n_workers is not None and n_workers <= 1:
        return
    if multiprocessing.current_process().name == 'MainProcess':
        return
    raise RuntimeError(
        "Parallel kaldo (n_workers > 1) was invoked inside a worker process.\n"
        "This usually means your script's top-level code calls\n"
        "    fc.second.calculate(..., n_workers=N)\n"
        "without an ``if __name__ == '__main__':`` guard, and spawn-based\n"
        "multiprocessing re-executed it in every worker.\n"
        "\n"
        "Fix: wrap the kaldo calls (and anything that depends on them) in\n"
        "the standard guard:\n"
        "    if __name__ == '__main__':\n"
        "        fc.second.calculate(..., n_workers=N)\n"
    )


def _init_worker_thread_caps(n_threads):
    """Cap a worker process's native-thread pools to ``n_threads``.

    Runs once at worker startup via ``ProcessPoolExecutor(initializer=...)``,
    before the calculator is imported or instantiated. Sets the standard
    OpenMP / BLAS env vars so that libgomp, MKL, and OpenBLAS see the cap
    when they first initialize their thread pools in this worker.

    User-set caps are respected: if a variable is already set in the
    environment (e.g. the caller ran with ``OMP_NUM_THREADS=4 python ...``),
    it is left untouched. Unset variables are capped to ``n_threads``.

    Why this matters: calculators with native multithreading (PyNEP,
    numpy+MKL, torch CPU) each default to one thread per core. With N
    process workers this produces ``N * n_cores`` threads contending
    for the same cores, which erases any parallel speedup.
    """
    n_str = str(int(n_threads))
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(var, n_str)


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


def get_executor(backend='process', n_workers=None, mp_context=None,
                 worker_threads=1, **kwargs):
    """Create a parallel executor.

    Parameters
    ----------
    backend : str
        One of ``'serial'``, ``'process'``, or ``'mpi'``.
    n_workers : int or None
        Number of worker processes. ``None`` uses all available CPUs.
    mp_context : multiprocessing context or None
        Start method for worker processes. If ``None`` (default), picks
        ``spawn`` when CUDA is active in this process (required for
        torch/GPU calculators) and the platform default otherwise. Only
        used by the ``'process'`` backend.
    worker_threads : int or None
        Cap each worker's native OpenMP / MKL / OpenBLAS thread pool.
        Default ``1`` — the right choice for calculators that already
        parallelize internally (PyNEP, torch CPU, numpy+MKL) because
        each worker otherwise spawns one thread per core, producing
        ``N_workers * N_cores`` threads that contend for the same cores.
        Set to ``None`` to leave the environment untouched (useful when
        the caller has set ``OMP_NUM_THREADS`` explicitly, or wants to
        hybridize e.g. ``n_workers=4`` × ``worker_threads=8`` on a
        32-core box). Only used by the ``'process'`` backend.
    **kwargs
        Passed to the underlying executor constructor.

    Returns
    -------
    executor : concurrent.futures.Executor
    """
    if backend == 'serial':
        return SerialExecutor()

    elif backend == 'process':
        _require_main_process(n_workers)
        if mp_context is None:
            mp_context = _default_mp_context()
        if worker_threads is not None:
            kwargs.setdefault('initializer', _init_worker_thread_caps)
            kwargs.setdefault('initargs', (int(worker_threads),))
        elif (n_workers is None or n_workers > 1) and not os.environ.get('OMP_NUM_THREADS'):
            warnings.warn(
                "get_executor(worker_threads=None) with an unset OMP_NUM_THREADS: "
                "each worker will spawn one OpenMP thread per core, producing "
                "n_workers * n_cores contending threads and likely zero speedup. "
                "Set worker_threads=1 (default) or export OMP_NUM_THREADS explicitly.",
                RuntimeWarning, stacklevel=2,
            )
        return ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context, **kwargs)

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

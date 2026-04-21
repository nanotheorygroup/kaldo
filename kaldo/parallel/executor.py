"""Pluggable executor backends for kaldo parallel computation.

All executors conform to the ``concurrent.futures.Executor`` interface,
so computation code is identical regardless of backend.
"""

import multiprocessing
import os
import warnings
from concurrent.futures import Future, ProcessPoolExecutor


def _default_mp_context():
    """Return a multiprocessing context safe for CUDA calculators.

    Fork (Linux default) cannot be used once CUDA is initialized in the parent:
    torch refuses to re-initialize CUDA in a forked child (BrokenProcessPool /
    'Cannot re-initialize CUDA in forked subprocess'). Spawn avoids this at a
    small cold-start cost per worker and is the default on macOS.

    Priority: spawn if torch+CUDA is initialized here, else the platform default.
    """
    try:
        import torch  # lazy: don't force-import torch for non-ML users
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return multiprocessing.get_context('spawn')
    except ImportError:
        pass
    return multiprocessing.get_context()  # platform default


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

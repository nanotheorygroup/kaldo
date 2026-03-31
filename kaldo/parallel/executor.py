"""Pluggable executor backends for kaldo parallel computation.

All executors conform to the ``concurrent.futures.Executor`` interface,
so computation code is identical regardless of backend.
"""

import multiprocessing
import os
import sys
import warnings
from concurrent.futures import Future, ProcessPoolExecutor


class SerialExecutor:
    """Executor that runs tasks synchronously in the calling process.

    Implements the ``concurrent.futures.Executor`` interface with zero
    overhead. Useful for debugging (breakpoints work), small systems,
    and environments where multiprocessing is unavailable.
    """

    def submit(self, fn, /, *args, **kwargs):
        """Execute *fn* immediately and return a resolved Future."""
        future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Apply *fn* to each element sequentially."""
        return map(fn, *iterables)

    def shutdown(self, wait=True, *, cancel_futures=False):
        """No-op (nothing to shut down)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


def _get_safe_mp_context():
    """Select a safe multiprocessing start method for the current platform.

    - macOS: ``'spawn'`` (fork is deprecated since Python 3.12).
    - Linux: ``'forkserver'`` preferred, falls back to ``'fork'`` with a
      warning if GPU/network state may be unsafe.
    """
    if sys.platform == 'darwin':
        return multiprocessing.get_context('spawn')

    try:
        return multiprocessing.get_context('forkserver')
    except ValueError:
        pass

    # Fallback to fork — warn if GPU context is live
    if 'tensorflow' in sys.modules:
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                warnings.warn(
                    "Forking with initialized GPU context. This may cause crashes "
                    "on HPC systems. Set KALDO_PARALLEL_BACKEND=serial to avoid.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        except Exception:
            pass
    return multiprocessing.get_context('fork')


def get_executor(backend='process', n_workers=None, **kwargs):
    """Factory for parallel executors.

    Parameters
    ----------
    backend : str
        One of ``'serial'``, ``'process'``, or ``'mpi'``.
    n_workers : int or None
        Number of worker processes. ``None`` auto-detects (all CPUs for
        ``'process'``, all MPI ranks for ``'mpi'``).
    **kwargs
        Additional keyword arguments passed to the underlying executor
        constructor.

    Returns
    -------
    executor : concurrent.futures.Executor
        A context-manager-compatible executor instance.

    Examples
    --------
    >>> with get_executor('serial') as ex:
    ...     future = ex.submit(lambda x: x**2, 5)
    ...     print(future.result())
    25
    """
    # Allow environment override for HPC job scripts
    backend = os.environ.get('KALDO_PARALLEL_BACKEND', backend)

    if backend == 'serial':
        return SerialExecutor()

    elif backend == 'process':
        ctx = _get_safe_mp_context()
        return ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            **kwargs,
        )

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
            f"Unknown backend {backend!r}. Choose from 'serial', 'process', 'mpi'."
        )

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

    else:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            "Choose from 'serial', 'process', 'mpi'."
        )

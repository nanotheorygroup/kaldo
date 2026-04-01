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


def _detect_fork_unsafe_signals():
    """Return a list of reasons why forking would be unsafe, or [] if safe.

    Checks for:
    - Active non-daemon threads beyond the main thread (may hold locks)
    - Initialized GPU context (CUDA is not fork-safe)
    """
    import threading
    reasons = []

    # Non-daemon threads beyond main may hold mutexes that become deadlocked
    # in a forked child. Daemon threads (e.g. Python's own GC) are excluded.
    non_main_threads = [
        t for t in threading.enumerate()
        if t is not threading.main_thread() and not t.daemon
    ]
    if non_main_threads:
        reasons.append(f"{len(non_main_threads)} active non-daemon thread(s)")

    if 'tensorflow' in sys.modules:
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                reasons.append("TensorFlow GPU devices present")
        except Exception:
            pass

    # Not typical kALDo imports, but possibly useful safeguards for
    # users with custom/exotic calculators with GPU backends
    if 'cupy' in sys.modules:
        reasons.append("CuPy imported (GPU context likely initialized)")

    if 'torch' in sys.modules:
        try:
            import torch
            if torch.cuda.is_initialized():
                reasons.append("PyTorch CUDA context initialized")
        except Exception:
            pass


    return reasons


def _get_safe_mp_context():
    """Select a safe multiprocessing start method for the current platform.

    - macOS: ``'spawn'`` (fork is deprecated since Python 3.12).
    - Linux: ``'fork'`` by default. Automatically falls back to
      ``'forkserver'`` if unsafe signals are detected (active non-daemon
      threads or an initialized GPU context). When falling back, a warning
      is emitted explaining the ``if __name__ == '__main__':`` requirement.
    """
    if sys.platform == 'darwin':
        return multiprocessing.get_context('spawn')

    unsafe_reasons = _detect_fork_unsafe_signals()
    if unsafe_reasons:
        warnings.warn(
            "kaldo detected conditions that make fork() unsafe "
            f"({'; '.join(unsafe_reasons)}), falling back to 'forkserver'. "
            "Your script must guard parallel calls with "
            "'if __name__ == \"__main__\":'. "
            "Set KALDO_PARALLEL_BACKEND=serial to suppress this warning.",
            RuntimeWarning,
            stacklevel=3,
        )
        try:
            return multiprocessing.get_context('forkserver')
        except ValueError:
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

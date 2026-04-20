"""Parallelization abstractions for kaldo.

Provides a unified interface for distributing computation across backends:
- ``'serial'``: No parallelism. Useful for debugging and small systems.
- ``'process'``: Shared-memory parallelism via ``ProcessPoolExecutor``.
- ``'mpi'``: Distributed parallelism via ``mpi4py.futures.MPIPoolExecutor``.

Usage::

    from kaldo.parallel import get_executor

    with get_executor(backend='process', n_workers=4) as executor:
        futures = [executor.submit(compute_fn, arg) for arg in work_items]
        results = [f.result() for f in futures]
"""

import pickle

from kaldo.parallel.executor import get_executor, SerialExecutor

__all__ = ['get_executor', 'SerialExecutor', 'is_parallel', 'validate_parallel_calculator']


def is_parallel(n_workers):
    """Return True if ``n_workers`` selects a parallel backend.

    ``n_workers > 1`` requests that many workers; ``n_workers is None`` asks
    the executor to pick (all available CPUs). ``n_workers == 1`` is serial.
    """
    return n_workers is None or n_workers > 1


def validate_parallel_calculator(calculator, method):
    """Check that ``calculator`` is safe to use with ``n_workers > 1``.

    Accepted:
      - ``None`` (calculator will be taken from ``replicated_atoms.calc``; its
        picklability is checked separately by the caller if needed).
      - Any callable factory (class, ``functools.partial``, lambda): each worker
        invokes it once to build a fresh instance in its own process.
      - A picklable ASE ``Calculator`` instance: serialized via multiprocessing.

    Rejected with a copy-pasteable fix message:
      - Non-picklable instances (MatterSim, MACE, CPUNEP, orb, ...), which hold
        PyTorch models, GPU contexts, or C handles that don't survive pickle.

    Parameters
    ----------
    calculator : None, callable, or ASE Calculator instance
        The calculator argument the user supplied to the parallel API.
    method : str
        Name of the API method the user called (e.g. ``"SecondOrder.calculate"``).
        Used in the error message so the user knows exactly which call to fix.

    Raises
    ------
    TypeError
        If ``calculator`` is a non-callable non-picklable instance.
    """
    if calculator is None or callable(calculator):
        return
    try:
        pickle.dumps(calculator)
    except Exception as exc:
        cls = type(calculator).__name__
        raise TypeError(
            f"{method} with n_workers > 1 requires a picklable calculator.\n"
            f"  {cls} instance cannot be pickled: {exc}\n"
            f"\n"
            f"Fix: pass a factory so each worker builds its own instance:\n"
            f"    from functools import partial\n"
            f"    calculator = partial({cls}, ...your_init_kwargs...)\n"
            f"    {method}(calculator, n_workers=N)\n"
            f"\n"
            f"If your calculator uses CUDA, workers run under the 'spawn' start\n"
            f"method — wrap your script's top-level code in:\n"
            f"    if __name__ == '__main__':\n"
            f"        ...your kaldo calls...\n"
            f"\n"
            f"Or run serially with n_workers=1 (the default)."
        ) from exc

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

from kaldo.parallel.executor import get_executor, SerialExecutor

__all__ = ['get_executor', 'SerialExecutor']

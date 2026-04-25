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
import warnings

from kaldo.parallel.executor import get_executor, SerialExecutor

__all__ = [
    'get_executor', 'SerialExecutor', 'is_parallel',
    'validate_parallel_calculator', 'maybe_warn_ml_delta_shift',
]


# Module prefixes whose calculators are typically torch-based and run in
# float32. Used as a heuristic to decide whether to warn about a too-small
# delta_shift. Conservative — we only warn when we're pretty sure the user
# is on an ML potential where finite-difference noise blows up.
_ML_CALCULATOR_MODULE_PREFIXES = (
    'orb_models.',
    'mace.',
    'mattersim.',
    'calorine.',     # NEP via calorine
    'sevenn.',       # SevenNet
    'pyace.',        # ACE
    'fairchem.',
)

# Below this delta_shift, float32 force noise (~1e-7 eV/Å) divided by delta
# produces an FD second derivative noise of ~1e-3, which dominates real
# physics. Above this delta, truncation error in the central difference
# starts to matter but is usually still small. Ranges 1e-2 to 5e-2 work
# well in practice for ML potentials at room-temperature phonon scales.
_ML_DELTA_RECOMMENDATION = 1e-2


def is_parallel(n_workers):
    """Return True if ``n_workers`` selects a parallel backend.

    ``n_workers > 1`` requests that many workers; ``n_workers is None`` asks
    the executor to pick (all available CPUs). ``n_workers == 1`` is serial.
    """
    return n_workers is None or n_workers > 1


def _looks_like_ml_calculator(calculator):
    """Heuristic: is ``calculator`` a torch/float32 ML potential?

    Returns True if its class lives in a module known to host an ML
    calculator (``orb_models``, ``mace``, ``mattersim``, ``calorine``,
    etc.). Returns False for None, callables we can't introspect cheaply,
    and analytical calculators (EMT, LJ, LAMMPS).
    """
    if calculator is None:
        return False
    if isinstance(calculator, type):
        cls = calculator
    elif callable(calculator):
        # Plain callables (lambdas, top-level functions, partials) can't be
        # introspected reliably; skip the warning rather than speculate.
        return False
    else:
        cls = type(calculator)
    module = getattr(cls, '__module__', '') or ''
    return module.startswith(_ML_CALCULATOR_MODULE_PREFIXES)


def maybe_warn_ml_delta_shift(calculator, delta_shift, method):
    """Warn once when ``delta_shift`` looks too small for an ML calculator.

    Float32 ML potentials produce force noise on the order of 1e-7 eV/Å.
    Finite-difference second derivatives divide this by ``delta_shift``, so
    ``delta_shift=1e-4`` gives ~1e-3 noise on each FC entry — same order
    of magnitude as the real physics. ``delta_shift >= 1e-2`` keeps noise
    comfortably below the signal.

    Heuristic: only fire when we recognize the calculator's module as ML.
    Stays silent for EMT, LAMMPS, or unknown calculators (where the user
    presumably knows their precision better than we do).
    """
    if delta_shift >= _ML_DELTA_RECOMMENDATION:
        return
    if not _looks_like_ml_calculator(calculator):
        return
    warnings.warn(
        f"{method}: delta_shift={delta_shift:.0e} is small for an ML "
        f"calculator. Float32 force noise (~1e-7 eV/Å) divided by this "
        f"delta gives FC noise ~{1e-7/delta_shift:.0e}, which can swamp "
        f"the real physics. Try delta_shift >= "
        f"{_ML_DELTA_RECOMMENDATION:.0e} for ML potentials.",
        UserWarning, stacklevel=3,
    )


def validate_parallel_calculator(calculator, method):
    """Check that ``calculator`` is safe to use with ``n_workers > 1``.

    Accepted:
      - ``None``: ``replicated_atoms.calc`` supplies the calculator; the
        caller is responsible for verifying picklability separately.
      - A callable factory (class, top-level function, ``functools.partial``,
        lambda): each worker invokes it once to build a fresh instance in
        its own process. This is the recommended form for any torch- or
        GPU-backed calculator.
      - A picklable ASE ``Calculator`` instance: serialized via
        multiprocessing.

    Rejected with a copy-pasteable fix message:
      - Non-picklable instances (MatterSim, MACE, CPUNEP, orb, ...), which
        hold PyTorch models, GPU contexts, or C handles that don't survive
        pickle.

    Parameters
    ----------
    calculator : None, callable, or ASE Calculator instance
        The calculator argument the user supplied to the parallel API.
    method : str
        Name of the API method the user called (e.g.
        ``"SecondOrder.calculate"``). Used in the error message so the user
        knows exactly which call to fix.

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
            f"Fix: define a no-arg factory function at module top level and\n"
            f"pass it (don't call it). Each worker invokes the function once\n"
            f"to build its own isolated calculator:\n"
            f"\n"
            f"    def make_calculator():\n"
            f"        return {cls}(...your_init_args...)\n"
            f"\n"
            f"    if __name__ == '__main__':\n"
            f"        {method}(make_calculator, n_workers=N)\n"
            f"\n"
            f"The ``if __name__ == '__main__':`` guard is required because\n"
            f"parallel kaldo uses the 'spawn' start method on CUDA-equipped\n"
            f"hosts, which re-imports your script in every worker.\n"
            f"\n"
            f"Or run serially with n_workers=1 (the default)."
        ) from exc

"""Memory estimation and worker-capping for parallel second- and third-order calculations.

Estimates per-worker memory cost before parallelizing and reduces n_workers
to a safe level when the system would otherwise exhaust memory through swap
thrashing (Linux fork+overcommit hides the true cost from the OOM killer).
"""

import gc
import math
import os
import warnings

import psutil

from kaldo.helpers.logger import get_logger

logging = get_logger()

# Memory estimation constants (MB unless noted)
#
# FORK_BASE_OVERHEAD_MB: With Linux fork+CoW, a child inherits parent memory
#   by sharing pages; only dirtied pages count against the child. For a Python
#   worker that mostly runs numpy on pre-existing arrays, dirty pages are
#   typically 5-20 MB of process-private state.
# SPAWN_BASE_OVERHEAD_MB: A spawned worker is a fresh interpreter and must
#   re-import numpy, scipy, ase, sparse, kaldo, etc. ~60 MB covers the
#   scientific stack.
# PARENT_RESERVE_MB: Headroom for parent process growth (mostly general
#   allocation noise; in-memory result lists are already covered by the
#   per-worker accumulation_buffer on the worker side).
# DEFAULT_HEADROOM_FRACTION: The amount of memory that should be reserved
#   for other processes. We assume you're ONLY launching kALDo so we just
#   leave a buffer of 10% of total RAM (e.g. for 32 Gb it leaves 3.2 Gb unused).
FORK_BASE_OVERHEAD_MB = 15
SPAWN_BASE_OVERHEAD_MB = 60
PARENT_RESERVE_MB = 50
DEFAULT_HEADROOM_FRACTION = 0.10
ACCUMULATION_SAFETY_FACTOR = 1.15
ASE_ATOMS_METADATA_BYTES = 4096


def probe_calculator_memory_mb(calculator, replicated_atoms):
    """Instantiate one calculator and run one force evaluation to measure RSS delta.

    Parameters
    ----------
    calculator : callable or ASE Calculator instance or None
        The calculator factory or instance. If callable, called with no
        arguments to produce a calculator. If None, probes using whatever
        calculator is already attached to ``replicated_atoms`` (matching
        how workers behave when ``calculator=None`` in ``calculate_third``).
    replicated_atoms : ASE Atoms
        The supercell atoms object used for the probe force evaluation.

    Returns
    -------
    float or None
        Measured RSS delta in MB, or None if the probe failed.
    """
    try:
        process = psutil.Process()
        gc.collect()
        rss_before = process.memory_info().rss

        atoms_copy = replicated_atoms.copy()
        if calculator is not None:
            atoms_copy.calc = calculator() if callable(calculator) else calculator
        # else: rely on whatever calc was already attached to replicated_atoms
        atoms_copy.get_forces()

        gc.collect()
        rss_after = process.memory_info().rss

        delta_mb = (rss_after - rss_before) / (1024 * 1024)
        # RSS can decrease due to GC of other objects; floor at 0
        delta_mb = max(0.0, delta_mb)

        del atoms_copy
        gc.collect()

        logging.info(f'Memory probe: calculator + one force evaluation = {delta_mb:.1f} MB')
        return delta_mb

    except Exception as exc:
        logging.warning(f'Memory probe failed ({exc}); skipping memory-based worker cap')
        return None


def estimate_worker_memory_mb(n_atoms, n_replicas, use_scratch, jat_flush_every,
                              start_method, calculator_memory_mb):
    """Estimate peak memory per worker in MB.

    Parameters
    ----------
    n_atoms : int
        Number of atoms in the unit cell.
    n_replicas : int
        Supercell multiplicity (product of supercell dimensions).
    use_scratch : bool
        Whether scratch_dir is set (bounds accumulation buffer).
    jat_flush_every : int
        Flush window size for scratch mode.
    start_method : str
        Multiprocessing start method: 'fork', 'spawn', or 'forkserver'.
    calculator_memory_mb : float
        Calculator memory cost in MB (from probe or default).

    Returns
    -------
    float
        Estimated peak memory per worker in MB.
    """
    n_total = n_replicas * n_atoms

    # Base Python interpreter overhead
    if start_method == 'fork':
        base_overhead = FORK_BASE_OVERHEAD_MB
    else:
        base_overhead = SPAWN_BASE_OVERHEAD_MB

    # replicated_atoms.copy(): positions + numbers + cell + ASE metadata
    atoms_copy_bytes = (
        n_total * 3 * 8       # positions (float64)
        + n_total * 8          # atomic numbers (int64)
        + 9 * 8               # cell (3x3 float64)
        + ASE_ATOMS_METADATA_BYTES
    )
    atoms_copy_mb = atoms_copy_bytes / (1024 * 1024)

    # Peak temporaries during calculate_gradient: ~6 concurrent arrays
    # (shift, phi_partial inner, phi_partial outer, atoms copy positions,
    #  forces from get_forces, reshaped grad)
    peak_temp_bytes = 6 * n_total * 3 * 8
    peak_temp_mb = peak_temp_bytes / (1024 * 1024)

    # Accumulation buffer
    n_k = n_total * 3  # length of force vector per computation
    if use_scratch:
        # Bounded by flush window: jat_flush_every jats × 9 (icoord,jcoord) pairs
        # Each entry: 5 int64 coords (40B) + 1 float64 value (8B) = 48B per k-index
        buffer_bytes = (jat_flush_every * 9 * n_k * 48
                        * ACCUMULATION_SAFETY_FACTOR)
    else:
        # In-memory: worst-case all jat pairs for one iat
        # Each scalar entry in Python lists: ~80B overhead (int/float objects + list pointers)
        n_entries = n_total * 9 * n_k
        buffer_bytes = n_entries * 80 * ACCUMULATION_SAFETY_FACTOR

    buffer_mb = buffer_bytes / (1024 * 1024)

    total = base_overhead + atoms_copy_mb + calculator_memory_mb + peak_temp_mb + buffer_mb
    return total


def cap_workers(requested_workers, n_atoms, n_replicas, use_scratch,
                jat_flush_every, calculator, replicated_atoms, start_method):
    """Determine safe number of workers based on available memory.

    Parameters
    ----------
    requested_workers : int or None
        User-requested n_workers. None means auto-detect (all CPUs).
    n_atoms : int
        Unit cell atom count.
    n_replicas : int
        Supercell multiplicity.
    use_scratch : bool
        Whether scratch_dir is set.
    jat_flush_every : int
        Flush window size for scratch mode.
    calculator : callable or ASE Calculator instance or None
        The calculator factory/instance for probing.
    replicated_atoms : ASE Atoms
        Supercell atoms for probing.
    start_method : str
        Multiprocessing start method.

    Returns
    -------
    safe_workers : int
        Safe number of workers (>= 1, <= requested_workers).
    estimate_mb : float
        Estimated peak memory per worker in MB.
    warning_msg : str or None
        Warning message if workers were reduced, None otherwise.
    """
    if requested_workers is None:
        requested_workers = os.cpu_count() or 1

    # Probe calculator memory. A calculator is required to compute force
    # constants, so we can always measure its real cost rather than guess.
    # If the probe fails (e.g., broken calculator), skip the memory check
    # and let the actual calculation surface the underlying error.
    calc_mb = probe_calculator_memory_mb(calculator, replicated_atoms)
    if calc_mb is None:
        warning_msg = (
            'Memory probe failed; skipping worker cap. '
            f'Proceeding with {requested_workers} workers as requested.'
        )
        return requested_workers, 0.0, warning_msg

    estimate_mb = estimate_worker_memory_mb(
        n_atoms, n_replicas, use_scratch, jat_flush_every,
        start_method, calc_mb,
    )

    try:
        vm = psutil.virtual_memory()
        available_mb = vm.available / (1024 * 1024)
        total_mb = vm.total / (1024 * 1024)
    except Exception:
        logging.warning('Could not query system memory; skipping worker cap')
        return requested_workers, estimate_mb, None

    headroom = float(os.environ.get('KALDO_MEMORY_HEADROOM', DEFAULT_HEADROOM_FRACTION))
    usable_mb = available_mb * (1 - headroom)

    safe_workers = max(1, math.floor((usable_mb - PARENT_RESERVE_MB) / estimate_mb))
    safe_workers = min(safe_workers, requested_workers)

    warning_msg = None
    if safe_workers < requested_workers:
        warning_msg = (
            f'Memory safety: reducing n_workers from {requested_workers} to {safe_workers}. '
            f'Estimated {estimate_mb:.0f} MB per worker; '
            f'{usable_mb:.0f} MB usable of {total_mb:.0f} MB total. '
            f'Set KALDO_SKIP_MEMORY_CHECK=1 to disable this check.'
        )

    return safe_workers, estimate_mb, warning_msg

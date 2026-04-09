"""Memory estimation and worker-capping for parallel third-order calculations.

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
FORK_BASE_OVERHEAD_MB = 50
SPAWN_BASE_OVERHEAD_MB = 80
PARENT_RESERVE_MB = 100
DEFAULT_CALCULATOR_FORK_MB = 10
DEFAULT_CALCULATOR_SPAWN_MB = 200
DEFAULT_HEADROOM_FRACTION = 0.80
ACCUMULATION_SAFETY_FACTOR = 1.5
ASE_ATOMS_METADATA_BYTES = 4096


def probe_calculator_memory_mb(calculator, replicated_atoms):
    """Instantiate one calculator and run one force evaluation to measure RSS delta.

    Parameters
    ----------
    calculator : callable or ASE Calculator instance
        The calculator factory or instance. If callable, called with no
        arguments to produce a calculator.
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
        atoms_copy.calc = calculator() if callable(calculator) else calculator
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
        logging.warning(f'Memory probe failed ({exc}); using conservative default')
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

    # Probe calculator memory
    calc_mb = None
    if calculator is not None:
        calc_mb = probe_calculator_memory_mb(calculator, replicated_atoms)

    if calc_mb is None:
        if start_method == 'fork':
            calc_mb = DEFAULT_CALCULATOR_FORK_MB
        else:
            calc_mb = DEFAULT_CALCULATOR_SPAWN_MB
            logging.warning(
                f'Using conservative default calculator memory estimate '
                f'({calc_mb} MB) for {start_method} start method.'
            )

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
    usable_mb = available_mb * headroom

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

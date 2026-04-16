import functools
import glob
import os
import pickle
import numpy as np
from concurrent.futures import as_completed
from kaldo.helpers.logger import get_logger
from kaldo.parallel import get_executor
from sparse import COO
logging = get_logger()



def list_of_replicas(atoms, replicated_atoms):
    n_atoms = atoms.positions.shape[0]
    n_replicas = int(replicated_atoms.positions.shape[0] / n_atoms)
    list_of_cells = (replicated_atoms.positions - atoms.positions).reshape((n_replicas, ))
    return list_of_cells


def calculate_gradient(x, input_atoms):
    """
    Construct the calculate_gradient based on the given structure and atom object
    Set a copy for the atom object so that
    the progress of the optimization is traceable
    Force is the negative of the calculate_gradient
    """

    atoms = input_atoms.copy()
    input_atoms.positions = np.reshape(x, (int(x.size / 3.), 3))
    gr = -1. * input_atoms.get_forces()
    grad = np.reshape(gr, gr.size)
    input_atoms.positions = atoms.positions
    return grad


def _validate_calculator(calculator, picklable=False):
    """Raise TypeError if calculator is not a callable or ASE Calculator.

    When ``picklable=True``, also verify that the calculator can be pickled,
    since parallel execution ships it across process boundaries.
    """
    if calculator is None:
        return
    if callable(calculator):
        pass
    else:
        try:
            from ase.calculators.calculator import Calculator
            is_ase_calc = isinstance(calculator, Calculator)
        except ImportError:
            is_ase_calc = False
        if not is_ase_calc:
            raise TypeError(
                f"calculator must be a callable (e.g. EMT) or ASE Calculator instance, "
                f"got {type(calculator).__name__}"
            )
    if picklable:
        try:
            pickle.dumps(calculator)
        except Exception as e:
            raise TypeError(
                f"calculator of type {type(calculator).__name__} is not picklable, "
                f"which is required for parallel execution. Wrap the class in "
                f"kaldo.parallel.CalculatorFactory or pass the class itself "
                f"(not an instance). Underlying error: {e}"
            ) from e


def calculate_second(atoms, replicated_atoms, second_order_delta, is_verbose=False, n_workers=1, calculator=None,
                     scratch_dir=None, keep_scratch=False):
    """
    Core method to compute second order force constant matrices
    Approximate the second order force constant matrices
    using central difference formula
    """
    if n_workers is not None and n_workers < 1:
        raise ValueError(f"n_workers must be >= 1 or None, got {n_workers}")
    use_parallel = n_workers is None or n_workers > 1
    _validate_calculator(calculator, picklable=use_parallel)

    logging.info('Calculating second order potential derivatives, ' + 'finite difference displacement: %.3e angstrom'%second_order_delta)
    n_unit_cell_atoms = len(atoms.numbers)
    n_replicated_atoms = len(replicated_atoms.numbers)
    n_atoms = n_unit_cell_atoms
    n_replicas = int(n_replicated_atoms / n_unit_cell_atoms)
    use_scratch = scratch_dir is not None
    if use_scratch:
        os.makedirs(scratch_dir, exist_ok=True)
        atoms_to_compute = [
            atom_id for atom_id in range(n_atoms)
            if not os.path.exists(os.path.join(scratch_dir, f'iat_{atom_id:05d}.done'))
        ]
        n_resumed = n_atoms - len(atoms_to_compute)
        if n_resumed:
            logging.info(f'Resuming: skipping {n_resumed} already-computed atom(s)')
    else:
        second = np.zeros((n_atoms, 3, n_replicated_atoms * 3))
        atoms_to_compute = list(range(n_atoms))
    backend = 'process' if use_parallel else 'serial'
    executor_workers = n_workers if use_parallel else None

    worker_fn = functools.partial(
        _compute_iat_second,
        calculator=calculator,
        scratch_dir=scratch_dir,
    )

    with get_executor(backend=backend, n_workers=executor_workers) as executor:
        futures = {
            executor.submit(worker_fn, i, replicated_atoms, second_order_delta): i
            for i in atoms_to_compute
        }
        for future in as_completed(futures):
            atom_id, second_per_atom = future.result()
            if is_verbose:
                logging.info('calculating forces on atom ' + str(atom_id))
            if not use_scratch:
                second[atom_id] = second_per_atom

    if use_scratch:
        second = _assemble_from_scratch_second(scratch_dir, n_atoms, n_replicated_atoms, keep_scratch)

    second = second.reshape((1, n_unit_cell_atoms, 3, n_replicas, n_unit_cell_atoms, 3))
    second = second / (2. * second_order_delta)
    asymmetry = np.sum(np.abs(second[0, :, :, 0, :, :] - np.transpose(second[0, :, :, 0, :, :], (2, 3, 0, 1))))
    logging.info('Symmetry of Dynamical Matrix ' + str(asymmetry))
    return second


def _compute_iat_second(atom_id, replicated_atoms, second_order_delta, calculator=None,
                        scratch_dir=None):
    """Compute second-order force constants for a single unit cell atom.

    Uses central difference: (forward force - backward force) for each
    Cartesian direction.
    """
    if calculator is not None:
        replicated_atoms = replicated_atoms.copy()
        replicated_atoms.calc = calculator() if callable(calculator) else calculator
    n_replicated_atoms = len(replicated_atoms.numbers)
    second_per_atom = np.zeros((3, n_replicated_atoms * 3))
    for alpha in range(3):
        for move in (-1, 1):
            shift = np.zeros((n_replicated_atoms, 3))
            shift[atom_id, alpha] += move * second_order_delta
            second_per_atom[alpha, :] += move * calculate_gradient(replicated_atoms.positions + shift,
                                                                   replicated_atoms)
    if scratch_dir is not None:
        np.save(os.path.join(scratch_dir, f'iat_{atom_id:05d}.npy'), second_per_atom)
        open(os.path.join(scratch_dir, f'iat_{atom_id:05d}.done'), 'w').close()
        return atom_id, None
    return atom_id, second_per_atom


def _assemble_from_scratch_second(scratch_dir, n_atoms, n_replicated_atoms, keep_scratch):
    second = np.empty((n_atoms, 3, n_replicated_atoms * 3), dtype=np.float64)
    for atom_id in range(n_atoms):
        path = os.path.join(scratch_dir, f'iat_{atom_id:05d}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing scratch file for atom {atom_id}: {path}')
        second[atom_id] = np.load(path)
        if not keep_scratch:
            os.remove(path)

    if not keep_scratch:
        for sentinel in glob.glob(os.path.join(scratch_dir, 'iat_*.done')):
            os.remove(sentinel)
        try:
            os.rmdir(scratch_dir)
        except OSError:
            pass

    return second


def calculate_third(atoms, replicated_atoms, third_order_delta, distance_threshold=None, is_verbose=False,
                    n_workers=1, calculator=None, scratch_dir=None, keep_scratch=False,
                    jat_flush_every=50):
    """
    Compute third order force constant matrices by using the central
    difference formula for the approximation.

    Parameters
    ----------
    atoms : ASE Atoms instance
    	An atoms object of the unit cell
    replicated_atoms : ASE Atoms instance
    	An atoms object of the super cell
    third_order_delta : float
    	How far to move the atoms
    distance_threshold : float or None
    	When an argument is provided, the third order force constants are only
    	calculated when the distance between atom i and atom j is less than
    	the cutoff (in units of Angstrom)
    n_workers : int or None
        Number of parallel worker processes. ``1`` runs serially (default).
        ``None`` uses all available CPUs. Values > 1 launch that many workers
        via ``concurrent.futures.ProcessPoolExecutor``.
    calculator : callable or ASE Calculator instance or CalculatorFactory
        Either an ASE calculator class, a ``CalculatorFactory``, a
        ``functools.partial``, or an already-constructed instance. When
        running in parallel (``n_workers > 1``), pass a class or factory so
        each worker can create its own instance::

            from ase.calculators.emt import EMT
            calculator=EMT

            from kaldo.parallel import CalculatorFactory
            calculator=CalculatorFactory(NEP, args=('nep.txt',))

        If None, replicated_atoms must already have a calculator attached.
    scratch_dir : str or None
        Path to a directory for scratch chunk files (``iat_NNNNN_chunk_MMMM.npz``).
        Workers flush to disk every ``jat_flush_every`` jat iterations, keeping
        peak memory proportional to one flush window rather than a full atom.
        The directory is created if it does not exist. If None (default), results
        are accumulated in memory (original behaviour).
    keep_scratch : bool
        If True, scratch files are preserved after successful assembly. If False
        (default), each file is deleted as it is consumed during assembly, and the
        directory is removed if empty.
    jat_flush_every : int
        Number of jat iterations to buffer before flushing to disk. Only used
        when ``scratch_dir`` is set. Default 50.
    """
    if n_workers is not None and n_workers < 1:
        raise ValueError(f"n_workers must be >= 1 or None, got {n_workers}")
    use_parallel = n_workers is None or n_workers > 1
    _validate_calculator(calculator, picklable=use_parallel)

    logging.info('Calculating third order potential derivatives, ' + 'finite difference displacement: %.3e angstrom'%third_order_delta)
    n_atoms = len(atoms.numbers)
    n_replicas = int(replicated_atoms.positions.shape[0] / n_atoms)
    use_scratch = scratch_dir is not None
    if use_scratch:
        os.makedirs(scratch_dir, exist_ok=True)
    else:
        i_at_sparse = []
        i_coord_sparse = []
        jat_sparse = []
        j_coord_sparse = []
        k_sparse = []
        value_sparse = []
    n_forces_to_calculate = n_replicas * (n_atoms * 3) ** 2
    n_forces_done = 0
    n_forces_skipped = 0

    # Determine which atoms need computing (resume support via scratch sentinels)
    if use_scratch:
        atoms_to_compute = [iat for iat in range(n_atoms)
                             if not os.path.exists(os.path.join(scratch_dir, f'iat_{iat:05d}.done'))]
        n_resumed = n_atoms - len(atoms_to_compute)
        if n_resumed:
            logging.info(f'Resuming: skipping {n_resumed} already-computed atom(s)')
    else:
        atoms_to_compute = list(range(n_atoms))

    # Select backend: 'process' for parallel, 'serial' for single-threaded
    backend = 'process' if use_parallel else 'serial'
    executor_workers = n_workers if use_parallel else None

    worker_fn = functools.partial(
        _compute_iat_third,
        calculator=calculator,
        scratch_dir=scratch_dir,
        jat_flush_every=jat_flush_every,
    )

    with get_executor(backend=backend, n_workers=executor_workers) as executor:
        futures = {
            executor.submit(worker_fn, iat, atoms, replicated_atoms,
                           third_order_delta, distance_threshold, is_verbose): iat
            for iat in atoms_to_compute
        }
        for future in as_completed(futures):
            iat = futures[future]
            local_i_at, local_i_coord, local_jat, local_j_coord, local_k, local_value, n_done, n_skipped = future.result()
            if not use_scratch:
                i_at_sparse.extend(local_i_at)
                i_coord_sparse.extend(local_i_coord)
                jat_sparse.extend(local_jat)
                j_coord_sparse.extend(local_j_coord)
                k_sparse.extend(local_k)
                value_sparse.extend(local_value)
            n_forces_done += n_done
            n_forces_skipped += n_skipped
            if use_parallel:
                logging.info(f'Completed atom {iat}: '
                             f'{int((n_forces_done + n_forces_skipped) / n_forces_to_calculate * 100)}% done')
            elif (n_forces_done + n_forces_skipped) % 300 == 0:
                logging.info('Calculate third derivatives ' + str(
                    int((n_forces_done + n_forces_skipped) / n_forces_to_calculate * 100)) + '%')
    logging.info('total forces to calculate third : ' + str(n_forces_to_calculate))
    logging.info('forces calculated : ' + str(n_forces_done))
    logging.info('forces skipped (outside distance threshold) : ' + str(n_forces_skipped))
    if use_scratch:
        return _assemble_from_scratch_third(scratch_dir, n_atoms, n_replicas, keep_scratch)
    coords = np.array([i_at_sparse, i_coord_sparse, jat_sparse, j_coord_sparse, k_sparse])
    shape = (n_atoms, 3, n_replicas * n_atoms, 3, n_replicas * n_atoms * 3)
    phifull = COO(coords, np.array(value_sparse), shape)
    phifull = phifull.reshape((n_atoms * 3, n_replicas * n_atoms * 3, n_replicas * n_atoms * 3))
    return phifull
    

def _compute_iat_third(iat, atoms, replicated_atoms, third_order_delta, distance_threshold, is_verbose,
                 calculator=None, scratch_dir=None, jat_flush_every=50):
    """Compute all third-order force constant terms for a single unit cell atom index.

    Parameters
    ----------
    calculator : callable, ASE Calculator instance, or None
        If a callable (class, CalculatorFactory, or functools.partial), it is
        called with no arguments to create a fresh ASE calculator instance,
        which is attached to a copy of replicated_atoms. If an ASE Calculator
        instance, it is attached to the copy directly. If None,
        replicated_atoms must already have a calculator attached.
    scratch_dir : str or None
        If provided, results are written directly to ``scratch_dir`` as a series of
        ``iat_NNNNN_chunk_MMMM.npz`` files. A ``iat_NNNNN.done`` sentinel is written
        on completion. Empty lists are returned in place of data to keep peak memory
        proportional to one flush window rather than the full atom.
    jat_flush_every : int
        Number of jat iterations to accumulate (as compact numpy arrays) before
        flushing to disk. Only used when ``scratch_dir`` is set.
    """
    if calculator is not None:
        replicated_atoms = replicated_atoms.copy()
        replicated_atoms.calc = calculator() if callable(calculator) else calculator
    n_atoms = len(atoms.numbers)
    n_replicas = int(replicated_atoms.positions.shape[0] / n_atoms)
    n_done = 0
    n_skipped = 0

    use_scratch = scratch_dir is not None
    chunk_id = 0
    jat_count_in_chunk = 0
    chunk_coords = []
    chunk_values = []

    for jat in range(n_replicas * n_atoms):
        is_computing = True
        if distance_threshold is not None:
            dxij = replicated_atoms.get_distance(iat, jat, mic=True, vector=False)
            if dxij > distance_threshold:
                is_computing = False
                n_skipped += 9
        if is_computing:
            if is_verbose:
                logging.info(f'calculating forces on atoms: {iat}, {jat}, '
                             f'{dxij if distance_threshold is not None else None}')
            for icoord in range(3):
                for jcoord in range(3):
                    value = calculate_single_third(atoms, replicated_atoms, iat, icoord, jat, jcoord,
                                                   third_order_delta)
                    n_k = value.shape[0]
                    chunk_coords.append(np.array([
                        np.full(n_k, iat,    dtype=np.int64),
                        np.full(n_k, icoord, dtype=np.int64),
                        np.full(n_k, jat,    dtype=np.int64),
                        np.full(n_k, jcoord, dtype=np.int64),
                        np.arange(n_k,       dtype=np.int64),
                    ]))
                    chunk_values.append(value.astype(np.float64))
            n_done += 9
        if use_scratch:
            jat_count_in_chunk += 1
            if jat_count_in_chunk >= jat_flush_every and chunk_values:
                _flush_chunk_third(scratch_dir, iat, chunk_id, chunk_coords, chunk_values)
                chunk_id += 1
                jat_count_in_chunk = 0
                chunk_coords = []
                chunk_values = []

    if use_scratch:
        if chunk_values:
            _flush_chunk_third(scratch_dir, iat, chunk_id, chunk_coords, chunk_values)
        open(os.path.join(scratch_dir, f'iat_{iat:05d}.done'), 'w').close()
        return [], [], [], [], [], [], n_done, n_skipped

    if not chunk_coords:
        return [], [], [], [], [], [], n_done, n_skipped
    coords = np.concatenate(chunk_coords, axis=1)
    values = np.concatenate(chunk_values)
    return (coords[0].tolist(), coords[1].tolist(), coords[2].tolist(),
            coords[3].tolist(), coords[4].tolist(), values.tolist(), n_done, n_skipped)


def calculate_single_third(atoms, replicated_atoms, iat, icoord, jat, jcoord, third_order_delta):
    n_in_unit_cell = len(atoms.numbers)
    n_replicated_atoms = len(replicated_atoms.numbers)
    n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
    phi_partial = np.zeros((n_supercell * n_in_unit_cell * 3))
    for isign in (1, -1):
        for jsign in (1, -1):
            shift = np.zeros((n_replicated_atoms, 3))
            shift[iat, icoord] += isign * third_order_delta
            shift[jat, jcoord] += jsign * third_order_delta
            phi_partial[:] += isign * jsign * (-1. * calculate_gradient(replicated_atoms.positions + shift, replicated_atoms))
    return phi_partial / (4. * third_order_delta * third_order_delta)


def _flush_chunk_third(scratch_dir, iat, chunk_id, chunk_coords, chunk_values):
    """Concatenate buffered numpy arrays and write one chunk file to disk."""
    coords = np.concatenate(chunk_coords, axis=1)   # shape (5, total_k)
    values = np.concatenate(chunk_values)            # shape (total_k,)
    path = os.path.join(scratch_dir, f'iat_{iat:05d}_chunk_{chunk_id:04d}.npz')
    np.savez_compressed(path, coords=coords, values=values)


def _assemble_from_scratch_third(scratch_dir, n_atoms, n_replicas, keep_scratch):
    """Build the final COO tensor from per-jat-chunk scratch files using two passes.

    Pass 1 reads only array metadata to count total non-zeros, allowing a single
    pre-allocated set of arrays (peak memory ~1x final COO size + one chunk).
    Pass 2 fills those arrays slice-by-slice and optionally deletes each file.
    """
    # Find and sort iat chunk files
    files = sorted(glob.glob(os.path.join(scratch_dir, 'iat_*_chunk_*.npz')))
    if not files:
        raise FileNotFoundError(f'No scratch chunk files found in {scratch_dir}')

    # Pass 1: count total non-zeros from metadata only (lazy load — no data read)
    total_nnz = 0
    for path in files:
        with np.load(path) as f:
            total_nnz += f['values'].shape[0]

    coords = np.empty((5, total_nnz), dtype=np.int64)
    values = np.empty(total_nnz, dtype=np.float64)

    # Pass 2: fill pre-allocated arrays, consuming each file as we go
    offset = 0
    for path in files:
        with np.load(path) as f:
            n = f['values'].shape[0]
            coords[:, offset:offset + n] = f['coords']
            values[offset:offset + n] = f['values']
        offset += n
        if not keep_scratch:
            os.remove(path)

    if not keep_scratch:
        for sentinel in glob.glob(os.path.join(scratch_dir, 'iat_*.done')):
            os.remove(sentinel)
        try:
            os.rmdir(scratch_dir)
        except OSError:
            pass  # not empty — leave it

    shape = (n_atoms, 3, n_replicas * n_atoms, 3, n_replicas * n_atoms * 3)
    phifull = COO(coords, values, shape)
    return phifull.reshape((n_atoms * 3, n_replicas * n_atoms * 3, n_replicas * n_atoms * 3))

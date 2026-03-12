
import glob
import multiprocessing
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from kaldo.helpers.logger import get_logger
from sparse import COO
logging = get_logger()

# Module-level stores for forked worker processes. Set in calculate_third before
# creating the executor; cleared in the finally block afterwards.
_calculator_store = None
_scratch_dir_store = None
_jat_flush_every_store = 50


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


def calculate_single_second(replicated_atoms, atom_id, second_order_delta):
    """
    Compute the numerator of the approximated second matrices
    (approximated force from forward difference -
    approximated force from backward difference )
     """
    n_replicated_atoms = len(replicated_atoms.numbers)
    second_per_atom = np.zeros((3, n_replicated_atoms * 3))
    for alpha in range(3):
        for move in (-1, 1):
            shift = np.zeros((n_replicated_atoms, 3))
            shift[atom_id, alpha] += move * second_order_delta
            second_per_atom[alpha, :] += move * calculate_gradient(replicated_atoms.positions + shift,
                                                                   replicated_atoms)
    return second_per_atom


def calculate_second(atoms, replicated_atoms, second_order_delta, is_verbose=False):
    # TODO: remove supercell
    """
    Core method to compute second order force constant matrices
    Approximate the second order force constant matrices
    using central difference formula
    """
    logging.info('Calculating second order potential derivatives, ' + 'finite difference displacement: %.3e angstrom'%second_order_delta)
    n_unit_cell_atoms = len(atoms.numbers)
    n_replicated_atoms = len(replicated_atoms.numbers)
    n_atoms = n_unit_cell_atoms
    n_replicas = int(n_replicated_atoms / n_unit_cell_atoms)
    second = np.zeros((n_atoms, 3, n_replicated_atoms * 3))
    for i in range(n_atoms):
        if is_verbose:
            logging.info('calculating forces on atom ' + str(i))
        second[i] = calculate_single_second(replicated_atoms, i, second_order_delta)
    second = second.reshape((1, n_unit_cell_atoms, 3, n_replicas, n_unit_cell_atoms, 3))
    second = second / (2. * second_order_delta)
    asymmetry = np.sum(np.abs(second[0, :, :, 0, :, :] - np.transpose(second[0, :, :, 0, :, :], (2, 3, 0, 1))))
    logging.info('Symmetry of Dynamical Matrix ' + str(asymmetry))
    return second


def calculate_third(atoms, replicated_atoms, third_order_delta, distance_threshold=None, is_verbose=False,
                    n_threads=1, calculator=None, scratch_dir=None, keep_scratch=False,
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
    n_threads : int or None
        Number of parallel worker processes. ``1`` runs serially (default).
        ``None`` uses all available CPUs. Values > 1 launch that many workers
        via ``concurrent.futures.ProcessPoolExecutor``.
    calculator : ASE Calculator instance or callable or None
        When ``n_threads == 1`` (serial): pass an already-constructed ASE
        calculator instance; it is assigned directly to replicated_atoms.
        When ``n_threads != 1`` (parallel): pass a zero-argument callable
        (e.g. a class or lambda) that each thread calls to create its own
        isolated instance.
        For file-based calculators (like LAMMPS) in parallel mode, configure 
        a unique working directory per process via the factory, e.g.::

            import os
            calculator=lambda: LAMMPS(tmp_dir=f'/tmp/kaldo_{os.getpid()}')
            
        Where the tmp_dir is the important feature. For an argument-less 
        class like ASE's EMT implementation use::
        
            from ase.calculators.emt import EMT
            calculator=EMT

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

    use_parallel = n_threads is None or n_threads > 1

    if use_parallel:
        max_workers = None if n_threads is None else n_threads
        if calculator is not None:
            # Strip the calculator so atoms are picklable; threads will call it as needed.
            # ASE Atoms.copy() does not copy the calculator.
            replicated_atoms_workers = replicated_atoms.copy()
        else:
            replicated_atoms_workers = replicated_atoms
        global _calculator_store, _scratch_dir_store, _jat_flush_every_store
        _calculator_store = calculator
        _scratch_dir_store = scratch_dir
        _jat_flush_every_store = jat_flush_every
        ctx = multiprocessing.get_context('fork')
        if use_scratch:
            # This list comprehension removes any atoms with sentinel files (empty files that track if
            # a calculation is done) from the calculation queue
            atoms_to_compute = [iat for iat in range(n_atoms)
                                 if not os.path.exists(os.path.join(scratch_dir, f'iat_{iat:05d}.done'))]
            n_resumed = n_atoms - len(atoms_to_compute)
            if n_resumed: # Inform user of progress
                logging.info(f'Resuming: skipping {n_resumed} already-computed atom(s)')
        else: # If no scratch directory, there is no progress - so prepare to iterate over every atom
            atoms_to_compute = list(range(n_atoms))
        try:
            # Core parallelization happens here by setting up the individual threads
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor: 
                futures = {
                    executor.submit(_compute_iat_forked, iat, atoms, replicated_atoms_workers,
                                   third_order_delta, distance_threshold, is_verbose): iat
                    for iat in atoms_to_compute
                } # As these complete this next code block executes >>
                for future in as_completed(futures): # Storing/Compiling the individually computed chunks into files/master list
                    iat = futures[future]
                    local_i_at, local_i_coord, local_jat, local_j_coord, local_k, local_value, n_done, n_skipped = future.result()
                    if not use_scratch:
                        i_at_sparse.extend(local_i_at)
                        i_coord_sparse.extend(local_i_coord)
                        jat_sparse.extend(local_jat)
                        j_coord_sparse.extend(local_j_coord)
                        k_sparse.extend(local_k)
                        value_sparse.extend(local_value)
                    # else: workers wrote their own chunk files directly
                    n_forces_done += n_done
                    n_forces_skipped += n_skipped
                    logging.info(f'Completed atom {iat}: '
                                 f'{int((n_forces_done + n_forces_skipped) / n_forces_to_calculate * 100)}% done')
        finally:
            _calculator_store = None
            _scratch_dir_store = None
    else: # Original serial code
        if calculator is not None:
            # Serial: attach the provided calculator instance directly.
            replicated_atoms = replicated_atoms.copy()
            replicated_atoms.calc = calculator
        for iat in range(n_atoms):
            if use_scratch:
                if os.path.exists(os.path.join(scratch_dir, f'iat_{iat:05d}.done')):
                    logging.info(f'Atom {iat}: already computed, skipping')
                    continue
            local_i_at, local_i_coord, local_jat, local_j_coord, local_k, local_value, n_done, n_skipped = \
                _compute_iat(iat, atoms, replicated_atoms, third_order_delta, distance_threshold, is_verbose,
                             calculator=None, scratch_dir=scratch_dir,
                             jat_flush_every=jat_flush_every)
            if not use_scratch:
                i_at_sparse.extend(local_i_at)
                i_coord_sparse.extend(local_i_coord)
                jat_sparse.extend(local_jat)
                j_coord_sparse.extend(local_j_coord)
                k_sparse.extend(local_k)
                value_sparse.extend(local_value)
            n_forces_done += n_done
            n_forces_skipped += n_skipped
            if (n_forces_done + n_forces_skipped) % 300 == 0:
                logging.info('Calculate third derivatives ' + str(
                    int((n_forces_done + n_forces_skipped) / n_forces_to_calculate * 100)) + '%')

    logging.info('total forces to calculate third : ' + str(n_forces_to_calculate))
    logging.info('forces calculated : ' + str(n_forces_done))
    logging.info('forces skipped (outside distance threshold) : ' + str(n_forces_skipped))
    if use_scratch:
        return _assemble_from_scratch(scratch_dir, n_atoms, n_replicas, keep_scratch)
    coords = np.array([i_at_sparse, i_coord_sparse, jat_sparse, j_coord_sparse, k_sparse])
    shape = (n_atoms, 3, n_replicas * n_atoms, 3, n_replicas * n_atoms * 3)
    phifull = COO(coords, np.array(value_sparse), shape)
    phifull = phifull.reshape((n_atoms * 3, n_replicas * n_atoms * 3, n_replicas * n_atoms * 3))
    return phifull


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
            phi_partial[:] += isign * jsign * calculate_single_third_with_shift(atoms, replicated_atoms, shift)
    return phi_partial / (4. * third_order_delta * third_order_delta)


def calculate_single_third_with_shift(atoms, replicated_atoms, shift):
    n_in_unit_cell = len(atoms.numbers)
    n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
    phi_partial = np.zeros((n_supercell * n_in_unit_cell * 3))
    phi_partial[:] = (-1. * calculate_gradient(replicated_atoms.positions + shift, replicated_atoms))
    return phi_partial
    

def _compute_iat(iat, atoms, replicated_atoms, third_order_delta, distance_threshold, is_verbose,
                 calculator=None, scratch_dir=None, jat_flush_every=50):
    """Compute all third-order force constant terms for a single unit cell atom index.

    Module-level worker function used by calculate_third for parallel execution.

    Parameters
    ----------
    calculator : callable or None
        If provided, called as ``calculator()`` to create a fresh ASE calculator
        instance that is attached to a copy of replicated_atoms. Used by forked
        workers — replicated_atoms must not already have a calculator in this path.
        If None, replicated_atoms must already have a calculator attached.
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
        replicated_atoms.calc = calculator()
    n_atoms = len(atoms.numbers)
    n_replicas = int(replicated_atoms.positions.shape[0] / n_atoms)
    n_done = 0
    n_skipped = 0

    if scratch_dir is not None:
        chunk_id = 0
        jat_count_in_chunk = 0
        chunk_coords = []   # list of (5, n_k) int64 arrays
        chunk_values = []   # list of (n_k,) float64 arrays
        for jat in range(n_replicas * n_atoms):
            is_computing = True
            if distance_threshold is not None:
                dxij = replicated_atoms.get_distance(iat, jat, mic=True, vector=False)
                if dxij > distance_threshold:
                    is_computing = False
                    n_skipped += 9
            if is_computing:
                if is_verbose:
                    logging.info(f'calculating forces on atoms: {iat}, {jat}, {np.linalg.norm(dxij) if distance_threshold is not None else None}')
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
            jat_count_in_chunk += 1
            if jat_count_in_chunk >= jat_flush_every and chunk_values:
                _flush_chunk(scratch_dir, iat, chunk_id, chunk_coords, chunk_values)
                chunk_id += 1
                jat_count_in_chunk = 0
                chunk_coords = []
                chunk_values = []
        if chunk_values:
            _flush_chunk(scratch_dir, iat, chunk_id, chunk_coords, chunk_values)
        open(os.path.join(scratch_dir, f'iat_{iat:05d}.done'), 'w').close()
        return [], [], [], [], [], [], n_done, n_skipped

    # Original in-memory path
    local_i_at = []
    local_i_coord = []
    local_jat = []
    local_j_coord = []
    local_k = []
    local_value = []
    for jat in range(n_replicas * n_atoms):
        is_computing = True
        if distance_threshold is not None:
            dxij = replicated_atoms.get_distance(iat, jat, mic=True, vector=False)
            if dxij > distance_threshold:
                is_computing = False
                n_skipped += 9
                if is_verbose:
                    logging.info(f'calculating forces on atoms: {iat}, {jat}, {np.linalg.norm(dxij)}')
        if is_computing:
            if is_verbose:
                logging.info(f'calculating forces on atoms: {iat}, {jat}, {np.linalg.norm(dxij) if distance_threshold is not None else None}')
            for icoord in range(3):
                for jcoord in range(3):
                    value = calculate_single_third(atoms, replicated_atoms, iat, icoord, jat, jcoord,
                                                   third_order_delta)
                    for id_k in range(value.shape[0]):
                        local_i_at.append(iat)
                        local_i_coord.append(icoord)
                        local_jat.append(jat)
                        local_j_coord.append(jcoord)
                        local_k.append(id_k)
                        local_value.append(value[id_k])
            n_done += 9
    return local_i_at, local_i_coord, local_jat, local_j_coord, local_k, local_value, n_done, n_skipped


def _compute_iat_forked(iat, atoms, replicated_atoms, third_order_delta, distance_threshold, is_verbose):
    """Parallel worker: reads per-run settings from module globals inherited via fork."""
    return _compute_iat(iat, atoms, replicated_atoms, third_order_delta, distance_threshold, is_verbose,
                        calculator=_calculator_store,
                        scratch_dir=_scratch_dir_store,
                        jat_flush_every=_jat_flush_every_store)


def _flush_chunk(scratch_dir, iat, chunk_id, chunk_coords, chunk_values):
    """Concatenate buffered numpy arrays and write one chunk file to disk."""
    coords = np.concatenate(chunk_coords, axis=1)   # shape (5, total_k)
    values = np.concatenate(chunk_values)            # shape (total_k,)
    path = os.path.join(scratch_dir, f'iat_{iat:05d}_chunk_{chunk_id:04d}.npz')
    np.savez_compressed(path, coords=coords, values=values)


def _assemble_from_scratch(scratch_dir, n_atoms, n_replicas, keep_scratch):
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

    #TODO: exact choices of dtype here should be analyzed
    coords = np.empty((5, total_nnz), dtype=np.int32)
    values = np.empty(total_nnz, dtype=np.float32) 

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

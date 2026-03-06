
import glob
import multiprocessing
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from kaldo.helpers.logger import get_logger
from sparse import COO
logging = get_logger()

# Stores calculator_factory during parallel execution so forked workers
# inherit it without pickling. Set before creating the executor; cleared after.
_calculator_factory_store = None


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


def _compute_iat(iat, atoms, replicated_atoms, third_order_delta, distance_threshold, is_verbose,
                 calculator_factory=None):
    """Compute all third-order force constant terms for a single unit cell atom index.

    Module-level worker function used by calculate_third for parallel execution.

    Parameters
    ----------
    calculator_factory : callable or None
        If provided, a fresh calculator is created via ``calculator_factory()`` and
        attached to a copy of replicated_atoms. Use this for file-based calculators
        that cannot be pickled, or when each worker needs an isolated scratch directory.
        If None, replicated_atoms must already have a calculator attached.
    """
    if calculator_factory is not None:
        replicated_atoms = replicated_atoms.copy()
        replicated_atoms.calc = calculator_factory()
    n_atoms = len(atoms.numbers)
    n_replicas = int(replicated_atoms.positions.shape[0] / n_atoms)
    local_i_at = []
    local_i_coord = []
    local_jat = []
    local_j_coord = []
    local_k = []
    local_value = []
    n_done = 0
    n_skipped = 0
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
    """Parallel worker: reads calculator_factory from module global inherited via fork."""
    return _compute_iat(iat, atoms, replicated_atoms, third_order_delta, distance_threshold, is_verbose,
                        calculator_factory=_calculator_factory_store)


def _save_iat_chunk(scratch_dir, iat, local_i_at, local_i_coord, local_jat, local_j_coord, local_k, local_value):
    """Save one atom's sparse IFC data to a compressed numpy file."""
    path = os.path.join(scratch_dir, f'iat_{iat:05d}.npz')
    np.savez_compressed(path,
                        i_at=np.array(local_i_at, dtype=np.int64),
                        i_coord=np.array(local_i_coord, dtype=np.int64),
                        jat=np.array(local_jat, dtype=np.int64),
                        j_coord=np.array(local_j_coord, dtype=np.int64),
                        k=np.array(local_k, dtype=np.int64),
                        value=np.array(local_value, dtype=np.float64))


def _assemble_from_scratch(scratch_dir, n_atoms, n_replicas, keep_scratch):
    """Build the final COO tensor from per-atom scratch files using two passes.

    Pass 1 reads only array metadata to count total non-zeros, allowing a single
    pre-allocated set of arrays (peak memory ~1x final COO size + one chunk).
    Pass 2 fills those arrays slice-by-slice and optionally deletes each file.
    """
    pattern = os.path.join(scratch_dir, 'iat_*.npz')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No scratch files found in {scratch_dir}')

    # Pass 1: count total non-zeros from metadata only (lazy load)
    total_nnz = 0
    for path in files:
        with np.load(path) as f:
            total_nnz += f['value'].shape[0]

    coords = np.empty((5, total_nnz), dtype=np.int64)
    values = np.empty(total_nnz, dtype=np.float64)

    # Pass 2: fill pre-allocated arrays
    offset = 0
    for path in files:
        with np.load(path) as f:
            n = f['value'].shape[0]
            coords[0, offset:offset + n] = f['i_at']
            coords[1, offset:offset + n] = f['i_coord']
            coords[2, offset:offset + n] = f['jat']
            coords[3, offset:offset + n] = f['j_coord']
            coords[4, offset:offset + n] = f['k']
            values[offset:offset + n] = f['value']
        offset += n
        if not keep_scratch:
            os.remove(path)

    if not keep_scratch:
        try:
            os.rmdir(scratch_dir)
        except OSError:
            pass  # not empty — leave it

    shape = (n_atoms, 3, n_replicas * n_atoms, 3, n_replicas * n_atoms * 3)
    phifull = COO(coords, values, shape)
    return phifull.reshape((n_atoms * 3, n_replicas * n_atoms * 3, n_replicas * n_atoms * 3))


def calculate_third(atoms, replicated_atoms, third_order_delta, distance_threshold=None, is_verbose=False,
                    n_threads=1, calculator_factory=None, scratch_dir=None, keep_scratch=False):
    """
    Compute third order force constant matrices by using the central
    difference formula for the approximation.

    Parameters
    ----------
    n_threads : int or None
        Number of parallel worker processes. ``1`` runs serially (default).
        ``None`` uses all available CPUs. Values > 1 launch that many workers
        via ``concurrent.futures.ProcessPoolExecutor``.
    calculator_factory : callable or None
        Zero-argument callable that returns a fresh ASE calculator. When provided,
        the calculator is not pickled — each worker (or the serial loop) calls
        the factory to create its own instance. Required for file-based calculators
        that cannot be pickled. When running in parallel, ensure the factory
        configures a unique working directory per process, e.g.::

            import os
            calculator_factory=lambda: LAMMPS(tmp_dir=f'/tmp/kaldo_{os.getpid()}')

        If None, the calculator must already be attached to replicated_atoms and
        must be picklable when n_threads != 1.
    scratch_dir : str or None
        Path to a directory where per-atom sparse data is saved as `iat_NNNNN.npz`
        files during calculation. Using scratch files keeps peak memory near
        1x the final COO size instead of accumulating all data in Python lists.
        The directory is created if it does not exist. If None (default), results
        are accumulated in memory (original behaviour).
    keep_scratch : bool
        If True, scratch files are preserved after successful assembly. If False
        (default), each file is deleted as it is consumed during assembly, and the
        directory is removed if empty.
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
        if calculator_factory is not None:
            # Strip the calculator so atoms are picklable; workers will call factory instead.
            # ASE Atoms.copy() does not copy the calculator.
            replicated_atoms_workers = replicated_atoms.copy()
        else:
            replicated_atoms_workers = replicated_atoms
        global _calculator_factory_store
        _calculator_factory_store = calculator_factory
        ctx = multiprocessing.get_context('fork')
        if use_scratch:
            atoms_to_compute = [iat for iat in range(n_atoms)
                                 if not os.path.exists(os.path.join(scratch_dir, f'iat_{iat:05d}.npz'))]
            n_resumed = n_atoms - len(atoms_to_compute)
            if n_resumed:
                logging.info(f'Resuming: skipping {n_resumed} already-computed atom(s)')
        else:
            atoms_to_compute = list(range(n_atoms))
        try:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(_compute_iat_forked, iat, atoms, replicated_atoms_workers,
                                   third_order_delta, distance_threshold, is_verbose): iat
                    for iat in atoms_to_compute
                }
                for future in as_completed(futures):
                    iat = futures[future]
                    local_i_at, local_i_coord, local_jat, local_j_coord, local_k, local_value, n_done, n_skipped = future.result()
                    if use_scratch:
                        _save_iat_chunk(scratch_dir, iat, local_i_at, local_i_coord,
                                        local_jat, local_j_coord, local_k, local_value)
                    else:
                        i_at_sparse.extend(local_i_at)
                        i_coord_sparse.extend(local_i_coord)
                        jat_sparse.extend(local_jat)
                        j_coord_sparse.extend(local_j_coord)
                        k_sparse.extend(local_k)
                        value_sparse.extend(local_value)
                    n_forces_done += n_done
                    n_forces_skipped += n_skipped
                    logging.info(f'Completed atom {iat}: '
                                 f'{int((n_forces_done + n_forces_skipped) / n_forces_to_calculate * 100)}% done')
        finally:
            _calculator_factory_store = None
    else:
        if calculator_factory is not None:
            # Serial: create one calculator instance and reuse across all iat iterations.
            replicated_atoms = replicated_atoms.copy()
            replicated_atoms.calc = calculator_factory()
        for iat in range(n_atoms):
            if use_scratch:
                chunk_path = os.path.join(scratch_dir, f'iat_{iat:05d}.npz')
                if os.path.exists(chunk_path):
                    logging.info(f'Atom {iat}: already computed, skipping')
                    continue
            local_i_at, local_i_coord, local_jat, local_j_coord, local_k, local_value, n_done, n_skipped = \
                _compute_iat(iat, atoms, replicated_atoms, third_order_delta, distance_threshold, is_verbose,
                             calculator_factory=None)
            if use_scratch:
                _save_iat_chunk(scratch_dir, iat, local_i_at, local_i_coord,
                                local_jat, local_j_coord, local_k, local_value)
            else:
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

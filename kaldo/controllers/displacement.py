import functools
import glob
import os
import numpy as np
from concurrent.futures import as_completed
from kaldo.helpers.logger import get_logger
from kaldo.parallel import (
    dispatch_with_resume, get_executor, is_parallel, validate_parallel_calculator,
)
from sparse import COO
logging = get_logger()


def get_equivalent_ifc_indices(atoms, supercell, symprec=1e-5, order=2):
    """Determine symmetrically equivalent IFC block indices for 2nd and 3rd order.

    Groups every (i, l_j, j) second-order and (i, l_j, j, l_k, k) third-order
    atom-tuple into equivalence classes under the crystal spacegroup. Two blocks
    belong to the same class when a spacegroup operation maps one to the other;
    their IFC tensors are then related by a Cartesian similarity transform:

        2nd order:  Phi[i,:,l_j,j,:]         = R @ Phi[canonical] @ R.T
        3rd order:  Phi[i,:,l_j,j,:,l_k,k,:] = einsum('ai,bj,ck,ijk', R, R, R, Phi[canonical])

    where (canonical) is the lowest flat-index representative of the class and
    R is the Cartesian rotation stored in the corresponding rot_map.

    Parameters
    ----------
    atoms : ase.Atoms
        Unit cell with valid cell vectors and periodic boundary conditions.
    supercell : tuple of int, length 3
        Supercell repetitions (nx, ny, nz).
    symprec : float, optional
        Symmetry detection tolerance passed to spglib (default 1e-5).
    order : int, optional
        2 or 3 (default 2). Passing 2 skips all third-order work.

    Returns
    -------
    irr_map_2 : ndarray, shape (n_unit, n_replicas, n_unit)
        Flat canonical index for each 2nd-order block.
    rot_map_2 : ndarray, shape (n_unit, n_replicas, n_unit, 3, 3)
        Cartesian rotation R such that Phi2[i,:,l,j,:] = R @ Phi2[canonical] @ R.T.
    irr_map_3 : ndarray or None, shape (n_unit, n_replicas, n_unit, n_replicas, n_unit)
        Flat canonical index for each 3rd-order block (None when order < 3).
    rot_map_3 : ndarray or None, shape (n_unit, n_replicas, n_unit, n_replicas, n_unit, 3, 3)
        Cartesian rotation for each 3rd-order block (None when order < 3).
    """
    try:
        import spglib
    except ImportError as exc:
        raise ImportError("spglib is required. Install with: pip install spglib") from exc

    n_unit = len(atoms)
    nx, ny, nz = supercell
    n_replicas = nx * ny * nz
    supercell_shape = np.array([nx, ny, nz], dtype=int)
    grid = np.array(list(np.ndindex(nx, ny, nz)), dtype=int)

    n_blocks_2 = n_unit * n_replicas * n_unit
    n_blocks_3 = n_unit * n_replicas * n_unit * n_replicas * n_unit if order >= 3 else 0

    s3_i  = n_replicas * n_unit * n_replicas * n_unit
    s3_lj = n_unit * n_replicas * n_unit
    s3_j  = n_replicas * n_unit
    s3_lk = n_unit

    lattice = atoms.cell[:]
    scaled_pos = atoms.get_scaled_positions()
    dataset = spglib.get_symmetry_dataset(
        (lattice, scaled_pos, atoms.numbers), symprec=symprec
    )
    if dataset is None:
        raise ValueError(
            f"spglib could not determine symmetry (symprec={symprec}). "
            "Try adjusting symprec or relaxing the structure."
        )

    rotations_full    = dataset.rotations
    translations_full = dataset.translations

    # Filter to spacegroup operations that the supercell shape can host.
    # An integer rotation R acts on a supercell lattice vector as
    #     g_new = R @ g + col_shift  (then mod-wrapped by the supercell shape)
    # For this to be a bijection on the supercell mesh — the precondition
    # for the equivalence-class union below to be valid — R must permute
    # axes of the supercell only among themselves when their lengths
    # differ. Concretely, on a diagonal supercell (nx, ny, nz) we require
    # R[i, j] == 0 whenever supercell_shape[i] != supercell_shape[j].
    # Cubic-shape supercells (N, N, N) keep all ops. Anisotropic shapes
    # (e.g. slab (N, N, 1)) keep the in-plane subgroup. Non-diagonal
    # supercells are caught by the caller (see calculate_second/third).
    n_ops_full = len(rotations_full)
    keep = np.ones(n_ops_full, dtype=bool)
    for k in range(n_ops_full):
        R = rotations_full[k]
        for i in range(3):
            for j in range(3):
                if R[i, j] != 0 and supercell_shape[i] != supercell_shape[j]:
                    keep[k] = False
                    break
            if not keep[k]:
                break
    rotations    = rotations_full[keep]
    translations = translations_full[keep]
    n_ops = len(rotations)

    logging.info(
        f"Space group: {dataset.international} (#{dataset.number}), "
        f"{n_ops_full} unit-cell ops, "
        f"{n_ops} compatible with supercell shape {tuple(supercell_shape.tolist())}"
    )

    AT = lattice.T
    rotations_cart = np.einsum('ij,kjl,lm->kim', AT, rotations, np.linalg.inv(AT))

    atom_map    = np.empty((n_ops, n_unit), dtype=int)
    cell_shifts = np.zeros((n_ops, n_unit, 3), dtype=int)

    for k in range(n_ops):
        R, t = rotations[k], translations[k]
        imgs    = scaled_pos @ R.T + t[np.newaxis, :]
        shifts  = np.floor(imgs + 0.5 * symprec).astype(int)
        imgs_w  = imgs - shifts
        for i in range(n_unit):
            diffs = imgs_w[i] - scaled_pos
            diffs -= np.round(diffs)
            norms = np.linalg.norm(diffs, axis=1)
            ip = int(np.argmin(norms))
            if norms[ip] > symprec * 10:
                raise RuntimeError(
                    f"Atom image lookup failed for op {k}, atom {i}. "
                    "Increase symprec or check the structure."
                )
            atom_map[k, i]    = ip
            cell_shifts[k, i] = shifts[i]

    parent2 = np.arange(n_blocks_2)
    parent3 = np.arange(n_blocks_3) if order >= 3 else None

    def _find(parent, x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    if order >= 3:
        LJ_flat, LK_flat = np.mgrid[0:n_replicas, 0:n_replicas]
        LJ_flat = LJ_flat.ravel()
        LK_flat = LK_flat.ravel()
        n_pairs = n_replicas * n_replicas

    for k in range(n_ops):
        R_int = rotations[k]
        for i in range(n_unit):
            ip = atom_map[k, i]
            for j in range(n_unit):
                jp = atom_map[k, j]
                col_shift_j = cell_shifts[k, j] - cell_shifts[k, i]
                g_new_j   = grid @ R_int.T + col_shift_j[np.newaxis, :]
                g_new_j_w = g_new_j % supercell_shape[np.newaxis, :]
                l_j_new   = (g_new_j_w[:, 0] * ny * nz
                             + g_new_j_w[:, 1] * nz
                             + g_new_j_w[:, 2])

                base_orig2 = i  * n_replicas * n_unit + j
                base_new2  = ip * n_replicas * n_unit + jp
                for lat in range(n_replicas):
                    f_orig = base_orig2 + lat                * n_unit
                    f_new  = base_new2  + int(l_j_new[lat]) * n_unit
                    r_o, r_n = _find(parent2, f_orig), _find(parent2, f_new)
                    if r_o != r_n:
                        if r_o < r_n:
                            parent2[r_n] = r_o
                        else:
                            parent2[r_o] = r_n

                if order >= 3:
                    for k_atom in range(n_unit):
                        kp = atom_map[k, k_atom]
                        col_shift_k = cell_shifts[k, k_atom] - cell_shifts[k, i]
                        g_new_k   = grid @ R_int.T + col_shift_k[np.newaxis, :]
                        g_new_k_w = g_new_k % supercell_shape[np.newaxis, :]
                        l_k_new   = (g_new_k_w[:, 0] * ny * nz
                                     + g_new_k_w[:, 1] * nz
                                     + g_new_k_w[:, 2])

                        LJ_new = l_j_new[LJ_flat]
                        LK_new = l_k_new[LK_flat]
                        f_orig3 = (i  * s3_i + LJ_flat * s3_lj + j  * s3_j + LK_flat * s3_lk + k_atom)
                        f_new3  = (ip * s3_i + LJ_new  * s3_lj + jp * s3_j + LK_new  * s3_lk + kp)

                        for idx in range(n_pairs):
                            r_o = _find(parent3, int(f_orig3[idx]))
                            r_n = _find(parent3, int(f_new3[idx]))
                            if r_o != r_n:
                                if r_o < r_n:
                                    parent3[r_n] = r_o
                                else:
                                    parent3[r_o] = r_n

    canonical2 = np.array([_find(parent2, f) for f in range(n_blocks_2)])
    irr_map_2  = canonical2.reshape(n_unit, n_replicas, n_unit)
    n_irr2 = len(np.unique(canonical2))
    logging.info(
        f"2nd-order IFC: {n_irr2} irreducible / {n_blocks_2} total blocks "
        f"({100.0 * n_irr2 / n_blocks_2:.1f}%)"
    )

    if order >= 3:
        canonical3 = np.array([_find(parent3, f) for f in range(n_blocks_3)])
        irr_map_3  = canonical3.reshape(n_unit, n_replicas, n_unit, n_replicas, n_unit)
        n_irr3 = len(np.unique(canonical3))
        logging.info(
            f"3rd-order IFC: {n_irr3} irreducible / {n_blocks_3} total blocks "
            f"({100.0 * n_irr3 / n_blocks_3:.1f}%)"
        )
    else:
        canonical3 = None
        irr_map_3  = None

    rot_map_2 = np.broadcast_to(np.eye(3), (n_unit, n_replicas, n_unit, 3, 3)).copy()
    if order >= 3:
        rot_map_3 = np.broadcast_to(
            np.eye(3), (n_unit, n_replicas, n_unit, n_replicas, n_unit, 3, 3)
        ).copy()
    else:
        rot_map_3 = None

    l_arr = np.arange(n_replicas)

    for k in range(n_ops):
        R_int = rotations[k]
        R_inv = rotations_cart[k].T
        for i in range(n_unit):
            ip = atom_map[k, i]
            for j in range(n_unit):
                jp = atom_map[k, j]
                col_shift_j = cell_shifts[k, j] - cell_shifts[k, i]
                g_new_j   = grid @ R_int.T + col_shift_j[np.newaxis, :]
                g_new_j_w = g_new_j % supercell_shape[np.newaxis, :]
                l_j_new   = (g_new_j_w[:, 0] * ny * nz
                             + g_new_j_w[:, 1] * nz
                             + g_new_j_w[:, 2])

                f_orig2 = i  * n_replicas * n_unit + l_arr * n_unit + j
                f_new2  = ip * n_replicas * n_unit + l_j_new * n_unit + jp
                mask2   = (canonical2[f_orig2] == f_new2) & (f_orig2 != f_new2)
                rot_map_2[i, l_arr[mask2], j] = R_inv

                if order >= 3:
                    for k_atom in range(n_unit):
                        kp = atom_map[k, k_atom]
                        col_shift_k = cell_shifts[k, k_atom] - cell_shifts[k, i]
                        g_new_k   = grid @ R_int.T + col_shift_k[np.newaxis, :]
                        g_new_k_w = g_new_k % supercell_shape[np.newaxis, :]
                        l_k_new   = (g_new_k_w[:, 0] * ny * nz
                                     + g_new_k_w[:, 1] * nz
                                     + g_new_k_w[:, 2])

                        LJ_new = l_j_new[LJ_flat]
                        LK_new = l_k_new[LK_flat]
                        f_orig3 = (i  * s3_i + LJ_flat * s3_lj + j  * s3_j + LK_flat * s3_lk + k_atom)
                        f_new3  = (ip * s3_i + LJ_new  * s3_lj + jp * s3_j + LK_new  * s3_lk + kp)
                        mask3   = (canonical3[f_orig3] == f_new3) & (f_orig3 != f_new3)
                        if np.any(mask3):
                            rot_map_3[i, LJ_flat[mask3], j, LK_flat[mask3], k_atom] = R_inv

    return irr_map_2, rot_map_2, irr_map_3, rot_map_3


def _diagonal_supercell_or_raise(atoms, replicated_atoms, tol=1e-6):
    """Recover the diagonal supercell tuple from atoms / replicated_atoms.

    Raises NotImplementedError when the supercell expansion is not diagonal.
    The realspace symmetry path indexes lattice vectors as
    ``ix * ny * nz + iy * nz + iz`` (C-order over a diagonal mesh) and
    has no representation for a sheared supercell.
    """
    M = replicated_atoms.cell @ np.linalg.inv(atoms.cell)
    M_int = np.round(M).astype(int)
    if not np.allclose(M, M_int, atol=tol):
        raise NotImplementedError(
            "use_symmetry=True requires an integer supercell expansion. "
            f"Got non-integer expansion matrix:\n{M}"
        )
    off_diag = M_int - np.diag(np.diag(M_int))
    if np.any(off_diag != 0):
        raise NotImplementedError(
            "use_symmetry=True only supports diagonal supercell expansions. "
            f"Got non-diagonal expansion matrix:\n{M_int}"
        )
    return tuple(np.diag(M_int))


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


def calculate_second(atoms, replicated_atoms, second_order_delta, is_verbose=False, n_workers=1, calculator=None,
                     scratch_dir=None, keep_scratch=False, use_symmetry=False, symprec=1e-5):
    """
    Core method to compute second order force constant matrices
    Approximate the second order force constant matrices
    using central difference formula
    """
    if n_workers is not None and n_workers < 1:
        raise ValueError(f"n_workers must be >= 1 or None, got {n_workers}")
    if is_parallel(n_workers):
        if calculator is not None:
            validate_parallel_calculator(calculator, method='calculate_second')
        elif getattr(replicated_atoms, 'calc', None) is not None:
            validate_parallel_calculator(replicated_atoms.calc, method='calculate_second')
    if use_symmetry and scratch_dir is not None:
        raise ValueError(
            "use_symmetry=True is not compatible with scratch_dir. "
            "Set scratch_dir=None when using symmetry reduction."
        )

    logging.info('Calculating second order potential derivatives, ' + 'finite difference displacement: %.3e angstrom'%second_order_delta)
    n_unit_cell_atoms = len(atoms.numbers)
    n_replicated_atoms = len(replicated_atoms.numbers)
    n_atoms = n_unit_cell_atoms
    n_replicas = int(n_replicated_atoms / n_unit_cell_atoms)
    use_scratch = scratch_dir is not None
    if not use_scratch:
        second = np.zeros((n_atoms, 3, n_replicated_atoms * 3))

    if use_symmetry:
        supercell = _diagonal_supercell_or_raise(atoms, replicated_atoms)
        irr_map_2, rot_map_2, _, _ = get_equivalent_ifc_indices(
            atoms, supercell, symprec=symprec, order=2
        )
        i0_all, l0_all, j0_all = np.unravel_index(
            irr_map_2.ravel(), (n_atoms, n_replicas, n_atoms)
        )
        atoms_to_compute = np.unique(i0_all).tolist()
        logging.info(
            f'Symmetry reduction: displacing {len(atoms_to_compute)}/{n_atoms} atoms'
        )
    else:
        irr_map_2 = rot_map_2 = None
        i0_all = l0_all = j0_all = None
        atoms_to_compute = range(n_atoms)

    worker_fn = functools.partial(
        _compute_iat_second,
        replicated_atoms=replicated_atoms,
        second_order_delta=second_order_delta,
        calculator=calculator,
        scratch_dir=scratch_dir,
    )

    for atom_id, result in dispatch_with_resume(
        atoms_to_compute, worker_fn,
        n_workers=n_workers,
        output_dir=scratch_dir,
        sentinel_prefix="iat_",
        log_progress=is_verbose,
    ):
        if not use_scratch:
            _, second_per_atom = result
            second[atom_id] = second_per_atom

    if use_scratch:
        second = _assemble_from_scratch_second(scratch_dir, n_atoms, n_replicated_atoms, keep_scratch)

    second = second.reshape((1, n_unit_cell_atoms, 3, n_replicas, n_unit_cell_atoms, 3))
    second = second / (2. * second_order_delta)

    if use_symmetry:
        # Reconstruct non-irreducible blocks: Phi[i,:,l,j,:] = R @ Phi[canonical] @ R.T
        phi = second[0]                                              # (n_unit, 3, n_repl, n_unit, 3)
        Phi_canon = phi[i0_all, :, l0_all, j0_all, :]               # (n_blocks_2, 3, 3)
        R = rot_map_2.reshape(-1, 3, 3)                              # (n_blocks_2, 3, 3)
        second_flat = R @ Phi_canon @ R.transpose(0, 2, 1)           # (n_blocks_2, 3, 3)
        second = (second_flat
                  .reshape(n_unit_cell_atoms, n_replicas, n_unit_cell_atoms, 3, 3)
                  .transpose(0, 3, 1, 2, 4)
                  [np.newaxis])

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
        # Sentinel is written by dispatch_with_resume after this returns.
        np.save(os.path.join(scratch_dir, f'iat_{atom_id:05d}.npy'), second_per_atom)
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
                    jat_flush_every=50, use_symmetry=False, symprec=1e-5):
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
        via ``concurrent.futures.ProcessPoolExecutor``. Each worker is capped
        to one OpenMP / MKL / OpenBLAS thread so calculators with internal
        multithreading don't oversubscribe; override by setting
        ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` in the environment.
    calculator : callable or ASE Calculator instance or None
        Either an ASE calculator class or an already-constructed instance.
        When running in parallel (``n_workers > 1``), pass a class so each
        worker can create its own instance::

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
    if n_workers is not None and n_workers < 1:
        raise ValueError(f"n_workers must be >= 1 or None, got {n_workers}")
    if is_parallel(n_workers):
        if calculator is not None:
            validate_parallel_calculator(calculator, method='calculate_third')
        elif getattr(replicated_atoms, 'calc', None) is not None:
            validate_parallel_calculator(replicated_atoms.calc, method='calculate_third')
    if use_symmetry and scratch_dir is not None:
        raise ValueError(
            "use_symmetry=True is not compatible with scratch_dir. "
            "Set scratch_dir=None when using symmetry reduction."
        )

    logging.info('Calculating third order potential derivatives, ' + 'finite difference displacement: %.3e angstrom'%third_order_delta)
    n_atoms = len(atoms.numbers)
    n_replicas = int(replicated_atoms.positions.shape[0] / n_atoms)
    use_scratch = scratch_dir is not None
    if not use_scratch:
        i_at_sparse = []
        i_coord_sparse = []
        jat_sparse = []
        j_coord_sparse = []
        k_sparse = []
        value_sparse = []
    n_forces_to_calculate = n_replicas * (n_atoms * 3) ** 2
    n_forces_done = 0
    n_forces_skipped = 0

    # Build per-iat allowed_jat mapping from irreducible pair set when using symmetry.
    # When use_symmetry is False, allowed_jat_per_iat stays empty and the worker
    # computes every (iat, jat) pair (resume via dispatch_with_resume sentinels).
    if use_symmetry:
        supercell = _diagonal_supercell_or_raise(atoms, replicated_atoms)
        _, _, irr_map_3, rot_map_3 = get_equivalent_ifc_indices(
            atoms, supercell, symprec=symprec, order=3
        )
        i0_all, lj0_all, j0_all, _, _ = np.unravel_index(
            irr_map_3.ravel(), (n_atoms, n_replicas, n_atoms, n_replicas, n_atoms)
        )
        irr_pairs = np.unique(
            np.stack([i0_all, lj0_all * n_atoms + j0_all], axis=1), axis=0
        )
        allowed_jat_per_iat = {}
        for i0, jat0 in irr_pairs:
            allowed_jat_per_iat.setdefault(int(i0), set()).add(int(jat0))
        atoms_to_compute = sorted(allowed_jat_per_iat.keys())
        logging.info(
            f'Symmetry reduction: {len(irr_pairs)} irreducible / '
            f'{n_atoms * n_replicas * n_atoms} total atom pairs'
        )
    else:
        irr_map_3 = rot_map_3 = None
        allowed_jat_per_iat = {}
        atoms_to_compute = range(n_atoms)

    worker_fn = functools.partial(
        _compute_iat_third,
        atoms=atoms,
        replicated_atoms=replicated_atoms,
        third_order_delta=third_order_delta,
        distance_threshold=distance_threshold,
        is_verbose=is_verbose,
        calculator=calculator,
        scratch_dir=scratch_dir,
        jat_flush_every=jat_flush_every,
        allowed_jat_per_iat=allowed_jat_per_iat,
    )

    for iat, result in dispatch_with_resume(
        atoms_to_compute, worker_fn,
        n_workers=n_workers,
        output_dir=scratch_dir,
        sentinel_prefix="iat_",
        log_progress=False,
    ):
        local_i_at, local_i_coord, local_jat, local_j_coord, local_k, local_value, n_done, n_skipped = result
        if not use_scratch:
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
    logging.info('total forces to calculate third : ' + str(n_forces_to_calculate))
    logging.info('forces calculated : ' + str(n_forces_done))
    logging.info('forces skipped (outside distance threshold) : ' + str(n_forces_skipped))
    if use_scratch:
        return _assemble_from_scratch_third(scratch_dir, n_atoms, n_replicas, keep_scratch)

    if use_symmetry:
        # Expand irreducible sparse entries to the full tensor via symmetry.
        n_rep_atoms = n_replicas * n_atoms

        phi_irr = {}
        for iat, ic, jat, jc, k, v in zip(i_at_sparse, i_coord_sparse, jat_sparse,
                                            j_coord_sparse, k_sparse, value_sparse):
            key = (int(iat), int(jat))
            if key not in phi_irr:
                phi_irr[key] = np.zeros((3, 3, n_rep_atoms * 3))
            phi_irr[key][int(ic), int(jc), int(k)] = float(v)

        irr_flat = irr_map_3.ravel()
        rot_flat = rot_map_3.reshape(-1, 3, 3)
        A_idx = np.arange(3, dtype=np.int32)
        B_idx = np.arange(3, dtype=np.int32)
        C_idx = np.arange(3, dtype=np.int32)

        i_at_sparse    = []
        i_coord_sparse = []
        jat_sparse     = []
        j_coord_sparse = []
        k_sparse       = []
        value_sparse   = []

        for canon_flat in np.unique(irr_flat):
            i0, lj0, j0, lk0, k0 = np.unravel_index(
                int(canon_flat), (n_atoms, n_replicas, n_atoms, n_replicas, n_atoms)
            )
            jat0 = lj0 * n_atoms + j0
            if (i0, jat0) not in phi_irr:
                continue

            kat0 = (lk0 * n_atoms + k0) * 3
            phi_333 = phi_irr[(i0, jat0)][:, :, kat0:kat0 + 3].copy()

            equiv_idxs = np.where(irr_flat == canon_flat)[0]
            i_all, lj_all, j_all, lk_all, k_all = np.unravel_index(
                equiv_idxs, (n_atoms, n_replicas, n_atoms, n_replicas, n_atoms)
            )
            n_eq = len(equiv_idxs)

            R_eq = rot_flat[equiv_idxs]
            T1   = (R_eq @ phi_333.reshape(3, 9)).reshape(n_eq, 3, 3, 3)
            T2   = (R_eq @ T1.transpose(0, 2, 1, 3).reshape(n_eq, 3, 9)
                    ).reshape(n_eq, 3, 3, 3).transpose(0, 2, 1, 3)
            T3   = (R_eq @ T2.transpose(0, 3, 1, 2).reshape(n_eq, 3, 9)
                    ).reshape(n_eq, 3, 3, 3).transpose(0, 2, 3, 1)

            jat_eq = (lj_all * n_atoms + j_all)
            kat_eq = (lk_all * n_atoms + k_all) * 3

            c_iat  = i_all[:, None, None, None]
            c_ic   = A_idx[None, :, None, None]
            c_jat  = jat_eq[:, None, None, None]
            c_jc   = B_idx[None, None, :, None]
            c_k    = kat_eq[:, None, None, None] + C_idx[None, None, None, :]

            # Drop near-zero entries from the COO. FD output is rarely exactly
            # zero; using `!= 0.0` would keep all the FD noise (~1e-7 for
            # delta=1e-5) and inflate the sparse tensor. atol=1e-12 is well
            # below physics-relevant scales (IFCs are 0.01-10).
            mask = ~np.isclose(T3, 0.0, atol=1e-12)
            bc   = (n_eq, 3, 3, 3)
            i_at_sparse.extend(np.broadcast_to(c_iat, bc)[mask].tolist())
            i_coord_sparse.extend(np.broadcast_to(c_ic,  bc)[mask].tolist())
            jat_sparse.extend(np.broadcast_to(c_jat, bc)[mask].tolist())
            j_coord_sparse.extend(np.broadcast_to(c_jc,  bc)[mask].tolist())
            k_sparse.extend(np.broadcast_to(c_k,    bc)[mask].tolist())
            value_sparse.extend(T3[mask].tolist())

    coords = np.array([i_at_sparse, i_coord_sparse, jat_sparse, j_coord_sparse, k_sparse])
    shape = (n_atoms, 3, n_replicas * n_atoms, 3, n_replicas * n_atoms * 3)
    phifull = COO(coords, np.array(value_sparse), shape)
    phifull = phifull.reshape((n_atoms * 3, n_replicas * n_atoms * 3, n_replicas * n_atoms * 3))
    return phifull
    

def _compute_iat_third(iat, atoms, replicated_atoms, third_order_delta, distance_threshold, is_verbose,
                 calculator=None, scratch_dir=None, jat_flush_every=50, allowed_jat_per_iat=None):
    """Compute all third-order force constant terms for a single unit cell atom index.

    Parameters
    ----------
    calculator : callable or None
        If provided, called as ``calculator()`` to create a fresh ASE calculator
        instance that is attached to a copy of replicated_atoms.
        If None, replicated_atoms must already have a calculator attached.
    scratch_dir : str or None
        If provided, results are written directly to ``scratch_dir`` as a series of
        ``iat_NNNNN_chunk_MMMM.npz`` files. A ``iat_NNNNN.done`` sentinel is written
        on completion. Empty lists are returned in place of data to keep peak memory
        proportional to one flush window rather than the full atom.
    jat_flush_every : int
        Number of jat iterations to accumulate (as compact numpy arrays) before
        flushing to disk. Only used when ``scratch_dir`` is set.
    allowed_jat_per_iat : dict[int, set[int]] or None
        When provided, only jat indices in ``allowed_jat_per_iat[iat]`` are
        computed; all others are skipped. Used by ``calculate_third`` when
        ``use_symmetry=True`` to restrict computation to irreducible atom
        pairs. The dict is bound via ``functools.partial`` once and the
        worker looks up its own ``iat`` entry.
    """
    if calculator is not None:
        replicated_atoms = replicated_atoms.copy()
        replicated_atoms.calc = calculator() if callable(calculator) else calculator
    n_atoms = len(atoms.numbers)
    n_replicas = int(replicated_atoms.positions.shape[0] / n_atoms)
    n_done = 0
    n_skipped = 0

    allowed_jat = allowed_jat_per_iat.get(iat) if allowed_jat_per_iat else None

    use_scratch = scratch_dir is not None
    chunk_id = 0
    jat_count_in_chunk = 0
    chunk_coords = []
    chunk_values = []

    for jat in range(n_replicas * n_atoms):
        is_computing = True
        if allowed_jat is not None and jat not in allowed_jat:
            is_computing = False
            n_skipped += 9
        if is_computing and distance_threshold is not None:
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
        # Sentinel is written by dispatch_with_resume after this returns.
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

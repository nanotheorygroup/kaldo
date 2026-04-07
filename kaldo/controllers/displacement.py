import functools
import glob
import os
import numpy as np
from concurrent.futures import as_completed
from kaldo.helpers.logger import get_logger
from kaldo.parallel import get_executor
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

    n_unit = len(atoms) # number of atoms in unit cell
    nx, ny, nz = supercell # supercell dimensions (diagonal)
    n_replicas = nx * ny * nz # number of replicas
    supercell_shape = np.array([nx, ny, nz], dtype=int)

    # Replica grid in C-order: grid[l] = (lx, ly, lz)
    # l = lx*ny*nz + ly*nz + lz
    grid = np.array(list(np.ndindex(nx, ny, nz)), dtype=int)  # C-order replica grid (n_replicas, 3)

    n_blocks_2 = n_unit * n_replicas * n_unit # Number of 3x3 Cartesian tensors (second order)
    n_blocks_3 = n_unit * n_replicas * n_unit * n_replicas * n_unit if order >= 3 else 0 # Number of 3x3x3 Cartesian tensors (third order)

    # Strides for 3rd-order flat index
    # shape: (n_unit, n_replicas, n_unit, n_replicas, n_unit)
    # Thus, the 'C-order cell index' (l) is: l = i*n_unit^2*n_replicas^2 + lj*n_unit^2*n_replicas + j*n_unit*n_replicas + lk*n_unit + k
    s3_i  = n_replicas * n_unit * n_replicas * n_unit
    s3_lj = n_unit * n_replicas * n_unit
    s3_j  = n_replicas * n_unit
    s3_lk = n_unit

    # ------------------------------------------------------------------ #
    # 1. Retrieve the spacegroup symmetry dataset
    # ------------------------------------------------------------------ #
    lattice = atoms.cell[:] # lattice matrix
    scaled_pos = atoms.get_scaled_positions() # fractional positions
    dataset = spglib.get_symmetry_dataset(
        (lattice, scaled_pos, atoms.numbers), symprec=symprec
    )
    if dataset is None:
        raise ValueError(
            f"spglib could not determine symmetry (symprec={symprec}). "
            "Try adjusting symprec or relaxing the structure."
        )

    rotations    = dataset.rotations      # (n_ops, 3, 3) – fractional rotations, integer
    translations = dataset.translations   # (n_ops, 3)   – fractional translations, float
    n_ops = len(rotations) # number of symmetry operations to check
    logging.info(
        f"Space group: {dataset.international} (#{dataset.number}), "
        f"{n_ops} symmetry operations"
    )

    # Convert rotations to Cartesian: Given symmetry operation {R_{p}, t_{p}} in Cartesian space and lattice atomic positions x_{ni}, x_{n'i'}
    # x_{n'i'} = R_{p}x_{ni} + t_{p} = R_{p}(A^T)(s_{i} + n) + (A^T)m_{p} = A^T(s_{i'} + n')
    # => s_{i'} + n' = ((A^T)^{-1}R_{p}A^T)(s_{i} + n) + m_{p} => R(frac) = (A^T)^{-1}R(cart)A^T => R(cart) = (A^T)R(frac)(A^T)^{-1}
    AT = lattice.T # ASE lattice is the transpose
    rotations_cart = np.einsum('ij,kjl,lm->kim', AT, rotations, np.linalg.inv(AT))

    # ------------------------------------------------------------------ #
    # 2. Build per-operation unit-cell atom mapping and integer cell shifts
    #
    #    For op k, atom i:  R_k @ s_i + t_k  =  s_{i'} + n_{k,i}
    #    where s_{i'} in [0,1)^3 and n_{k,i} in {0, 1, 2, ..., Nx/y/z}^3 is the integer cell shift.
    # ------------------------------------------------------------------ #
    atom_map    = np.empty((n_ops, n_unit), dtype=int) # mapping from i s-> i'
    cell_shifts = np.zeros((n_ops, n_unit, 3), dtype=int) # mapping from n s-> n' where "s->" is the symmetry mapping taking (n,i) s-> (n',i')

    # loop over operations to find (n,i) s-> (n',i')
    for k in range(n_ops):
        R, t = rotations[k], translations[k] # rotation+translation k
        imgs    = scaled_pos @ R.T + t[np.newaxis, :] # Symmetry operated scaled positions
        shifts  = np.floor(imgs + 0.5 * symprec).astype(int)
        imgs_w  = imgs - shifts  # wrapped into [0, 1)
        for i in range(n_unit):
            diffs = imgs_w[i] - scaled_pos
            diffs -= np.round(diffs)                # wraps atoms to first unit cell to find i s-> i'
            norms = np.linalg.norm(diffs, axis=1)
            ip = int(np.argmin(norms)) # smallest norm is where the mapping maps i to i'
            if norms[ip] > symprec * 10: # check the norm is basically zero (ideally within numerics)
                raise RuntimeError(
                    f"Atom image lookup failed for op {k}, atom {i}. "
                    "Increase symprec or check the structure."
                )
            atom_map[k, i]    = ip # array that maps operation k and atom i to new atom i'
            cell_shifts[k, i] = shifts[i] # array that holds the lattice vector shifts of atomic positions for operation k and atom i

    # ------------------------------------------------------------------ #
    # 3. Group IFC blocks into equivalence classes with Union-Find
    #
    #    2nd order — op k maps (i, l, j) -> (i', l'', j'):
    #      i'      = atom_map[k, i]
    #      j'      = atom_map[k, j]
    #      g_{l''} = R_k @ g_l + cell_shifts[k,j] - cell_shifts[k,i]
    #      l''     = grid_index( g_{l''} % supercell_shape )
    #
    #    3rd order — op k maps (i, l_j, j, l_k, k_atom) -> (i', l_j'', j', l_k'', k'):
    #      Same atom mapping; each replica index is transformed independently
    #      with its own col_shift relative to atom i.
    #
    #    The cell_shifts[k,i] correction ensures the first atom always sits
    #    in cell 0, matching kALDo's IFC convention.
    # ------------------------------------------------------------------ #
    parent2 = np.arange(n_blocks_2) # index of 3x3 Cartesian tensors (block)
    parent3 = np.arange(n_blocks_3) if order >= 3 else None # index of 3x3x3 Cartesian tensors (blocks)

    def _find(parent, x):
        # Simple find algorithm. Iteratively constructs set of all "roots" (irreducible atoms)
        # from disjoint forest of trees such that {i1, i2, ..., is} -> {i_irreducible}; {j1, j2, ..., jq} -> {j_irreducible}
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    # Precompute all (l_j, l_k) replica-pair flat indices for vectorised
    # 3rd-order flat-index computation (only needed when order >= 3). C-order meshgrid.
    if order >= 3:
        LJ_flat, LK_flat = np.mgrid[0:n_replicas, 0:n_replicas]
        LJ_flat = LJ_flat.ravel()   # (n_replicas^2,)
        LK_flat = LK_flat.ravel()   # (n_replicas^2,)
        n_pairs = n_replicas * n_replicas # number of pairwise cell interactions (lj,lk)

    for k in range(n_ops):
        R_int = rotations[k] # fractional rotation
        for i in range(n_unit):
            ip = atom_map[k, i] # i' such that operation k maps i s-> i'
            for j in range(n_unit):
                jp = atom_map[k, j] # j' such that operation k maps j s-> j'
                col_shift_j = cell_shifts[k, j] - cell_shifts[k, i] # difference in lattice shift
                g_new_j   = grid @ R_int.T + col_shift_j[np.newaxis, :] # symmetry map on fractional replicas
                g_new_j_w = g_new_j % supercell_shape[np.newaxis, :] # operated supercell wrapped to fractional grid
                l_j_new   = (g_new_j_w[:, 0] * ny * nz
                             + g_new_j_w[:, 1] * nz
                             + g_new_j_w[:, 2])   # Index mapping from lj s-> lj' (n_replicas,)

                # -- 2nd order --
                base_orig2 = i  * n_replicas * n_unit + j # C-ordering index relating (0, i, j) -> flattened (I)
                base_new2  = ip * n_replicas * n_unit + jp # C-ordering index relating (0, i', j') -> flattened (I')
                for l in range(n_replicas):
                    f_orig = base_orig2 + l                  * n_unit # replica index added to get full I: (0, i, l, j) -> I
                    f_new  = base_new2  + int(l_j_new[l])   * n_unit # replica index added to get full I': (0, i', l', j') -> I'
                    r_o, r_n = _find(parent2, f_orig), _find(parent2, f_new) # trims index tree to find reducible atoms
                    if r_o != r_n: # if statement uses the smallest index as the irreducible atom in map
                        if r_o < r_n: parent2[r_n] = r_o
                        else:         parent2[r_o] = r_n

                # -- 3rd order (skipped when order < 3) --
                # Process is the exact same but between (0,i) and (lk, k) instead of (lj, j)
                if order >= 3:
                    for k_atom in range(n_unit):
                        kp = atom_map[k, k_atom]
                        col_shift_k = cell_shifts[k, k_atom] - cell_shifts[k, i]
                        g_new_k   = grid @ R_int.T + col_shift_k[np.newaxis, :]
                        g_new_k_w = g_new_k % supercell_shape[np.newaxis, :]
                        l_k_new   = (g_new_k_w[:, 0] * ny * nz
                                     + g_new_k_w[:, 1] * nz
                                     + g_new_k_w[:, 2])   # (n_replicas,)

                        # Vectorised flat indices over all (l_j, l_k) pairs. Needs to be taken into account to relate lj, j, lk, and k
                        LJ_new = l_j_new[LJ_flat]   # (n_pairs,)
                        LK_new = l_k_new[LK_flat]   # (n_pairs,)
                        f_orig3 = (i  * s3_i + LJ_flat * s3_lj + j  * s3_j + LK_flat * s3_lk + k_atom)
                        f_new3  = (ip * s3_i + LJ_new  * s3_lj + jp * s3_j + LK_new  * s3_lk + kp)

                        for idx in range(n_pairs):
                            r_o = _find(parent3, int(f_orig3[idx]))
                            r_n = _find(parent3, int(f_new3[idx]))
                            if r_o != r_n:
                                if r_o < r_n: parent3[r_n] = r_o
                                else:         parent3[r_o] = r_n

    # ------------------------------------------------------------------ #
    # 4. Build the canonical (irreducible) index maps
    # ------------------------------------------------------------------ #
    canonical2 = np.array([_find(parent2, f) for f in range(n_blocks_2)]) # get irreducible atomic sites by symmetry for each atom index
    irr_map_2  = canonical2.reshape(n_unit, n_replicas, n_unit) # reshaped so that irr_map2[i, l, j] = I' <-> (i', l', j')

    n_irr2 = len(np.unique(canonical2)) # number of unique blocks by symmetry
    logging.info(
        f"2nd-order IFC: {n_irr2} irreducible / {n_blocks_2} total blocks "
        f"({100.0 * n_irr2 / n_blocks_2:.1f}%)"
    )

    # same process but for third order
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

    # ------------------------------------------------------------------ #
    # 5. Find the Cartesian rotation relating each block to its canonical
    #
    #    The f_orig != f_new guard prevents site-symmetry ops that map a
    #    canonical block to itself from overwriting its identity rotation.
    # ------------------------------------------------------------------ #
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
        R_inv = rotations_cart[k].T   # orthogonal: R^{-1} = R^T
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

                # -- 2nd order: vectorised over l --
                f_orig2 = i  * n_replicas * n_unit + l_arr * n_unit + j
                f_new2  = ip * n_replicas * n_unit + l_j_new * n_unit + jp
                mask2   = (canonical2[f_orig2] == f_new2) & (f_orig2 != f_new2) # identity maps are not touched
                rot_map_2[i, l_arr[mask2], j] = R_inv # gets the rotation matrix necessary for mapping (i, l, j) to (i', l', j')

                # -- 3rd order: vectorised over (l_j, l_k) pairs (skipped when order < 3) --
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

    return irr_map_2, rot_map_2, irr_map_3, rot_map_3 # returns irreducible maps and rotational maps


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


def _validate_calculator(calculator):
    """Raise TypeError if calculator is not a callable or ASE Calculator."""
    if calculator is None:
        return
    if callable(calculator):
        return
    try:
        from ase.calculators.calculator import Calculator
        if isinstance(calculator, Calculator):
            return
    except ImportError:
        pass
    raise TypeError(
        f"calculator must be a callable (e.g. EMT) or ASE Calculator instance, "
        f"got {type(calculator).__name__}"
    )


def calculate_second(atoms, replicated_atoms, second_order_delta, is_verbose=False, n_workers=1, calculator=None,
                     scratch_dir=None, keep_scratch=False, use_symmetry=False, supercell=None, symprec=1e-5):
    """
    Core method to compute second order force constant matrices
    Approximate the second order force constant matrices
    using central difference formula
    """
    if n_workers is not None and n_workers < 1:
        raise ValueError(f"n_workers must be >= 1 or None, got {n_workers}")
    _validate_calculator(calculator)
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

    if use_symmetry and supercell is not None:
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

    use_parallel = n_workers is None or n_workers > 1
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

    if use_symmetry and supercell is not None:
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
                    jat_flush_every=50, use_symmetry=False, supercell=None, symprec=1e-5):
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
    use_symmetry : bool, optional
        Reduce atom-pair displacements using crystal spacegroup symmetry
        (default False). Only irreducible (iat, jat) pairs are computed;
        equivalent pairs are reconstructed via rotation. Incompatible with
        ``scratch_dir``.
    supercell : tuple of int or None
        Required when ``use_symmetry=True``.
    symprec : float, optional
        Symmetry tolerance passed to spglib (default 1e-5).
    """
    if n_workers is not None and n_workers < 1:
        raise ValueError(f"n_workers must be >= 1 or None, got {n_workers}")
    _validate_calculator(calculator)
    if use_symmetry and scratch_dir is not None:
        raise ValueError(
            "use_symmetry=True is not compatible with scratch_dir. "
            "Set scratch_dir=None when using symmetry reduction."
        )

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

    # Build per-iat allowed_jat mapping from irreducible pair set when using symmetry
    if use_symmetry and supercell is not None:
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
        # Determine which atoms need computing (resume support via scratch sentinels)
        if use_scratch:
            atoms_to_compute = [iat for iat in range(n_atoms)
                                 if not os.path.exists(os.path.join(scratch_dir, f'iat_{iat:05d}.done'))]
            n_resumed = n_atoms - len(atoms_to_compute)
            if n_resumed:
                logging.info(f'Resuming: skipping {n_resumed} already-computed atom(s)')
        else:
            atoms_to_compute = list(range(n_atoms))

    use_parallel = n_workers is None or n_workers > 1

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
                           third_order_delta, distance_threshold, is_verbose,
                           allowed_jat=allowed_jat_per_iat.get(iat)): iat
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

    if use_symmetry and supercell is not None:
        # Expand irreducible sparse entries to the full tensor via symmetry.
        # For each canonical equivalence class, slice the (3,3,3) sub-tensor from the
        # irreducible data, apply the 3-mode batched rotation, and emit COO entries
        # for every block in the class.
        n_rep_atoms = n_replicas * n_atoms

        # Group irreducible entries by (iat, jat) -> dense (3, 3, n_rep_atoms*3) block
        phi_irr = {}
        for iat, ic, jat, jc, k, v in zip(i_at_sparse, i_coord_sparse, jat_sparse,
                                            j_coord_sparse, k_sparse, value_sparse):
            key = (int(iat), int(jat))
            if key not in phi_irr:
                phi_irr[key] = np.zeros((3, 3, n_rep_atoms * 3))
            phi_irr[key][int(ic), int(jc), int(k)] = float(v)

        irr_flat = irr_map_3.ravel()               # (n_blocks_3,)
        rot_flat = rot_map_3.reshape(-1, 3, 3)      # (n_blocks_3, 3, 3)

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
                continue  # pair skipped by distance threshold

            # Extract the (3, 3, 3) sub-tensor for this canonical class
            kat0 = (lk0 * n_atoms + k0) * 3
            phi_333 = phi_irr[(i0, jat0)][:, :, kat0:kat0 + 3].copy()  # (3, 3, 3)

            # Find all blocks belonging to this canonical class
            equiv_idxs = np.where(irr_flat == canon_flat)[0]
            i_all, lj_all, j_all, lk_all, k_all = np.unravel_index(
                equiv_idxs, (n_atoms, n_replicas, n_atoms, n_replicas, n_atoms)
            )
            n_eq = len(equiv_idxs)

            # 3-mode batched rotation: T3[f,A,B,C] = R[f,A,a] R[f,B,b] R[f,C,c] phi[a,b,c]
            R_eq = rot_flat[equiv_idxs]                                       # (n_eq, 3, 3)
            T1   = (R_eq @ phi_333.reshape(3, 9)).reshape(n_eq, 3, 3, 3)
            T2   = (R_eq @ T1.transpose(0, 2, 1, 3).reshape(n_eq, 3, 9)
                    ).reshape(n_eq, 3, 3, 3).transpose(0, 2, 1, 3)
            T3   = (R_eq @ T2.transpose(0, 3, 1, 2).reshape(n_eq, 3, 9)
                    ).reshape(n_eq, 3, 3, 3).transpose(0, 2, 3, 1)           # (n_eq, A, B, C)

            jat_eq = (lj_all * n_atoms + j_all)                               # (n_eq,)
            kat_eq = (lk_all * n_atoms + k_all) * 3                           # (n_eq,) k-axis start

            # Broadcast coordinate arrays over (n_eq, 3, 3, 3)
            c_iat  = i_all[:, None, None, None]
            c_ic   = A_idx[None, :, None, None]
            c_jat  = jat_eq[:, None, None, None]
            c_jc   = B_idx[None, None, :, None]
            c_k    = kat_eq[:, None, None, None] + C_idx[None, None, None, :]

            mask = T3 != 0.0
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
                 calculator=None, scratch_dir=None, jat_flush_every=50, allowed_jat=None):
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
    allowed_jat : set or None
        When provided, only jat indices in this set are computed; all others are
        skipped. Used by ``calculate_third`` when ``use_symmetry=True`` to restrict
        computation to irreducible atom pairs.
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
                logging.info(f'calculating forces on atoms: {iat}, {jat}')
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

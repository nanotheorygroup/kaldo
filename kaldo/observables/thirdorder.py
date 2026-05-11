from kaldo.observables.forceconstant import ForceConstant
from ase import Atoms
import os
import ase.io
import numpy as np
from scipy.sparse import load_npz, save_npz
from sparse import COO
from kaldo.interfaces.eskm_io import import_from_files
from kaldo.interfaces.tdep_io import parse_tdep_third_forceconstant
import kaldo.interfaces.shengbte_io as shengbte_io
import ase.units as units
from kaldo.controllers.displacement import calculate_third
from kaldo.parallel import is_parallel, validate_parallel_calculator, maybe_warn_ml_delta_shift
from kaldo.helpers.logger import get_logger
from tqdm import tqdm

logging = get_logger()

REPLICATED_ATOMS_THIRD_FILE = 'replicated_atoms_third.xyz'
REPLICATED_ATOMS_FILE = 'replicated_atoms.xyz'
THIRD_ORDER_FILE_SPARSE = 'third.npz'
THIRD_ORDER_FILE = 'third.npy'


def detect_path(files: list[str], folder: str = ""):
    """return the path and the filename of the first existed file in the ``files`` list in ``folder``.
    Raise an error if none of the files in the list is found in ``folder``.
    """
    # file_list = list(map(lambda f: os.path.join(folder, f), files))
    results = list(filter(lambda f: os.path.isfile(os.path.join(folder, f)), files))
    if results:
        return os.path.join(folder, results[0]), results[0]
    else:
        raise ValueError(f"{' or '.join(files)} are not found.")


class ThirdOrder(ForceConstant):

    @classmethod
    def load(cls,
             folder: str,
             supercell: tuple[int, int, int] = (1, 1, 1),
             format: str = 'sparse',
             third_energy_threshold: float = 0.,
             chunk_size: int = 100000):
        """
        Load third order force constants from a folder in the given format, used for library internally.

        To load force constants data, ``ForceConstants.from_folder`` is recommended.

        Parameters
        ----------
        folder : str
            Specifies where to load the data files.
        supercell : tuple[int, int, int]
            The supercell for the third order force constant matrix.
            Default: (1, 1, 1)
        format : str
            Format of the third order force constant information being loaded into ForceConstant object.
            Default: 'sparse'
        third_energy_threshold : float, optional
            When importing sparse third order force constant matrices, energies below
            the threshold value in magnitude are ignored. Units: eV/A^3
            Default: `None`
        chunk_size : int, optional
            Number of entries to process per chunk when reading sparse third order files.
            Larger values use more memory but may be faster for very large files.
            Default: 100000

        Returns
        -------
        third_order : ThirdOrder object
            A new instance of the ThirdOrder class
        """

        match format:
            case 'sparse' | 'numpy':
                config_path, _ = detect_path([REPLICATED_ATOMS_THIRD_FILE, REPLICATED_ATOMS_FILE], folder)
                replicated_atoms = ase.io.read(config_path, format='extxyz')

                n_replicas = np.prod(supercell)
                n_total_atoms = replicated_atoms.positions.shape[0]
                n_unit_atoms = n_total_atoms // n_replicas
                unit_symbols = []
                unit_positions = []
                for i in range(n_unit_atoms):
                    unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                    unit_positions.append(replicated_atoms.positions[i])
                unit_cell = replicated_atoms.cell / supercell

                atoms = Atoms(unit_symbols,
                              positions=unit_positions,
                              cell=unit_cell,
                              pbc=[1, 1, 1])

                _third_order = COO.from_scipy_sparse(load_npz(os.path.join(folder, THIRD_ORDER_FILE_SPARSE))) \
                    .reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3)) \
                    .astype(np.float64)
                third_order = ThirdOrder(atoms=atoms,
                                         replicated_positions=replicated_atoms.positions,
                                         supercell=supercell,
                                         value=_third_order,
                                         folder=folder)

            case 'eskm' | 'lammps':
                if format == 'eskm':
                    config_file = os.path.join(folder, "CONFIG")
                    replicated_atoms = ase.io.read(config_file, format='dlp4')
                elif format == 'lammps':
                    config_file = os.path.join(folder, "replicated_atoms.xyz")
                    replicated_atoms = ase.io.read(config_file, format='extxyz')

                third_file = os.path.join(folder, "THIRD")
                n_replicas = np.prod(supercell)
                n_total_atoms = replicated_atoms.positions.shape[0]
                n_unit_atoms = n_total_atoms // n_replicas
                unit_symbols = []
                unit_positions = []
                for i in range(n_unit_atoms):
                    unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                    unit_positions.append(replicated_atoms.positions[i])
                unit_cell = replicated_atoms.cell / supercell

                atoms = Atoms(unit_symbols,
                              positions=unit_positions,
                              cell=unit_cell,
                              pbc=[1, 1, 1])

                out = import_from_files(replicated_atoms=replicated_atoms,
                                        third_file=third_file,
                                        supercell=supercell,
                                        third_energy_threshold=third_energy_threshold,
                                        chunk_size=chunk_size)
                third_order = ThirdOrder(atoms=atoms,
                                         replicated_positions=replicated_atoms.positions,
                                         supercell=supercell,
                                         value=out[1],
                                         folder=folder)

            case ("vasp-sheng" | "shengbte") | ("qe-sheng" | "shengbte-qe") | ("qe-d3q" | "shengbte-d3q") | "vasp-d3q":
                grid_type = 'F'
                config_path, config_file = detect_path(['CONTROL', 'POSCAR'], folder)
                match config_file:
                    case 'CONTROL':
                        atoms, _supercell, charges = shengbte_io.import_control_file(config_path)
                    case 'POSCAR':
                        logging.info('Trying to open POSCAR')
                        atoms = ase.io.read(config_path)

                match format:
                    case ("vasp-sheng" | "shengbte") | ("qe-sheng" | "shengbte-qe"):
                        # load VASP third order force constant
                        third_file = os.path.join(folder, 'FORCE_CONSTANTS_3RD')
                        third_order = shengbte_io.read_third_order_matrix(third_file, atoms, supercell, order='C')
                    case _:
                        # load d3q third order force constant
                        third_file = os.path.join(folder, 'FORCE_CONSTANTS_3RD_D3Q')
                        third_order = shengbte_io.read_third_d3q(third_file, atoms, supercell, order='C')
                third_order = ThirdOrder.from_supercell(atoms=atoms,
                                                        grid_type=grid_type,
                                                        supercell=supercell,
                                                        value=third_order,
                                                        folder=folder)

            case 'hiphive':
                filename = 'atom_prim.xyz'
                # TODO: add replicated filename in example
                replicated_filename = 'replicated_atoms.xyz'
                try:
                    import kaldo.interfaces.hiphive_io as hiphive_io
                except ImportError:
                    logging.error('In order to use hiphive along with kaldo, hiphive is required. \
                        Please consider installing hihphive. More info can be found at: \
                        https://hiphive.materialsmodeling.org/')

                atom_prime_file = os.path.join(folder, filename)
                replicated_atom_prime_file = os.path.join(folder, replicated_filename)
                # TODO: Make this independent of replicated file
                atoms = ase.io.read(atom_prime_file)
                if os.path.isfile(replicated_atom_prime_file):
                    replicated_atoms = ase.io.read(replicated_atom_prime_file)
                else:
                    logging.warning('Replicated atoms file not found. Please check if the file exists. Use the unit cell atoms instead.')
                    replicated_atoms = atoms * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])

                if 'model3.fcs' in os.listdir(str(folder)):
                    # Derive constants used for third-order reshape
                    supercell = np.array(supercell)
                    n_prim = atoms.copy().get_masses().shape[0]
                    n_sc = np.prod(supercell)
                    pbc_conditions = replicated_atoms.get_pbc()
                    dim = len(pbc_conditions[pbc_conditions == True])
                    _third_order = hiphive_io.import_third_from_hiphive(atoms, supercell, folder)
                    _third_order = _third_order[0].reshape(n_prim * dim, n_sc * n_prim * dim,
                                                           n_sc * n_prim * dim)
                    third_order = cls(atoms=atoms,
                                      replicated_positions=replicated_atoms.positions,
                                      supercell=supercell,
                                      value=_third_order,
                                      folder=folder)

            case 'tdep':
                from kaldo.interfaces.tdep_io import (
                    build_nondiag_observable_kwargs,
                    attach_snf_metadata,
                )

                uc = ase.io.read(os.path.join(folder, 'infile.ucposcar'), format='vasp')
                sc = ase.io.read(os.path.join(folder, 'infile.ssposcar'), format='vasp')
                M = np.linalg.solve(np.asarray(uc.cell), np.asarray(sc.cell))
                M_int = np.round(M).astype(int)

                if not np.allclose(M, M_int, atol=1e-4):
                    raise ValueError(
                        f"Mapping from unitll to supercell (M matrix) was not integer-valued, got\n{M}"
                    )

                M_diag = np.diag(np.diag(M_int))
                M_is_not_diagonal = not np.allclose(M_int - M_diag, 0.0, atol=1e-6)

                if M_is_not_diagonal:
                    kw = build_nondiag_observable_kwargs(uc, sc)
                    mapping = kw.pop("_mapping")
                    third_ifcs = parse_tdep_third_forceconstant(
                        fc_filename=os.path.join(folder, 'infile.forceconstant_thirdorder'),
                        primitive=uc,
                        grid=kw["grid"],
                    )
                    third_order = cls(value=third_ifcs, folder=folder, **kw)
                    return attach_snf_metadata(third_order, mapping)

                third_ifcs = parse_tdep_third_forceconstant(
                    fc_filename=os.path.join(folder, 'infile.forceconstant_thirdorder'),
                    primitive=uc,
                    supercell=supercell,
                )

                third_order = cls(atoms=uc,
                                  replicated_positions=sc.positions,
                                  supercell=supercell,
                                  value=third_ifcs,
                                  folder=folder)

            case _:
                logging.error('Third order format not recognized: ' + str(format))
                raise ValueError

        return third_order



    @classmethod
    def remap(cls, ifc3 : "ThirdOrder", new_supercell : Atoms, tol : float = 1e-5) -> "ThirdOrder":
        """
        Remaps the force constants in ifc2 to a super-cell representation. The result is a new
        SecondOrder object whose internal IFCs have size: (n_rep, n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)
        where n_rep and n_uc are set by the `new_supercell`.
        """

        from kaldo.interfaces.tdep_io import (
            build_nondiag_observable_kwargs,
            attach_snf_metadata,
            _map2prim
        )
        from kaldo.distance_table import DistanceTable
        from kaldo.grid import wrap_coordinates

        uc = ifc3.atoms
        new_cell = np.asarray(new_supercell.cell)
        new_cell_inv = np.linalg.inv(new_cell)
        uc_cell = np.asarray(uc.cell)
        n_uc = len(uc)
        n_rep = ifc3.n_replicas

        # TODO FIGUREOUT CUTOFF FROM ifc3 object passed
        rc = 4.0 #TODO
        rc_sq = rc**2
        dt = DistanceTable(new_supercell, rc)

        # map2prim[i] gives the primitive atom index corresponding to
        # atom i in new_supercell.
        map2prim = _map2prim(uc, new_supercell)  

        kw = build_nondiag_observable_kwargs(uc, new_supercell)
        mapping = kw.pop("_mapping")

        # Build the output sparsely as COO: collect coordinates and values.
        # Output shape matches kaldo's replica-factorized IFC3 convention:
        #   (n_rep, n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)
        out_shape = (n_rep, n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)
        out_coords: list[list[int]] = []
        out_data: list[float] = []

        # Build progress bar:
        total_pairs = sum(atom.N_neighbors**2 for atom in dt)   # or N_neighbors*(N_neighbors-1) if you exclude self/self
        pbar = tqdm(total=total_pairs, desc="Remap IFC3 pairs", mininterval=0.5)
    
        for i, atom_i in enumerate(dt):
            uc_index_i = map2prim[i]

            # Figure out which blocks actually interact (i.e., non-zero)
            # to identify which (rep_j, uc_j, rep_k, uc_k) blocks are nonzero.
            phi3_i = ifc3.value[uc_index_i]
            mask = np.any(phi3_i != 0.0, axis=(0, 3, 6))   # shape (n_rep, n_uc, n_rep, n_uc)
            R2s, js, R3s, ks = np.where(mask)

            neighbor_indices_i = atom_i.inds

            for j, (vj, _, _, _, _) in enumerate(atom_i.neighbors()):
                for k, (vk, _, _, _, _) in enumerate(atom_i.neighbors()):
                    pbar.update(1)

                    # Check neighbor-neighbor distance is also within cutoff
                    if np.sum(np.square(vj - vk)) > rc_sq:
                        continue

                    found_match = False
                    for rep_j, uc_j, rep_k, uc_k in zip(R2s, js, R3s, ks):
                        # Candidate displacement built from IFC indices:
                        # Δr(i->j) = (r_uc[uc_j] - r_uc[uc_i]) + R2 @ uc_cell
                        # where R2 is the replica lattice vector (primitive basis) for rep_j.
                        R2 = ifc3._direct_grid.id_to_grid_index(int(rep_j))
                        rv_ij = (uc.positions[int(uc_j)] - uc.positions[int(uc_index_i)]) + (R2 @ uc_cell)
                        rv_ij = wrap_coordinates(rv_ij, new_cell, new_cell_inv)

                        if np.sum(np.square(rv_ij - vj)) > tol*tol:
                            continue
                        
                        # Second neighbor k
                        R3 = ifc3._direct_grid.id_to_grid_index(int(rep_k))
                        rv_ik = (uc.positions[int(uc_k)] - uc.positions[int(uc_index_i)]) + (R3 @ uc_cell)
                        rv_ik = wrap_coordinates(rv_ik, new_cell, new_cell_inv)

                        if np.sum(np.square(rv_ik - vk)) > tol*tol:
                            continue

                        # Convert linear supercell indices to (rep, uc) indices
                        uc_i_new = int(mapping["atom_of_sc"][i])
                        uc_j_new = int(mapping["atom_of_sc"][neighbor_indices_i[j]])
                        uc_k_new = int(mapping["atom_of_sc"][neighbor_indices_i[k]])
                        rep_i_new = int(mapping["replica_id_of_sc"][i])
                        rep_j_new = int(mapping["replica_id_of_sc"][neighbor_indices_i[j]])
                        rep_k_new = int(mapping["replica_id_of_sc"][neighbor_indices_i[k]])

                        # Store coords/value for output
                        block = ifc3.value[uc_index_i, :, rep_j, uc_j, :, rep_k, uc_k, :]
                        bcoords = np.asarray(block.coords)  # (3, nnz) for (alpha,beta,gamma)
                        bdata = np.asarray(block.data)
                        for nnz_idx in range(bdata.shape[0]):
                            alpha = int(bcoords[0, nnz_idx])
                            beta = int(bcoords[1, nnz_idx])
                            gamma = int(bcoords[2, nnz_idx])
                            out_coords.append([
                                rep_i_new, uc_i_new, alpha,
                                rep_j_new, uc_j_new, beta,
                                rep_k_new, uc_k_new, gamma,
                            ])
                            out_data.append(float(bdata[nnz_idx]))

                        found_match = True
                        break

                    if not found_match:
                        j_sc = int(neighbor_indices_i[j])
                        k_sc = int(neighbor_indices_i[k])
                        failed_atom = f"i_sc={i}, uc_index_i={uc_index_i}, j_sc={j_sc}, k_sc={k_sc}"
                        raise ValueError(
                            f"Could not find matching triplet for {failed_atom}. Cells are likely inconsistent."
                        )
        pbar.close()

        coords_arr = np.asarray(out_coords, dtype=np.int64).T  # (9, nnz)
        data_arr = np.asarray(out_data, dtype=float)
        third_ifcs = COO(coords=coords_arr, data=data_arr, shape=out_shape)#.sum_duplicates()
        third_order = ThirdOrder(value=third_ifcs, folder="Remapped IFCs, no source folder", **kw)

        return attach_snf_metadata(third_order, mapping)


    def save(self, filename='THIRD', format='sparse', min_force=1e-6):
        folder = self.folder
        filename = folder + '/' + filename
        n_atoms = self.atoms.positions.shape[0]
        match format:
            case 'eskm':
                logging.info('Exporting third in eskm format')
                n_replicas = self.n_replicas
                n_replicated_atoms = n_atoms * n_replicas
                tenjovermoltoev = 10 * units.J / units.mol
                third = self.value.reshape((n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3)) / tenjovermoltoev
                with open(filename, 'w') as out_file:
                    for i in range(n_atoms):
                        for alpha in range(3):
                            for j in range(n_replicated_atoms):
                                for beta in range(3):
                                    value = third[i, alpha, j, beta].todense()
                                    mask = np.argwhere(np.linalg.norm(value, axis=1) > min_force)
                                    if mask.any():
                                        for k in mask:
                                            k = k[0]
                                            out_file.write("{:5d} ".format(i + 1))
                                            out_file.write("{:5d} ".format(alpha + 1))
                                            out_file.write("{:5d} ".format(j + 1))
                                            out_file.write("{:5d} ".format(beta + 1))
                                            out_file.write("{:5d} ".format(k + 1))
                                            for gamma in range(3):
                                                out_file.write(' {:16.6f}'.format(third[i, alpha, j, beta, k, gamma]))
                                            out_file.write('\n')
                logging.info('Done exporting third.')
            case 'sparse' | 'numpy':
                config_file = folder + REPLICATED_ATOMS_THIRD_FILE
                ase.io.write(config_file, self.replicated_atoms, format='extxyz')

                save_npz(folder + '/' + THIRD_ORDER_FILE_SPARSE, self.value.reshape((n_atoms * 3 * self.n_replicas *
                                                                            n_atoms * 3, self.n_replicas *
                                                                            n_atoms * 3)).to_scipy_sparse())
            case _:
                super(ThirdOrder, self).save(filename, format)



    def calculate(self, calculator=None, delta_shift=1e-4, distance_threshold=None, is_storing=True, is_verbose=False,
                  n_workers=1, scratch_dir=None, keep_scratch=False, jat_flush_every=50, use_symmetry=False, symprec=1e-5):
        """Calculate the third order force constants.

        This is the method typically reached through ``fc.third.calculate(...)``.
        It can load an existing stored result from ``self.folder`` when
        ``is_storing`` is enabled, or compute the anharmonic force constants
        directly from finite-difference force evaluations.

        See the *Parallel runs with ML calculators* section of the
        ForceConstants documentation for the recommended pattern when
        running torch-based calculators (Orb, MACE, MatterSim, CPUNEP) in
        parallel: define a no-arg factory function at module top level
        and pass it (without parentheses) as ``calculator``.

        Parameters
        ----------
        calculator : callable or ASE Calculator instance
            For serial runs, pass an ASE Calculator instance (the existing
            kaldo idiom). For parallel runs, pass a callable that returns
            a fresh ASE Calculator: a class with a no-arg constructor, a
            top-level factory function, ``functools.partial``, etc. Each
            worker invokes the callable once to build its own isolated
            calculator::

                from ase.calculators.emt import EMT
                calculator=EMT

            If None, replicated_atoms must already have a calculator attached.
        delta_shift : float
            Finite-difference displacement in Angstrom. The default ``1e-4``
            is tuned for analytical calculators (EMT, LAMMPS). ML potentials
            in float32 (Orb, MACE, MatterSim, ...) need ``1e-2`` or larger
            because float32 force noise (~1e-7 eV/Å) divided by a tiny
            delta produces FC noise that swamps the physics. A warning
            fires when ``delta_shift < 1e-2`` and the calculator looks
            ML-based.
            Default: 1e-4
        n_workers : int or None
            Number of parallel worker processes. ``1`` runs serially.
            ``None`` uses all available CPUs. Each worker is capped to one
            OpenMP / MKL / OpenBLAS thread so calculators with internal
            multithreading (PyNEP, torch CPU, numpy+MKL) don't oversubscribe.
            Override by setting ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` in
            the environment before invoking.
            Default: 1 (serial)
        scratch_dir : str or None
            Directory for scratch chunk files written during calculation to keep
            peak memory low. Pass an explicit path to override. Pass an
            empty string ``''`` to disable scratch files and fall back to
            in-memory accumulation.
            Default: ``{folder}/third_order`` when ``self.folder`` is set,
            ``n_workers > 1``, and ``use_symmetry=False``. With
            ``use_symmetry=True`` the auto-default is suppressed (the two
            modes are mutually incompatible — see the ``use_symmetry``
            docstring below).
        keep_scratch : bool
            If True, scratch files are kept after assembly.
            Default: False
        jat_flush_every : int
            Number of jat iterations each worker buffers before flushing to disk.
            Smaller values use less memory at the cost of more I/O. Default 50.
        use_symmetry : bool, optional
            If True, use the crystal spacegroup to reduce the number of
            atom pairs (i, jat) computed by the FD method. Only spacegroup
            operations compatible with the supercell shape are used (e.g.
            an in-plane subgroup for slab supercells). Requires a
            diagonal integer supercell expansion. Not compatible with
            ``scratch_dir`` — pass ``scratch_dir=None`` (the default)
            when enabling.
            Default: False
        symprec : float, optional
            precision for symmetry using spglib.
            Default: 1e-5
        """
        if is_parallel(n_workers):
            validate_parallel_calculator(calculator, method='ThirdOrder.calculate')
        maybe_warn_ml_delta_shift(calculator, delta_shift, method='ThirdOrder.calculate')
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        # Attach the calculator instance to replicated_atoms once and skip the
        # per-atom rebind in _compute_iat_third. Some calculator libraries
        # require a calculator to stay bound to a single atoms object.
        if n_workers == 1 and calculator is not None and not callable(calculator):
            replicated_atoms.calc = calculator
            worker_calculator = None
        else:
            worker_calculator = calculator
        # Auto-resolve the default scratch directory only for parallel runs;
        # serial stays in memory to avoid creating unexpected directories.
        # use_symmetry is incompatible with scratch_dir (calculate_third
        # raises ValueError on the combo), so don't auto-assign in that case.
        if (scratch_dir is None and self.folder and is_parallel(n_workers)
                and not use_symmetry):
            scratch_dir = os.path.join(self.folder, 'third_order')
        elif scratch_dir == '':
            scratch_dir = None
        if is_storing:
            try:
                self.value = ThirdOrder.load(folder=self.folder, supercell=self.supercell).value

            except FileNotFoundError:
                logging.info('Third order not found. Calculating.')
                self.value = calculate_third(atoms,
                                             replicated_atoms,
                                             delta_shift,
                                             distance_threshold=distance_threshold,
                                             is_verbose=is_verbose,
                                             n_workers=n_workers,
                                             calculator=worker_calculator,
                                             scratch_dir=scratch_dir,
                                             keep_scratch=keep_scratch,
                                             jat_flush_every=jat_flush_every,
                                             use_symmetry=use_symmetry,
                                             symprec=symprec)
                self.save('third')
                ase.io.write(self.folder + '/' + REPLICATED_ATOMS_THIRD_FILE, self.replicated_atoms, 'extxyz')
            else:
                logging.info('Reading stored third')
        else:
            self.value = calculate_third(atoms,
                                         replicated_atoms,
                                         delta_shift,
                                         distance_threshold=distance_threshold,
                                         is_verbose=is_verbose,
                                         n_workers=n_workers,
                                         calculator=worker_calculator,
                                         scratch_dir=scratch_dir,
                                         keep_scratch=keep_scratch,
                                         jat_flush_every=jat_flush_every,
                                         use_symmetry=use_symmetry,
                                         symprec=symprec)
            if is_storing:
                self.save('third')
                ase.io.write(self.folder + '/' + REPLICATED_ATOMS_THIRD_FILE, self.replicated_atoms, 'extxyz')




    def __str__(self):
        return 'third'

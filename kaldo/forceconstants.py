"""
kaldo
Anharmonic Lattice Dynamics
"""
import numpy as np
from kaldo.grid import SupercellGrid
from kaldo.grid import Grid, NonDiagonalGrid, wrap_lattice_vector_to_replica
from kaldo.observables.secondorder import SecondOrder
from kaldo.observables.thirdorder import ThirdOrder
from kaldo.observables.fourthorder import FourthOrder
from kaldo.helpers.logger import get_logger
from kaldo.observables.harmonic_with_q import HarmonicWithQ, _HarmonicIFCInterpolation
import ase.units as units
logging = get_logger()

MAIN_FOLDER = 'displacement'


def _normalize_supercell(supercell: tuple[int, int, int] | np.ndarray | None):
    """Return an exact diagonal triple or integer expansion matrix.

    Supercell indices define a finite lattice quotient and therefore cannot be
    rounded approximately. Length-three repetitions must also be positive;
    general matrices are checked for nonsingularity by ``SupercellGrid``.
    """
    if supercell is None:
        return None
    array = np.asarray(supercell)
    if array.shape == (3,):
        return tuple(int(value) for value in array)
    if array.shape == (3, 3):
        rounded = np.rint(array).astype(int)
        if not np.allclose(array, rounded, rtol=0, atol=1e-12):
            raise ValueError("supercell matrix must be integer-valued")
        return rounded
    raise ValueError("supercell must be a length-3 diagonal or integer 3x3 matrix")


class ForceConstants:
    """
    A ForceConstants class object is used to create or load the second or third order force constant matrices as well as
    store information related to the geometry of the system.

    Parameters
    ----------
    atoms: Tabulated xyz files or ASE Atoms object
        The atoms to work on.
    supercell: tuple[int, int, int] or array-like (3, 3), optional
        Diagonal repetitions ``(l, m, n)`` or an integer expansion matrix
        whose determinant gives the number of primitive cells.
        Default: (1, 1, 1)
    third_supercell: tuple[int, int, int] or array-like (3, 3), optional
        Same as supercell, but for the third order force constant matrix.
        If not provided, it's copied from supercell.
        Default: ``self.supercell``
    folder: str, optional
        Name to be used for the displacement information folder.
        Default: ``'displacement'``
    distance_threshold: float, optional
        If the distance between two atoms exceeds threshold, the interatomic
        force is ignored.
        Default: None
    second_order: SecondOrder, optional
        Preloaded second-order force constants attached to the instance.
        Default: ``None`` (lazy construction)
    third_order: ThirdOrder, optional
        Preloaded third-order force constants attached to the instance.
        Default: ``None`` (lazy construction)
    fourth_order: FourthOrder, optional
        Preloaded fourth-order force constants attached to the instance.
        Default: ``None`` (not constructed lazily — must be loaded explicitly)
    is_acoustic_sum: bool, optional
        If True, apply the acoustic sum rule to second-order force constants
        computed via ``second.calculate``. For force constants read from
        files use ``from_folder(..., is_acoustic_sum=True)`` instead. Note
        that on transpose-asymmetric force constants (e.g. raw finite
        differences from an ML potential that breaks the crystal symmetry)
        the sum rule alone cannot zero the Gamma acoustic modes.
        Default: False

    Attributes
    ----------
    n_atoms: int
        Number of atoms in the unit cell
    n_modes: int
        The number of possible vibrational modes in the system from a lattice dynamics perspective. Equivalent to
        3*n_atoms where the factor of 3 comes from the 3 Cartesian directions.
    n_replicas: int
        Number of periodic supercell classes, ``abs(det(supercell_matrix))``.
        For a diagonal three-vector this is ``np.prod(supercell)``.
    n_replicated_atoms: int
        Number of atoms in the physical replicated structure,
        ``n_atoms * n_replicas``.
    cell_inv: np.array(3, 3)
        A 3x3 matrix which satisfies AB=I where A is the matrix of cell vectors, I is the identity matrix, and B is the
        cell_inv matrix.


    """
    def __init__(self,
                 atoms,
                 supercell: tuple[int, int, int] | np.ndarray = (1, 1, 1),
                 third_supercell: tuple[int, int, int] | np.ndarray | None = None,
                 folder: str = MAIN_FOLDER,
                 distance_threshold: float | None = None,
                 second_order: SecondOrder | None = None,
                 third_order: ThirdOrder | None = None,
                 fourth_order: FourthOrder | None = None,
                 is_acoustic_sum: bool = False):

        # Store the user defined information to the object
        self.atoms = atoms
        self.supercell = _normalize_supercell(supercell)
        normalized_third = _normalize_supercell(third_supercell)
        self.third_supercell = (
            self.supercell if normalized_third is None else normalized_third
        )
        self.n_atoms = atoms.positions.shape[0]
        self.n_modes = self.n_atoms * 3
        if second_order is not None:
            self.supercell_grid = second_order.supercell_grid
        else:
            matrix = np.asarray(self.supercell)
            if matrix.shape == (3,):
                matrix = np.diag(matrix)
            self.supercell_grid = SupercellGrid(matrix)
        self.n_replicas = self.supercell_grid.size
        self.n_replicated_atoms = self.n_replicas * self.n_atoms
        self.cell_inv = np.linalg.inv(atoms.cell)
        self.folder = folder
        self.distance_threshold = distance_threshold
        self.is_acoustic_sum = is_acoustic_sum
        self._list_of_replicas = None
        self._second = second_order
        self._third = third_order
        self._fourth = fourth_order

        if distance_threshold is not None:
            logging.info('Using folded IFC matrices.')

    @property
    def second(self):
        if self._second is None:
            # initialize an empty SecondOrder object for computing force constants later.
            self._second = SecondOrder.from_supercell(self.atoms,
                                                      supercell=self.supercell,
                                                      grid_type='C',
                                                      is_acoustic_sum=self.is_acoustic_sum,
                                                      folder=self.folder)
        return self._second

    @property
    def third(self):
        if self._third is None:
            self._third = ThirdOrder.from_supercell(self.atoms,
                                                    supercell=self.third_supercell,
                                                    grid_type='C',
                                                    folder=self.folder)
        return self._third

    @property
    def fourth(self):
        """Fourth-order force constants (FourthOrder).

        Unlike ``second`` and ``third`` there is no lazy construction path
        today: IFC4 must be loaded explicitly via ``from_folder`` (currently
        only format='tdep' is supported) or passed as ``fourth_order=`` at
        construction time. Returns ``None`` if not loaded.
        """
        return self._fourth

    @classmethod
    def from_folder(cls,
                    folder: str,
                    supercell: tuple[int, int, int] | np.ndarray = (1, 1, 1),
                    format: str = 'numpy',
                    third_energy_threshold: float = 0.,
                    third_supercell: tuple[int, int, int] | np.ndarray | None = None,
                    is_acoustic_sum: bool = False,
                    only_second: bool = False,
                    include_fourth: bool = False,
                    distance_threshold: float | None = None,
                    chunk_size: int = 100000,
                    supercell_matrix: np.ndarray | None = None):
        """
        Create a finite difference object from a folder

        The folder should contain the a set of files whose names and contents are dependent on the "format" parameter.
        Below is the list required for each format (also found in the api_forceconstants documentation if you prefer
        to read it with nicer formatting and explanations).

        - numpy: replicated_atoms.xyz, second.npy, third.npz
        - eskm: CONFIG, replicated_atoms.xyz, Dyn.form, THIRD
        - lammps: replicated_atoms.xyz, Dyn.form, THIRD
        - vasp-sheng: CONTROL/POSCAR, FORCE_CONSTANTS_2ND/FORCE_CONSTANTS, FORCE_CONSTANTS_3RD
        - qe-sheng: CONTROL/POSCAR, espresso.ifc2, FORCE_CONSTANTS_3RD
        - vasp-d3q: CONTROL/POSCAR, FORCE_CONSTANTS_2ND/FORCE_CONSTANTS, FORCE_CONSTANTS_3RD_D3Q
        - qe-d3q: CONTROL/POSCAR, espresso.ifc2, FORCE_CONSTANTS_3RD_D3Q
        - hiphive: atom_prim.xyz, replicated_atoms.xyz, model2.fcs, model3.fcs
        - tdep: infile.ucposcar, infile.ssposcar, infile.forceconstant, infile.forceconstant_thirdorder
                (+ infile.forceconstant_fourthorder if ``include_fourth=True``)
        - gpumd: gpumd_fc.npz (single compact archive; supercell/geometry embedded)

        Parameters
        ----------
        folder : str
            Chosen folder to load in system information.
        supercell : (int, int, int) or array-like (3, 3), optional
            Diagonal repetitions or an integer primitive-to-supercell
            expansion matrix. ``format='tdep'`` is the complete matrix-aware
            combined IFC2/IFC3 file route; legacy compact readers require a
            diagonal supercell. For TDEP, the matrix inferred from
            ``infile.ucposcar`` and ``infile.ssposcar`` is authoritative.
            Default: (1, 1, 1)
        format : 'numpy', 'eskm', 'lammps', 'vasp-sheng', 'qe-sheng', 'vasp-d3q', 'qe-d3q', 'hiphive', 'tdep', 'gpumd'
            Format of force constant information being loaded into ForceConstants object.
            Default is ``'numpy'``
        third_energy_threshold : float, optional
            When importing sparse third order force constant matrices, energies below
            the threshold value in magnitude are ignored. Units: eV/Angstrom^3
            Default is None
        distance_threshold : float, optional
            When calculating force constants, contributions from atoms further than the
            distance threshold will be ignored.
        third_supercell : (int, int, int) or array-like (3, 3), optional
            Supercell topology for the third-order force constants. If not
            supplied, the resolved second-order topology is used.
            Default: ``None``
        is_acoustic_sum : Bool, optional
            If true, the acoustic sum rule is applied to the dynamical matrix.
            Default is False
        only_second : bool, optional
            Load only harmonic force constants and leave IFC3 unconstructed.
            Default: False
        include_fourth : bool, optional
            Also load ``infile.forceconstant_fourthorder``. Currently valid
            only for ``format='tdep'`` and ignored when ``only_second=True``.
            Default: False
        chunk_size : int, optional
            Number of entries to process per chunk when reading sparse third order files.
            Larger values use more memory but may be faster for very large files.
            Default: 100000
        supercell_matrix : array-like (3, 3), optional
            Expected integer expansion matrix for TDEP input. The structure
            files define the authoritative matrix; a mismatch raises an error.
            Other formats normally express the topology through ``supercell``
            and ``third_supercell``.
            Default: None

        Returns
        -------
        forceconstants: ForceConstants object
            A new instance of the ForceConstants class
        """
        supercell = _normalize_supercell(supercell)
        third_supercell = _normalize_supercell(third_supercell)

        # Validate include_fourth early so we don't waste time loading IFC2/3
        # only to discover the format is wrong.
        if include_fourth and format != 'tdep':
            raise ValueError(
                f"include_fourth=True is only supported for format='tdep'"
                f" (got format={format!r})"
            )

        effective_second_format = format
        effective_third_format = format

        second_order = SecondOrder.load(folder=folder,
                                        supercell=supercell,
                                        format=effective_second_format,
                                        is_acoustic_sum=is_acoustic_sum,
                                        supercell_matrix=supercell_matrix)
        atoms = second_order.atoms
        resolved_supercell = _normalize_supercell(second_order.supercell)

        third_order = None
        target_third_supercell = (
            resolved_supercell if third_supercell is None else third_supercell
        )

        if not only_second:
            third_order = ThirdOrder.load(folder=folder,
                                          supercell=target_third_supercell,
                                          format=effective_third_format,
                                          third_energy_threshold=third_energy_threshold,
                                          chunk_size=chunk_size,
                                          supercell_matrix=supercell_matrix,
                                          atoms_override=(second_order.atoms
                                                          if effective_third_format in
                                                          ('qe-sheng', 'shengbte-qe', 'qe-d3q', 'shengbte-d3q')
                                                          else None))
            target_third_supercell = _normalize_supercell(third_order.supercell)

        fourth_order = None
        if include_fourth and not only_second:
            # Fourth-order loading is opt-in because most existing datasets
            # ship only IFC2 + IFC3. Today only format='tdep' is wired
            # (validated at the top of this method).
            fourth_order = FourthOrder.load(folder=folder,
                                            supercell=target_third_supercell,
                                            format='tdep',
                                            supercell_matrix=supercell_matrix)

        return cls(atoms=atoms,
                   supercell=resolved_supercell,
                   third_supercell=target_third_supercell,
                   folder=folder,
                   distance_threshold=distance_threshold,
                   second_order=second_order,
                   third_order=third_order,
                   fourth_order=fourth_order,
                   is_acoustic_sum=is_acoustic_sum)

    @staticmethod
    def _build_shifted_rep(grid) -> np.ndarray:
        """Return shifted_rep[rep_i, rel] = index of (grid[rep_i] + grid[rel]) in the replica table.

        For a diagonal Grid uses fast modular arithmetic + ravel_multi_index.
        For a NonDiagonalGrid uses wrap_lattice_vector_to_replica so that the
        non-diagonal PBC is handled correctly.
        """
        if isinstance(grid, NonDiagonalGrid):
            table = grid._replica_table                              # (n_rep, 3) int
            M = np.rint(grid._M).astype(int)
            n_rep = len(table)

            # All pairwise sums: (n_rep, n_rep, 3)
            sums = table[:, None, :] + table[None, :, :]

            # Step 1: [0,1) sc-fractional wrap (vectorized).
            # Mirrors what build_supercell_replica_mapping does before norm-min.
            inv_M = np.linalg.inv(M.astype(float))
            frac = sums.astype(float) @ inv_M                              # (n_rep, n_rep, 3)
            wrapped = np.rint((frac - np.floor(frac + 1e-4)) @ M).astype(int)  # (n_rep, n_rep, 3)

            # Step 2: norm-minimal search over {-1,0,1}^3 — same shifts used
            # when the table was built, so exactly one candidate matches each (i,j).
            sv = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                                      indexing='ij')).reshape(3, -1).T   # (27, 3)
            candidates = wrapped[:, :, None, :] - (sv @ M)[None, None, :, :]  # (n_rep, n_rep, 27, 3)

            # matches[i, j, s, k] = True when candidates[i,j,s] == table[k]
            matches = (candidates[:, :, :, None, :] == table[None, None, None, :, :]).all(axis=-1)
            matched = matches.any(axis=2)   # (n_rep, n_rep, n_rep): any shift hits table[k]?

            if not matched.any(axis=-1).all():
                raise ValueError("Replica sum not found; the replica table may be incomplete.")

            return np.argmax(matched, axis=-1)   # (n_rep, n_rep)
        else:
            supercell = grid.grid_shape
            grid_arr = grid.grid(is_wrapping=False)                                    # (n_rep, 3)
            combined = (grid_arr[:, np.newaxis, :] + grid_arr[np.newaxis, :, :]) % np.array(supercell)
            return np.ravel_multi_index(
                combined.reshape(-1, 3).T, supercell, order=grid.order
            ).reshape(grid.grid_size, grid.grid_size)


    @staticmethod
    def _project_second_onto_snf_class_table(raw_pair, ifc_obj, mapping):
        """Reindex per-pair IFC2 onto the det(M) congruence-class table.

        After PR #301, ``SecondOrder`` on a non-diagonal TDEP supercell stores
        IFC2 on the unique per-pair lattice vectors from the file (correct for
        Fourier phases) while ``_snf_mapping['replica_table']`` keeps the full
        ``det(M)`` class table (closed under supercell PBC). Translational
        expansion via :meth:`irred_to_full` needs that closed table.

        Returns
        -------
        raw_class : ndarray, shape (n_uc, 3, n_class, n_uc, 3)
        class_grid : NonDiagonalGrid
        """
        class_table = np.asarray(mapping["replica_table"], dtype=int)
        M = np.rint(mapping["M"]).astype(int)
        pair_table = np.asarray(ifc_obj._direct_grid._replica_table, dtype=int)
        n_uc = raw_pair.shape[0]
        n_class = len(class_table)
        raw_class = np.zeros((n_uc, 3, n_class, n_uc, 3), dtype=np.float64)
        for r, R in enumerate(pair_table):
            class_id = int(wrap_lattice_vector_to_replica(R, class_table, M))
            # Two per-pair R's in the same class are periodic images
            # (they differ by a supercell lattice vector); the supercell
            # force constant is the sum over images. With TDEP cutoffs
            # below half the box each class has a single image, so this
            # reduces to a plain assignment.
            raw_class[:, :, class_id, :, :] += raw_pair[:, :, r, :, :]
        class_grid = NonDiagonalGrid(replica_table=class_table, M=M)
        return raw_class, class_grid

    def irred_to_full(self, order: int, grid: Grid | None = None) -> np.ndarray:
        """Reconstruct the full IFC tensor from the irreducible part stored in this object.

        The irreducible tensor (``self.second`` or ``self.third``) suppresses the replica index
        of the first atom (always cell 0 by translational convention).  Translational symmetry
        gives:

            fc_full[rep_i, uc_i, :, abs_j, uc_j, :, ...]
                = ifc_irred[uc_i, :, rel_j, uc_j, :, ...]

        where  abs_j = shifted_rep[rep_i, rel_j]
        and    shifted_rep = ForceConstants._build_shifted_rep(grid).

        For the loop-based assignment pattern (resolving ``???`` for 3rd order)::

            shifted_rep = ForceConstants._build_shifted_rep(grid)
            for rep_i in range(n_rep):
                for uc_i, rep_j, uc_j, rep_k, uc_k in ...:
                    fc3_out[rep_i, uc_i, :,
                            shifted_rep[rep_i, rep_j], uc_j, :,
                            shifted_rep[rep_i, rep_k], uc_k, :] += ifc3_irred[uc_i, :, rep_j, uc_j, :, rep_k, uc_k, :]

        Parameters
        ----------
        order : int
            IFC order; 2 or 3.
        grid : Grid, optional
            Supercell grid.  Defaults to the grid of the corresponding order object
            (``self.second._direct_grid`` or ``self.third._direct_grid``).

        Returns
        -------
        fc_full : np.ndarray
            Full IFC tensor of shape ``(n_rep, n_unit, 3) * order``.
        """
        if order not in (2, 3):
            raise ValueError(f'order must be 2 or 3, got {order}.')

        n_unit = self.n_atoms

        if order == 2:
            ifc_obj = self.second
            raw = np.asarray(ifc_obj.value[0], dtype=np.float64)
        else:
            ifc_obj = self.third
            n_rep_obj = ifc_obj.n_replicas
            raw = ifc_obj.value
            if hasattr(raw, 'todense'):
                raw = raw.todense()
            raw = np.asarray(raw, dtype=np.float64).reshape(n_unit, 3, n_rep_obj, n_unit, 3, n_rep_obj, n_unit, 3)

        if grid is None:
            grid = ifc_obj._direct_grid

        # Per-pair IFC2 on SNF is not closed under replica addition; project
        # onto the det(M) class table before building shifted_rep.
        if (
            order == 2
            and isinstance(grid, NonDiagonalGrid)
            and getattr(ifc_obj, "_snf_mapping", None) is not None
        ):
            mapping = ifc_obj._snf_mapping
            class_table = np.asarray(mapping["replica_table"], dtype=int)
            if len(grid._replica_table) != len(class_table):
                raw, grid = self._project_second_onto_snf_class_table(
                    raw, ifc_obj, mapping,
                )

        n_rep = grid.grid_size
        # Relative-replica axes in the irreducible tensor: 2, 5, ... (one per non-first atom)
        replica_axes = [3 * j - 1 for j in range(1, order)]

        shifted_rep = self._build_shifted_rep(grid)        # (n_rep, n_rep)
        inv_shifted_rep = np.argsort(shifted_rep, axis=1)  # inv_shifted_rep[rep_i, abs] = rel

        full_shape = (n_rep, n_unit, 3) * order
        fc_full = np.zeros(full_shape, dtype=np.float64)

        for rep_i in range(n_rep):
            inv_perm = inv_shifted_rep[rep_i]
            temp = raw
            for ax in replica_axes:
                temp = np.take(temp, inv_perm, axis=ax)
            fc_full[rep_i] = temp

        return fc_full


    def elastic_prop(self):
        """
        Return the stiffness tensor (aka elastic modulus tensor) of the system in GPa.

        This describes the stress-strain relationship of the material and can sometimes
        be used as a loose predictor for thermal conductivity. Requires the dynamical
        matrix to be loaded or calculated.

        Returns
        -------
        np.ndarray
            Elasticity tensor C_ijkl with shape (3, 3, 3, 3) in GPa.

        Notes
        -----
        Notation follows: Theory of the elastic constants of graphite and graphene,
        DOI 10.1002/pssb.200879604
        """
        # Extract key parameters
        atoms = self.atoms
        masses = atoms.get_masses()
        volume = atoms.get_volume()
        n_unit = atoms.positions.shape[0]
        # Elasticity is the q->0 expansion of the same interpolated harmonic
        # matrix used for frequencies and velocities. Reuse its pair-specific
        # Wigner--Seitz geometry so a skew cell cannot pair one raw replica
        # representative with every basis-atom pair.
        interpolation = _HarmonicIFCInterpolation.build(self.second, "auto")
        d1, d2 = interpolation.real_space_moments()

        # Compute Gamma tensor as eq.6
        h0 = HarmonicWithQ(np.array([0, 0, 0]), self.second, storage='numpy')

        # Optical eigenvectors
        e_mu = np.array(h0._eigensystem[1:, :]).reshape((n_unit, 3, 3 * n_unit))

        # Optical eigenfrequencies (w/(2*pi) = f) in THz
        w_mu = np.abs(np.array(h0._eigensystem[0, :])) ** 0.5

        gamma = np.einsum(
            'iav,jbv,v->iajb',
            e_mu[:, :, 3:],
            e_mu[:, :, 3:],
            1 / w_mu[3:] ** 2
        )

        # Compute component square bracket (`b`) and round bracket (`r`) terms
        # Keep the real component only

        # Square bracket term, eq.4
        # [ij, kl] = b_{ijkl} = 1/(2 v_c) \sum_{n,m} \sqrt{M_n} \sqrt{M_m} D^{nm}_{ij,kl}^{(2)}
        sqrt_masses = masses ** 0.5
        b = (1 / (2 * volume)) * np.einsum(
            'n,m,nimjkl->ijkl',
            sqrt_masses,
            sqrt_masses,
            d2
        ).real

        # Include mass in first order term
        d1r = np.einsum('nhmij,m->nhmij', d1, sqrt_masses)

        # Round bracket term, eq.5, mass is included in d1r
        r = -(1 / volume) * np.einsum(
            'nhmij,nhrp,rpskl->ijkl',
            d1r,
            gamma,
            d1r
        ).real

        # Compute elastic constants C_{ij,kl} as eq.3
        cijkl = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        cijkl[i, j, k, l] = (
                            b[i, k, j, l] + b[j, k, i, l] -
                            b[i, j, k, l] + r[i, j, k, l]
                        )

        # Unit conversion constants
        ev_to_tenjovermol = units.mol / (10 * units.J)
        # units._e = 1.602×10^-19 J
        # units.Angstrom = 1.0 = 1e-10 m
        # (units.Angstrom)^3 = 1e-30 m^3 / 1e9 from Pa to GPa
        # Combined: 1e-21
        evperang3togpa = units._e / (units.Angstrom * 1e-21)

        # Return elastic tensor in GPa
        return evperang3togpa * cijkl / ev_to_tenjovermol

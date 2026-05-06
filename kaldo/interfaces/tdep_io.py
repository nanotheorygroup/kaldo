from ase import Atoms
import ase.io
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from ase.geometry import get_distances
from kaldo.helpers.logger import get_logger
from kaldo.grid import Grid
from sparse import COO

logging = get_logger()


# ---------------------------------------------------------------------------
# SNF-style supercell enumeration (for non-diagonal primitive -> ssposcar)
# ---------------------------------------------------------------------------

def build_supercell_replica_mapping(primitive_atoms, supercell_atoms, tol=1e-4):
    """Map TDEP ssposcar atoms to (primitive_atom_index, lattice_vector).

    Given a primitive cell and a TDEP ``infile.ssposcar`` that may be a
    non-diagonal tiling of the primitive (rhombo primitive + cubic
    conventional supercell, etc.), compute:

    * ``atom_of_sc``  : (n_sc,) int — which primitive atom each sc atom is.
    * ``replica_vector_of_sc``  : (n_sc, 3) int — the integer lattice vector
      R in the primitive basis of each sc atom.
    * ``replica_table`` : (n_rep, 3) int — deduplicated list of unique
      replica lattice vectors R.
    * ``replica_id_of_sc`` : (n_sc,) int — index into replica_table for
      each sc atom.

    Wraps replica vectors to their minimum-image form under the supercell
    lattice so the replica table covers only n_rep = n_sc / n_uc unique
    lattice points.
    """
    uc_pos = np.asarray(primitive_atoms.positions)
    uc_cell = np.asarray(primitive_atoms.cell)
    sc_pos = np.asarray(supercell_atoms.positions)
    sc_cell = np.asarray(supercell_atoms.cell)
    n_uc = len(primitive_atoms)
    n_sc = len(supercell_atoms)
    if n_sc % n_uc != 0:
        raise ValueError(
            f"ssposcar n_atoms={n_sc} not divisible by primitive n_atoms={n_uc}"
        )

    inv_uc = np.linalg.inv(uc_cell)
    inv_sc = np.linalg.inv(sc_cell)
    # M_prim_to_sc satisfies sc_cell = M @ uc_cell (row-vector lattice matrices).
    M = np.linalg.solve(uc_cell, sc_cell)

    atom_of_sc = np.full(n_sc, -1, dtype=int)
    replica_vector_of_sc = np.zeros((n_sc, 3), dtype=int)

    for i, rsc in enumerate(sc_pos):
        # Try each primitive atom: rsc = uc_pos[j] + R @ uc_cell + k @ sc_cell
        # For each j, solve R = (rsc - uc_pos[j]) @ inv_uc in primitive basis;
        # then wrap through the supercell lattice (R might point outside one
        # period of the ssposcar). We want the R that lives inside the
        # Wigner-Seitz cell of the supercell (min-image).
        for j in range(n_uc):
            R_frac_prim = (rsc - uc_pos[j]) @ inv_uc
            # Wrap into the [0, 1)^3 Brillouin zone of the SUPERCELL lattice,
            # expressed in primitive units. The supercell lattice points are
            # integer linear combinations of M rows. So we express R in sc
            # fractional coords and wrap those.
            R_frac_sc = R_frac_prim @ np.linalg.inv(M)
            R_frac_sc_wrap = R_frac_sc - np.floor(R_frac_sc + tol)
            # Convert back to primitive basis
            R_frac_prim_wrap = R_frac_sc_wrap @ M
            R_int = np.round(R_frac_prim_wrap).astype(int)
            residual = np.max(np.abs(R_frac_prim_wrap - R_int))
            if residual < tol:
                atom_of_sc[i] = j
                replica_vector_of_sc[i] = R_int
                break
        if atom_of_sc[i] == -1:
            raise ValueError(
                f"sc atom {i} at {rsc} did not map to any primitive atom"
            )

    # Build deduplicated replica table (atom 0's replicas = all unique R)
    uniq_rs, inverse = np.unique(
        replica_vector_of_sc, axis=0, return_inverse=True,
    )
    n_rep_expected = n_sc // n_uc
    if len(uniq_rs) != n_rep_expected:
        raise ValueError(
            f"expected {n_rep_expected} unique replicas, got {len(uniq_rs)}"
        )
    # Re-wrap each replica to the Cartesian-norm-minimal form under the
    # supercell lattice. An entry R from the [0,1) sc-fractional wrap can
    # equivalently be R - k @ M for any integer k; we pick k that minimizes
    # ||R @ uc_cell||. This matches the TDEP IFC file convention (signed,
    # small) and keeps exp(iq.R) phases correct on q-meshes smaller than the
    # supercell.
    uniq_rs_min = np.zeros_like(uniq_rs)
    M_rows = M.astype(int)
    shifts = np.array(
        [[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)],
        dtype=int,
    )
    for idx, R in enumerate(uniq_rs):
        best_R = R
        best_norm = np.linalg.norm(R @ uc_cell)
        for s in shifts:
            if not np.any(s):
                continue
            R_shift = R - s @ M_rows
            norm = np.linalg.norm(R_shift @ uc_cell)
            if norm < best_norm - 1e-8:
                best_R = R_shift
                best_norm = norm
        uniq_rs_min[idx] = best_R
    replica_table = uniq_rs_min.astype(int)
    replica_id_of_sc = inverse

    return dict(
        atom_of_sc=atom_of_sc,
        replica_vector_of_sc=replica_vector_of_sc,
        replica_table=replica_table,
        replica_id_of_sc=replica_id_of_sc,
        M=M,
    )


# wrap_lattice_vector_to_replica was moved to kaldo.grid (it is pure SNF
# math with no TDEP-format dependency). Re-export here for back-compat.
from kaldo.grid import wrap_lattice_vector_to_replica  # noqa: E402, F401


# ---------------------------------------------------------------------------
# Shared helpers for TDEP non-diagonal observable loaders
# ---------------------------------------------------------------------------

# def validate_tdep_supercell_matrix(supercell_matrix, M_inferred, supercell, tol=1e-4):
#     """Cross-check a user-supplied ``supercell_matrix`` against ssposcar.

#     Returns ``M_int`` (integer ``np.ndarray``, shape (3, 3)) when valid.
#     Raises ``ValueError`` on:
#       * ``supercell_matrix`` not integer-valued (within ``tol``)
#       * mismatch with the M inferred from ``ucposcar`` / ``ssposcar``

#     If ``supercell_matrix`` is ``None``, validates the **diagonal** path
#     instead: M must be diagonal and its diagonal must match ``supercell``.
#     Returns ``None`` to signal "use the diagonal path".
#     """
#     if supercell_matrix is None:
#         M_diag = np.diag(np.diag(M_inferred))
#         if not np.allclose(M_inferred - M_diag, 0.0, atol=1e-6):
#             raise ValueError(
#                 "format='tdep' requires a diagonal primitive-to-supercell"
#                 " mapping, but the ssposcar is non-diagonal:\n"
#                 f"  M = ucposcar^-1 * ssposcar =\n{M_inferred}\n"
#                 "Pass supercell_matrix= (a 3x3 integer matrix) to enable"
#                 " non-diagonal SNF support."
#             )
#         expected_diag = np.array(supercell, dtype=float)
#         if not np.allclose(np.diag(M_inferred), expected_diag, atol=1e-6):
#             raise ValueError(
#                 f"format='tdep' supercell={tuple(supercell)} does not match"
#                 f" the diagonal tiling M={np.diag(M_inferred).astype(int).tolist()}"
#                 " implied by ucposcar/ssposcar. Pass the matching supercell"
#                 " tuple."
#             )
#         return None

#     M_given = np.asarray(supercell_matrix, dtype=float)
#     M_given_round = np.round(M_given)
#     if not np.allclose(M_given, M_given_round, atol=tol):
#         raise ValueError(
#             f"supercell_matrix must be integer-valued, got\n{M_given}"
#         )
#     if not np.allclose(M_given_round, M_inferred, atol=tol):
#         raise ValueError(
#             f"supercell_matrix does not match ucposcar->ssposcar"
#             f" mapping:\n given M=\n{M_given_round.astype(int)}\n"
#             f" inferred M=\n{M_inferred}"
#         )
#     return M_given_round.astype(int)


def build_nondiag_observable_kwargs(uc, sc):
    """Build the shared kwargs SecondOrder/ThirdOrder/FourthOrder need to
    construct themselves on a non-diagonal SNF replica mapping.

    Returns a dict with keys::

        {
          "atoms": uc,
          "replicated_positions": (n_rep * n_uc, 3) Cartesian,
          "supercell": (n_rep, 1, 1)  (linearized; the real M is in mapping),
          "folder": ...,                # caller fills
          "grid": NonDiagonalGrid(...),
          "_mapping": <SNF mapping dict>,
        }

    Caller appends ``value=`` plus any observable-specific kwargs
    (``is_acoustic_sum`` for SecondOrder, etc.), constructs the observable,
    then attaches ``_snf_mapping`` / ``_supercell_matrix`` / ``_replica_table``
    metadata via :func:`attach_snf_metadata` below.
    """
    from kaldo.grid import NonDiagonalGrid
    mapping = build_supercell_replica_mapping(uc, sc)
    nd_grid = NonDiagonalGrid(
        replica_table=mapping["replica_table"], M=mapping["M"],
    )
    rep_pos = (mapping["replica_table"] @ np.asarray(uc.cell))[:, None, :] \
              + np.asarray(uc.positions)[None, :, :]
    n_rep = len(mapping["replica_table"])
    return dict(
        atoms=uc,
        replicated_positions=rep_pos.reshape(-1, 3),
        supercell=(n_rep, 1, 1),
        grid=nd_grid,
        _mapping=mapping,
    )


def attach_snf_metadata(observable, mapping):
    """Stamp the SNF mapping onto a freshly-built observable so downstream
    code (cumulant helpers, future BTE on non-diagonal) can read it back."""
    observable._snf_mapping = mapping
    observable._supercell_matrix = mapping["M"].astype(int)
    observable._replica_table = mapping["replica_table"]
    return observable


# --------------------------------
# Second order force constant method

def parse_tdep_forceconstant(
    fc_file: str = "infile.forceconstants",
    primitive: str | Atoms = "infile.ucposcar",
    supercell: str | Atoms = "infile.ssposcar",
    two_dim: bool = True,
    symmetrize: bool = False,
    eps: float = 1e-13,
    tol: float = 1e-5,
    format: str = "vasp",
    grid: Grid | None = None,
):
    """
    Parse TDEP second-order force constants.

    Two paths, selected by whether ``grid=`` is supplied:

    ``grid`` is None  (default — back-compatible with the historical API)
        Returns force constants via the geometric-match path:
        ``two_dim=True`` returns the ``(3*n_sc, 3*n_sc)`` matrix; otherwise
        the ``(n_sc, n_sc, 3, 3)`` 4D form. Used by
        :func:`kaldo.controllers.sigma2.calculate_sigma2`.

    ``grid`` is not None
        Returns a dense ndarray of shape ``(1, n_uc, 3, n_rep, n_uc, 3)``,
        the kaldo replica-factorized storage convention. ``grid`` may be
        :class:`Grid` (diagonal supercell) or :class:`NonDiagonalGrid`
        (SNF non-diagonal supercell). Used by
        :class:`SecondOrder.from_folder`. When this branch is taken
        ``supercell``, ``two_dim``, and ``symmetrize`` are unused.

    Parameters
    ----------
    fc_file : str, optional
        Path to the force constant file.
    primitive : str or Atoms, optional
        Path to the primitive cell file or an Atoms object.
    supercell : str or Atoms, optional
        Path to the supercell file or an Atoms object. Only used when
        ``grid is None``.
    two_dim : bool, optional
        Geometric-match path only. If True, returns the
        ``(3*n_sc, 3*n_sc)`` matrix. Default: True.
    symmetrize : bool, optional
        Geometric-match path only. Forwarded to
        :func:`remap_force_constants`. Default: False.
    eps, tol : float, optional
        Numerical tolerances for ASE ``wrap`` and supercell-position
        matching.
    format : str, optional
        ASE format string for reading structures. Default ``"vasp"``.
    grid : Grid or NonDiagonalGrid or None, optional
        If supplied, route through the kaldo replica-factorized path
        (above). When given, ``supercell``, ``two_dim`` and ``symmetrize``
        are ignored.
    """
    # Load primitive
    if isinstance(primitive, Atoms):
        uc = primitive
    elif Path(primitive).exists():
        uc = ase.io.read(primitive, format=format)
    else:
        raise RuntimeError("Primitive cell missing")

    n_uc = len(uc)

    if grid is not None:
        # --- kaldo replica-factorized path ---
        n_rep = grid.grid_size if hasattr(grid, "grid_size") else int(np.prod(grid.grid_shape))
        tensor = np.zeros((1, n_uc, 3, n_rep, n_uc, 3), dtype=float)
        with open(fc_file) as f:
            na = int(f.readline().split()[0])
            _cutoff = float(f.readline().split()[0])
            if na != n_uc:
                raise AssertionError(
                    f"IFC2 file n_atoms={na} != primitive n_atoms={n_uc}"
                )
            for a1 in range(n_uc):
                n_nbr = int(f.readline().split()[0])
                for _ in range(n_nbr):
                    a2 = int(f.readline().split()[0]) - 1
                    lv = np.array(f.readline().split(), dtype=float)
                    phi = np.array(
                        [f.readline().split() for _ in range(3)], dtype=float,
                    )
                    R = np.round(lv).astype(int)
                    rep_ids = grid.grid_index_to_id(R, is_wrapping=True)
                    if len(rep_ids) == 0:
                        raise ValueError(
                            f"IFC2 entry (a1={a1}, a2={a2}, R={R}) did not"
                            " resolve to any replica in the supplied grid"
                        )
                    rep_id = int(rep_ids[0])
                    tensor[0, a1, :, rep_id, a2, :] += phi
        return tensor

    # --- grid is None: geometric-match path (sigma2 / historical) ---
    if isinstance(supercell, Atoms):
        sc = supercell
    elif Path(supercell).exists():
        sc = ase.io.read(supercell, format=format)
    else:
        raise RuntimeError("Supercell missing")

    uc.wrap(eps=tol)
    sc.wrap(eps=tol)
    n_sc = len(sc)

    force_constants = np.zeros((n_uc, n_sc, 3, 3))

    with open(fc_file) as file:
        n_atoms = int(file.readline().split()[0])
        _cutoff = float(file.readline().split()[0])
        if n_atoms != n_uc:
            raise AssertionError(f"n_atoms == {n_atoms}, should be {n_uc}")

        for i1 in range(n_atoms):
            n_neighbors = int(file.readline().split()[0])
            for _ in range(n_neighbors):
                i2 = int(file.readline().split()[0]) - 1
                lp = np.array(file.readline().split(), dtype=float)
                phi = np.array(
                    [file.readline().split() for _ in range(3)],
                    dtype=float,
                )
                r_target = uc.positions[i2] + np.dot(lp, uc.cell[:])
                for ii, r1 in enumerate(sc.positions):
                    r_diff = np.abs(r_target - r1)
                    sc_cell = sc.get_cell(complete=True)
                    r_diff = np.linalg.solve(sc_cell.T, r_diff.T).T % 1.0
                    r_diff -= np.floor(r_diff + eps)
                    if np.sum(r_diff) < tol:
                        force_constants[i1, ii, :, :] += phi

    force_constants = remap_force_constants(
        force_constants, uc, sc, symmetrize=symmetrize,
    )
    if two_dim:
        return force_constants.swapaxes(2, 1).reshape(2 * (3 * n_sc,))
    return force_constants


# remap_force_constants is reachable only via parse_tdep_forceconstant(grid=None,
# two_dim=True) — the kaldo.controllers.sigma2 path. It maps an
# (n_prim, n_sc, 3, 3) IFC2 onto the full supercell-supercell (n_sc, n_sc, 3, 3)
# array, then optionally flattens to (3*n_sc, 3*n_sc) for sigma2.
def remap_force_constants(
    force_constants: NDArray,
    primitive: Atoms,
    supercell: Atoms,
    new_supercell: Atoms = None,
    two_dim: bool = False,
    symmetrize: bool = True,
    tol: float = 1e-5,
    eps: float = 1e-13,
) -> NDArray:
    """
    Remap force constants from [N_prim, N_sc, 3, 3] to [N_sc, N_sc, 3, 3].

    Parameters
    ----------
    force_constants : NDArray
        Force constants in shape [N_prim, N_sc, 3, 3]
    primitive : Atoms
        Primitive cell for reference
    supercell : Atoms
        Supercell for reference
    new_supercell : Atoms, optional
        Supercell to map to. Default: None (uses supercell copy)
    two_dim : bool, optional
        If True, return in [3*N_sc, 3*N_sc] shape. Default: False
    symmetrize : bool, optional
        Make force constants symmetric. Default: True
    tol : float, optional
        Tolerance to discern pairs. Default: 1e-5
    eps : float, optional
        Finite zero tolerance. Default: 1e-13

    Returns
    -------
    NDArray
        The remapped force constants
    """
    if new_supercell is None:
        new_supercell = supercell.copy()

    primitive_cell = primitive.cell.copy()
    primitive.cell = supercell.cell

    primitive.wrap(eps=tol)
    supercell.wrap(eps=tol)

    n_sc_new = len(new_supercell)

    # Calculate distance vectors for each primitive atom
    sc_r = np.zeros((force_constants.shape[0], force_constants.shape[1], 3))
    for aa, a1 in enumerate(primitive):
        diff = supercell.positions - a1.position
        p2s = np.where(np.linalg.norm(diff, axis=1) < tol)[0][0]
        spos = supercell.positions
        sc_r[aa], _ = get_distances(
            [spos[p2s]], spos, cell=supercell.cell, pbc=True
        )

    primitive.cell = primitive_cell
    map2prim = _map2prim(primitive, new_supercell)

    ref_struct_pos = new_supercell.get_scaled_positions(wrap=True)
    sc_cell = new_supercell.get_cell(complete=True)

    fc_out = np.zeros((n_sc_new, n_sc_new, 3, 3))

    # Remap force constants to new supercell
    for a1, (r0, uc_index) in enumerate(zip(new_supercell.positions, map2prim)):
        for sc_a2, sc_r2 in enumerate(sc_r[uc_index]):
            r_pair = r0 + sc_r2
            r_pair = np.linalg.solve(sc_cell.T, r_pair.T).T % 1.0

            r_diff = np.abs(r_pair - ref_struct_pos)

            # Treat positions near 1.0 as equivalent to 0.0.
            r_diff -= np.floor(r_diff + eps)

            norms = np.linalg.norm(r_diff, axis=1)
            matches = np.where(norms < tol)[0]

            if len(matches) != 1:
                raise ValueError(
                    f"Failed to remap IFCs. Expected exactly one match, found {len(matches)} "
                    f"for a1={a1}, uc_index={uc_index}, sc_a2={sc_a2}. "
                    f"Minimum distance = {norms.min()}"
                )

            fc_out[a1, matches[0], :, :] += force_constants[
                uc_index, sc_a2, :, :
            ]

    # Convert to 2D format if requested
    if two_dim:
        fc_out = fc_out.swapaxes(1, 2).reshape(2 * (3 * fc_out.shape[1],))

        # Check symmetry
        violation = np.linalg.norm(fc_out - fc_out.T)
        if violation > 1e-5:
            logging.warning(
                f"Force constants are not symmetric by {violation:.2e}."
            )
            if symmetrize:
                logging.info("Symmetrize force constants.")
                fc_out = 0.5 * (fc_out + fc_out.T)

        # Check sum rules
        violation = abs(fc_out.sum(axis=0)).mean()
        if violation > 1e-9:
            logging.warning(f"Sum rule violated by {violation:.2e} (axis 1).")

        violation = abs(fc_out.sum(axis=1)).mean()
        if violation > 1e-9:
            logging.warning(f"Sum rule violated by {violation:.2e} (axis 2).")

        return fc_out

    return fc_out


def _map2prim(primitive: Atoms, supercell: Atoms, tol: float = 1e-5) -> list:
    """
    Create mapping from supercell atoms to primitive cell atoms.

    Parameters
    ----------
    primitive : Atoms
        Primitive cell
    supercell : Atoms
        Supercell
    tol : float, optional
        Distance tolerance for atom matching. Default: 1e-5

    Returns
    -------
    list
        Map from supercell to primitive cell indices
    """
    map2prim = []
    primitive = primitive.copy()
    supercell = supercell.copy()

    supercell_with_prim_cell = supercell.copy()
    supercell_with_prim_cell.cell = primitive.cell.copy()

    primitive.wrap(eps=tol)
    supercell_with_prim_cell.wrap(eps=tol)

    for a1 in supercell_with_prim_cell:
        diff = primitive.positions - a1.position
        map2prim.append(np.where(np.linalg.norm(diff, axis=1) < tol)[0][0])

    _, counts = np.unique(map2prim, return_counts=True)
    if counts.std() != 0:
        raise AssertionError(f"Inconsistent mapping counts: {counts}")

    return map2prim


# --------------------------------
# Third order force constant method

def parse_tdep_third_forceconstant(
    fc_filename: str,
    primitive: str | Atoms,
    supercell: tuple[int, int, int] | None = None,
    grid: Grid | None = None,
):
    """Parse TDEP third-order force constants.

    Parameters
    ----------
    fc_filename : str
        Path to ``infile.forceconstant_thirdorder``.
    primitive : str or Atoms
        Path to ``infile.ucposcar`` (or an ASE ``Atoms``).
    supercell : tuple[int, int, int], optional
        Diagonal supercell tiling. Builds ``Grid(supercell, order='C')``
        internally. Mutually exclusive with ``grid=``.
    grid : Grid or NonDiagonalGrid, optional
        Replica grid. Use this to pass a ``NonDiagonalGrid`` for SNF
        non-diagonal tilings, or a pre-built ``Grid`` for the diagonal
        case. Mutually exclusive with ``supercell=``.

    Returns
    -------
    sparse.COO
        Shape ``(n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)``.
    """
    if (supercell is None) == (grid is None):
        raise ValueError(
            "parse_tdep_third_forceconstant requires exactly one of"
            " supercell= or grid="
        )
    if grid is None:
        grid = Grid(supercell, order="C")
    if isinstance(primitive, Atoms):
        uc = primitive
    else:
        uc = ase.io.read(primitive, format="vasp")
    n_uc = len(uc)
    n_rep = grid.grid_size if hasattr(grid, "grid_size") else int(np.prod(grid.grid_shape))

    print(n_rep)

    dense = np.zeros(
        (n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3), dtype=float,
    )

    with open(fc_filename) as f:
        na = int(f.readline().split()[0])
        _cutoff = float(f.readline().split()[0])
        if na != n_uc:
            raise AssertionError(
                f"IFC3 file n_atoms={na} != primitive n_atoms={n_uc}"
            )
        for a1 in range(n_uc):
            n_trips = int(f.readline().split()[0])
            for _ in range(n_trips):
                i1 = int(f.readline().split()[0]) - 1
                a2 = int(f.readline().split()[0]) - 1
                a3 = int(f.readline().split()[0]) - 1
                if i1 != a1:
                    raise ValueError(
                        f"IFC3 record at outer atom {a1} has central index"
                        f" i1={i1} (expected {a1}); file is malformed."
                    )
                _lv1 = np.array(f.readline().split(), dtype=float)
                if not np.allclose(_lv1, 0.0, atol=1e-6):
                    raise ValueError(
                        f"IFC3 R1 lattice vector for central atom {a1} is"
                        f" {_lv1} (expected [0,0,0]); file is malformed."
                    )
                lv2 = np.array(f.readline().split(), dtype=float)
                lv3 = np.array(f.readline().split(), dtype=float)
                flat = np.empty(27)
                idx = 0
                while idx < 27:
                    for tok in f.readline().split():
                        flat[idx] = float(tok)
                        idx += 1
                        if idx >= 27:
                            break
                phi = flat.reshape(3, 3, 3)

                R2 = np.round(lv2).astype(int)
                R3 = np.round(lv3).astype(int)
                r2_ids = grid.grid_index_to_id(R2, is_wrapping=True)
                r3_ids = grid.grid_index_to_id(R3, is_wrapping=True)
                if len(r2_ids) == 0 or len(r3_ids) == 0:
                    raise ValueError(
                        f"IFC3 triplet (a1={a1}, a2={a2}, a3={a3},"
                        f" R2={R2}, R3={R3}) did not resolve in the grid"
                    )
                r2_id = int(r2_ids[0])
                r3_id = int(r3_ids[0])
                dense[a1, :, r2_id, a2, :, r3_id, a3, :] += phi

    return COO.from_numpy(dense)


# --------------------------------
# Fourth order force constant method

def parse_tdep_fourth_forceconstant(
    fc_filename: str,
    primitive: str | Atoms,
    supercell: tuple[int, int, int] | None = None,
    grid: Grid | None = None,
):
    """Parse TDEP fourth-order force constants.

    Reads ``infile.forceconstant_fourthorder`` and returns a sparse rank-11
    COO tensor in kaldo's storage convention:

      shape = (n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)

    Parameters
    ----------
    fc_filename : str
        Path to ``infile.forceconstant_fourthorder``.
    primitive : str or Atoms
    supercell : tuple[int, int, int], optional
        Diagonal supercell tiling. Builds ``Grid(supercell, order='C')``
        internally. Mutually exclusive with ``grid=``.
    grid : Grid or NonDiagonalGrid, optional
        Replica grid. Use this to pass a ``NonDiagonalGrid`` for SNF
        non-diagonal tilings, or a pre-built ``Grid`` for the diagonal
        case. Mutually exclusive with ``supercell=``.
    """
    if (supercell is None) == (grid is None):
        raise ValueError(
            "parse_tdep_fourth_forceconstant requires exactly one of"
            " supercell= or grid="
        )
    if grid is None:
        grid = Grid(supercell, order="C")
    if isinstance(primitive, Atoms):
        uc = primitive
    else:
        uc = ase.io.read(primitive, format='vasp')
    n_uc = len(uc)
    n_rep = grid.grid_size if hasattr(grid, "grid_size") else int(np.prod(grid.grid_shape))

    shape = (
        n_uc, 3, n_rep,
        n_uc, 3, n_rep,
        n_uc, 3, n_rep,
        n_uc, 3,
    )

    def _read_vec3(fh):
        return np.array(fh.readline().split(), dtype=float)

    def _read_phi4(fh):
        flat = np.empty(81)
        idx = 0
        while idx < 81:
            for tok in fh.readline().split():
                flat[idx] = float(tok)
                idx += 1
                if idx >= 81:
                    break
        return flat.reshape(3, 3, 3, 3)

    dense = np.zeros(shape, dtype=float)
    with open(fc_filename) as fh:
        na = int(fh.readline().split()[0])
        _cutoff = float(fh.readline().split()[0])
        if na != n_uc:
            raise AssertionError(
                f"IFC4 file n_atoms={na} != n_uc={n_uc}"
            )
        for a1 in range(n_uc):
            n_quartets = int(fh.readline().split()[0])
            for _ in range(n_quartets):
                i1 = int(fh.readline().split()[0]) - 1
                i2 = int(fh.readline().split()[0]) - 1
                i3 = int(fh.readline().split()[0]) - 1
                i4 = int(fh.readline().split()[0]) - 1
                if i1 != a1:
                    raise ValueError(
                        f"IFC4 record at outer atom {a1} has central index"
                        f" i1={i1} (expected {a1}); file is malformed."
                    )
                _R1 = _read_vec3(fh)
                if not np.allclose(_R1, 0.0, atol=1e-6):
                    raise ValueError(
                        f"IFC4 R1 lattice vector for central atom {a1} is"
                        f" {_R1} (expected [0,0,0]); file is malformed."
                    )
                R2 = _read_vec3(fh)
                R3 = _read_vec3(fh)
                R4 = _read_vec3(fh)
                phi = _read_phi4(fh)

                R2_int = np.round(R2).astype(int)
                R3_int = np.round(R3).astype(int)
                R4_int = np.round(R4).astype(int)
                r2_ids = grid.grid_index_to_id(R2_int, is_wrapping=True)
                r3_ids = grid.grid_index_to_id(R3_int, is_wrapping=True)
                r4_ids = grid.grid_index_to_id(R4_int, is_wrapping=True)
                if len(r2_ids) == 0 or len(r3_ids) == 0 or len(r4_ids) == 0:
                    raise ValueError(
                        f"IFC4 quartet (a1={a1}, a2={i2}, a3={i3}, a4={i4},"
                        f" R2={R2_int}, R3={R3_int}, R4={R4_int}) did not"
                        " resolve in the grid"
                    )
                r2_id = int(r2_ids[0])
                r3_id = int(r3_ids[0])
                r4_id = int(r4_ids[0])
                dense[
                    a1, :, r2_id, i2, :, r3_id, i3, :, r4_id, i4, :,
                ] += phi

    return COO.from_numpy(dense)

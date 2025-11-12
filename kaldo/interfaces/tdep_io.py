from ase import Atoms
import ase.io
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from ase.geometry import get_distances
from kaldo.helpers.logger import get_logger
from kaldo.grid import Grid
from sparse import COO
from kaldo.interfaces.common import ForceConstantData, ensure_replicas

logging = get_logger()


# --------------------------------
# Second order force constant method

def parse_tdep_forceconstant(
    fc_file: str = "infile.forceconstants",
    primitive: str = "infile.ucposcar",
    supercell: str = "infile.ssposcar",
    fortran: bool = True,
    two_dim: bool = True,
    symmetrize: bool = False,
    reduce_fc: bool = True,
    eps: float = 1e-13,
    tol: float = 1e-5,
    format: str = "vasp",
):
    """
    Parse TDEP second order force constants.

    Parameters
    ----------
    fc_file : str, optional
        Path to the force constant file. Default: "infile.forceconstants"
    primitive : str or Atoms, optional
        Path to primitive cell file or Atoms object. Default: "infile.ucposcar"
    supercell : str or Atoms, optional
        Path to supercell file or Atoms object. Default: "infile.ssposcar"
    fortran : bool, optional
        Unused parameter for compatibility. Default: True
    two_dim : bool, optional
        If True, return 2D array shape (3*N_sc, 3*N_sc). Default: True
    symmetrize : bool, optional
        If True, symmetrize force constants. Default: False
    reduce_fc : bool, optional
        If True, return reduced shape. Default: True
    eps : float, optional
        Finite zero tolerance. Default: 1e-13
    tol : float, optional
        Distance tolerance for atom matching. Default: 1e-5
    format : str, optional
        File format for reading structures. Default: "vasp"

    Returns
    -------
    np.ndarray
        Force constants in requested shape.
    """
    # Load or validate primitive cell
    if isinstance(primitive, Atoms):
        uc = primitive
    elif Path(primitive).exists():
        uc = ase.io.read(primitive, format=format)
    else:
        raise RuntimeError("Primitive cell missing")

    # Load or validate supercell
    if isinstance(supercell, Atoms):
        sc = supercell
    elif Path(supercell).exists():
        sc = ase.io.read(supercell, format=format)
    else:
        raise RuntimeError("Supercell missing")

    uc.wrap(eps=tol)
    sc.wrap(eps=tol)
    n_uc = len(uc)
    n_sc = len(sc)

    force_constants = np.zeros((n_uc, n_sc, 3, 3))

    # Parse force constant file
    with open(fc_file) as file:
        n_atoms = int(file.readline().split()[0])
        cutoff = float(file.readline().split()[0])

        if n_atoms != n_uc:
            raise AssertionError(f"n_atoms == {n_atoms}, should be {n_uc}")

        for i1 in range(n_atoms):
            n_neighbors = int(file.readline().split()[0])
            for _ in range(n_neighbors):
                i2 = int(file.readline().split()[0]) - 1
                lp = np.array(file.readline().split(), dtype=float)
                phi = np.array(
                    [file.readline().split() for _ in range(3)],
                    dtype=float
                )
                r_target = uc.positions[i2] + np.dot(lp, uc.cell[:])

                # Find matching atom in supercell
                for ii, r1 in enumerate(sc.positions):
                    r_diff = np.abs(r_target - r1)
                    sc_cell = sc.get_cell(complete=True)
                    r_diff = np.linalg.solve(sc_cell.T, r_diff.T).T % 1.0
                    r_diff -= np.floor(r_diff + eps)
                    if np.sum(r_diff) < tol:
                        force_constants[i1, ii, :, :] += phi

    # Remap if needed
    if not reduce_fc or two_dim:
        force_constants = remap_force_constants(
            force_constants, uc, sc, symmetrize=symmetrize
        )

    # Return in 2D format if requested
    if two_dim:
        return force_constants.swapaxes(2, 1).reshape(2 * (3 * n_sc,))

    return force_constants


# TODO: remap_force_constants and reduce_force_constants functions are mainly from vibes.
# needs to check original code
def remap_force_constants(
    force_constants: NDArray,
    primitive: Atoms,
    supercell: Atoms,
    new_supercell: Atoms = None,
    reduce_fc: bool = False,
    two_dim: bool = False,
    symmetrize: bool = True,
    tol: float = 1e-5,
    eps: float = 1e-13,
) -> NDArray:
    """
    Remap force constants from [N_prim, N_sc, 3, 3] to [N_sc, N_sc, 3, 3].

    Note: This function mostly follows vibes.force_constants.py from Vibes library.

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
    reduce_fc : bool, optional
        If True, return in [N_prim, N_sc, 3, 3] shape. Default: False
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
            r_diff -= np.floor(r_diff + eps)

            norms = np.linalg.norm(r_diff, axis=1)
            below_tolerance = np.where(norms < tol)

            fc_out[a1, below_tolerance, :, :] += force_constants[
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

    # Reduce to primitive representation if requested
    if reduce_fc:
        p2s_map = np.zeros(len(primitive), dtype=int)

        primitive.cell = new_supercell.cell

        new_supercell.wrap(eps=tol)
        primitive.wrap(eps=tol)

        for aa, a1 in enumerate(primitive):
            diff = new_supercell.positions - a1.position
            p2s_map[aa] = np.where(np.linalg.norm(diff, axis=1) < tol)[0][0]

        primitive.cell = primitive_cell
        primitive.wrap(eps=tol)

        return reduce_force_constants(fc_out, p2s_map)

    return fc_out


def reduce_force_constants(fc_full: NDArray, map2prim: NDArray) -> NDArray:
    """
    Reduce force constants from [N_sc, N_sc, 3, 3] to [N_prim, N_sc, 3, 3].

    Parameters
    ----------
    fc_full : NDArray
        The non-reduced force constant matrix
    map2prim : NDArray
        Map from supercell to unitcell index

    Returns
    -------
    NDArray
        The reduced force constants
    """
    _, uc_index = np.unique(map2prim, return_index=True)
    fc_out = np.zeros((len(uc_index), fc_full.shape[1], 3, 3))
    for ii, uc_ind in enumerate(uc_index):
        fc_out[ii, :, :, :] = fc_full[uc_ind, :, :, :]

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
    primitive: str,
    supercell: tuple[int, int, int],
):
    """
    Parse TDEP third order force constants.

    Parameters
    ----------
    fc_filename : str
        Path to the third order force constant file
    primitive : str
        Path to the primitive cell file
    supercell : tuple[int, int, int]
        Supercell dimensions

    Returns
    -------
    COO
        Sparse third order force constants tensor
    """
    uc = ase.io.read(primitive, format='vasp')
    n_unit_atoms = uc.positions.shape[0]
    n_replicas = np.prod(supercell)
    order = 'C'

    current_grid = Grid(supercell, order=order)

    # Read file and parse header
    with open(fc_filename, 'r') as file:
        line = file.readline()
        num1 = int(line.split()[0])
        line = file.readline()
        lines = file.readlines()

    # Parse triplet structure
    num_triplets = []
    new_ind = 0
    count = 0
    if count == 0:
        n_t = int(lines[0].split()[0])
        num_triplets.append(n_t)
        new_ind += int(n_t * 15 + 1)
        count += 1
    while count != 0 and new_ind < len(lines):
        n_t = int(lines[new_ind].split()[0])
        num_triplets.append(n_t)
        new_ind += int(n_t * 15 + 1)

    coords = []
    frcs = np.zeros((
        n_unit_atoms, 3, n_replicas,
        n_unit_atoms, 3, n_replicas,
        n_unit_atoms, 3
    ))

    # Parse force constant triplets
    for count1 in range(num1):
        for j in range(len(num_triplets)):
            n_trip = num_triplets[j]
            lower = sum(int(num_triplets[i] * 15 + 1) for i in range(j))
            upper = lower + int(n_trip * 15 + 1)
            subset = lines[lower:upper][1:]
            num2 = int(len(subset) / 15)

            for count2 in range(num2):
                lower2 = int(count2 * 15)
                upper2 = int((count2 + 1) * 15)
                ssubset = subset[lower2:upper2]

                # Parse atom indices
                atom_i = int(ssubset[0].split()[0]) - 1
                atom_j = int(ssubset[1].split()[0]) - 1
                atom_k = int(ssubset[2].split()[0]) - 1

                # Parse lattice vectors (R1 unused)
                R2 = np.array(ssubset[4].split(), dtype=float)
                R3 = np.array(ssubset[5].split(), dtype=float)

                # Parse 3x3 phi matrix
                phi = np.array([
                    [ssubset[6].split(), ssubset[7].split(), ssubset[8].split()],
                    [ssubset[9].split(), ssubset[10].split(), ssubset[11].split()],
                    [ssubset[12].split(), ssubset[13].split(), ssubset[14].split()]
                ], dtype=float)

                second_cell_id = current_grid.grid_index_to_id(
                    R2, is_wrapping=True
                )
                third_cell_id = current_grid.grid_index_to_id(
                    R3, is_wrapping=True
                )

                # Store force constants
                for alpha in range(3):
                    for beta in range(3):
                        for gamma in range(3):
                            frcs[
                                atom_i, alpha, second_cell_id[0],
                                atom_j, beta, third_cell_id[0],
                                atom_k, gamma
                            ] = phi[alpha, beta, gamma]

    # Build sparse array
    sparse_frcs = []
    for n1 in range(n_unit_atoms):
        for a in range(3):
            for nr1 in range(n_replicas):
                for n2 in range(n_unit_atoms):
                    for b in range(3):
                        for nr2 in range(n_replicas):
                            for n3 in range(n_unit_atoms):
                                for c in range(3):
                                    coords.append((n1, a, nr1, n2, b, nr2, n3, c))
                                    sparse_frcs.append(
                                        frcs[n1, a, nr1, n2, b, nr2, n3, c]
                                    )

    third_ifcs = COO(
        np.array(coords).T,
        np.array(sparse_frcs),
        shape=(
            n_unit_atoms, 3, n_replicas,
            n_unit_atoms, 3, n_replicas,
            n_unit_atoms, 3
        )
    )

    third_ifcs.reshape((
        n_unit_atoms * 3,
        n_replicas * n_unit_atoms * 3,
        n_replicas * n_unit_atoms * 3
    ))

    return third_ifcs


def load_second_tdep(*, folder: Path, resolved, filename: str = "infile.forceconstant", **_) -> ForceConstantData:
    d2 = parse_tdep_forceconstant(
        fc_file=str(folder / filename),
        primitive=str(folder / "infile.ucposcar"),
        supercell=str(folder / "infile.ssposcar"),
        reduce_fc=False,
    )
    n_unit = resolved.unit_atoms.positions.shape[0]
    n_rep = int(np.prod(resolved.supercell))
    value = d2.reshape((n_rep, n_unit, 3, n_rep, n_unit, 3))[0, np.newaxis, ...]
    replicas = ensure_replicas(resolved, folder, ("infile.ssposcar",))
    return ForceConstantData(
        order=2,
        value=value,
        unit_atoms=resolved.unit_atoms,
        supercell=resolved.supercell,
        replicated_atoms=replicas,
    )


def load_third_tdep(*, folder: Path, resolved, filename: str = "infile.forceconstant_thirdorder", **_) -> ForceConstantData:
    raw = parse_tdep_third_forceconstant(
        fc_filename=str(folder / filename),
        primitive=str(folder / "infile.ucposcar"),
        supercell=resolved.supercell,
    )
    n_unit = resolved.unit_atoms.positions.shape[0]
    n_rep = int(np.prod(resolved.supercell))
    value = raw.reshape((3 * n_unit, 3 * n_rep * n_unit, 3 * n_rep * n_unit))
    replicas = ensure_replicas(resolved, folder, ("infile.ssposcar",))
    return ForceConstantData(
        order=3,
        value=value,
        unit_atoms=resolved.unit_atoms,
        supercell=resolved.supercell,
        replicated_atoms=replicas,
    )

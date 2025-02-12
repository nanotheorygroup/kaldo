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
    """Parse tdep second order force constant.
    """

    if isinstance(primitive, Atoms):
        uc = primitive
    elif Path(primitive).exists():
        uc = ase.io.read(primitive, format=format)
    else:
        raise RuntimeError("primitive cell missing")

    if isinstance(supercell, Atoms):
        sc = supercell
    elif Path(supercell).exists():
        sc = ase.io.read(supercell, format=format)
    else:
        raise RuntimeError("supercell missing")

    uc.wrap(eps=tol)
    sc.wrap(eps=tol)
    n_uc = len(uc)
    n_sc = len(sc)

    force_constants = np.zeros((len(uc), len(sc), 3, 3))

    with open(fc_file) as file:
        n_atoms = int(file.readline().split()[0])
        cutoff = float(file.readline().split()[0])

        assert n_atoms == n_uc, f"n_atoms == {n_atoms}, should be {n_uc}"

        for i1 in range(n_atoms):
            n_neighbors = int(file.readline().split()[0])
            for _ in range(n_neighbors):
                i2 = int(file.readline().split()[0]) - 1
                lp = np.array(file.readline().split(), dtype=float)
                phi = np.array([file.readline().split() for _ in range(3)], dtype=float)
                r_target = uc.positions[i2] + np.dot(lp, uc.cell[:])
                for ii, r1 in enumerate(sc.positions):
                    r_diff = np.abs(r_target - r1)
                    sc_temp = sc.get_cell(complete=True)
                    r_diff = np.linalg.solve(sc_temp.T, r_diff.T).T % 1.0
                    r_diff -= np.floor(r_diff + eps)
                    if np.sum(r_diff) < tol:
                        force_constants[i1, ii, :, :] += phi

    if not reduce_fc or two_dim:
        force_constants = remap_force_constants(force_constants, uc, sc, symmetrize=symmetrize)

    if two_dim:
        return force_constants.swapaxes(2, 1).reshape(2 * (3 * n_sc,))

    return force_constants


# TODO: remap_force_constants and reduce_force_constants functions are mainly from vibes. needs to check original code
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
    remap force constants [N_prim, N_sc, 3, 3] to [N_sc, N_sc, 3, 3]
    Note: This function mostly follows vibes.force_constants.py from Vibes library.

    Args:
    ----
        force_constants: force constants in [N_prim, N_sc, 3, 3] shape
        primitive: primitive cell for reference
        supercell: supercell for reference
        new_supercell: supercell to map to
        reduce_fc: return in [N_prim, N_sc, 3, 3]  shape
        two_dim: return in [3*N_sc, 3*N_sc] shape
        symmetrize: make force constants symmetric
        tol: tolerance to discern pairs
        eps: finite zero

    Returns:
    -------
        The remapped force constants

    """

    if new_supercell is None:
        new_supercell = supercell.copy()

    primitive_cell = primitive.cell.copy()
    primitive.cell = supercell.cell

    primitive.wrap(eps=tol)
    supercell.wrap(eps=tol)

    n_sc_new = len(new_supercell)

    sc_r = np.zeros((force_constants.shape[0], force_constants.shape[1], 3))
    for aa, a1 in enumerate(primitive):
        diff = supercell.positions - a1.position
        p2s = np.where(np.linalg.norm(diff, axis=1) < tol)[0][0]
        spos = supercell.positions
        sc_r[aa], _ = get_distances([spos[p2s]], spos, cell=supercell.cell, pbc=True)

    primitive.cell = primitive_cell
    map2prim = _map2prim(primitive, new_supercell)

    ref_struct_pos = new_supercell.get_scaled_positions(wrap=True)
    sc_temp = new_supercell.get_cell(complete=True)

    fc_out = np.zeros((n_sc_new, n_sc_new, 3, 3))

    for a1, (r0, uc_index) in enumerate(zip(new_supercell.positions, map2prim)):

        for sc_a2, sc_r2 in enumerate(sc_r[uc_index]):

            r_pair = r0 + sc_r2
            r_pair = np.linalg.solve(sc_temp.T, r_pair.T).T % 1.0

            r_diff = np.abs(r_pair - ref_struct_pos)
            r_diff -= np.floor(r_diff + eps)

            norms = np.linalg.norm(r_diff, axis=1)
            below_tolerance = np.where(norms < tol)

            fc_out[a1, below_tolerance, :, :] += force_constants[uc_index, sc_a2, :, :]

    if two_dim:
        fc_out = fc_out.swapaxes(1, 2).reshape(2 * (3 * fc_out.shape[1],))

        violation = np.linalg.norm(fc_out - fc_out.T)
        if violation > 1e-5:
            logging.warning(f"Force constants are not symmetric by {violation:.2e}.")
            if symmetrize:
                logging.info("Symmetrize force constants.")
                fc_out = 0.5 * (fc_out + fc_out.T)

        violation = abs(fc_out.sum(axis=0)).mean()
        if violation > 1e-9:
            logging.warning(f"Sum rule violated by {violation:.2e} (axis 1).")

        violation = abs(fc_out.sum(axis=1)).mean()
        if violation > 1e-9:
            logging.warning(f"Sum rule violated by {violation:.2e} (axis 2).")

        return fc_out

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


def reduce_force_constants(fc_full: NDArray, map2prim: NDArray):
    """
    reduce force constants from [N_sc, N_sc, 3, 3] to [N_prim, N_sc, 3, 3]

    Args:
    ----
        fc_full: The non-reduced force constant matrix
        map2prim: map from supercell to unitcell index

    Returns:
    -------
        The reduced force constants

    """
    _, uc_index = np.unique(map2prim, return_index=True)
    fc_out = np.zeros((len(uc_index), fc_full.shape[1], 3, 3))
    for ii, uc_ind in enumerate(uc_index):
        fc_out[ii, :, :, :] = fc_full[uc_ind, :, :, :]

    return fc_out


def _map2prim(primitive, supercell, tol=1e-5):
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
    assert counts.std() == 0, counts

    return map2prim


# --------------------------------
# Third order force constant method

def parse_tdep_third_forceconstant(
    fc_filename: str,
    primitive: str,
    supercell: tuple[int, int, int],
):
    """Parse tdep third order force constant.
    """
    uc = ase.io.read(primitive, format='vasp')
    n_unit_atoms = uc.positions.shape[0]
    n_replicas = np.prod(supercell)
    order = 'C'

    second_cell_list = []
    third_cell_list = []

    current_grid = Grid(supercell, order=order).grid(is_wrapping=True)
    list_of_index = current_grid
    list_of_replicas = list_of_index.dot(uc.cell)

    with open(fc_filename, 'r') as file:
        line = file.readline()
        num1 = int(line.split()[0])
        line = file.readline()
        lines = file.readlines()

    num_triplets = []
    new_ind = 0
    count = 0
    if count == 0:
        n_t = int(lines[0].split()[0])
        num_triplets.append(n_t)
        new_ind += int(n_t*15+1)
        count += 1
    while count != 0 and new_ind < len(lines):
        n_t = int(lines[new_ind].split()[0])
        num_triplets.append(n_t)
        new_ind += int(n_t * 15 + 1)

    coords = []
    frcs = np.zeros((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))

    for count1 in range(num1):
        for j in range(len(num_triplets)):
            n_trip = num_triplets[j]
            lower = 0
            for i in range(j):
                lower += int(num_triplets[i] * 15 + 1)
            upper = lower + int(n_trip*15+1)
            subset = lines[lower:upper]
            subset = subset[1:]
            num2 = int(len(subset) / 15)
            for count2 in range(num2):
                lower2 = int(count2 * 15)
                upper2 = int((count2 + 1) * 15)
                ssubset = subset[lower2:upper2]
                atom_i = int(ssubset[0].split()[0]) - 1
                atom_j = int(ssubset[1].split()[0]) - 1
                atom_k = int(ssubset[2].split()[0]) - 1
                R1 = np.array(ssubset[3].split(), dtype=float)
                R2 = np.array(ssubset[4].split(), dtype=float)
                R3 = np.array(ssubset[5].split(), dtype=float)
                phi1 = ssubset[6].split()
                phi2 = ssubset[7].split()
                phi3 = ssubset[8].split()
                phi4 = ssubset[9].split()
                phi5 = ssubset[10].split()
                phi6 = ssubset[11].split()
                phi7 = ssubset[12].split()
                phi8 = ssubset[13].split()
                phi9 = ssubset[14].split()
                phi = np.array(
                    [[[phi1[0], phi1[1], phi1[2]], [phi2[0], phi2[1], phi2[2]], [phi3[0], phi3[1], phi3[2]]],
                    [[phi4[0], phi4[1], phi4[2]], [phi5[0], phi5[1], phi5[2]], [phi6[0], phi6[1], phi6[2]]],
                    [[phi7[0], phi7[1], phi7[2]], [phi8[0], phi8[1], phi8[2]], [phi9[0], phi9[1], phi9[2]]]],
                    dtype=float)
                second_cell_list.append(R2)
                # TODO: abstract these code into a function in Grid
                second_cell_id = (list_of_index[:] == R2).prod(axis=1)
                second_cell_id = np.argwhere(second_cell_id).flatten()
                third_cell_list.append(R3)
                third_cell_id = (list_of_index[:] == R3).prod(axis=1)
                third_cell_id = np.argwhere(third_cell_id).flatten()
                for alpha in range(3):
                    for beta in range(3):
                        for gamma in range(3):
                            #coords.append((atom_i, alpha, second_cell_id[0], atom_j, beta, third_cell_id[0], atom_k, gamma))
                            #frcs.append(phi[alpha, beta, gamma])
                            frcs[atom_i, alpha, second_cell_id[0], atom_j, beta, third_cell_id[0], atom_k, gamma] = phi[alpha, beta, gamma]

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
                                    sparse_frcs.append(frcs[n1, a, nr1, n2, b, nr2, n3, c])

    third_ifcs = COO(np.array(coords).T, np.array(sparse_frcs), shape=(n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))

    third_ifcs.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))

    return third_ifcs

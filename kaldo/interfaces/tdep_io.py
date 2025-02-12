from ase import Atoms
import ase.io
import numpy as np
from pathlib import Path
from ase.geometry import get_distances
from kaldo.helpers.logger import get_logger

logging = get_logger()


def parse_tdep_forceconstant(
    fc_file="infile.forceconstants",
    primitive="infile.ucposcar",
    supercell="infile.ssposcar",
    fortran=True,
    two_dim=True,
    symmetrize=False,
    reduce_fc=True,
    eps=1e-13,
    tol=1e-5,
    format="vasp",
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
    force_constants: np.ndarray,
    primitive: Atoms,
    supercell: Atoms,
    new_supercell: Atoms = None,
    reduce_fc: bool = False,
    two_dim: bool = False,
    symmetrize: bool = True,
    tol: float = 1e-5,
    eps: float = 1e-13,
) -> np.ndarray:
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


def reduce_force_constants(fc_full: np.ndarray, map2prim: np.ndarray):
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

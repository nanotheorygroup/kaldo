import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from numpy.typing import NDArray

from kaldo.observables.secondorder import SecondOrder
from kaldo.observables.thirdorder import ThirdOrder
from kaldo.grid import wrap_coordinates

def _match_unique_periodic_position(
    target_cart: NDArray,
    ref_scaled_pos: NDArray,
    cell: NDArray,
    tol: float,
) -> int:
    """
    Find the unique atom in ref_scaled_pos whose periodic position matches target_cart.
    """
    target_frac = np.linalg.solve(cell.T, target_cart.T).T % 1.0

    delta_frac = target_frac - ref_scaled_pos
    delta_frac -= np.round(delta_frac)

    delta_cart = delta_frac @ cell
    norms = np.linalg.norm(delta_cart, axis=1)

    matches = np.flatnonzero(norms < tol)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one periodic position match, found {len(matches)}. "
            f"Closest distance = {norms.min():.6e}"
        )

    return int(matches[0])


def _build_displacement_table(
    primitive: Atoms,
    replicated_atoms : Atoms,
) -> NDArray:
    """
    Build an unflattened displacement table with shape (n_uc, n_rep, n_uc, 3)

    The index [uc_i, R_j, j_uc] is the minimum-image displacement from
    primitive atom uc_i to the reference-supercell atom represented by
    atom j_uc in replica R_j.
    """

    rp = replicated_atoms.positions #  (n_rep, n_uc, 3)
    cell_inv = np.linalg.inv(replicated_atoms.cell)
    cell = replicated_atoms.cell

    n_rep, n_uc, _ = rp.shape

    sc_r = np.zeros((n_uc, *rp.shape))

    for (uc_i, r_i) in enumerate(primitive.positions):
        for rep in range(n_rep):
            for uc_j in range(n_uc):
                dr = rp[rep, uc_j] - r_i
                sc_r[uc_i, rep, uc_j] = wrap_coordinates(dr, cell=cell, cell_inv=cell_inv)

    return sc_r

def _build_translated_index_map(
    sc_r: NDArray,
    map2prim: NDArray,
    new_supercell: Atoms,
    tol: float,
    eps: float = 1e-13,
) -> NDArray:
    """
    Build translated_index array with shape [i_sc, R_j, j_uc] = j_sc.

    sc_r has shape

        (n_prim, n_rep, n_prim, 3)

    and translated_index[i_sc, R_j, j_uc] gives the atom in new_supercell
    located at

        position(i_sc) + sc_r[map2prim[i_sc], R_j, j_uc]
    """

    if sc_r.ndim != 4 or sc_r.shape[-1] != 3:
        raise ValueError(
            "Expected sc_r shape (n_prim, n_rep, n_prim, 3). "
            f"Got {sc_r.shape}."
        )

    n_prim_i, n_rep, n_prim_j, _ = sc_r.shape
    if n_prim_i != n_prim_j:
        raise ValueError(
            "Expected matching primitive dimensions in sc_r. "
            f"Got {n_prim_i} and {n_prim_j}."
        )

    map2prim = np.asarray(map2prim, dtype=np.int64)

    n_sc_new = len(new_supercell)
    ref_scaled_pos = new_supercell.get_scaled_positions(wrap=True)
    cell = np.asarray(new_supercell.get_cell(complete=True))

    translated_index = np.empty(
        (n_sc_new, n_rep, n_prim_j),
        dtype=np.int64,
    )

    for i_sc, (r_i, uc_i) in enumerate(zip(new_supercell.positions, map2prim)):
        for R_j in range(n_rep):
            for j_uc in range(n_prim_j):
                target_cart = r_i + sc_r[uc_i, R_j, j_uc]

                translated_index[i_sc, R_j, j_uc] = _match_unique_periodic_position(
                    target_cart,
                    ref_scaled_pos,
                    cell,
                    tol=tol,
                )

    return translated_index


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


def remap_raw_second_force_constants(
    ifc2 : NDArray,
    uc : Atoms,
    new_supercell : Atoms,
    two_dim : bool = False,
    symmetrize : bool = True,
    tol: float = 1e-5,
    eps: float = 1e-13,
):

    """
    Remap second-order force constants from compact broken-out form
        (1, n_prim, 3, n_rep, n_prim, 3)
    to 
        (n_rep, n_prim, 3, n_rep, n_prim, 3)
    or, if two_dim=True,
        (3*n_sc, 3*n_sc).
    """

    n_uc = len(uc)
    n_rep = ifc2.shape[-3]

    expected_dim = (1, n_uc, 3, n_rep, n_uc, 3)
    if not np.all(ifc2.shape == expected_dim):
        raise ValueError(f"Expected second-order IFC shape {expected_dim}, got {ifc2.shape}")

    map2prim = np.asarray(
        _map2prim(uc, new_supercell, tol=tol),
        dtype=np.int64,
    )


    sc_r = _build_displacement_table(
        uc, new_supercell, map2prim, n_uc, n_rep, tol
    )

    rep_of_sc, inferred_n_rep = _replica_indices_from_map2prim(
        map2prim, n_prim=n_uc,
    )

    if inferred_n_rep != n_rep:
        raise ValueError(
            f"IFC tensor has n_rep={n_rep}, but new_supercell mapping "
            f"implies n_rep={inferred_n_rep}."
        )

    translated_index = _build_translated_index_map(
        sc_r=sc_r,
        map2prim=map2prim,
        new_supercell=new_supercell,
        tol=tol,
        eps=eps,
    )

    fc_out = np.zeros(
        (n_rep, n_uc, 3, n_rep, n_uc, 3),
        dtype=ifc2.dtype,
    )

    for i_sc, i_uc in enumerate(map2prim):
        R_i = rep_of_sc[i_sc]

        for R_ref in range(n_rep):
            for j_uc_ref in range(n_uc):
                j_sc = translated_index[i_sc, R_ref, j_uc_ref]

                R_j = rep_of_sc[j_sc]
                j_uc = map2prim[j_sc]

                fc_out[R_i, i_uc, :, R_j, j_uc, :] += ifc2[
                    0, i_uc, :, R_ref, j_uc_ref, :
                ]

    if symmetrize:
        fc_out = 0.5 * (
            fc_out + fc_out.transpose(3, 4, 5, 0, 1, 2)
        )

    fc_out[np.abs(fc_out) < eps] = 0.0

    if two_dim:
        fc_out = fc_out.reshape(
            n_rep * n_uc,
            3,
            n_rep * n_uc,
            3,
        )
        fc_out = fc_out.transpose(0, 1, 2, 3).reshape(
            3 * n_rep * n_uc,
            3 * n_rep * n_uc,
        )

    return fc_out


def remap_second_force_constants(
    ifc2 : SecondOrder,
    new_supercell : Atoms,
    two_dim: bool = False,
    symmetrize: bool = True,
    tol: float = 1e-5,
    eps: float = 1e-13,
) -> NDArray:

    return remap_raw_second_force_constants(
        ifc2.value,
        ifc2.atoms,
        new_supercell,
        symmetrize,
        two_dim,
        tol,
        eps,
    )


def remap_third_force_constants(
    ifc3: ThirdOrder,
    tol: float = 1e-5,
    eps: float = 1e-13,
) -> NDArray:

    """
    Converts IFCs from primitive cell representation to supercell representation. Currentlly, will
    only remap to the supercell stored in the ThirdOrder object.
    """

    uc = ifc3.atoms
    sc = ifc3.supercell
    n_uc = len(uc)
    n_sc = len(sc)
    n_rep = ifc3.n_replicas


    expected_shape = (n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)
    if not np.all(ifc3.shape == expected_shape):
        raise ValueError(f"Expected third-order IFC shape {expected_shape}, got {ifc3.shape}")


    sc_r = _build_displacement_table(
        primitive=uc,
        supercell=sc,
        n_prim=n_uc,
        n_sc=n_sc,
        tol=tol,
    )

    # map2prim[i] gives the primitive atom index corresponding to
    # atom i in new_supercell.
    map2prim = _map2prim(uc, sc)  

    # translated_index[i, sc_j] gives the output-supercell atom j
    # corresponding to compact reference atom sc_j when centered on i.
    translated_index = _build_translated_index_map(
        sc_r=sc_r,
        map2prim=map2prim,
        new_supercell=sc,
        tol=tol,
    )

    phi3 = ifc3.value

    for i in range(n_uc):
        # Get all non-zero blocks for atom i
        # This acts as a neighbor list
        mask = np.any(phi3[i] != 0.0, axis=(0, 3, 6))   # shape (n_rep, n_uc, n_rep, n_uc)
        R2s, js, R3s, ks = np.where(mask)
        for R2, j, R3, k in zip(R2s, js, R3s, ks):
            block = phi3[i, :, R2, j, :, R3, k, :]

            # Figure out the remapping
            j_remapped = translated_index[i, j]
            k_remapped = translated_index[i, k]

            # Remap the block
            block_remapped = block[j_remapped, k_remapped, :, :]

            # Add the block to the output
            fc3_out[i, j_remapped, k_remapped, :, :] += block_remapped

    return fc3_out
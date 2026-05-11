from numpy.typing import NDArray
from ase import Atoms
from dataclasses import dataclass
from ase.geometry import get_distances
import numpy as np
from kaldo.grid import wrap_coordinates
from kaldo.interfaces.tdep_io import build_nondiag_observable_kwargs, attach_snf_metadata
from sparse import COO
from kaldo.observables.thirdorder import ThirdOrder
from kaldo.observables.secondorder import SecondOrder
from tqdm import tqdm

@dataclass
class DistanceTableAtom:
    central_atom : int
    N_neighbors : int
    vs : NDArray # Vector from central atom to neighbor atoms (with mic), (N_neighbors x 3)
    lvs : NDArray # Lattice vector for cell containing neighbor atom (with mic), (N_neighbors x 3)
    image_flags : NDArray # Image flags for neighbor atoms, (N_neighbors x 3)
    inds : NDArray # Neighbor indices, (N_neighbors)
    dists : NDArray # Neighbor distances

    def neighbors(self):
        return zip(self.vs, self.lvs, self.image_flags, self.inds, self.dists)

class DistanceTable:


    def __init__(self, atoms : Atoms, distance_threshold : float):

        self.distance_table_atoms = []
        
        dist_vecs, magnitudes = get_distances(atoms.positions, cell = atoms.cell, pbc = True)

        neighbor_mask = magnitudes < distance_threshold

        magnitudes = np.where(neighbor_mask, magnitudes, 0.0)

        x_frac = atoms.get_scaled_positions()

        for i in range(len(atoms)):
            vs = []; lvs = []; ns = []; dists = []

            # Will include self interactions
            neighbor_indices = np.where(neighbor_mask[i])[0]

            for j in neighbor_indices:
                df = x_frac[j] - x_frac[i]
                n = np.rint(df).astype(int) # in [-1, 0, 1]
                # wrapped = df - n # in [-0.5, 0.5]
                vs.append(dist_vecs[i, j])
                lvs.append((-n) @ atoms.cell)
                ns.append(n)
                dists.append(magnitudes[i, j])

            self.distance_table_atoms.append(DistanceTableAtom(
                central_atom = i,
                N_neighbors = len(dists),
                vs = np.array(vs),
                lvs = np.array(lvs),
                image_flags = np.array(ns),
                inds = neighbor_indices,
                dists = np.array(dists)
            ))

    def __len__(self):
        return len(self.distance_table_atoms)

    def __getitem__(self, index):
        return self.distance_table_atoms[index]

    def __iter__(self):
        return iter(self.distance_table_atoms)


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


def remap_second_force_constants(
    ifc2: SecondOrder,
    new_supercell: Atoms,
    tol: float = 1e-5,
) -> NDArray:

    """
    Converts IFCs from primitive cell representation to supercell representation.
    """

    uc = ifc2.atoms
    new_cell = np.asarray(new_supercell.cell)
    new_cell_inv = np.linalg.inv(new_cell)
    uc_cell = np.asarray(uc.cell)
    n_uc = len(uc)
    n_rep = ifc2.n_replicas

    # TODO FIGUREOUT CUTOFF FROM ifc2 object passed
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
    #   (n_rep, n_uc, 3, n_rep, n_uc, 3)
    out_shape = (n_rep, n_uc, 3, n_rep, n_uc, 3)
    out_coords: list[list[int]] = []
    out_data: list[float] = []

    # Build progress bar:
    total_pairs = sum(atom.N_neighbors**2 for atom in dt)   # or N_neighbors*(N_neighbors-1) if you exclude self/self
    pbar = tqdm(total=total_pairs, desc="Remap IFC2 pairs", mininterval=0.5)
  
    for i, atom_i in enumerate(dt):
        uc_index_i = map2prim[i]

        # Figure out which blocks actually interact (i.e., non-zero)
        # to identify which (rep_j, uc_j) blocks are nonzero.
        phi2_i = ifc2.value[uc_index_i]
        mask = np.any(phi2_i != 0.0, axis=(0, 3))   # shape (n_rep, n_uc)
        R2s, js = np.where(mask)

        neighbor_indices_i = atom_i.inds

        for j, (vj, _, _, _, _) in enumerate(atom_i.neighbors()):
            pbar.update(1)

            found_match = False
            for rep_j, uc_j in zip(R2s, js):
                R2 = ifc2._direct_grid.id_to_grid_index(int(rep_j))
                rv_ij = (uc.positions[int(uc_j)] - uc.positions[int(uc_index_i)]) + (R2 @ uc_cell)
                rv_ij = wrap_coordinates(rv_ij, new_cell, new_cell_inv)

                if np.sum(np.square(rv_ij - vj)) > tol*tol:
                    continue

                uc_i_new = int(mapping["atom_of_sc"][i])
                uc_j_new = int(mapping["atom_of_sc"][neighbor_indices_i[j]])
                rep_i_new = int(mapping["replica_id_of_sc"][i])
                rep_j_new = int(mapping["replica_id_of_sc"][neighbor_indices_i[j]])

                # Store coords/value for output
                block = ifc2.value[uc_index_i, :, rep_j, uc_j, :]
                bcoords = np.asarray(block.coords)  # (3, nnz) for (alpha,beta,gamma)
                bdata = np.asarray(block.data)
                for nnz_idx in range(bdata.shape[0]):
                    alpha = int(bcoords[0, nnz_idx])
                    beta = int(bcoords[1, nnz_idx])
                    out_coords.append([
                        rep_i_new, uc_i_new, alpha,
                        rep_j_new, uc_j_new, beta,
                    ])
                    out_data.append(float(bdata[nnz_idx]))

                found_match = True
                break

            if not found_match:
                j_sc = int(neighbor_indices_i[j])
                failed_atom = f"i_sc={i}, uc_index_i={uc_index_i}, j_sc={j_sc}"
                raise ValueError(
                    f"Could not find matching pair for {failed_atom}. Cells are likely inconsistent."
                )
        pbar.close()

    coords_arr = np.asarray(out_coords, dtype=np.int64).T  # (9, nnz)
    data_arr = np.asarray(out_data, dtype=float)
    second_ifcs = COO(coords=coords_arr, data=data_arr, shape=out_shape)#.sum_duplicates()
    second_order = SecondOrder(value=second_ifcs, folder="Remapped IFCs, no source folder", **kw)

    return attach_snf_metadata(second_order, mapping)


def remap_third_force_constants(
    ifc3: ThirdOrder,
    new_supercell : Atoms,
    tol: float = 1e-5,
) -> NDArray:

    """
    Converts IFCs from primitive cell representation to supercell representation.
    """

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


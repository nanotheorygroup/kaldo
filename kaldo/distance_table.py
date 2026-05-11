import numpy as np
from numpy.typing import NDArray
from ase.geometry import get_distances
from ase import Atoms
from dataclasses import dataclass

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
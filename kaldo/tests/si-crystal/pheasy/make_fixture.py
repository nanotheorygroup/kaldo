"""Generate the committed pheasy-format Si fixture from the existing hiphive fixture.

Run from the kaldo repo root:
    PYTHONPATH=. python kaldo/tests/si-crystal/pheasy/make_fixture.py

Files are written in pheasy's exact output conventions (atom-major supercell
ordering with the first lattice vector fastest; phonopy-style compact
FORCE_CONSTANTS; ShengBTE FORCE_CONSTANTS_3RD; fc2/fc3.hdf5 with datasets
'fc2'/'fc3'), so the importer test is a differential oracle against the
trusted hiphive route. FORCE_CONSTANTS_4TH carries two synthetic quartets
(kaldo ships no Si IFC4 reference) to exercise include_fourth plumbing.
"""
import itertools
import os

import ase.io
import h5py
import numpy as np

from ase import Atoms
from kaldo.forceconstants import ForceConstants

HERE = os.path.dirname(__file__)
HIPHIVE = os.path.join(HERE, '..', 'hiphive')
SUPERCELL = (3, 3, 3)

# fixed synthetic IFC4 content, asserted verbatim by test_si_pheasy_fourth_order
PHI4_A = np.arange(81, dtype=float).reshape(3, 3, 3, 3) / 100.0
PHI4_B = -np.arange(81, dtype=float).reshape(3, 3, 3, 3) / 200.0


def kaldo_id_to_pheasy_c(grid, supercell):
    d0, d1, d2 = supercell
    offsets = grid.grid(is_wrapping=False)
    return np.array([int(ox + oy * d0 + oz * d0 * d1) for (ox, oy, oz) in offsets])


def main():
    fc = ForceConstants.from_folder(folder=HIPHIVE, supercell=SUPERCELL, format='hiphive')
    atoms = fc.atoms
    n_uc = atoms.positions.shape[0]
    d0, d1, d2 = SUPERCELL
    n_cells = d0 * d1 * d2
    n_sc = n_uc * n_cells

    ase.io.write(os.path.join(HERE, 'POSCAR'), atoms, format='vasp')

    # supercell in pheasy's atom-major, x-fastest order
    scaled = atoms.get_scaled_positions()
    positions, numbers = [], []
    for i in range(n_uc):
        for c in range(n_cells):
            ox, oy, oz = c % d0, (c // d0) % d1, c // (d0 * d1)
            positions.append((scaled[i] + np.array([ox, oy, oz])) / np.array([d0, d1, d2]))
            numbers.append(atoms.numbers[i])
    sposcar = Atoms(numbers=numbers, scaled_positions=positions,
                    cell=np.array(atoms.cell) * np.array([[d0], [d1], [d2]]), pbc=True)
    ase.io.write(os.path.join(HERE, 'SPOSCAR'), sposcar, format='vasp')

    # second order
    to_c2 = kaldo_id_to_pheasy_c(fc.second._direct_grid, SUPERCELL)
    fc2 = np.asarray(fc.second.value, dtype=np.float64)
    ifc2 = np.zeros((n_uc, n_sc, 3, 3))
    for i in range(n_uc):
        for rep in range(n_cells):
            for j in range(n_uc):
                ifc2[i, j * n_cells + to_c2[rep]] = fc2[0, i, :, rep, j, :]
    with open(os.path.join(HERE, 'FORCE_CONSTANTS'), 'w') as fd:
        fd.write(f"{n_uc:>5d}{n_sc:5d}\n")
        for i, j in np.ndindex((n_uc, n_sc)):
            fd.write(f"{i * n_cells + 1:5d}{j + 1:5d}\n")
            for alpha in range(3):
                fd.write("".join(f"{x:25.15f}" for x in ifc2[i, j, alpha]) + "\n")
    with h5py.File(os.path.join(HERE, 'fc2.hdf5'), 'w') as fd:
        fd.create_dataset('fc2', data=ifc2, compression='gzip')

    # third order
    to_c3 = kaldo_id_to_pheasy_c(fc.third._direct_grid, SUPERCELL)
    third = np.asarray(fc.third.value).reshape((n_uc, 3, n_cells, n_uc, 3, n_cells, n_uc, 3))
    replicas3 = fc.third.list_of_replicas
    blocks = []
    for i in range(n_uc):
        for r2 in range(n_cells):
            for j in range(n_uc):
                for r3 in range(n_cells):
                    for k in range(n_uc):
                        phi = third[i, :, r2, j, :, r3, k, :]
                        if np.abs(phi).max() == 0.0:
                            continue
                        blocks.append((replicas3[r2], replicas3[r3], (i, j, k), phi))
    with open(os.path.join(HERE, 'FORCE_CONSTANTS_3RD'), 'w') as fd:
        fd.write(f"{len(blocks)}\n")
        for n, (c2, c3, (i, j, k), phi) in enumerate(blocks, start=1):
            fd.write(f"\n{n}\n")
            fd.write("".join(f"{x:25.15f}" for x in c2) + "\n")
            fd.write("".join(f"{x:25.15f}" for x in c3) + "\n")
            fd.write(f"{i + 1:6d}{j + 1:6d}{k + 1:6d}\n")
            for (a, b, c) in itertools.product([1, 2, 3], repeat=3):
                fd.write(f"{a:4d}{b:4d}{c:4d}{phi[a - 1, b - 1, c - 1]:25.15f}\n")
    fc3_h5 = np.zeros((n_uc, n_sc, n_sc, 3, 3, 3))
    for i in range(n_uc):
        for r2 in range(n_cells):
            for j in range(n_uc):
                for r3 in range(n_cells):
                    for k in range(n_uc):
                        phi = third[i, :, r2, j, :, r3, k, :]
                        if np.abs(phi).max() == 0.0:
                            continue
                        fc3_h5[i, j * n_cells + to_c3[r2], k * n_cells + to_c3[r3]] = phi
    with h5py.File(os.path.join(HERE, 'fc3.hdf5'), 'w') as fd:
        fd.create_dataset('fc3', data=fc3_h5, compression='gzip')

    # synthetic fourth order: two quartets, offsets (0,0,0) and one lattice vector along x
    a1 = np.array(atoms.cell)[0]
    with open(os.path.join(HERE, 'FORCE_CONSTANTS_4TH'), 'w') as fd:
        fd.write("2\n")
        for n, (r2, quartet, phi) in enumerate(
                [(np.zeros(3), (0, 1, 1, 0), PHI4_A), (a1, (0, 1, 0, 1), PHI4_B)], start=1):
            fd.write(f"\n{n}\n")
            fd.write("".join(f"{x:25.15f}" for x in r2) + "\n")
            fd.write("".join(f"{x:25.15f}" for x in np.zeros(3)) + "\n")
            fd.write("".join(f"{x:25.15f}" for x in np.zeros(3)) + "\n")
            i, j, k, m = quartet
            fd.write(f"{i + 1:6d}{j + 1:6d}{k + 1:6d}{m + 1:6d}\n")
            for (a, b, c, e) in itertools.product([1, 2, 3], repeat=4):
                fd.write(f"{a:4d}{b:4d}{c:4d}{e:4d}{phi[a - 1, b - 1, c - 1, e - 1]:25.15f}\n")
    print('wrote pheasy Si fixture to', HERE)


if __name__ == '__main__':
    main()

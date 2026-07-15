"""Generate the committed pheasy-format MgO NAC fixture from the qe-d3q reference.

Run from the kaldo repo root:
    PYTHONPATH=. python kaldo/tests/mgo/pheasy/make_fixture.py

Stores second order as fc2.hdf5 (pheasy's --hdf5 output, exercising the hdf5
route) plus born.fmt (dielectric tensor + Born charges) and POSCAR. Values
come from the espresso.ifc2 reference already in kaldo/tests/mgo, so the NAC
test is a differential oracle against the qe route.
"""
import os

import ase.io
import h5py
import numpy as np

from kaldo.forceconstants import ForceConstants

HERE = os.path.dirname(__file__)
MGO = os.path.join(HERE, '..')
SUPERCELL = (5, 5, 5)


def main():
    fc = ForceConstants.from_folder(folder=MGO, supercell=SUPERCELL, only_second=True, format='qe-d3q')
    atoms = fc.atoms
    n_uc = atoms.positions.shape[0]
    d0, d1, d2 = SUPERCELL
    n_cells = d0 * d1 * d2
    n_sc = n_uc * n_cells

    offsets = fc.second._direct_grid.grid(is_wrapping=False)
    to_c = np.array([int(ox + oy * d0 + oz * d0 * d1) for (ox, oy, oz) in offsets])
    fc2 = np.asarray(fc.second.value, dtype=np.float64)
    ifc2 = np.zeros((n_uc, n_sc, 3, 3))
    for i in range(n_uc):
        for rep in range(n_cells):
            for j in range(n_uc):
                ifc2[i, j * n_cells + to_c[rep]] = fc2[0, i, :, rep, j, :]
    with h5py.File(os.path.join(HERE, 'fc2.hdf5'), 'w') as fd:
        fd.create_dataset('fc2', data=ifc2, compression='gzip')

    rows = np.vstack([atoms.info['dielectric']] + [atoms.get_array('charges')[i] for i in range(n_uc)])
    np.savetxt(os.path.join(HERE, 'born.fmt'), rows, fmt='%22.15f')
    ase.io.write(os.path.join(HERE, 'POSCAR'), atoms, format='vasp')
    print('wrote pheasy MgO fixture to', HERE)


if __name__ == '__main__':
    main()

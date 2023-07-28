from ase.io import read
import numpy as np

npoints = 200
sympath_string = 'GXULG'

atoms = read('forces/POSCAR', format='vasp')
cell = atoms.cell
lat = cell.get_bravais_lattice()
sympath = cell.bandpath(sympath_string, npoints=npoints)

np.savetxt('path.out.txt', sympath.kpts)

# Example: (10,0) Carbon Nanotube, Tersoff optimized potential
# Generates: input files for LAMMPS USER-Phonon calcualtions 
# External files: unit.xyz : (10,0) Carbon Nanotube unit cell (40 atoms per cell)

from ase.io import read,write
import numpy as np
import os

# Generate fc_CNT folder if not previously defined
fc_folder = 'fc_CNT/'
if not os.path.exists(fc_folder):
        os.makedirs(fc_folder)

# Read in the 10,0 CNT unit cell 
atoms = read('unit.xyz')

# Sanity check for right hand convention
if np.linalg.det(atoms.cell) < 0:
    print('Non right-handed cell convetion detected. Execute cell index swapping !')
    right_handed_convetion_cell = atoms.cell.get_bravais_lattice().tocell()
    atoms.cell = right_handed_convetion_cell

# Replicate the unit cell 'nrep'=3 times
nrep = 3
supercell = np.array([1, 1, nrep])

# Generate input files for LAMMPS and replicated super cell structure
write('CNT.lmp',format = 'lammps-data',images = atoms)
write('fc_CNT/replicated_atoms.xyz',format ='extxyz',images=atoms.copy().repeat(supercell))

# Print reminder information
print('Supercell structures and LAMMPS input generated.') 
print('Supercell dimension is: ' + str(supercell))

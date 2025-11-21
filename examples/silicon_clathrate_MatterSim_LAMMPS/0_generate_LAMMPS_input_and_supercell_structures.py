from ase.build import bulk
from ase.constraints import StrainFilter
from ase.io import read
from ase.optimize import BFGS
from ase.io import write
from mattersim.forcefield import MatterSimCalculator
import numpy as np
import os
import torch
from loguru import logger

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running MatterSim on {device}")


# Generate fc_Si46 folder if not previously defined
fc_folder = 'fc_Si46/'
if not os.path.exists(fc_folder):
    os.makedirs(fc_folder)

# Create unitcell for GaAs
raw_atoms = read('Si46.lmp', format='lammps-data', Z_of_type = {1:14})
raw_atoms.calc =  MatterSimCalculator(device=device)
sf = StrainFilter(raw_atoms)
dyn = BFGS(sf, trajectory='Si46.traj')
dyn.run(fmax=0.001)
atoms = read('Si46.traj')

# Sanity check for right hand convention
if np.linalg.det(atoms.cell) < 0:
    print('Non right-handed cell convetion detected. Execute cell index swapping !')
    right_handed_convetion_cell = atoms.cell.get_bravais_lattice().tocell()
    atoms.cell = right_handed_convetion_cell

# Replicate the unit cell 'nrep'=3 times
nrep_second = 3
supercell = np.array([nrep_second, nrep_second, nrep_second])

# Generate input files for LAMMPS and replicated super cell structure
write('Si46_opt.lmp',format = 'lammps-data',images = atoms)
write('fc_Si46/replicated_atoms.xyz',format ='extxyz',images=atoms.copy().repeat(supercell))

# Print reminder information
print('Supercell structures and LAMMPS input generated.') 
print('Supercell dimension is: ' + str(supercell))

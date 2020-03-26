"""
Compute force constants (fcs) for desired orders.

"""
from ase.build import bulk
from ase.io import write
from hiphive import StructureContainer, ForceConstantPotential
from hiphive.fitting import Optimizer
import numpy as np
import os
import shutil

a0 = 5.432
atoms_prim = bulk('Si', 'diamond', a=a0)
supercell = np.array([3,3,3])
n_sc = np.prod(supercell)

# Read in the replicated structures from previous rattle gneration

atoms_replicated = atoms_prim.copy()*(supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])

# Read structure containers and cluster spaces

sc2 = StructureContainer.read('structure_container2')
sc3 = StructureContainer.read('structure_container3')
cs2 = sc2.cluster_space
cs3 = sc3.cluster_space

# Fit models

opt = Optimizer(sc2.get_fit_data())
opt.train()
print(opt)
fcp2 = ForceConstantPotential(cs2, opt.parameters)

opt = Optimizer(sc3.get_fit_data())
opt.train()
print(opt)
fcp3 = ForceConstantPotential(cs3, opt.parameters)

# Derive and save force constants from force potential

fcs2 = fcp2.get_force_constants(atoms_replicated)
fcs3 = fcp3.get_force_constants(atoms_replicated)

fcs2.write('model2.fcs')
fcs3.write('model3.fcs')

# Set up hiphive folder

hiphive_filename = 'hiphive_si_bulk/'
primitive_fname = 'hiphive_si_bulk/atom_prim.xyz'
if not os.path.isdir(hiphive_filename):
    os.mkdir(hiphive_filename)

write(primitive_fname, atoms_prim,format= 'xyz')
shutil.move('model2.fcs',hiphive_filename)
shutil.move('model3.fcs',hiphive_filename)
print('\n')
print('fcs computations are completed!')
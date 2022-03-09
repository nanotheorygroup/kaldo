from ase.io import read,write
from ase.lattice.hexagonal import *
import numpy as np
a = 2.5
c = 40
supercell=(25, 12, 1)
atoms = Graphene('C', latticeconstant={'a':a, 'c':c}, size=(1, 1, 1), pbc=(1,1,1))
write('graphene.lmp', format='lammps-data', images=atoms)
write('replicated_atoms.xyz', format ='extxyz', images=atoms.copy().repeat(supercell))

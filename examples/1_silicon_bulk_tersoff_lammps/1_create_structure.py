from ase.io import read, write
from ase.build import bulk

atoms = bulk('Si', 'diamond', a=5.432) * (3, 3, 3)
write(filename='structures/replicated_atoms.lmp', images=atoms, format='lammps-data')
write(filename='structures/replicated_atoms.xyz', images=atoms, format='extxyz')

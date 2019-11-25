from ase.io import read, write
from ase.build import bulk
import numpy as np

atoms = bulk('Si', 'diamond', a=5.43)
write(filename='structures/coords.lmp', images=atoms, format='lammps-data')
print('Box info:')
q, r = np.linalg.qr(atoms.cell)
cell_prime = np.abs(np.linalg.inv(q) @ atoms.cell)
transformation = cell_prime @ np.linalg.inv(atoms.cell)
pos_prime = np.tensordot(transformation, atoms.positions, (1, 1)).T
atoms.positions = pos_prime
atoms.cell = cell_prime
xhi, yhi, zhi, xy, xz, yz = cell_prime[0, 0], cell_prime[1, 1], cell_prime[2, 2], cell_prime[0, 1], cell_prime[0, 2], cell_prime[1, 2]
print(xhi, yhi, zhi, xy, xz, yz)

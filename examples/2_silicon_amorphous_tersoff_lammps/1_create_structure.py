from ase.io import read, write
from ase.build import bulk
import numpy as np

atoms = read(filename='structures/CONFIG', format='dlp4')
write(filename='structures/coords.lmp', images=atoms, format='lammps-data')

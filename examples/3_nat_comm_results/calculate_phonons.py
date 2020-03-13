from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
from ballistico.finitedifference import FiniteDifference
from ballistico.phonons import Phonons
from ballistico.conductivity import Conductivity
from sklearn.neighbors.kde import KernelDensity
from ballistico.controllers.plotter import plot_dispersion

# Uncomment the following to import from lammps

# folder = 'structure_lammps'
# config_file = str(folder) + '/replicated_coords.lmp'
# dynmat_file = str(folder) + '/dynmat.dat'
# third_file = str(folder) + '/third.dat'
# atoms = read(config_file, format='lammps-data', style='atomic')
#
# atomic_numbers = atoms.get_atomic_numbers()
# atomic_numbers[atomic_numbers == 1] = 14
# atoms.set_atomic_numbers(atomic_numbers)
#
# finite_difference = FiniteDifference.from_files(replicated_atoms=atoms, dynmat_file=dynmat_file, folder=folder)

finite_difference = FiniteDifference.from_folder('structure_dlpoly', format='eskm')

# Conductivity AF
phonons = Phonons(finite_difference=finite_difference,
                  is_classic=False,
                  temperature=300,
                  # 1 THz
                  # third_bandwidth=1/4.135,
                  broadening_shape='triangle',
                  is_tf_backend=True,
                  folder='ald-output')

# Unfortunately at the moment the input is in 2 * pi * THZ
# 0.5 THz
phonons.diffusivity_bandwidth = 0.5 * 2 * np.pi / 4.135
cond = Conductivity(phonons=phonons, method='qhgk').conductivity.sum(axis=0)
cond = np.abs(np.mean(cond.diagonal()))
print('AF conductivity: ', cond)



# Conductivity QHGK
phonons = Phonons(finite_difference=finite_difference,
                  is_classic=False,
                  temperature=300,
                  # 1 THz
                  third_bandwidth=1/4.135,
                  broadening_shape='triangle',
                  is_tf_backend=True,
                  folder='ald-output')


cond = Conductivity(phonons=phonons, method='qhgk').conductivity.sum(axis=0)
cond = np.abs(np.mean(cond.diagonal()))
print('QHGK conductivity: ', cond)

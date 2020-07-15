from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import matplotlib.pyplot as plt
from kaldo.conductivity import Conductivity
from ase.io import read
import numpy as np

supercell = np.array([3, 3, 3])
# forceconstants = ForceConstants.import_from_dlpoly_folder('si-dlpoly', supercell)


forceconstants = ForceConstants.from_folder('structures', supercell=supercell, format='lammps')

k = 5
kpts = [k, k, k]
is_classic = False
temperature = 300

# # Create a phonon object
phonons = Phonons(forceconstants=forceconstants,
                  kpts=kpts,
                  is_classic=is_classic,
                  temperature=temperature)

print('AF conductivity')
print(Conductivity(phonons=phonons, method='qhgk').conductivity.sum(axis=0))

plt.scatter(phonons.frequency.flatten()[3:], phonons.bandwidth.flatten()[3:], s=5)
plt.ylabel('gamma_THz', fontsize=16, fontweight='bold')
plt.xlabel("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
plt.show()

plt.scatter(phonons.frequency.flatten()[3:], phonons.phase_space.flatten()[3:], s=5)
plt.ylabel('ps', fontsize=16, fontweight='bold')
plt.xlabel("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
plt.show()

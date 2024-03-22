from ase.io import read
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.controllers import plotter
from pynep.calculate import NEP
import os
import numpy as np

import matplotlib.pyplot as plt
from kaldo.helpers.storage import get_folder_from_label
from kaldo.controllers import plotter

from ase.optimize import BFGS
from ase.constraints import StrainFilter

################################################### 
# Read initial structure and optimize it using BFGS
###################################################

atoms = read('graph.cif', format='cif')
calc = NEP('forcefields/C_2022_NEP3.txt')

atoms.calc = calc
sf = StrainFilter(atoms)
opt = BFGS(sf) # Optimization of crystal using BFGS algorithm with NEP input
opt.run(0.005)
print("Cell: ",atoms.get_cell()) 

###################################################
# Set up 5x5x3 supercell with 15x15x7 grid
###################################################

# Declares GPU useage
os.environ["CUDA_VISIBLE_DEVICES"]="2"

nrepxy = 5 # Replicas in x and y directions
nrepz = 3 # Replicas in the z direction
kpxy = 15
kpz = 7


# Replicate the unit cell 'nrep'=3 times
supercell = np.array([nrepxy, nrepxy, nrepz])

###################################################
# Create Forceconstants object and Calculate IFCs
###################################################

# Create a finite difference object
forceconstants_config  = {'atoms':atoms,'supercell': supercell,'folder':'fd'}
forceconstants = ForceConstants(**forceconstants_config)

# Compute 2nd and 3rd IFCs with the defined calculators
nep_calculator = calc
forceconstants.second.calculate(calculator = nep_calculator, delta_shift=1e-4)
forceconstants.third.calculate(calculator = nep_calculator, delta_shift=1e-4)

# Define the k-point mesh using 'kpts' parameter
is_classic = False
phonons_config = {'kpts': [kpxy, kpxy, kpz],
                  'is_classic': is_classic,
                  'temperature': 300,  # 'temperature'=300K
                  'folder': 'ALD',
                  'storage': 'formatted'}

# Create phonons object from calculated force constants
phonons = Phonons(forceconstants=forceconstants, **phonons_config)

###  Plot Phonon Dispersion and Phonon Velocities
plt.style.use('seaborn-poster')
folder = get_folder_from_label(phonons, base_folder='plots')
if not os.path.exists(folder):
    os.makedirs(folder)
## 
plotter.plot_dispersion(phonons, with_velocity=True, is_showing=False)

###################################################
# Calculate conductivity from Inverse and RTA
###################################################

print('\n')
inv_cond_matrix = (Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0)) # Calculation of conductivity by inversion
print('Conductivity from inversion (W/m-K): %.3f' % (np.mean(np.diag(inv_cond_matrix))))
print(inv_cond_matrix)

print('\n')
rta_cond_matrix = (Conductivity(phonons=phonons, method='rta').conductivity.sum(axis=0)) # Calculation of conductivity by RTA
print('Conductivity from RTA (W/m-K): %.3f' % (np.mean(np.diag(rta_cond_matrix))))
print(rta_cond_matrix)

###################################################
# Calculate Phonon properties and print them
###################################################

frequency = phonons.frequency.flatten(order='C') # phonon frequency
velocity = phonons.velocity.real.reshape((phonons.n_phonons, 3)) # phonon velocity
phase_space = phonons.phase_space.flatten(order='C')  # scattering phase space
band_width = phonons.bandwidth.flatten(order='C')     # bandwidth from RTA
mfp = np.abs(Conductivity(phonons=phonons,
                          method='inverse',
                          storage='numpy').mean_free_path[:,2])/10. # A-->nm mean free path from inversion

data = np.stack((frequency, mfp, phase_space, band_width)).T
property_header_str = 'frequency (THz)   mfp (nm)    phase_space     band_width (1/ps)'
if is_classic:
	np.savetxt('CL_properties.dat', data, delimiter=' ', fmt='%10.5f', header  = property_header_str)
else:
	np.savetxt('QM_properties.dat', data, delimiter=' ', fmt='%10.5f', header  = property_header_str)

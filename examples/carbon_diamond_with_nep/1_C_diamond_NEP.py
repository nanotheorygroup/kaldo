from ase.build import bulk
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.controllers import plotter
from pynep.calculate import NEP
import os
import numpy as np


###################################################
# Create initial structure
###################################################

os.environ["CUDA_VISIBLE_DEVICES"]="2"

a0 = 3.534

# Generate initial structure
atoms = bulk('C', 'diamond', a=a0)

###################################################
# Set up 5x5x5 supercell with 5x5x5 k-grid
###################################################

nrep = 5
k_points = 5

# Replicate the unit cell 'nrep'=3 times
supercell = np.array([nrep, nrep, nrep])

###################################################
# Create Forceconstants object and Calculate IFCs
###################################################

# Create a finite difference object
forceconstants_config  = {'atoms':atoms,'supercell': supercell,'folder':'fd'}
forceconstants = ForceConstants(**forceconstants_config)

# Compute 2nd and 3rd IFCs with the defined calculators
nep_calculator = NEP('forcefields/C_2022_NEP3.txt')
forceconstants.second.calculate(calculator = nep_calculator, delta_shift=1e-3)
forceconstants.third.calculate(calculator = nep_calculator, delta_shift=1e-3)


# Define the k-point mesh using 'kpts' parameter
is_classic = False
phonons_config = {'kpts': [k_points, k_points, k_points],
                  'is_classic': is_classic,
                  'temperature': 300,  # 'temperature'=300K
                  'folder': 'ALD',
                  'storage': 'formatted'}

# Create phonons object from calculated force constants
phonons = Phonons(forceconstants=forceconstants, **phonons_config)

###################################################
# Calculate conductivity from Inverse and RTA
###################################################

print('\n')
inv_cond_matrix = (Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0))
print('Conductivity from inversion (W/m-K): %.3f' % (np.mean(np.diag(inv_cond_matrix))))
print(inv_cond_matrix)

print('\n')
rta_cond_matrix = (Conductivity(phonons=phonons, method='rta').conductivity.sum(axis=0))
print('Conductivity from RTA (W/m-K): %.3f' % (np.mean(np.diag(rta_cond_matrix))))
print(rta_cond_matrix)

###################################################
# Calculate Phonon properties and print them
###################################################

frequency = phonons.frequency.flatten(order='C')
velocity = phonons.velocity.real.reshape((phonons.n_phonons, 3))
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

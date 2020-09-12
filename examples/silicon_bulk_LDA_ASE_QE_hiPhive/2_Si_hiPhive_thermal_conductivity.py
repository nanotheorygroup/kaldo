# Example: silicon bulk, hiPhive 
# Computes: anharmonic properties and thermal conductivity for silicon bulk (2 atoms per cell)
# Uses: hiPhive 

from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import numpy as np

# Config force constants object by loading in the IFCs
# from hiphive calculations
forceconstants = ForceConstants.from_folder('hiPhive_si_bulk', supercell=[3, 3, 3],format='hiphive')

### Set up the phonon object and the anharmonic properties calculations ####

# Configure phonon object
# 'k_points': number of k-points
# 'is_classic': specify if the system is classic, True for classical and False for quantum
# 'temperature: temperature (Kelvin) at which simulation is performed
# 'folder': name of folder containing phonon property and thermal conductivity calculations
# 'storage': Format to storage phonon properties ('formatted' for ASCII format data, 'numpy' 
#            for python numpy array and 'memory' for quick calculations, no data stored)

# Define the k-point mesh using 'kpts' parameter
k_points = 7 #'k_points'=7 k points in each direction
phonons_config = {'kpts': [k_points, k_points, k_points],
                  'is_classic': False, 
                  'temperature': 300, #'temperature'=300K
                  'folder': 'ALD_si_bulk',
		   'storage': 'formatted'}
# Set up phonon object by passing in configuration details and the forceconstants object computed above
phonons = Phonons(forceconstants=forceconstants, **phonons_config)

### Set up the Conductivity object and thermal conductivity calculations ####

# Compute thermal conductivity (t.c.) by solving Boltzmann Transport
# Equation (BTE) with various of methods

# 'phonons': phonon object obtained from the above calculations
# 'method': specify methods to solve for BTE  
# ('rta' for RTA,'sc' for self-consistent and 'inverse' for direct inversion of the scattering matrix)

print('\n')
inv_cond_matrix = (Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0))
print('Conductivity from inversion (W/m-K): %.3f' % (np.mean(np.diag(inv_cond_matrix))))
print(inv_cond_matrix)

print('\n')
sc_cond_matrix = Conductivity(phonons=phonons, method='sc', n_iterations=20).conductivity.sum(axis=0)
print('Conductivity from self-consistent (W/m-K): %.3f' % (np.mean(np.diag(sc_cond_matrix))))
print(sc_cond_matrix)

print('\n')
rta_cond_matrix = Conductivity(phonons=phonons, method='rta').conductivity.sum(axis=0)
print('Conductivity from RTA (W/m-K): %.3f' % (np.mean(np.diag(rta_cond_matrix))))
print(rta_cond_matrix)

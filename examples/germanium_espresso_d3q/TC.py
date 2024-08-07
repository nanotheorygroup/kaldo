# Example: germanium bulk 
# Computes: anharmonic properties and thermal conductivity for germanium bulk (2 atoms per cell)
# force constants generated by d3q

from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import numpy as np
import os
from ase.io import read,write

ini_sc=[10,10,10]
root='./fc/'
### Set up forceconstants object
# possible different supercell for second and third
forceconstants = ForceConstants.from_folder(root,supercell=ini_sc,third_supercell=np.array([5,5,5]),format='shengbte-d3q')
if not os.path.exists(root+'second.npy'):
    forceconstants.second.save('second',format='numpy')
    write(root+'replicated_atoms.xyz',forceconstants.second.replicated_atoms)
### Set up the phonon object and the anharmonic properties calculations ####

# Configure phonon object
# 'k_points': number of k-points
# 'is_classic': specify if the system is classic, True for classical and False for quantum
# 'temperature: temperature (Kelvin) at which simulation is performed
# 'folder': name of folder containing phonon property and thermal conductivity calculations
# 'storage': Format to storage phonon properties ('formatted' for ASCII format data, 'numpy' 
#            for python numpy array and 'memory' for quick calculations, no data stored)

# Define the k-point mesh using 'kpts' parameter
k_points = 10 #'k_points'=7 k points in each direction
temperature=300

phonons_config = {'kpts': [k_points, k_points, k_points],
                  'is_classic': False, 
                  'temperature': temperature, #'temperature'=300K
                  'folder': 'ald',
		   'storage': 'numpy'}
# Set up phonon object by passing in configuration details and the forceconstants object computed above
phonons = Phonons(forceconstants=forceconstants, **phonons_config)
### Set up the Conductivity object and thermal conductivity calculations ####

# Compute thermal conductivity (t.c.) by solving Boltzmann Transport
# Equation (BTE) with various of methods

# 'phonons': phonon object obtained from the above calculations
# 'method': specify methods to solve for BTE  
# ('rta' for RTA,'sc' for self-consistent and 'inverse' for direct inversion of the scattering matrix)

print('\n')
qhgk_cond_matrix = Conductivity(phonons=phonons, method='qhgk', storage='numpy').conductivity.sum(axis=0)
print('Conductivity from QHGK (W/m-K): %.3f' % (np.mean(np.diag(qhgk_cond_matrix) ) ) )
print(qhgk_cond_matrix)

print('\n')
rta_cond_matrix = Conductivity(phonons=phonons, method='rta',storage='numpy').conductivity.sum(axis=0)
print('Conductivity from RTA (W/m-K): %.3f' % (np.mean(np.diag(rta_cond_matrix))))
print(rta_cond_matrix)

print("comparison methods:",k_points,phonons.temperature,np.abs(np.mean(np.diag(qhgk_cond_matrix))),np.mean(np.diag(rta_cond_matrix)) )

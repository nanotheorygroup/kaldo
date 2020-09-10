# Example 1.2: carbon diamond, Tersoff potential 
# Computes:anharmonic properties and thermal conductivity for carbon diamond (2 atoms per cell)
# Uses: ASE, LAMMPS
# External files: forcefields/C.tersoff

# Import necessary packages

from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from kaldo.conductivity import Conductivity
from kaldo.controllers import plotter
from kaldo.forceconstants import ForceConstants
from kaldo.helpers.storage import get_folder_from_label
from kaldo.phonons import Phonons
import numpy as np
import os

### Set up the coordinates of the system and the force constant calculations ####

# Define the system according to ASE style. 'a': lattice parameter (Angstrom)
atoms = bulk('C', 'diamond', a=3.566)

# Replicate the unit cell 'nrep'=3 times
nrep = 3
supercell = np.array([nrep, nrep, nrep])

# Configure force constant calculator
forceconstants_config = {'atoms': atoms, 'supercell': supercell, 'folder': 'fc_c_diamond'}
forceconstants = ForceConstants(**forceconstants_config)

# Define input information for the ase LAMMPSlib calculator
# Terosff potential for carbon (C.tersoff) is used for this example.
lammps_inputs = {'lmpcmds': [
    'pair_style tersoff',
    'pair_coeff * * forcefields/C.tersoff C'],
    'keep_alive': True,
    'log_file': 'lammps-c-diamond.log'}

# Compute 2nd and 3rd IFCs with the defined calculators
forceconstants.second.calculate(LAMMPSlib(**lammps_inputs))
forceconstants.third.calculate(LAMMPSlib(**lammps_inputs))

### Set up the phonon object and the anharmonic properties calculations ####

# Configure phonon object
# 'k_points': number of k-points
# 'is_classic': specify if the system is classic, True for classical and False for quantum
# 'temperature: temperature (Kelvin) at which simulation is performed
# 'folder': name of folder containing phonon property and thermal conductivity calculations
# 'storage': Format to storage phonon properties ('formatted' for ASCII format data, 'numpy' 
#            for python numpy array and 'memory' for quick calculations, no data stored)

# Define the k-point mesh using 'kpts' parameter
k_points = 5 #'k_points'=5 k points in each direction
phonons_config = {'kpts': [k_points, k_points, k_points],
                  'is_classic': False, 
                  'temperature': 300, #'temperature'=300K
                  'folder': 'ALD_c_diamond',
		   'storage': 'formatted'}
# Set up phonon object by passing in configuration details and the forceconstants object computed above
phonons = Phonons(forceconstants=forceconstants, **phonons_config)

### Set up the Conductivity object and thermal conductivity calculations ####

# Compute thermal conductivity (t.c.) by solving Boltzmann Transport
# Equation (BTE) with various of methods.

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

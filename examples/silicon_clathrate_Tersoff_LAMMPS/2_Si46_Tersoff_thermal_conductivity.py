# Example 4.2: Type I silicon clathrate, Tersoff potential 
# Computes: anharmonic properties and thermal conductivity for type I silicon clathrate (46 atoms per cell)
# Uses: LAMMPS
# External files: forcefields/Si.tersoff

# Import necessary packages

from ase.io import read
from kaldo.conductivity import Conductivity
from kaldo.controllers import plotter
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.helpers.storage import get_folder_from_label
import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('seaborn-poster')

### Set up force constant objects via interface to LAMMPS ####

# Replicate the unit cell 'nrep'=3 times
nrep = 3
supercell = np.array([nrep, nrep, nrep])

# Load in computed 2nd, 3rd IFCs from LAMMPS outputs
forceconstants = ForceConstants.from_folder(folder='fc_Si46',supercell=supercell,format='lammps')


# Configure phonon object
# 'k_points': number of k-points
# 'is_classic': specify if the system is classic, True for classical and False for quantum
# 'temperature: temperature (Kelvin) at which simulation is performed
# 'folder': name of folder containing phonon property and thermal conductivity calculations
# 'storage': Format to storage phonon properties ('formatted' for ASCII format data, 'numpy' 
#            for python numpy array and 'memory' for quick calculations, no data stored)

# Define the k-point mesh using 'kpts' parameter
k_points = 3#'k_points'=3 k points in each direction
phonons_config = {'kpts': [k_points, k_points, k_points],
                  'is_classic': False, 
                  'temperature': 300, #'temperature'=300K
                  'folder': 'ALD_Si46',
		   'storage': 'formatted'}

# Set up phonon object by passing in configuration details and the forceconstants object computed above
phonons = Phonons(forceconstants=forceconstants, **phonons_config)

### Set up the Conductivity object and thermal conductivity calculations ####

# Compute thermal conductivity (t.c.) by solving Boltzmann Transport
# Equation (BTE) with various of methods

# 'phonons': phonon object obtained from the above calculations
# 'method': specify methods to solve for BTE  
# ('rta' for Relaxiation Time Approxmiation (RTA))


print('\n')
rta_cond_matrix = Conductivity(phonons=phonons, method='rta').conductivity.sum(axis=0)
print('Conductivity from RTA (W/m-K): %.3f' % (np.mean(np.diag(rta_cond_matrix))))
print(rta_cond_matrix)

# Define the base folder to contain plots
# 'base_folder':name of the base folder
folder = get_folder_from_label(phonons, base_folder='plots')
if not os.path.exists(folder):
  os.makedirs(folder)

# Define a Boolean flag to specify if figure window pops during simulation
is_show_fig = False

# Plot cumulative conductivity from RTA method
rta_full_cond = Conductivity(phonons=phonons, method='rta').conductivity
frequency = phonons.frequency.flatten(order='C')
rta_cumulative_cond = plotter.cumulative_cond_cal(frequency,rta_full_cond,phonons.n_phonons)
plt.figure()
plt.plot(frequency,rta_cumulative_cond,'.')
plt.xlabel(r'frequency($THz$)', fontsize=16)
plt.ylabel(r'$\kappa_{cum,RTA}(W/m/K)$', fontsize=16)
plt.xlim([1, 18])
plt.savefig(folder + '/rta_cum_cond_vs_freq.png', dpi=300)
if not is_show_fig:
  plt.close()
else:
  plt.show()

# Example: amorphous silicon, Tersoff potential 
# Computes: Quasi Harmonic Green Kubo (QHGK) properties for amorphous silicon (512 atoms)
# Uses: LAMMPS
# External files: forcefields/Si.tersoff

# Import necessary packages

from ase.io import read
from kaldo.conductivity import Conductivity
from kaldo.controllers import plotter
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('seaborn-poster')

### Set up force constant objects via interface to LAMMPS ####

# Replicate the unit cell 'nrep'=1 time
nrep = 1
supercell = np.array([nrep, nrep, nrep])

# Load in computed 2nd, 3rd IFCs from LAMMPS outputs
forceconstants = ForceConstants.from_folder(folder='fc_aSi512',supercell=supercell,format='lammps')

# Configure phonon object
# 'is_classic': specify if the system is classic, True for classical and False for quantum
# 'temperature: temperature (Kelvin) at which simulation is performed
# 'folder': name of folder containing phonon property and thermal conductivity calculations
# 'storage': Format to storage phonon properties ('formatted' for ASCII format data, 'numpy' 
#            for python numpy array and 'memory' for quick calculations, no data stored)

THz_to_meV = 4.136

phonons_config = {'is_classic': False, 
                  'temperature': 300, #'temperature'=300K
                  'folder': 'ALD_aSi512_example2',
                   'third_bandwidth':0.5/THz_to_meV, # 0.5 meV is used here.
                   'broadening_shape':'triangle',
		   'storage': 'numpy'}

# Set up phonon object by passing in configuration details and the forceconstants object computed above
phonons = Phonons(forceconstants=forceconstants, **phonons_config)

### Set up the Conductivity object and thermal conductivity calculations ####

# Compute thermal conductivity (t.c.) by solving Boltzmann Transport
# Equation (BTE) with various of methods.

# 'phonons': phonon object obtained from the above calculations
# 'method': specify methods to solve for BTE  
#   ('qhgk' for Quasi-Harmonic Green Kubo (QHGK))
# 'storage': Format to storage phonon properties ('formatted' for ASCII format data, 'numpy' 
#            for python numpy array and 'memory' for quick calculations, no data stored)

### Set up the Conductivity object and diffusivity calculations ####
# One needs to compute conducvity from QHGK , and then compute diffusivity 
print('\n')
qhgk_cond = Conductivity(phonons=phonons, method='qhgk', storage='numpy')
qhgk_cond.diffusivity_bandwidth = phonons.bandwidth
print(np.abs(np.mean(qhgk_cond.conductivity.sum(axis=0).diagonal())))
qhgk_cond.diffusivity
	
# Define the base folder to contain plots
# 'base_folder':name of the base folder
folder = phonons.get_folder_from_label(base_folder='plots')
if not os.path.exists(folder):
  os.makedirs(folder)

# Define a Boolean flag to specify if figure window pops during simulation
is_show_fig = False

# Visualize anharmonic phonon properties by using matplotlib
# The following show examples of plotting
# phase space vs frequency and 
#  life tims using RTA vs frequency
# 'order': Index order to reshape array, 
# 'order'='C' for C-like index order; 'F' for Fortran-like index order
# 'band_width': phonon bandwdith (THz) computed from diagonal elements

frequency = phonons.frequency.flatten(order='C')
diffusivity = qhgk_cond.diffusivity.flatten(order='C')

plt.figure()
plt.scatter(frequency[3:],diffusivity[3:], s=5)
plt.xlabel("$\\nu$ (THz)", fontsize=16)
plt.ylabel("$D (mm/s)$", fontsize=16)
plt.xlim([0, 25])
plt.savefig(folder + '/diffusivity_vs_freq.png', dpi=300)
if not is_show_fig:
  plt.close()
else:
  plt.show()

# Plot cumulative conductivity from QHGK methods
qhgk_full_cond = Conductivity(phonons=phonons, method='qhgk').conductivity
qhgk_cumulative_cond = plotter.cumulative_cond_cal(frequency,qhgk_full_cond,phonons.n_phonons)
plt.figure()
plt.plot(frequency[3:],qhgk_cumulative_cond[3:],'.')
plt.xlabel(r'frequency($THz$)', fontsize=16)
plt.ylabel(r'$\kappa_{cum,QHGK}(W/m/K)$', fontsize=16)
plt.xlim([0, 20])
plt.savefig(folder + '/qhgk_cum_cond_vs_freq.png', dpi=300)
if not is_show_fig:
  plt.close()
else:
  plt.show()

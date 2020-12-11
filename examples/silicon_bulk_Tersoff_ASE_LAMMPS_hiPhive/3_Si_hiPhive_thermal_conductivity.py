# Example: silicon bulk, Tersoff potential
# Computes: anharmonic properties and thermal conductivity for silicon bulk (2 atoms per cell)
# Uses: hiPhive, ASE, LAMMPS
# External files: Si.tersoff

from kaldo.conductivity import Conductivity
from kaldo.controllers import plotter
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.helpers.storage import get_folder_from_label
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-poster')

# Config force constants object by loading in the IFCs
# from hiPhive calculations
forceconstants = ForceConstants.from_folder('hiPhive_si_bulk', supercell=[3, 3, 3], format='hiphive')

### Set up the phonon object and the anharmonic properties calculations ####

# Configure phonon object
# 'k_points': number of k-points
# 'is_classic': specify if the system is classic, True for classical and False for quantum
# 'temperature: temperature (Kelvin) at which simulation is performed
# 'folder': name of folder containing phonon property and thermal conductivity calculations
# 'storage': Format to storage phonon properties ('formatted' for ASCII format data, 'numpy'
#            for python numpy array and 'memory' for quick calculations, no data stored)

# Define the k-point mesh using 'kpts' parameter
k_points = 5  # 'k_points'=5 k points in each direction
phonons_config = {'kpts': [k_points, k_points, k_points],
                  'is_classic': False,
                  'temperature': 300,  # 'temperature'=300K
                  'folder': 'ALD_si_bulk',
                  'storage': 'formatted'}

# Set up phonon object by passing in configuration details and the force constants object computed above
phonons = Phonons(forceconstants=forceconstants, **phonons_config)

### Set up the Conductivity object and thermal conductivity calculations ####

# Compute thermal conductivity (t.c.) by solving Boltzmann Transport
# Equation (BTE) with various of methods

# 'phonons': phonon object obtained from the above calculations
# 'method': specify methods to solve for BTE
# ('rta' for RTA,'sc' for self-consistent and 'inverse' for direct inversion of the scattering matrix)

print('\n')
inv_cond_matrix = (Conductivity(phonons=phonons, method='inverse', storage='formatted').conductivity.sum(axis=0))
print('Conductivity from inversion (W/m-K): %.3f' % (np.mean(np.diag(inv_cond_matrix))))
print(inv_cond_matrix)

print('\n')
sc_cond_matrix = Conductivity(phonons=phonons, method='sc', n_iterations=20, storage='formatted').conductivity.sum(
    axis=0)
print('Conductivity from self-consistent (W/m-K): %.3f' % (np.mean(np.diag(sc_cond_matrix))))
print(sc_cond_matrix)

print('\n')
rta_cond_matrix = Conductivity(phonons=phonons, method='rta', storage='formatted').conductivity.sum(axis=0)
print('Conductivity from RTA (W/m-K): %.3f' % (np.mean(np.diag(rta_cond_matrix))))
print(rta_cond_matrix)

# Define the base folder to contain plots
# 'base_folder':name of the base folder
folder = get_folder_from_label(phonons, base_folder='plots')
if not os.path.exists(folder):
    os.makedirs(folder)
# Define a boolean flag to specify if figure window pops during simulation
is_show_fig = False

# Plot cumulative functions for conductivity
frequency = phonons.frequency.flatten(order='C')
rta_full_cond = Conductivity(phonons=phonons, method='rta').conductivity
rta_cumulative_cond = plotter.cumulative_cond_cal(frequency, rta_full_cond, phonons.n_phonons)
inv_full_cond = Conductivity(phonons=phonons, method='inverse').conductivity
inv_cumulative_cond = plotter.cumulative_cond_cal(frequency, inv_full_cond, phonons.n_phonons)

plt.figure()
plt.plot(frequency[3:], rta_cumulative_cond[3:], 'r.', label=r'$\kappa_{cum,RTA}$')
plt.plot(frequency[3:], inv_cumulative_cond[3:], 'k.', label=r'$\kappa_{cum,inverse}$')
plt.xlabel(r'frequency($THz$)', fontsize=16)
plt.ylabel(r'$\kappa_{cum}(W/m/K)$', fontsize=16)
plt.legend(loc=2, frameon=False)
plt.savefig(folder + '/kappa_cum_vs_freq.png', dpi=300)
if not is_show_fig:
    plt.close()
else:
    plt.show()

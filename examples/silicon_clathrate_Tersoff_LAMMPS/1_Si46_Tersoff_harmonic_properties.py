# Example 4.1: Type I silicon clathrate, Tersoff potential 
# Computes: harmonic properties for type I silicon clathrate (46 atoms per cell)
# Uses: LAMMPS
# External files: forcefields/Si.tersoff

# Import necessary packages

from ase.io import read
from kaldo.controllers import plotter
from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.helpers.storage import get_folder_from_label
from kaldo.phonons import Phonons
import matplotlib.pyplot as plt
import numpy as np
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

# Visualize phonon dispersion, group velocity and density of states with 
# the build-in plotter.

# 'with_velocity': specify whether to plot both group velocity and dispersion relation
# 'is_showing':specify if figure window pops up during simulation
plotter.plot_dispersion(phonons,with_velocity =True,is_showing=False)
plotter.plot_dos(phonons,is_showing=False)

# Visualize heat capacity vs frequency and 
# 'order': Index order to reshape array, 
# 'order'='C' for C-like index order; 'F' for Fortran-like index order

# Define the base folder to contain plots
# 'base_folder':name of the base folder
folder = get_folder_from_label(phonons, base_folder='plots')
if not os.path.exists(folder):
        os.makedirs(folder)
# Define a boolean flag to specify if figure window pops during sumuatlion
is_show_fig = False

frequency = phonons.frequency.flatten(order='C')
heat_capacity = phonons.heat_capacity.flatten(order='C')
plt.figure()
plt.scatter(frequency[3:], 1e23 * heat_capacity[3:], s=5)
plt.xlabel("$\\nu$ (THz)", fontsize=16)
plt.ylabel("$C_{v} \ (10^{23} \ J/K)$", fontsize=16)
plt.savefig(folder + '/cv_vs_freq.png', dpi=300)
if not is_show_fig:
  plt.close()
else:
  plt.show()

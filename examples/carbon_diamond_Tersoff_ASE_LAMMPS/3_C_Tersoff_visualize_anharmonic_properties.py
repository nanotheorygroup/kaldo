# Example 1.3: carbon diamond, Tersoff potential 
# Computes:anharmonic propertiesfor carbon diamond (2 atoms per cell)
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
import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('seaborn-poster')

### Set up the coordinates of the system and the force constant calculations ####

# Define the system according to ASE style. 'a': lattice parameter(Angstrom)
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
	
# Define the base folder to contain plots
# 'base_folder':name of the base folder
folder = get_folder_from_label(phonons, base_folder='plots')
if not os.path.exists(folder):
        os.makedirs(folder)

# Define a boolean flag to specify if figure window pops during sumuatlion
is_show_fig = False

# Visualize anharmonic phonon properties by using matplotlib. 
# The following show examples of plotting
# phase space vs frequency
# 'order': Index order to reshape array, 
# 'order'='C' for C-like index order; 'F' for Fortran-like index order
frequency = phonons.frequency.flatten(order='C')
phase_space = phonons.phase_space.flatten(order='C')
plt.figure()
plt.scatter(frequency[3:], phase_space[3:], s=5)
plt.xlabel("$\\nu$ (THz)", fontsize=16)
plt.ylabel("Phase Space", fontsize=16)
plt.savefig(folder + '/ps_vs_freq.png', dpi=300)
if not is_show_fig:
  plt.close()
else:
  plt.show()


### Compare phonon life times at different level of theory ########

# The following shows a comparison of phonon life times
# computed using Relaxation Time Approximation (RTA) and at direct inversion
# of scattering matrix (inverse) methods.

# 'n_phonons': number of phonons in the simulation
# 'band_width': phonon bandwdith (THz) computed from diagonal elements
#  of scattering matrix
band_width = phonons.bandwidth.flatten(order='C')
tau_RTA = (band_width[3:]) ** (-1)

# Compute life times from direct inversion by dividing
# the mean free path from inversion by the group velocities
velocity = phonons.velocity.real.reshape((phonons.n_phonons, 3))
mean_free_path_inversion = Conductivity(phonons=phonons, method='inverse', storage='numpy').mean_free_path
tau_inversion = np.zeros_like(mean_free_path_inversion)

for alpha in range(3):
  for mu in range(len(velocity)):
    if velocity[mu, alpha]!=0:
      tau_inversion[mu, alpha] = np.abs(np.divide(mean_free_path_inversion[mu, alpha],
                              velocity[mu, alpha]))
    else:
        # phonon life times remain zero at zero group velocities
        tau_inversion[mu, alpha] = 0

plt.figure()
plt.plot(frequency[3:], tau_inversion[3:, 0], 'r.', label=r'$\tau_{inv,x}$')
plt.plot(frequency[3:], tau_inversion[3:, 1], 'b.', label=r'$\tau_{inv,y}$')
plt.plot(frequency[3:], tau_inversion[3:, 2], 'k.', label=r'$\tau_{inv,z}$')
plt.plot(frequency[3:], tau_RTA, 'g.', label=r'$\tau_{RTA}$')
plt.xlabel("$\\nu$ (THz)", fontsize=16)
plt.ylabel("$\\tau \ (ps)$", fontsize=16)
plt.yscale('log')
plt.legend(loc='center', fontsize=20)
plt.xlim([10, 50])
plt.savefig(folder + '/phonon_life_time.png', dpi=300)
if not is_show_fig:
  plt.close()
else:
  plt.show()

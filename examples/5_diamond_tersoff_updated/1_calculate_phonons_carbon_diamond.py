# Exaple 5_diamond_tersoff illustrates the kALDo's functionality of
# performing thermal transport simulation for carbon diamond,
# by using ase LAMMPSlib as the force constant calculator.

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

plt.style.use('seaborn-poster')

################# Set up force constant calculations ###########################

# We start from creating the atoms object, here the carbon diamond
# unit cell structure is created with "bulk" routine from ase and
# and the lattice constant of the unit cell is 3.566 Angstrom. Other
# input parameters for forconstants object include super cell structure
# and name of the folder (optional) which will
# contain the calculated 2nd and 3rd IFCs.
atoms = bulk('C', 'diamond', a=3.566)
supercell = np.array([3, 3, 3])
forceconstants_config = {'atoms': atoms, 'supercell': supercell, 'folder': 'fc'}
forceconstants = ForceConstants(**forceconstants_config)

# Define calculator input information for the ase LAMMPSlib calculator
# Terosff potential for carbon (C.tersoff) is used for this example.
lammps_inputs = {'lmpcmds': [
    'pair_style tersoff',
    'pair_coeff * * forcefields/C.tersoff C'],
    'keep_alive': True,
    'log_file': 'lammps-c-diamond.log'}

# Compute 2nd and 3rd IFCs with the defined calculators
forceconstants.second.calculate(LAMMPSlib(**lammps_inputs))
forceconstants.third.calculate(LAMMPSlib(**lammps_inputs))
######################## Set up phonon objects #################################

# After computing IFCS using the forceconstants object, we then set
# up the phonon object and use it to compute 2nd and 3rd (phonon)
# properties. Here, the forcecostants object computed from the above
# section will be used as an input parameter.Other input parameters 
# include the k-point grid dimension, a Boolean flag to specify 
# whether to use a classic or quantum distribution for the phonon population,
# the temperature of the simulation
# in Kelvin, name of the folder (optional) and a Boolean flag to
# specify whether to accelerate the calculation
# for 3rd order properties with TensorFlow (GPU) backend.
# Lastly, various of format to properties computed using phonon object
# are available (e.g. "formatted" for ASCII format data, "numpy" for python 
# numpy array format and "memory" for quick calculations, no data stored).
k = 5
phonons_config = {'kpts': [k, k, k],
                  'is_classic': False,
                  'temperature': 300,
                  'folder': 'ald_c_diamond',
                  'is_tf_backend': False,
                  'storage': 'formatted'}

phonons = Phonons(forceconstants=forceconstants, **phonons_config)

################# 2nd and 3rd order quantities calculations ####################

# With phonon object being set up, the 2nd and 3rd order properties can
# be computed and accessed from the phonon object. For example, the phonon
# dispersion, group velocity and phase space can be computed
# and visualized with the build-in plotter
folder = get_folder_from_label(phonons, base_folder='plots')
plotter.plot_dispersion(phonons)
plotter.plot_dos(phonons)
plotter.plot_vs_frequency(phonons, observable=phonons.phase_space,
                          observable_name=r'Phase Space')

# Other properties, such as heat capacity, can also be computed
# from the phonon object. Meanwhile, the visualization of
# phonon properties can also be customized by using matplotlib. The
# following shows an example of plotting
# heat capacity vs frequency
frequency = phonons.frequency.flatten(order='C')
heat_capacity = phonons.heat_capacity.flatten(order='C')
plt.figure()
plt.scatter(frequency[3:], 1e23 * heat_capacity[3:], s=5)
plt.xlabel("$\\nu$ (THz)", fontsize=16)
plt.ylabel("$C_{v} \ (10^{23} \ J/K)$", fontsize=16)
plt.savefig(folder + '/cv_vs_freq.png', dpi=300)

##################### Thermal conductivities calculations ######################

# After computing 2nd and 3rd (phonon) properties, the thermal
# conductivities (t.c.) can be computed by solving Boltzmann Transport
# Equation (BTE) with various of methods. To compute t.c., one needs to set
# up a Conductivity object, which takes the phonon object computed
# from the above section and a method string
# (e.g. 'rta' for RTA,'sc' for self-consistent and 'inverse'
# for direct inversion of the scattering matrix)
# to solve for BTE. Meanwhile, various of format to store conductivity 
# and mean free path are also available in the Conductivity object.

print('\n')
inv_cond_matrix = (Conductivity(phonons=phonons, method='inverse', storage='memory').conductivity.sum(axis=0))
print('Inverse conductivity (W/m-K): %.3f' % (np.mean(np.diag(inv_cond_matrix))))
print(inv_cond_matrix)

print('\n')
sc_cond_matrix = Conductivity(phonons=phonons, method='sc', n_iterations=20, storage='memory').conductivity.sum(axis=0)
print('Self-consistent conductivity (W/m-K): %.3f' % (np.mean(np.diag(sc_cond_matrix))))
print(sc_cond_matrix)

print('\n')
rta_cond_matrix = Conductivity(phonons=phonons, method='rta', storage='memory').conductivity.sum(axis=0)
print('RTA conductivity (W/m-K): %.3f' % (np.mean(np.diag(rta_cond_matrix))))
print(rta_cond_matrix)

################ Compare properties at different level of theory ###############

# By accessing the phonon objects, one can also derive and compare
# phonon properties at different level of theory. The following lines
# show a comparison between phonon life times from Fermi Golden Rule
# vs. life times from direction inversion of scattering matrix. The
# life times from direction inversion is computed from dividing
# the mean free path from inversion by the group velocities.
tau_RTA = (phonons.bandwidth.flatten(order='C')) ** (-1)
physical_mode = phonons.physical_mode.reshape(phonons.n_phonons)
velocity = phonons.velocity.real.reshape((phonons.n_phonons, 3))
mean_free_path_inversion = Conductivity(phonons=phonons, method='inverse', storage='numpy').mean_free_path
tau_inversion = np.zeros_like(mean_free_path_inversion)
for alpha in range(3):
    tau_inversion[physical_mode, alpha] = np.abs(np.divide(mean_free_path_inversion[physical_mode, alpha],
                                                           velocity[physical_mode, alpha]))

plt.figure()
plt.plot(frequency, tau_inversion[:, 0], 'r.', label=r'$\tau_{inv,x}$')
plt.plot(frequency, tau_inversion[:, 1], 'b.', label=r'$\tau_{inv,y}$')
plt.plot(frequency, tau_inversion[:, 2], 'k.', label=r'$\tau_{inv,z}$')
plt.plot(frequency, tau_RTA, 'g.', label=r'$\tau_{RTA}$')
plt.xlabel("$\\nu$ (THz)", fontsize=16)
plt.ylabel("$\\tau \ (ps)$", fontsize=16)
plt.yscale('log')
plt.legend(loc='best', fontsize=20)
plt.xlim([5, 50])
plt.savefig(folder + '/phonon_life_time.png', dpi=300)

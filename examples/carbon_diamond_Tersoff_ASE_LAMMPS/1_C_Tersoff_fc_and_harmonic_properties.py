# Example: carbon diamond, Tersoff potential
# Computes: 2nd, 3rd order force constants and harmonic properties for carbon diamond (2 atoms per cell)
# Uses: ASE, LAMMPS, LAMMPSlib
# External files: forcefields/C.tersoff

# Import necessary packages

from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from kaldo.conductivity import Conductivity
from kaldo.controllers import plotter
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('seaborn-poster')

# --  Set up the coordinates of the system and the force constant calculations -- #

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

# Compute 2nd and 3rd IFCs with the defined calculator
# delta_shift: finite difference displacement, in angstrom
forceconstants.second.calculate(LAMMPSlib(**lammps_inputs), delta_shift=1e-4)
forceconstants.third.calculate(LAMMPSlib(**lammps_inputs), delta_shift=1e-4)

# -- Set up the phonon object and the harmonic property calculations -- #

# Configure phonon object
# 'k_points': number of k-points
# 'is_classic': specify if the system is classic, True for classical and False for quantum
# 'temperature: temperature (Kelvin) at which simulation is performed
# 'folder': name of folder containing phonon property and thermal conductivity calculations
# 'storage': Format to storage phonon properties ('formatted' for ASCII format data, 'numpy'
#            for python numpy array and 'memory' for quick calculations, no data stored")


# Define the k-point mesh using 'kpts' parameter
k_points = 5  # 'k_points'=5 k points in each direction
phonons_config = {'kpts': [k_points, k_points, k_points],
                  'is_classic': False,
                  'temperature': 300,  # 'temperature'=300K
                  'folder': 'ALD_c_diamond',
                  'storage': 'formatted'}

# Set up phonon object by passing in configuration details and the forceconstants object computed above
phonons = Phonons(forceconstants=forceconstants, **phonons_config)

# Visualize phonon dispersion, group velocity and density of states with
# the build-in plotter.

# 'with_velocity': specify whether to plot both group velocity and dispersion relation
# 'is_showing':specify if figure window pops up during simulation
plotter.plot_dispersion(phonons, with_velocity=True, is_showing=False)
plotter.plot_dos(phonons, is_showing=False)

# Visualize heat capacity vs frequency and
# 'order': Index order to reshape array,
# 'order'='C' for C-like index order; 'F' for Fortran-like index order

# Define the base folder to contain plots
# 'base_folder':name of the base folder
folder = phonons.get_folder_from_label(base_folder='plots')
if not os.path.exists(folder):
    os.makedirs(folder)
# Define a Boolean flag to specify if figure window pops during simulation
is_show_fig = False

frequency = phonons.frequency.flatten(order='C')
heat_capacity = phonons.heat_capacity.flatten(order='C')
plt.figure()
plt.scatter(frequency[3:], 1e23 * heat_capacity[3:],
            s=5)  # Get rid of the first three non-physical modes while plotting
plt.xlabel("$\\nu$ (THz)", fontsize=16)
plt.ylabel(r"$C_{v} \ (10^{23} \ J/K)$", fontsize=16)
plt.savefig(folder + '/cv_vs_freq.png', dpi=300)
if not is_show_fig:
    plt.close()
else:
    plt.show()

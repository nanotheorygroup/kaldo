# Example 5.1: carbon diamond, Tersoff potential 
# Computes: 2nd, 3rd order force constants and (phonon) harmonic properties for carbon diamond (2 atoms per cell)
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
plt.style.use('seaborn-poster')

### Set up the coordinates of the system and the force constant calculations ####

# Define the system according to ASE style. 'a': lattice parameter(Angstrom)
atoms = bulk('C', 'diamond', a=3.566)

# Replicate the unit cell 'nrep'=3 times
nrep = 3
supercell = np.array([nrep, nrep, nrep])

# Configure force constant calculator
forceconstants_config = {'atoms': atoms, 'supercell': supercell, 'folder': 'fc'}
forceconstants = ForceConstants(**forceconstants_config)

# Define input information for the ase LAMMPSlib calculator
# Terosff potential for carbon (C.tersoff) is used for this example.
lammps_inputs = {'lmpcmds': [
    'pair_style tersoff',
    'pair_coeff * * forcefields/C.tersoff C'],
    'keep_alive': True,
    'log_file': 'lammps-c-diamond.log'}

# Compute 2nd and 3rd IFCs with the defined calculator
forceconstants.second.calculate(LAMMPSlib(**lammps_inputs))
forceconstants.third.calculate(LAMMPSlib(**lammps_inputs))

### Set up the phonon object and the harmonic property calculations ####

# Configure phonon object
# 'k_points': number of k-points
# 'is_classic': specify if the system is classic, True for classic and False for quantum
# 'temperature: temperature (Kelvin) at which simulation is performed
# 'folder': name of folder containing phonon property and thermal conductivity calculations
# 'is_tf_backend': specify if 3rd order phonons scattering calculations should be performed on tensorflow (True) or numpy (False)
# 'storage': Format to storage phonon properties ('formatted' for ASCII format data, 'numpy' 
#            for python numpy array and 'memory' for quick calculations, no data stored")


# Define the k-point mesh using 'kpts' parameter
k_points = 5 #'k_points'=5 k-points in each direction
phonons_config = {'kpts': [k_points, k_points, k_points],
                  'is_classic': False, 
                  'temperature': 300, #'temperature'=300K
                  'folder': 'ald_c_diamond',
                  'is_tf_backend': False,
		   'storage': 'formatted'}

# Set up phonon object by passing configuration details and the forceconstants object computed above
phonons = Phonons(forceconstants=forceconstants, **phonons_config)

# Visualize phonon dispersion, group velocity and density of states with 
# the build-in plotter by passing in the phonon object set up above
# 'with_velocity': specify whether to plot both velocity and dispersion relation
# 'is_showing':speify if figure pops up during simulation
plotter.plot_dispersion(phonons,with_velocity =True,is_showing=False)
plotter.plot_dos(phonons,is_showing=False)

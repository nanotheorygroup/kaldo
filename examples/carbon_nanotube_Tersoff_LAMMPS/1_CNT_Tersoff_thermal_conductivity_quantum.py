# Example: (10,0) Carbon Nanotube, Tersoff optimized potential
# Computes: infinite size and finite size thermal conductivity for(10,0) CNT (40 atoms per cell)
# Uses: LAMMPS
# External files: forcefields/C.optimize.tersoff C(O)

# Import necessary packages
from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import numpy as np

# -- Set up force constant objects via interface to LAMMPS --#

# Replicate the unit cell 'nrep'=3 times
# Please be consistent with supercell structures generated 
# from 0_generate_LAMMPS_input_and_supercell_structures.py 
nrep = 3
supercell = np.array([1, 1, nrep])

# Load in computed 2nd, 3rd IFCs from LAMMPS outputs
forceconstants = ForceConstants.from_folder(folder='fc_CNT', supercell=supercell, format='lammps')

# -- Set up the phonon object and the anharmonic properties calculations --#

# Configure phonon object
# 'k_points': number of k-points
# 'is_classic': specify if the system is classic, True for classical and False for quantum
# 'temperature: temperature (Kelvin) at which simulation is performed
# 'folder': name of folder containing phonon property and thermal conductivity calculations
# 'storage': Format to storage phonon properties ('formatted' for ASCII format data, 'numpy'
#            for python numpy array and 'memory' for quick calculations, no data stored)

# Define the k-point mesh using 'kpts' parameter
k_points = 151
# For one dimension system, only replicate k-point mesh
# along the direction of transport (z-axis)
kpts = [1, 1, k_points]
temperature = 300
# Create a phonon object
phonons = Phonons(forceconstants=forceconstants,
                  kpts=kpts,
                  is_classic=False,
                  temperature=temperature,
                  is_nw=True,
                  folder='ALD_CNT',
                  storage='numpy')

# Compute conductivity from direct inversion of
# scattering matrix for infinite size samples
# 29.92 is the rescale factor between
# the volume of the simulation box and the
# volume of the 10,0 Carbon Nanotube
inverse_conductivity = Conductivity(phonons=phonons, method='inverse').conductivity
inverse_conductivity_matrix = inverse_conductivity.sum(axis=0)
print('Infinite size conductivity from inversion (W/m-K): %.3f' % (29.92 * inverse_conductivity_matrix[2, 2]))

# Config finite size conductivity from direct inversion of scattering matrix
# Specific finite size length, in angstrom,
# along the direction of transport (z-axis)
# finite_length_method ='ms' for the Mckelvey-Schockley method
finite_size_conductivity_config = {'method': 'inverse',
                                   'length': (0, 0, 1000000000),
                                   'finite_length_method': 'ms',
                                   'storage': 'numpy'}
finite_size_inverse_conductivity = Conductivity(phonons=phonons, **finite_size_conductivity_config)
finite_size_inverse_conductivity_matrix = finite_size_inverse_conductivity.conductivity.sum(axis=0)
# Verify finite size conductivity at sufficient long length approaches to the infinite size limit.
print('Finite size conductivity from inversion (W/m-K): %.3f' % (29.92 * finite_size_inverse_conductivity_matrix[2, 2]))

# Example 2.1: amorphous silicon, Tersoff potential 
# Computes: Quasi Harmonic Green Kubo (QHGK) thermal conductivity for amorphous silicon (512 atoms)
# Uses: LAMMPS
# External files: forcefields/Si.tersoff

# Import necessary packages

from ase.io import read
from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import numpy as np

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

phonons_config = {'is_classic': False, 
                  'temperature': 300, #'temperature'=300K
                  'folder': 'ALD_aSi512',
                   'third_bandwidth':0.5/4.135, # 0.5 eV is used here.
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

print('\n')
qhgk_cond = Conductivity(phonons=phonons, method='qhgk', storage='numpy')
qhgk_cond.diffusivity_bandwidth = phonons.bandwidth
print('Conductivity from QHGK (W/m-K): %.3f' % (np.mean(np.diag(qhgk_cond.conductivity.sum(axis=0)))))
print(qhgk_cond.conductivity.sum(axis=0))

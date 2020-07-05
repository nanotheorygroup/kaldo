from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import numpy as np
import os

# We start from the atoms object
atoms = bulk('C', 'diamond', a=3.566)
supercell = np.array([3,3,3])
forceconstants_config  = {'atoms':atoms,'supercell': supercell,'folder':'fd'}
forceconstants = ForceConstants(**forceconstants_config)

# Define calculator input information
lammps_inputs = {'lmpcmds': [
                                  'pair_style tersoff',
                                 'pair_coeff * * forcefields/C.tersoff C'],
                             'keep_alive':True,
                     'log_file': 'lammps-c-diamond.log'}
    


# Compute 2nd and 3rd IFCs with the defined calculators
forceconstants.calculate_second(LAMMPSlib(**lammps_inputs))
forceconstants.calculate_third(LAMMPSlib(**lammps_inputs))

# Config and create a phonon object 
k = 5
phonons_config = {'kpts': [k, k, k],
                  'is_classic': False,
                  'temperature': 300,
                  'folder':'ald_c_diamond',
                  'is_tf_backend':False,
                  'storage':'numpy'}
phonons = Phonons(forceconstants=forceconstants, **phonons_config)


# Compute thermal conductivity with various methods
print('\n')
rta_cond_matrix = Conductivity(phonons=phonons, method='rta').conductivity.sum(axis=0)
print('Rta conductivity (W/mK): %.3f'%(np.mean(np.diag(rta_cond_matrix))))
print(rta_cond_matrix)

print('\n')
sc_cond_matrix = Conductivity(phonons=phonons, method='sc',n_iterations=20).conductivity.sum(axis=0)
print('Self-consistent conductivity (W/mK): %.3f'%(np.mean(np.diag(sc_cond_matrix))))
print(sc_cond_matrix)

print('\n')
qhgk_cond_matrix = Conductivity(phonons=phonons, method='qhgk').conductivity.sum(axis=0)
print('Qhgk conductivity (W/mK): %.3f'%(np.abs(np.mean(np.diag(qhgk_cond_matrix)))))
print(qhgk_cond_matrix)

print('\n')
inv_cond_matrix = (Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0))
print('Inverse conductivity (W/mK): %.3f'%(np.mean(np.diag(inv_cond_matrix))))
print(inv_cond_matrix)

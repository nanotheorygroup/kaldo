from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ballistico.conductivity import Conductivity
from ballistico.finitedifference import FiniteDifference
from ballistico.phonons import Phonons
import numpy as np

# We start from the atoms object
    
atoms = bulk('C', 'diamond', a=3.566)

# Define calculator input infomration

lammps_inputs = {'lmpcmds': [
                                  'pair_style tersoff',
                                 'pair_coeff * * forcefields/C.tersoff C'],
                     
                     'log_file': 'lammps-c-diamond.log'}
    
    
# Create a finite difference object

finite_difference_config  = {'atoms':atoms,
                            'supercell':[3,3,3],
                            'calculator':LAMMPSlib,
                            'calculator_inputs':lammps_inputs,
                            'folder':'fd'}

finite_difference = FiniteDifference(**finite_difference_config)

k = 5
phonons_config = {'kpts': [k, k, k],
                  'is_classic': False,
                  'temperature': 300,
                  'folder':'ald',
                  'is_tf_backend':False,
                  'storage':'numpy'}
phonons = Phonons(finite_difference=finite_difference, **phonons_config)

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

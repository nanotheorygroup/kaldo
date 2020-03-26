from ballistico.finitedifference import FiniteDifference
import numpy as np
from ballistico.phonons import Phonons
from ballistico.conductivity import Conductivity

finite_difference = FiniteDifference.from_folder('hiphive_si_bulk', [3, 3, 3],format='hiphive')
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
sc_cond_matrix = Conductivity(phonons=phonons, method='sc').conductivity.sum(axis=0)
print('Self-consistent conductivity (W/mK): %.3f'%(np.mean(np.diag(sc_cond_matrix))))
print(rta_cond_matrix)

print('\n')
qhgk_cond_matrix = Conductivity(phonons=phonons, method='qhgk').conductivity.sum(axis=0)
print('Qhgk conductivity (W/mK): %.3f'%(np.mean(np.diag(qhgk_cond_matrix))))
print(rta_cond_matrix)

inv_cond_matrix = (Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0))
print('Inverse conductivity (W/mK): %.3f'%(np.mean(np.diag(inv_cond_matrix))))
print(inv_cond_matrix)
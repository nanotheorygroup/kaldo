import numpy as np
import timeit
import inspect
from kaldo.observables.secondorder import SecondOrder
from kaldo.observables.thirdorder import ThirdOrder
from ase.calculators.lammpslib import LAMMPSlib
from kaldo.forceconstants import ForceConstants
from kaldo.conductivity import Conductivity
from kaldo.phonons import Phonons
from ase.io import read
import numpy as np

ge_concentration='10'
# Set up Forceconstants Object
folder_string = 'structures/1728_atom/aSiGe_C'+ge_concentration+'/'
print(folder_string)
atoms = read(folder_string+'replicated_atoms.xyz',format='xyz')

forceconstants = ForceConstants(atoms=atoms,
    folder=folder_string+'/ald')

# Create Phonon Object
phonons = Phonons(forceconstants=forceconstants,
    is_classic=False, # quantum stats
    temperature=300, # 300 K
    folder=folder_string,
    third_bandwidth=0.5/4.135, # 0.5 eV smearing
    broadening_shape='gauss') # shape of smearing

print(phonons.frequency)

def check_zeros(matrix):
    raise ValueError('Matrix of zeros!')
    if not np.sum(matrix):
        return True
    else:
        return False


# if check_zeros(zeros):
#     print('error')
# else:
#     print('not all zeros!')
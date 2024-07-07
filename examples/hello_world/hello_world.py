"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity

print("Preparing phonons object.")
forceconstants = ForceConstants.from_folder(folder='kaldo/tests/si-crystal',
                                            supercell=[3, 3, 3],
                                            format='eskm')
phonons = Phonons(forceconstants=forceconstants,
                  kpts=[5, 5, 5],
                  is_classic=False,
                  temperature=300,
                  storage='memory')
cond = Conductivity(phonons=phonons, method='inverse', storage='memory').conductivity.sum(axis=0)
print(cond)

"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import pytest


@pytest.yield_fixture(scope="session")
def phonons():
    print ("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(folder='kaldo/tests/si-crystal',
                                                     supercell=[3, 3, 3],
                                                     format='eskm')
    phonons = Phonons(forceconstants=forceconstants,
                      kpts=[5, 5, 5],
                      is_classic=False,
                      temperature=300,
                      include_isotopes=True,
                      storage='memory')
    return phonons





def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory').conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_approx_equal(cond, 187.5, significant=4)

def test_rta_conductivity(phonons):
    cond = np.abs(np.mean(Conductivity(phonons=phonons, method='rta', storage='memory').conductivity.sum(axis=0).diagonal()))
    np.testing.assert_approx_equal(cond, 183.3, significant=4)




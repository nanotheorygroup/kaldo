"""
Unit and regression test for the kaldo package.
"""

# Imports
from kaldo.forceconstants import ForceConstants
from kaldo.conductivity import Conductivity
from kaldo.phonons import Phonons
import ase.units as units
import numpy as np
import pytest

# NOTE: the scope of this fixture needs to be 'function' for these tests to work properly.
@pytest.yield_fixture(scope="function")
def phonons():
    print ("Preparing phonons object.")

    # Create a finite difference object
    forceconstants = ForceConstants.from_folder(folder='kaldo/tests/si-crystal',
                                                     supercell=[3, 3, 3],
                                                     format='eskm')
    phonons = Phonons(forceconstants=forceconstants,
                      kpts=[5, 5, 5],
                      is_classic=False,
                      temperature=300,
                      storage='memory')
    return phonons

def test_triangle_broadening(phonons):
    phonons.broadening_shape='triangle'
    np.testing.assert_approx_equal(phonons.bandwidth[0][3], 0.10344, significant=4)

def test_gaussian_broadening(phonons):
    phonons.broadening_shape='gauss'
    np.testing.assert_approx_equal(phonons.bandwidth[0][3], 0.12086, significant=4)

def test_lorentz_broadening(phonons):
    phonons.broadening_shape='lorentz'
    np.testing.assert_approx_equal(phonons.bandwidth[0][3], 0.09793, significant=4)
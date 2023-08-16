"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import ase.units as units
import pytest


@pytest.fixture(scope="session")
def phonons():
    print ("Preparing phonons object.")

    # Create a finite difference object
    forceconstants = ForceConstants.from_folder(folder='kaldo/tests/si-amorphous', format='eskm')

    # # Create a phonon object
    phonons = Phonons(forceconstants=forceconstants,
                      is_classic=True,
                      temperature=300,
                      third_bandwidth= 0.05 / 4.135,
                      broadening_shape='triangle',
                      storage='memory')
    return phonons


def test_first_gamma(phonons):
    thztomev = units.J * units._hbar * 2 * np.pi * 1e15
    np.testing.assert_approx_equal(phonons.bandwidth[0, 3] * thztomev / (2 * np.pi), 22.451, significant=3)


def test_second_gamma(phonons):
    thztomev = units.J * units._hbar * 2 * np.pi * 1e15
    np.testing.assert_approx_equal(phonons.bandwidth[0, 4] * thztomev / (2 * np.pi), 23.980, significant=3)


def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory').conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_approx_equal(cond, 0.996, significant=2)


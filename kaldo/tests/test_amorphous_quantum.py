"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
import ase.units as units
import pytest


@pytest.yield_fixture(scope="session")
def phonons():
    print ("Preparing phonons object.")

    # Create a finite difference object
    forceconstants = ForceConstants.from_folder(folder='kaldo/tests/si-amorphous',
                                                     format='eskm')

    # # Create a phonon object
    phonons = Phonons(forceconstants=forceconstants,
                      is_classic=False,
                      temperature=300,
                      third_bandwidth= 0.05 / 4.135,
                      broadening_shape='triangle',
                      storage='memory')
    return phonons

def test_first_gamma(phonons):
    thztomev = units.J * units._hbar * 2 * np.pi * 1e15
    np.testing.assert_approx_equal(phonons.bandwidth[0, 3] * thztomev / (2 * np.pi), 22.216, significant=3)


def test_second_gamma(phonons):
    thztomev = units.J * units._hbar * 2 * np.pi * 1e15
    np.testing.assert_approx_equal(phonons.bandwidth[0, 4] * thztomev / (2 * np.pi), 23.748, significant=3)

def test_participation_ratio(phonons):
    participation = phonons.participation_ratio.squeeze()
    np.testing.assert_approx_equal(participation[100], 0.52007, significant=3)

def test_velocity_amorphous(phonons):
    np.testing.assert_approx_equal(phonons.velocity.squeeze()[10, 2], 0, significant=2)



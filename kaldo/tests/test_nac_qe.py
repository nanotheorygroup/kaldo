"""
Unit and regression test for the kaldo package.
Tests the unfolding routine for second (and soon, third) order
force constants.
"""

# Import package, test suite, and other packages as needed
import numpy as np
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import pytest


@pytest.fixture(scope="session")
def phonons():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/mgo",
        supercell=[5, 5, 5],
        only_second=True,
        format="shengbte-d3q",
    )
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[3, 3, 3],
        is_classic=False,
        temperature=300,
        is_unfolding=True,
        storage="memory",
    )
    return phonons


def test_unfolding_dispersion(phonons):
    q_point = np.array([0.3, 0, 0.3])  # chosen to check we get a degenerate pair for both acoustic and optical
    frequency_expected = np.array([7.05676324,  7.1162959 , 10.96372809, 12.71796071, 12.75355179, 17.50524562])
    frequency_actual = HarmonicWithQ(q_point=q_point, second=phonons.forceconstants.second, is_unfolding=True).frequency
    frequency_actual = frequency_actual.flatten()  # HWQ outputs a 2d array
    np.testing.assert_array_almost_equal(frequency_expected, frequency_actual, decimal=2)

def test_unfolding_velocity(phonons):
    q_point = np.array([0.3, 0, 0.3])
    velocity_expected = np.array([34.64999537, 36.2252077 , 35.78366205,  7.71079558,  8.99215236, 23.52709405])
    velocity = HarmonicWithQ(q_point=q_point, second=phonons.forceconstants.second, is_unfolding=True).velocity
    velocity_actual = np.linalg.norm(velocity, axis=-1).flatten()
    np.testing.assert_array_almost_equal(velocity_expected, velocity_actual, decimal=2)

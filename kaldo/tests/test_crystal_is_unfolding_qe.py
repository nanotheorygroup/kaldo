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
        folder="kaldo/tests/si-crystal/qe", supercell=[3, 3, 3], format="qe-vasp"
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
    frequency_expected = np.array([4.11380807, 4.11380825, 8.44285067, 14.00947531, 14.00947536, 14.37330857])
    frequency_actual = HarmonicWithQ(q_point=q_point, second=phonons.forceconstants.second, is_unfolding=True).frequency
    frequency_actual = frequency_actual.flatten()  # HWQ outputs a 2d array
    np.testing.assert_array_almost_equal(frequency_expected, frequency_actual, decimal=2)

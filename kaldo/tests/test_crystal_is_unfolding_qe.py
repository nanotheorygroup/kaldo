"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
import numpy as np
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import pytest


@pytest.yield_fixture(scope="session")
def phonons():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(folder='kaldo/tests/si-crystal/qe',
                                                supercell=[3, 3, 3],
                                                format='shengbte-qe')
    phonons = Phonons(forceconstants=forceconstants,
                      kpts=[3, 3, 3],
                      is_classic=False,
                      temperature=300,
                      is_unfolding=True,
                      storage='memory')
    return phonons

def test_unfolding_dispersion(phonons):
    freqs = np.array([ 4.11380807,  4.11380825,  8.44285067, 14.00947531, 14.00947536, 14.37330857])
    q_point = np.array([0.3, 0, 0.3])
    phonon = HarmonicWithQ(q_point=q_point, second=phonons.forceconstants.second, is_unfolding=True)
    np.testing.assert_array_almost_equal(freqs, phonon.frequency, decimal=2)



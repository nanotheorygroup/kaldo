"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import pytest


@pytest.fixture(scope="session")
def phonons():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/ge-crystal",
        supercell=[10, 10, 10],
        format="qe-d3q",
        only_second=True,
    )
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[7, 7, 7],
        is_classic=False,
        temperature=300,
        include_isotopes=True,
        storage="memory",
    )
    return phonons


def test_frequency(phonons):
    freq = phonons.frequency.flatten()
    np.testing.assert_approx_equal(freq[3], 8.99, significant=3)


def test_iso_bw(phonons):
    freq = phonons.isotopic_bandwidth.flatten()
    np.testing.assert_approx_equal(freq[3], 7.85 * 1e-3, significant=3)

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
    iso_bw = phonons.isotopic_bandwidth.flatten()
    # Modes 3,4,5 form a degenerate triplet. Individual values depend on
    # the eigenvector basis; the mean over the group is basis-invariant.
    degen_mean = np.mean(iso_bw[3:6])
    np.testing.assert_approx_equal(degen_mean, 7.75 * 1e-3, significant=3)

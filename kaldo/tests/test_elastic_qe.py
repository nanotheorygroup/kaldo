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
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-crystal/qe", supercell=[3, 3, 3], format="shengbte-qe")
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[5, 5, 5],
        is_classic=False,
        temperature=300,
        storage="memory",
    )
    return phonons


def test_c11(phonons):
    cijkl = phonons.forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[0, 0, 0, 0], 74.49, significant=3)

def test_c12(phonons):
    cijkl = phonons.forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[0, 0, 1, 1], 38.00, significant=3)


def test_c44(phonons):
    cijkl = phonons.forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[1, 2, 1, 2], 74.49, significant=3)

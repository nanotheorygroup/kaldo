"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
import pytest


@pytest.fixture(scope="session")
def forceconstants():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-crystal/qe", supercell=[3, 3, 3], format="shengbte-qe")
    return forceconstants


def test_c11(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[0, 0, 0, 0], 227.5, significant=3)


def test_c12(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[0, 0, 1, 1], 51.71, significant=3)


def test_c44(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[1, 2, 1, 2], 74.49, significant=3)

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
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-crystal/vasp", supercell=[5, 5, 5], format="vasp-sheng")
    return forceconstants


def test_c11(forceconstants):
    cijkl = forceconstants.elastic_prop()
    # The old values used one replica vector for every basis pair.  These
    # references pin the pair-shortest q->0 IFC moments instead.
    np.testing.assert_allclose(cijkl[0, 0, 0, 0], 158.732508, rtol=5e-4, atol=0.0)


def test_c12(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_allclose(cijkl[0, 0, 1, 1], 63.470281, rtol=5e-4, atol=0.0)


def test_c44(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_allclose(cijkl[1, 2, 1, 2], 77.864122, rtol=5e-4, atol=0.0)

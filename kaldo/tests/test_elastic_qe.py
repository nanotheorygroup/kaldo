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
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-crystal/qe", supercell=[3, 3, 3], format="qe-sheng")
    return forceconstants


def test_c11(forceconstants):
    cijkl = forceconstants.elastic_prop()
    # This nonpolar q2r fixture retains QE's direct-periodic convention; the
    # q->0 moments must therefore preserve its existing matdyn-compatible result.
    np.testing.assert_allclose(cijkl[0, 0, 0, 0], 227.509722, rtol=5e-4, atol=0.0)


def test_c12(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_allclose(cijkl[0, 0, 1, 1], 51.710100, rtol=5e-4, atol=0.0)


def test_c44(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_allclose(cijkl[1, 2, 1, 2], 74.494384, rtol=5e-4, atol=0.0)

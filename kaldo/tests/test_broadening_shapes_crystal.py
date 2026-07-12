"""
Unit and regression test for the kaldo package.
"""

# Imports
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import numpy as np
import pytest


# NOTE: the scope of this fixture needs to be 'function' for these tests to work properly.
@pytest.fixture(scope="function")
def phonons():
    print("Preparing phonons object.")

    # Create a finite difference object
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-crystal", supercell=[3, 3, 3], format="eskm")
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[5, 5, 5],
        is_classic=False,
        temperature=300,
        broadening_kernel="tdep",  # gauge-invariant sigma; the shapes are what these tests probe (#290)
        storage="memory",
    )
    return phonons


def test_triangle_broadening(phonons):
    phonons.broadening_shape = "triangle"
    np.testing.assert_allclose(np.mean(phonons.bandwidth[0][3:6]), 0.331071, rtol=1e-3, atol=0.0)


def test_gaussian_broadening(phonons):
    phonons.broadening_shape = "gauss"
    np.testing.assert_allclose(np.mean(phonons.bandwidth[0][3:6]), 0.467721, rtol=1e-3, atol=0.0)


def test_lorentz_broadening(phonons):
    phonons.broadening_shape = "lorentz"
    np.testing.assert_allclose(np.mean(phonons.bandwidth[0][3:6]), 0.364596, rtol=1e-3, atol=0.0)

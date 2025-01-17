"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import pytest


@pytest.fixture(scope="function")
def phonons():
    print("Preparing phonons object.")

    # Create a finite difference object
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-amorphous", format="eskm")

    # # Create a phonon object
    phonons = Phonons(forceconstants=forceconstants, is_classic=False, storage="memory")
    return phonons


def test_af_conductivity_50(phonons):
    phonons.temperature = 50
    cond = (
        Conductivity(phonons=phonons, method="qhgk", storage="memory", diffusivity_bandwidth=0.025)
        .conductivity.sum(axis=0)
        .diagonal()
        .mean()
    )
    expected_cond = 0.098
    np.testing.assert_approx_equal(cond, expected_cond, significant=2)


def test_af_conductivity_300(phonons):
    phonons.temperature = 300
    cond = (
        Conductivity(phonons=phonons, method="qhgk", storage="memory", diffusivity_bandwidth=0.025)
        .conductivity.sum(axis=0)
        .diagonal()
        .mean()
    )
    expected_cond = 0.532
    np.testing.assert_approx_equal(cond, expected_cond, significant=2)


def test_frequency(phonons):
    frequencies = phonons.frequency.flatten()
    np.testing.assert_approx_equal(frequencies[5], 1.64, significant=2)

"""
Unit and regression test for the ballistico package.
"""

# Import package, test suite, and other packages as needed
from ballistico.finitedifference import FiniteDifference
import numpy as np
from ballistico.phonons import Phonons
import ballistico.controllers.conductivity as bac
from tempfile import TemporaryDirectory
import pytest


@pytest.yield_fixture(scope="function")
def phonons():
    print ("Preparing phonons object.")

    # Create a finite difference object
    finite_difference = FiniteDifference.from_folder(folder='ballistico/tests/si-amorphous', format='eskm')

    # # Create a phonon object
    with TemporaryDirectory() as td:
        phonons = Phonons(finite_difference=finite_difference,
                          is_classic=False,
                          folder=td)

        yield phonons
    print ("Cleaning up.")


def test_af_conductivity_50(phonons):
    phonons.temperature = 50
    gamma_in = 0.025
    cond = bac.conductivity(phonons, method='qhgk', gamma_in=gamma_in).sum(axis=0).diagonal().mean()
    expected_cond = 0.098
    np.testing.assert_approx_equal(cond, expected_cond, significant=2)


def test_af_conductivity_300(phonons):
    phonons.temperature = 300
    gamma_in = 0.025
    cond = bac.conductivity(phonons, method='qhgk', gamma_in=gamma_in).sum(axis=0).diagonal().mean()
    expected_cond = 0.532
    np.testing.assert_approx_equal(cond, expected_cond, significant=2)

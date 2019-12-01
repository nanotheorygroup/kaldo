"""
Unit and regression test for the ballistico package.
"""

# Import package, test suite, and other packages as needed
from ballistico.finitedifference import FiniteDifference
import numpy as np
from ballistico.phonons import Phonons
import ase.units as units
import shutil
import pytest

TMP_FOLDER = 'ballistico/tests/tmp-folder'


@pytest.yield_fixture(scope="session")
def phonons():
    print ("Preparing phonons object.")

    # Create a finite difference object
    finite_difference = FiniteDifference.from_folder(folder='ballistico/tests/si-amorphous',
                                                     format='eskm')

    # # Create a phonon object
    phonons = Phonons(finite_difference=finite_difference,
                      is_classic=False,
                      temperature=300,
                      folder=TMP_FOLDER,
                      sigma_in= 0.05 / 4.135,
                      broadening_shape='triangle')

    yield phonons
    print ("Cleaning up.")
    shutil.rmtree(TMP_FOLDER, ignore_errors=True)


def test_first_gamma(phonons):
    THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
    np.testing.assert_approx_equal(phonons.gamma[3] * THZTOMEV / (2 * np.pi), 22.216, significant=3)


def test_second_gamma(phonons):
    THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
    np.testing.assert_approx_equal(phonons.gamma[4] * THZTOMEV / (2 * np.pi), 23.748, significant=3)



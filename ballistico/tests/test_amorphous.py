"""
Unit and regression test for the ballistico package.
"""

# Import package, test suite, and other packages as needed
from finitedifference.finitedifference import FiniteDifference
import numpy as np
from ballistico.phonons import Phonons
import ballistico.conductivity as bac
import ase.units as units
TMP_FOLDER = 'ballistico/tests/tmp-folder'


def create_phonons():

    # Create a finite difference object
    finite_difference = FiniteDifference.import_from_dlpoly_folder(folder='ballistico/tests/si-amorphous')

    # # Create a phonon object
    phonons = Phonons(finite_difference=finite_difference,
                      is_classic=True,
                      temperature=300,
                      folder=TMP_FOLDER,
                      sigma_in= 0.05 / 4.135,
                      broadening_shape='triangle')
    return phonons


def test_first_gamma():
    phonons = create_phonons()
    THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
    np.testing.assert_approx_equal(phonons.gamma[3] * THZTOMEV / (2 * np.pi), 22.451, significant=3)


def test_second_gamma():
    phonons = create_phonons()
    THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
    np.testing.assert_approx_equal(phonons.gamma[4] * THZTOMEV / (2 * np.pi), 23.980, significant=3)


def test_qhgk_conductivity():
    phonons = create_phonons()
    cond = bac.conductivity(phonons, method='qhgk').sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_approx_equal(cond, 0.99, significant=2)


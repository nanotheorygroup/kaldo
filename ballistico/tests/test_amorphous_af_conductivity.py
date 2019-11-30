"""
Unit and regression test for the ballistico package.
"""

# Import package, test suite, and other packages as needed
from ballistico.finitedifference import FiniteDifference
import numpy as np
from ballistico.phonons import Phonons
import ballistico.controllers.conductivity as bac

TMP_FOLDER = 'ballistico/tests/tmp-folder'


def create_phonons(temperature):

    # Create a finite difference object
    finite_difference = FiniteDifference.from_folder(folder='ballistico/tests/si-amorphous', format='dlpoly')

    # Create a phonon object# # Create a phonon object
    phonons = Phonons (finite_difference=finite_difference,
                       is_classic=False,
                       temperature=temperature,
                       folder=TMP_FOLDER)
    return phonons

def test_af_conductivity_50():
    temperature = 50
    gamma_in = 0.025
    phonons = create_phonons(temperature)
    cond = bac.conductivity(phonons, method='qhgk', gamma_in=gamma_in).sum(axis=0).diagonal().mean()
    expected_cond = 0.101
    np.testing.assert_approx_equal(cond, expected_cond, significant=2)


def test_af_conductivity_300():
    temperature = 300
    gamma_in = 0.025
    phonons = create_phonons(temperature)
    cond = bac.conductivity(phonons, method='qhgk', gamma_in=gamma_in).sum(axis=0).diagonal().mean()
    expected_cond = 0.532
    np.testing.assert_approx_equal(cond, expected_cond, significant=2)

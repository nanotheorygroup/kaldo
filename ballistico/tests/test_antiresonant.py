from ballistico.finitedifference import FiniteDifference
from ballistico.phonons import Phonons
from ballistico.conductivity import Conductivity
import matplotlib.pyplot as plt
import ase.io
"""
Unit and regression test for the ballistico package.
"""

# Import package, test suite, and other packages as needed
from ballistico.finitedifference import FiniteDifference
import numpy as np
from ballistico.phonons import Phonons
from ballistico.conductivity import Conductivity
import pytest


@pytest.yield_fixture(scope="function")
def phonons():
    print ("Preparing phonons object.")
    finite_difference = FiniteDifference.from_folder(folder='ballistico/tests/si-amorphous', format='eskm')
    phonons = Phonons(finite_difference=finite_difference,
                      is_classic=False,
                      temperature=300,
                      third_bandwidth=1 / 4.135,
                      broadening_shape='triangle',
                      is_tf_backend=True,
                      storage='memory')
    phonons.diffusivity_bandwidth = phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes))
    return phonons


def test_af_conductivity_without_antiresonant(phonons):
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory').conductivity.sum(axis=0)
    cond = (cond.diagonal().mean())
    np.testing.assert_approx_equal(cond, 0.804, significant=3)


def test_af_conductivity_with_antiresonant(phonons):
    phonons.is_diffusivity_including_antiresonant = True
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory').conductivity.sum(axis=0)
    cond = (cond.diagonal().mean())
    np.testing.assert_approx_equal(cond, 0.825, significant=3)


def test_af_conductivity_without_antiresonant_gauss(phonons):
    phonons.diffusivity_shape = 'gauss'
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory').conductivity.sum(axis=0)
    cond = (cond.diagonal().mean())
    np.testing.assert_approx_equal(cond, 0.8305, significant=3)


def test_af_conductivity_with_antiresonant_gauss(phonons):
    phonons.diffusivity_shape = 'gauss'
    phonons.is_diffusivity_including_antiresonant = True
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory').conductivity.sum(axis=0)
    cond = (cond.diagonal().mean())
    np.testing.assert_approx_equal(cond, 0.8335, significant=3)

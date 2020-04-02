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
    np.testing.assert_approx_equal(cond, 0.8002305990273381, significant=3)


def test_af_conductivity_with_antiresonant(phonons):
    phonons.is_diffusivity_including_antiresonant = True
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory').conductivity.sum(axis=0)
    cond = (cond.diagonal().mean())
    np.testing.assert_approx_equal(cond, 0.821142483615389, significant=3)


def test_af_conductivity_without_antiresonant_gauss(phonons):
    phonons.diffusivity_shape = 'gauss'
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory').conductivity.sum(axis=0)
    cond = (cond.diagonal().mean())
    np.testing.assert_approx_equal(cond, 0.8299142655117929, significant=3)


def test_af_conductivity_with_antiresonant_gauss(phonons):
    phonons.diffusivity_shape = 'gauss'
    phonons.is_diffusivity_including_antiresonant = True
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory').conductivity.sum(axis=0)
    cond = (cond.diagonal().mean())
    np.testing.assert_approx_equal(cond, 0.8328416261335327, significant=3)


calculated_diffusivities = np.array([0.        , 0.        , 0.        , 0.56008541, 0.55693832,
                                     0.496376  , 0.56369572, 0.58721542, 0.68280378, 0.71898442])

def test_diffusivity(phonons):
    np.testing.assert_array_almost_equal(phonons.diffusivity.flatten().real[:10], calculated_diffusivities, decimal=3)

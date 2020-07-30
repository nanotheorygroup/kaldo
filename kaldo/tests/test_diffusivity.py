"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import tensorflow as tf
import pytest


@pytest.yield_fixture(scope="function")
def phonons():
    print ("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(folder='kaldo/tests/si-amorphous', format='eskm')
    phonons = Phonons(forceconstants=forceconstants,
                      is_classic=False,
                      temperature=300,
                      third_bandwidth=1 / 4.135,
                      broadening_shape='triangle',
                      storage='memory')
    return phonons


calculated_diffusivities_full = np.array([0.   , 0.   , 0.   , 0.560 , 0.557, 0.496, 0.564, 0.587, 0.683, 0.719])

calculated_diffusivities_two_sigma = np.array([0.   , 0.   , 0.   , 0.271, 0.313, 0.401, 0.485, 0.524, 0.627, 0.658])


def test_diffusivity(phonons):
    cond = Conductivity(phonons=phonons,
                        method='qhgk',
                        storage='memory',
                        diffusivity_bandwidth=phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes)))

    cond.conductivity
    np.testing.assert_array_almost_equal(cond.diffusivity.flatten().real[:10], calculated_diffusivities_full,
                                         decimal=3)


def test_diffusivity_small_threshold(phonons):
    cond = Conductivity(phonons=phonons,
                 method='qhgk', storage='memory')
    cond.diffusivity_threshold = 2

    cond.diffusivity_bandwidth = phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes))
    cond.conductivity
    np.testing.assert_array_almost_equal(cond.diffusivity.flatten().real[:10], calculated_diffusivities_two_sigma,
                                         decimal=3)


def test_diffusivity_large_threshold(phonons):

    cond = Conductivity(phonons=phonons,
                 method='qhgk', storage='memory')
    cond.diffusivity_threshold = 20
    cond.diffusivity_bandwidth = phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes))
    cond.conductivity
    np.testing.assert_array_almost_equal(cond.diffusivity.flatten().real[:10], calculated_diffusivities_full,
                                         decimal=3)



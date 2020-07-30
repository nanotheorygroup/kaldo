from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import matplotlib.pyplot as plt
import ase.io
"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
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


def test_af_conductivity_without_antiresonant(phonons):
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory')
    cond.diffusivity_bandwidth = phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes))
    cond = (cond.conductivity.sum(axis=0).diagonal().mean())
    np.testing.assert_approx_equal(cond, 0.804, significant=3)


def test_af_conductivity_with_antiresonant(phonons):
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory')
    cond.is_diffusivity_including_antiresonant = True
    cond.diffusivity_bandwidth = phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes))
    cond = (cond.conductivity.sum(axis=0).diagonal().mean())
    np.testing.assert_approx_equal(cond, 0.825, significant=3)


def test_af_conductivity_without_antiresonant_gauss(phonons):
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory')
    cond.diffusivity_shape = 'gauss'
    cond.diffusivity_bandwidth = phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes))
    cond = (cond.conductivity.sum(axis=0).diagonal().mean())
    np.testing.assert_approx_equal(cond, 0.8305, significant=3)


def test_af_conductivity_with_antiresonant_gauss(phonons):
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory')
    cond.diffusivity_shape = 'gauss'
    cond.is_diffusivity_including_antiresonant = True
    cond.diffusivity_bandwidth = phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes))
    cond = (cond.conductivity.sum(axis=0).diagonal().mean())
    np.testing.assert_approx_equal(cond, 0.8335, significant=3)

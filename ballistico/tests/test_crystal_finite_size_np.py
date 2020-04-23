"""
Unit and regression test for the ballistico package.
"""

# Import package, test suite, and other packages as needed
from ballistico.finitedifference import FiniteDifference
import numpy as np
from ballistico.phonons import Phonons
from ballistico.conductivity import Conductivity
import pytest
import os

@pytest.yield_fixture(scope="session")
def phonons():
    print ("Preparing phonons object.")
    finite_difference = FiniteDifference.from_folder(folder='ballistico/tests/si-crystal',
                                                     supercell=[3, 3, 3],
                                                     format='eskm')

    # Create a phonon object
    phonons = Phonons(finite_difference=finite_difference,
                      kpts=[5, 5, 5],
                      is_classic=False,
                      temperature=300,
                      is_tf_backend=False,
                      storage='memory')

    return phonons


def test_sc_finite_size_conductivity(phonons):
    cond_c = np.abs(Conductivity(phonons=phonons, method='sc', max_n_iterations = 71, storage='memory',length=(1e4,0,0), finite_length_method='caltech').conductivity.sum(axis=0)[0, 0])
    np.testing.assert_approx_equal(cond_c, 162.476, significant=3)
	

    cond_m = np.abs(Conductivity(phonons=phonons, method='sc', max_n_iterations = 71, storage='memory',length=(1e4,0,0), finite_length_method='matthiessen').conductivity.sum(axis=0)[0, 0])
    np.testing.assert_approx_equal(cond_m, 212.854, significant=3)
	
    cond_ms = np.abs(Conductivity(phonons=phonons, method='sc', max_n_iterations = 71, storage='memory',length=(1e4,0,0), finite_length_method='ms').conductivity.sum(axis=0)[0, 0])
    np.testing.assert_approx_equal(cond_ms, 187.841, significant=3)



def test_rta_finite_size_conductivity(phonons):

    cond_c = np.abs(Conductivity(phonons=phonons, method='rta', storage='memory',length=(1e4,0,0), finite_length_method='caltech').conductivity.sum(axis=0)[0, 0])
    np.testing.assert_approx_equal(cond_c, 146.273, significant=3)
	

    cond_m = np.abs(Conductivity(phonons=phonons, method='rta', storage='memory',length=(1e4,0,0), finite_length_method='matthiessen').conductivity.sum(axis=0)[0, 0])
    np.testing.assert_approx_equal(cond_m, 187.157, significant=3)
	
    cond_ms = np.abs(Conductivity(phonons=phonons, method='rta', storage='memory',length=(1e4,0,0), finite_length_method='ms').conductivity.sum(axis=0)[0, 0])
    np.testing.assert_approx_equal(cond_ms, 163.756, significant=3)


def test_inverse_finite_size_conductivity(phonons):

    cond_c = np.abs(Conductivity(phonons=phonons, method='inverse', storage='memory',length=(1e4,0,0), finite_length_method='caltech').conductivity.sum(axis=0)[0, 0])
    np.testing.assert_approx_equal(cond_c, 162.985, significant=3)
	

    cond_m = np.abs(Conductivity(phonons=phonons, method='inverse', storage='memory',length=(1e4,0,0), finite_length_method='matthiessen').conductivity.sum(axis=0)[0, 0])
    np.testing.assert_approx_equal(cond_m, 210.417, significant=3)
	
    cond_ms = np.abs(Conductivity(phonons=phonons, method='inverse', storage='memory',length=(1e4,0,0), finite_length_method='ms').conductivity.sum(axis=0)[0, 0])
    np.testing.assert_approx_equal(cond_ms, 182.302, significant=3)

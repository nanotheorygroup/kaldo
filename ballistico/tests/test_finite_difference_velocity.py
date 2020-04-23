import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ballistico.finitedifference import FiniteDifference
from ballistico.phonons import Phonons
import pytest
from tempfile import TemporaryDirectory


@pytest.yield_fixture(scope='session')
def velocity_stored():
    return np.load('ballistico/tests/EMT/test_velocity_EMT.npy')


@pytest.yield_fixture(scope='session')
def phonons():
    print ("Preparing phonons object.")
    # Setup crystal and EMT calculator
    atoms = bulk('Al', 'fcc', a=4.05)
    N=5
    supercell = (N, N, N)
    with TemporaryDirectory() as td:
        finite_difference = FiniteDifference(atoms=atoms,
                                             supercell=supercell,
                                             calculator=EMT,
                                             is_reduced_second=True,
                                             folder=td)
        is_classic = False
        k = 5
        phonons_config = {'kpts': [k, k, k],
                          'is_classic': is_classic,
                          'temperature': 300,
                          'is_tf_backend':False,
                          'storage':'memory',
                          'third_bandwidth':.1,
                          'broadening_shape':'gauss'}
        phonons = Phonons(finite_difference=finite_difference, **phonons_config)
        return phonons


def test_velocity_with_finite_difference_x(phonons, velocity_stored):
    calculated_vx = phonons.velocity[:,:,0]
    vx = velocity_stored[:,:,0]
    np.testing.assert_array_almost_equal(calculated_vx,vx, 2)


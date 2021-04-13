from kaldo.forceconstants import ForceConstants
import numpy as np
import os
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import pytest

@pytest.yield_fixture(scope="session")
def amorphous():
    # Create a finite difference object
    forceconstants = ForceConstants.from_folder(folder='kaldo/tests/si-amorphous',
                                                     format='eskm')
    return forceconstants


@pytest.yield_fixture(scope="session")
def crystal():
    # Create a finite difference object
    forceconstants = ForceConstants.from_folder(folder='kaldo/tests/si-crystal',
                                                     supercell=[3, 3, 3],
                                                     format='eskm')
    return forceconstants


def test_xyz_outputs():
    print(crystal)
    crystal.second.store_displacements(delta_shift=1)
    amorphous.second.store_displacements(delta_shift=1)
    crystal_success = os.path.isfile('kaldo/tests/si-crystal/second_order_displacements/0_xp')
    amorph_success = os.path.isfile('kaldo/tests/si-amorphous/second_order_displacements/0_xp')
    np.testing.assert_equal(crystal_success == amorph_success, True)
"""
Unit and regression test for the kaldo package.
"""
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import pytest


@pytest.fixture(scope="session")
def phonons():
    print ("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(folder='kaldo/tests/si-crystal',
                                                     supercell=[3, 3, 3],
                                                     format='eskm')
    phonons = Phonons(forceconstants=forceconstants,
                      kpts=[5, 5, 5],
                      is_classic=False,
                      temperature=300,
                      storage='memory')
    return phonons


def test_phonon_free_energy(phonons):
    physical_mode = phonons.physical_mode.reshape(phonons.frequency.shape)
    free_energy = phonons.free_energy[physical_mode].sum()
    assert -43 < free_energy < -42, f"Unexpected free energy: {free_energy}"

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
    # Free energy now includes zero-point energy: F = k_B*T*ln(1 - exp(-hbar*omega/(k_B*T))) + hbar*omega/2
    # At 300K, ZPE dominates over thermal contribution, giving positive total free energy
    # Expected range in eV for Si at 300K with ZPE included
    assert 0.090 < free_energy < 0.095, f"Unexpected free energy: {free_energy} eV"

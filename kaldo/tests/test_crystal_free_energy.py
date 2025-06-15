"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from ase import units
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

    # Extract physical mode mask
    physical_mode = phonons.physical_mode.reshape(phonons.frequency.shape)

    # Free energy is in THz
    free_energy_thz = phonons.free_energy[physical_mode]

    # Normalize by number of q-points
    n_qpoints = phonons.n_k_points
    total_energy_thz = np.sum(free_energy_thz) / n_qpoints

    # Convert THz → J
    total_energy_joules = total_energy_thz * units._hplanck * 1e12

    # Normalize per atom
    energy_per_atom_joules = total_energy_joules / phonons.n_atoms
    energy_per_atom_eV = energy_per_atom_joules / units._e
    energy_per_mol_joules = energy_per_atom_joules * units.mol

    # Assert reasonable physical values for Si at 300 K
    assert 0.03 < energy_per_atom_eV < 0.06, f"Unexpected free energy per atom: {energy_per_atom_eV} eV"
    assert 3000 < energy_per_mol_joules < 6000, f"Unexpected free energy per mol: {energy_per_mol_joules} J/mol"

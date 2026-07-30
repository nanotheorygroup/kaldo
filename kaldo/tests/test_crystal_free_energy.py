"""
Unit and regression test for the kaldo package.
"""
import numpy as np
import pytest
from ase import units

from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons


@pytest.fixture(scope="session")
def forceconstants():
    return ForceConstants.from_folder(
        folder='kaldo/tests/si-crystal',
        supercell=[3, 3, 3],
        format='eskm',
    )


@pytest.fixture(scope="session")
def phonons(forceconstants):
    print("Preparing phonons object.")
    return Phonons(
        forceconstants=forceconstants,
        kpts=[5, 5, 5],
        is_classic=False,
        temperature=300,
        storage='memory',
    )


def test_phonon_free_energy(phonons):
    physical_mode = phonons.physical_mode.reshape(phonons.frequency.shape)
    free_energy = phonons.free_energy[physical_mode].sum()
    # Free energy now includes zero-point energy: F = k_B*T*ln(1 - exp(-hbar*omega/(k_B*T))) + hbar*omega/2
    # At 300K, ZPE dominates over thermal contribution, giving positive total free energy
    # Expected range in eV for Si at 300K with ZPE included
    assert 0.090 < free_energy < 0.095, f"Unexpected free energy: {free_energy} eV"


def test_phonon_free_energy_classical(forceconstants):
    """Classical F = kT ln(hbar omega / kT) per mode; no ZPE."""
    T = 300.0
    ph = Phonons(
        forceconstants=forceconstants,
        kpts=[5, 5, 5],
        is_classic=True,
        temperature=T,
        storage='memory',
    )
    physical = ph.physical_mode.reshape(ph.frequency.shape)
    valid = physical & (ph.frequency > 0)
    omega = ph.frequency[valid] * 2.0 * np.pi * 1.0e12
    kBT_J = units._k * T
    expected = ((kBT_J / units._e) * np.log(units._hbar * omega / kBT_J)).sum() / ph.n_k_points
    got = ph.free_energy[physical].sum()
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    # Distinct from the quantum result (ZPE-dominated at 300 K for Si)
    ph_q = Phonons(
        forceconstants=forceconstants,
        kpts=[5, 5, 5],
        is_classic=False,
        temperature=T,
        storage='memory',
    )
    assert not np.isclose(got, ph_q.free_energy[physical].sum())

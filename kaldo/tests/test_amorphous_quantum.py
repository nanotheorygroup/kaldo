"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
import ase.units as units
import pytest


@pytest.fixture(scope="session")
def phonons():
    print("Preparing phonons object.")

    # Create a finite difference object
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-amorphous", format="eskm")

    # # Create a phonon object
    phonons = Phonons(
        forceconstants=forceconstants,
        is_classic=False,
        temperature=300,
        third_bandwidth=0.05 / 4.135,
        broadening_shape="triangle",
        storage="memory",
    )
    return phonons


def test_phase_space(phonons):
    phase_space = phonons.phase_space.sum()
    np.testing.assert_approx_equal(phase_space, 1752052, significant=7)

def test_first_gamma(phonons):
    thztomev = units.J * units._hbar * 2 * np.pi * 1e15
    np.testing.assert_approx_equal(phonons.bandwidth[0, 3] * thztomev / (2 * np.pi), 22.216, significant=3)


def test_second_gamma(phonons):
    thztomev = units.J * units._hbar * 2 * np.pi * 1e15
    np.testing.assert_approx_equal(phonons.bandwidth[0, 4] * thztomev / (2 * np.pi), 23.748, significant=3)


def test_participation_ratio(phonons):
    participation = phonons.participation_ratio.squeeze()
    np.testing.assert_approx_equal(participation[100], 0.52007, significant=3)


def test_velocity_amorphous(phonons):
    np.testing.assert_approx_equal(phonons.velocity.squeeze()[10, 2], 0, significant=2)


def test_eigensystem_shape(phonons):
    """Test that _eigensystem has correct shape after storage/loading.
    
    This test catches the bug where _eigensystem could have shape (1, n_modes+1, n_modes)
    instead of (n_modes+1, n_modes) after being stored and loaded from disk.
    """
    # Access a HarmonicWithQ object at gamma point
    from kaldo.observables.harmonic_with_q import HarmonicWithQ
    
    q_point = np.array([0., 0., 0.])
    phonon = HarmonicWithQ(
        q_point=q_point,
        second=phonons.forceconstants.second,
        distance_threshold=phonons.forceconstants.distance_threshold,
        folder=phonons.folder,
        storage=phonons.storage,
        is_amorphous=True
    )
    
    # Check eigensystem has correct 2D shape
    eigensystem = phonon._eigensystem
    assert eigensystem.ndim == 2, f"Expected 2D eigensystem, got {eigensystem.ndim}D with shape {eigensystem.shape}"
    assert eigensystem.shape == (phonon.n_modes + 1, phonon.n_modes), \
        f"Expected shape ({phonon.n_modes + 1}, {phonon.n_modes}), got {eigensystem.shape}"
    
    # Verify we can extract eigenvectors without errors
    eigenvectors = eigensystem[1:, :]
    assert eigenvectors.shape == (phonon.n_modes, phonon.n_modes), \
        f"Expected eigenvectors shape ({phonon.n_modes}, {phonon.n_modes}), got {eigenvectors.shape}"
    
    # Verify sij calculation works (this was the original failing operation)
    sij_x = phonon._sij_x
    assert sij_x.shape == (phonon.n_modes, phonon.n_modes), \
        f"Expected sij_x shape ({phonon.n_modes}, {phonon.n_modes}), got {sij_x.shape}"

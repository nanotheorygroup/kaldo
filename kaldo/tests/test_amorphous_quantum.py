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
    Uses third_bandwidth=0 for fast testing.
    """
    # Access a HarmonicWithQ object at gamma point
    from kaldo.observables.harmonic_with_q import HarmonicWithQ
    
    # Create a quick phonons object with third_bandwidth=0 for fast testing
    quick_phonons = Phonons(
        forceconstants=phonons.forceconstants,
        is_classic=False,
        temperature=300,
        third_bandwidth=0.0,
        broadening_shape="triangle",
        storage="memory",
    )
    
    q_point = np.array([0., 0., 0.])
    phonon = HarmonicWithQ(
        q_point=q_point,
        second=quick_phonons.forceconstants.second,
        distance_threshold=quick_phonons.forceconstants.distance_threshold,
        folder=quick_phonons.folder,
        storage=quick_phonons.storage,
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


def test_gamma_projection_chunked_workers_and_resume(phonons, tmp_path, monkeypatch):
    """Chunked parallel Gamma projection matches serial, and resumes from disk."""
    import kaldo.phonons as phonons_module

    monkeypatch.setattr(phonons_module, "GAMMA_MODE_CHUNK", 100)
    kwargs = dict(
        forceconstants=phonons.forceconstants, is_classic=False, temperature=300,
        third_bandwidth=0.05 / 4.135, broadening_shape="triangle", storage="memory",
        n_workers=2, projection_output_dir=str(tmp_path),
    )
    parallel = Phonons(**kwargs)
    np.testing.assert_allclose(parallel.bandwidth, phonons.bandwidth, rtol=1e-10, atol=1e-12)
    assert len(list(tmp_path.rglob("gamma_*.done"))) == 7
    # A different temperature must not reuse these rows.
    other = Phonons(**{**kwargs, "temperature": 400})
    assert len(list(tmp_path.rglob("gamma_*.done"))) == 7
    assert not np.allclose(other.bandwidth, phonons.bandwidth)
    assert len(list(tmp_path.rglob("gamma_*.done"))) == 14

    def fail(*args, **kwargs):
        raise AssertionError("resume should not recompute any chunk")

    monkeypatch.setattr(phonons_module, "_compute_gamma_mode_chunk", fail)
    resumed = Phonons(**kwargs)
    np.testing.assert_allclose(resumed.bandwidth, phonons.bandwidth, rtol=1e-10, atol=1e-12)

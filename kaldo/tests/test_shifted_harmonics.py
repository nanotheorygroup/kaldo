import numpy as np
import pytest

from kaldo.controllers._shifted_harmonics import compute_shifted_harmonics
from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.phonons import Phonons

SHIFT = (0.5, 0.5, 0.5)


@pytest.fixture(scope="session")
def amorphous_fc():
    return ForceConstants.from_folder(folder="kaldo/tests/si-amorphous", format="eskm")


@pytest.fixture
def shifted_phonons(amorphous_fc, tmp_path):
    return Phonons(forceconstants=amorphous_fc, kpts=(2, 2, 2), kpts_shift=SHIFT, temperature=300, storage="memory",
                   folder=str(tmp_path))


def test_shifted_phonons_frequency_and_velocity_match_per_q_reference(amorphous_fc, shifted_phonons):
    frequency = shifted_phonons.frequency
    velocity = shifted_phonons.velocity
    points = shifted_phonons._reciprocal_grid.fractional_points
    for ik, q in enumerate(points):
        reference = HarmonicWithQ(q_point=np.array(q), second=amorphous_fc.second, storage="memory",
                                  is_amorphous=False)
        np.testing.assert_allclose(frequency[ik], reference.frequency[0], rtol=1e-10, atol=1e-10)
        reference_velocity = reference.velocity[0]
        np.testing.assert_allclose(velocity[ik], reference_velocity, rtol=0,
                                   atol=1e-10 * np.abs(reference_velocity).max())


def test_shifted_eigensystem_layout_and_eigenvalues(shifted_phonons):
    eigensystem = shifted_phonons._eigensystem
    n = shifted_phonons.n_modes
    assert eigensystem.shape == (8, n + 1, n)
    assert eigensystem.dtype == np.complex128
    w = eigensystem[:, 0, :].real
    expected = np.abs(w) ** 0.5 * np.sign(w) / (2 * np.pi)
    np.testing.assert_allclose(expected, shifted_phonons.frequency, rtol=1e-12, atol=1e-12)


def test_shifted_velocities_are_nonzero(shifted_phonons):
    velocity = shifted_phonons.velocity
    physical = shifted_phonons.physical_mode
    speed = np.linalg.norm(velocity, axis=-1)[physical]
    assert np.median(speed) > 0.1


def test_shifted_harmonics_use_no_tensorflow(amorphous_fc, tmp_path, monkeypatch):
    import kaldo.phonons
    import kaldo.observables.harmonic_with_q
    import kaldo.observables.harmonic_with_q_temp
    import kaldo.observables.secondorder
    import kaldo.controllers.anharmonic

    class _NoTensorFlow:
        def __getattr__(self, name):
            raise AssertionError(f"TensorFlow attribute {name!r} used on the shifted harmonic path")

    for module in (kaldo.phonons, kaldo.observables.harmonic_with_q, kaldo.observables.harmonic_with_q_temp,
                   kaldo.observables.secondorder, kaldo.controllers.anharmonic):
        if hasattr(module, "tf"):
            monkeypatch.setattr(module, "tf", _NoTensorFlow())
    phonons = Phonons(forceconstants=amorphous_fc, kpts=(2, 2, 2), kpts_shift=SHIFT, temperature=300,
                      storage="memory", folder=str(tmp_path))
    assert phonons.frequency.shape == (8, phonons.n_modes)
    assert phonons.velocity.shape == (8, phonons.n_modes, 3)
    assert np.all(phonons.heat_capacity >= 0)
    assert np.all(phonons.population >= 0)


def test_partner_grid_is_real_and_gamma_acoustic_masked(shifted_phonons):
    harmonics = shifted_phonons._shifted_harmonics
    partner = harmonics.partner()
    assert partner is harmonics.partner()
    assert partner.eigenvectors.dtype == np.float64
    assert partner.eigenvectors.shape == (8, shifted_phonons.n_modes, shifted_phonons.n_modes)
    assert not partner.physical[0, :3].any()
    # No shifted point is Gamma, so only the default min_frequency=0 window can mask modes.
    np.testing.assert_array_equal(harmonics.physical, harmonics.frequency >= 0.)


def test_representatives_and_time_reversal_partners(shifted_phonons):
    harmonics = shifted_phonons._shifted_harmonics
    grid = shifted_phonons._reciprocal_grid
    assert harmonics.representatives.dtype == np.int64
    assert len(harmonics.representatives) == 4
    for r in harmonics.representatives:
        p = grid.partner_id(r)
        assert r < p
        np.testing.assert_array_equal(harmonics.frequency[r], harmonics.frequency[p])
        np.testing.assert_allclose(harmonics.velocity[p], -harmonics.velocity[r], rtol=0, atol=0)
        np.testing.assert_array_equal(harmonics.eigenvectors[p], np.conj(harmonics.eigenvectors[r]))


def test_shifted_harmonics_reject_distance_threshold(amorphous_fc, tmp_path):
    amorphous_fc.distance_threshold = 5.0
    try:
        phonons = Phonons(forceconstants=amorphous_fc, kpts=(2, 2, 2), kpts_shift=SHIFT, temperature=300,
                          storage="memory", folder=str(tmp_path))
        with pytest.raises(NotImplementedError, match="distance_threshold"):
            phonons.frequency
    finally:
        amorphous_fc.distance_threshold = None


def test_frequency_window_masks_physical_modes(amorphous_fc):
    grid = Phonons(forceconstants=amorphous_fc, kpts=(2, 2, 2), kpts_shift=SHIFT, storage="memory",
                   folder="/nonexistent-unused")._reciprocal_grid
    wide = compute_shifted_harmonics(amorphous_fc.second, grid, "auto", 0., None, 1)
    cut = float(np.median(wide.frequency))
    narrow = compute_shifted_harmonics(amorphous_fc.second, grid, "auto", cut, None, 1)
    np.testing.assert_array_equal(narrow.physical, wide.physical & (wide.frequency >= cut))


def test_shifted_harmonics_reject_active_nac(amorphous_fc, tmp_path, monkeypatch):
    # The a-Si fixture has no polar data, so explicit is_nac=True is rejected upstream; force the resolution instead.
    monkeypatch.setattr("kaldo.phonons._resolve_nac_activation", lambda atoms, requested: True)
    phonons = Phonons(forceconstants=amorphous_fc, kpts=(2, 2, 2), kpts_shift=SHIFT, temperature=300,
                      storage="memory", folder=str(tmp_path))
    with pytest.raises(NotImplementedError, match="NAC"):
        phonons.frequency

import numpy as np
import pytest

from kaldo.conductivity import Conductivity
from kaldo.controllers import _bloch_eigensystem
from kaldo.controllers._shifted_projection import project_shifted
from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q_temp import CLASSICAL_HBAR_SCALE
from kaldo.phonons import Phonons

SHIFT = (0.5, 0.5, 0.5)
SIGMA = 0.05


def _make_phonons(fc, **kwargs):
    return Phonons(forceconstants=fc, kpts=(2, 2, 2), kpts_shift=SHIFT, temperature=300, third_bandwidth=SIGMA,
                   n_workers=8, **kwargs)


@pytest.fixture(scope="session")
def amorphous_fc():
    return ForceConstants.from_folder(folder="kaldo/tests/si-amorphous", format="eskm")


@pytest.fixture(scope="module")
def shared_phonons(amorphous_fc):
    return _make_phonons(amorphous_fc, storage="memory")


def test_shifted_bandwidth_matches_route(shared_phonons):
    phonons = shared_phonons
    n = phonons.n_modes
    harmonics = phonons._shifted_harmonics
    partner = harmonics.partner()
    population = phonons.population.reshape(8, n).astype(np.float64)
    partner_population = _bloch_eigensystem.population(partner.frequency, partner.physical, 300, phonons.is_classic)
    hbar_factor = CLASSICAL_HBAR_SCALE if phonons.is_classic else 1
    routed = project_shifted(phonons.forceconstants.third, harmonics, phonons._reciprocal_grid,
                             phonons.ifc_interpolation, SIGMA, population, partner_population, hbar_factor, 8)
    np.testing.assert_allclose(phonons.bandwidth.ravel(), routed[:, 1], rtol=1e-14, atol=0)
    np.testing.assert_allclose(phonons.phase_space.ravel(), routed[:, 0], rtol=1e-14, atol=0)


def test_shifted_rta_conductivity_is_finite_and_positive(shared_phonons):
    kappa = Conductivity(phonons=shared_phonons, method="rta", storage="memory").conductivity.sum(axis=0)
    assert np.all(np.isfinite(kappa))
    assert np.trace(kappa) > 0


def test_shifted_rta_uses_no_tensorflow(amorphous_fc, monkeypatch):
    import kaldo.phonons
    import kaldo.conductivity
    import kaldo.observables.harmonic_with_q
    import kaldo.observables.harmonic_with_q_temp
    import kaldo.observables.secondorder
    import kaldo.controllers.anharmonic

    class _NoTensorFlow:
        def __getattr__(self, name):
            raise AssertionError(f"TensorFlow attribute {name!r} used on the shifted RTA path")

    for module in (kaldo.phonons, kaldo.conductivity, kaldo.observables.harmonic_with_q,
                   kaldo.observables.harmonic_with_q_temp, kaldo.observables.secondorder,
                   kaldo.controllers.anharmonic):
        if hasattr(module, "tf"):
            monkeypatch.setattr(module, "tf", _NoTensorFlow())
    phonons = _make_phonons(amorphous_fc, storage="memory")
    kappa = Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0)
    assert np.all(np.isfinite(kappa))
    assert np.trace(kappa) > 0


def test_shifted_rta_rejects_gamma_tensor(shared_phonons):
    with pytest.raises(ValueError):
        shared_phonons._select_algorithm_for_phase_space_and_gamma(is_gamma_tensor_enabled=True)


@pytest.mark.performance
def test_shifted_storage_roundtrip(amorphous_fc, tmp_path):
    first = _make_phonons(amorphous_fc, storage="numpy", folder=str(tmp_path))
    bandwidth = first.bandwidth.copy()
    second = _make_phonons(amorphous_fc, storage="numpy", folder=str(tmp_path))
    np.testing.assert_allclose(second.bandwidth, bandwidth, rtol=0, atol=0)


@pytest.mark.performance
def test_shifted_and_unshifted_caches_do_not_mix(amorphous_fc, tmp_path):
    shifted = _make_phonons(amorphous_fc, storage="numpy", folder=str(tmp_path))
    assert np.abs(shifted.velocity).max() > 1e-3
    shifted.bandwidth
    unshifted = Phonons(forceconstants=amorphous_fc, kpts=(2, 2, 2), temperature=300, third_bandwidth=SIGMA,
                        storage="numpy", folder=str(tmp_path))
    assert np.abs(unshifted.velocity).max() < 1e-10

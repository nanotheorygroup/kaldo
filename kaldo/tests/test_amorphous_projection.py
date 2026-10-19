"""Numerical and resource contracts for Gamma-only cubic projection."""

from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from sparse import COO

from kaldo.controllers import anharmonic
from kaldo.phonons import Phonons


def _projection_input(values: np.ndarray, bandwidth: float = 0.12) -> SimpleNamespace:
    """Prepare real Gamma input from IFC3 ``(n,n,n)`` in eV/Angstrom^3.

    Frequencies are in THz and the mass-rescaled eigenvectors have inverse
    square-root atomic-mass units, with unit masses used in this fixture.
    """
    n_modes = values.shape[0]
    random = np.random.default_rng(301)
    eigenvectors = np.linalg.qr(random.normal(size=(n_modes, n_modes)))[0]
    return SimpleNamespace(
        frequency=np.arange(1, n_modes + 1, dtype=float)[None, :],
        _rescaled_eigenvectors=eigenvectors[None, :, :],
        forceconstants=SimpleNamespace(
            third=SimpleNamespace(gamma_contracted_value=lambda: COO.from_numpy(values))
        ),
        n_modes=n_modes,
        n_phonons=n_modes,
        n_k_points=1,
        physical_mode=np.ones((1, n_modes), dtype=bool),
        third_bandwidth=bandwidth,
        broadening_shape="gauss",
    )


@pytest.mark.parametrize("bandwidth", [0.0, 0.12])
def test_projection_releases_folded_ifc3_before_mode_loop(monkeypatch, bandwidth):
    """Projection must retain the source but release its temporary folded COO."""
    values = np.random.default_rng(302).normal(size=(6, 6, 6))
    source = COO.from_numpy(values)
    folded_references = []

    def folded_value():
        folded = source.copy()
        folded_references.append(weakref.ref(folded))
        return folded

    phonons = _projection_input(values, bandwidth)
    phonons.forceconstants.third.value = source
    phonons.forceconstants.third.gamma_contracted_value = folded_value
    expected_phases, expected_potentials = Phonons._project_amorphous(_projection_input(values, bandwidth))
    calculate_delta = anharmonic.calculate_dirac_delta_amorphous

    def check_lifetime(*args, **kwargs):
        assert folded_references[0]() is None
        assert phonons.forceconstants.third.value is source
        return calculate_delta(*args, **kwargs)

    monkeypatch.setattr(anharmonic, "calculate_dirac_delta_amorphous", check_lifetime)
    phases, potentials = Phonons._project_amorphous(phonons)

    assert len(phases) == len(potentials) == 6
    assert all(len(channels) == 2 for channels in phases)
    np.testing.assert_array_equal(source.todense(), values)
    for actual_modes, expected_modes in ((phases, expected_phases), (potentials, expected_potentials)):
        for actual_channels, expected_channels in zip(actual_modes, expected_modes):
            for actual, expected in zip(actual_channels, expected_channels):
                if expected is None:
                    assert actual is None
                else:
                    np.testing.assert_array_equal(actual.indices.numpy(), expected.indices.numpy())
                    np.testing.assert_array_equal(actual.values.numpy(), expected.values.numpy())
    if bandwidth == 0:
        assert all(tensor is None for channels in phases for tensor in channels)
    else:
        assert all(any(tensor is not None for tensor in channel) for channel in phases)

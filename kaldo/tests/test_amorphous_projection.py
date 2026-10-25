"""Numerical and resource contracts for Gamma-only cubic projection."""

from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import ase.units as units
import numpy as np
import pytest
from sparse import COO
import tensorflow as tf

from kaldo.controllers import anharmonic
from kaldo.observables.harmonic_with_q_temp import CLASSICAL_HBAR_SCALE, HarmonicWithQTemp
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


def _dense_projection_reference(
    values: np.ndarray, phonons: SimpleNamespace
) -> tuple[list[list[tf.SparseTensor | None]], list[list[tf.SparseTensor | None]]]:
    """Contract IFC3 ``(n,n,n)`` in eV/Angstrom^3 with all three mode legs.

    Eigenvectors ``(n,n)`` carry inverse square-root atomic-mass units and
    angular frequencies ``(n,)`` are in rad/ps. Return the production sparse
    phase and squared-potential conventions without its staged contraction.
    """
    eigenvectors = phonons._rescaled_eigenvectors[0]
    angular_frequency = 2 * np.pi * phonons.frequency.flatten()
    mode_potential = np.einsum(
        "abc,an,bp,cq->npq", values, eigenvectors, eigenvectors, eigenvectors
    )
    squared_potential = mode_potential ** 2
    squared_potential /= (
        angular_frequency[:, None, None]
        * angular_frequency[None, :, None]
        * angular_frequency[None, None, :]
    )
    squared_potential *= np.pi * units._hbar / 4 * anharmonic.GAMMA_TO_THZ
    phases = []
    potentials = []
    for mode_index in range(phonons.n_modes):
        if not phonons.physical_mode[0, mode_index]:
            phases.append([None] * 4)
            potentials.append([None] * 4)
            continue
        phases.append([])
        potentials.append([])
        for is_plus in (0, 1):
            phase = anharmonic.calculate_dirac_delta_amorphous(
                is_plus,
                mode_index,
                angular_frequency[None, :],
                phonons.physical_mode,
                tf.constant(phonons.third_bandwidth, dtype=tf.float64),
                phonons.broadening_shape,
                phonons.n_phonons,
            )
            phases[-1].append(phase)
            if phase is None:
                potentials[-1].append(None)
            else:
                partner_indices = phase.indices.numpy()
                potentials[-1].append(tf.SparseTensor(
                    phase.indices,
                    squared_potential[mode_index, partner_indices[:, 0], partner_indices[:, 1]],
                    phase.dense_shape,
                ))
    return phases, potentials


@pytest.mark.parametrize("broadening_shape", ["gauss", "triangle", "lorentz"])
@pytest.mark.parametrize("bandwidth", [0.0, 0.12])
def test_projection_contracts_once_at_first_nonempty_channel(monkeypatch, bandwidth, broadening_shape):
    values = np.random.default_rng(303).normal(size=(6, 6, 6))
    phonons = _projection_input(values, bandwidth)
    phonons.broadening_shape = broadening_shape
    phonons.physical_mode[0, 3] = False
    operations = Mock()
    operations.delta = Mock(wraps=anharmonic.calculate_dirac_delta_amorphous)
    operations.sparse = Mock(wraps=tf.sparse.sparse_dense_matmul)
    operations.dense = Mock(wraps=tf.einsum)
    monkeypatch.setattr(anharmonic, "calculate_dirac_delta_amorphous", operations.delta)
    monkeypatch.setattr(tf.sparse, "sparse_dense_matmul", operations.sparse)
    monkeypatch.setattr(tf, "einsum", operations.dense)

    phases, potentials = Phonons._project_amorphous(phonons)

    active_modes = []
    expected_order = []
    for mode_index, channels in enumerate(phases):
        if not phonons.physical_mode[0, mode_index]:
            assert channels == potentials[mode_index] == [None] * 4
            continue
        assert len(channels) == len(potentials[mode_index]) == 2
        contracted = False
        for phase in channels:
            expected_order.append("delta")
            if phase is not None and not contracted:
                active_modes.append(mode_index)
                expected_order.extend(["sparse", "dense"])
                contracted = True
    if bandwidth:
        assert [phase is not None for phase in phases[0]] == [False, True]
        assert [phase is not None for phase in phases[-1]] == [True, False]
        assert any(all(phase is not None for phase in channels) for channels in phases)
    else:
        assert not active_modes
    assert operations.sparse.call_count == len(active_modes)
    assert operations.dense.call_count == len(active_modes)
    assert [operation[0] for operation in operations.mock_calls] == expected_order
    for contraction, mode_index in zip(operations.sparse.call_args_list, active_modes):
        np.testing.assert_array_equal(
            contraction.args[1].numpy().ravel(), phonons._rescaled_eigenvectors[0, :, mode_index]
        )


@pytest.mark.parametrize("broadening_shape", ["gauss", "triangle", "lorentz"])
@pytest.mark.parametrize("is_classic", [False, True])
def test_projection_matches_dense_contraction_and_reducer(broadening_shape, is_classic):
    random = np.random.default_rng(304)
    values = random.normal(size=(6, 6, 6))
    values[random.random(values.shape) < 0.5] = 0.0
    phonons = _projection_input(values)
    phonons.broadening_shape = broadening_shape
    phonons.physical_mode[0, 3] = False

    phases, potentials = Phonons._project_amorphous(phonons)
    expected_phases, expected_potentials = _dense_projection_reference(values, phonons)

    for actual_modes, expected_modes in ((phases, expected_phases), (potentials, expected_potentials)):
        assert len(actual_modes) == len(expected_modes) == phonons.n_modes
        for actual_channels, expected_channels in zip(actual_modes, expected_modes):
            assert len(actual_channels) == len(expected_channels)
            for actual, expected in zip(actual_channels, expected_channels):
                if expected is None:
                    assert actual is None
                else:
                    assert isinstance(actual, tf.SparseTensor)
                    assert actual.indices.dtype == tf.int64
                    assert actual.values.dtype == tf.float64
                    np.testing.assert_array_equal(actual.indices.numpy(), expected.indices.numpy())
                    np.testing.assert_array_equal(actual.dense_shape.numpy(), expected.dense_shape.numpy())
                    np.testing.assert_allclose(actual.values.numpy(), expected.values.numpy(), rtol=1e-12, atol=1e-14)

    hbar_factor = CLASSICAL_HBAR_SCALE if is_classic else 1.0
    phonons.hbar = units._hbar * hbar_factor
    phonons.temperature = 300.0
    population = HarmonicWithQTemp._calculate_population(phonons).flatten()
    for is_balanced in (False, True):
        actual = anharmonic.calculate_ps_and_gamma(
            phases, potentials, population, is_balanced, phonons.n_phonons, True, hbar_factor=hbar_factor
        )
        expected = anharmonic.calculate_ps_and_gamma(
            expected_phases, expected_potentials, population, is_balanced, phonons.n_phonons, True,
            hbar_factor=hbar_factor,
        )
        assert np.any(expected[:, 0] > 0)
        assert np.any(expected[:, 1] > 0)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


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

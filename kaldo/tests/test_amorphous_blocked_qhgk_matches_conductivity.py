"""Amorphous Gamma ``Conductivity(method='qhgk')`` reproduces the per-q-point pair-field calculation."""

from pathlib import Path

import numpy as np
import pytest

from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons

# Exactly representable in float32, which Conductivity uses for scalar bandwidths.
DIFFUSIVITY_BANDWIDTH = 0.0625
TEMPERATURE = 300.


@pytest.fixture(scope="module")
def forceconstants():
    return ForceConstants.from_folder(folder=str(Path(__file__).parent / "si-amorphous"), format="eskm")


def _conductivity(forceconstants, is_classic, **options):
    phonons = Phonons(forceconstants=forceconstants, temperature=TEMPERATURE, is_classic=is_classic,
                      storage="memory")
    return Conductivity(phonons=phonons, method="qhgk", storage="memory",
                        diffusivity_bandwidth=DIFFUSIVITY_BANDWIDTH, **options)


def _assert_matches_pair_fields(conductivity):
    """Compare the blocked dispatch with the existing per-q loop (float32 accumulation)."""
    assert conductivity.phonons._is_amorphous
    bandwidth = DIFFUSIVITY_BANDWIDTH * np.ones((1, conductivity.phonons.n_modes), dtype=np.float32)
    expected_tensor, expected_diffusivity = conductivity._calculate_qhgk_by_q_point(bandwidth)
    tensor, diffusivity = conductivity.calculate_conductivity_and_diffusivity_qhgk()
    assert tensor.shape == expected_tensor.shape and diffusivity.shape == expected_diffusivity.shape
    for actual, expected in ((tensor, expected_tensor), (diffusivity, np.real(expected_diffusivity))):
        scale = np.max(np.abs(expected))
        assert scale > 0
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6 * scale)
    total = expected_tensor.sum(axis=0)
    np.testing.assert_allclose(conductivity.conductivity.sum(axis=0), total, rtol=1e-5,
                               atol=1e-6 * np.max(np.abs(total)))


@pytest.mark.parametrize("is_classic", [False, True])
@pytest.mark.parametrize("shape", ["lorentz", "gauss", "triangle"])
def test_blocked_qhgk_matches_pair_fields(forceconstants, is_classic, shape):
    _assert_matches_pair_fields(_conductivity(forceconstants, is_classic, diffusivity_shape=shape))


@pytest.mark.parametrize("options", [{"diffusivity_threshold": 2},
                                     {"is_diffusivity_including_antiresonant": True}],
                         ids=["threshold", "antiresonant"])
def test_blocked_qhgk_options_match_pair_fields(forceconstants, options):
    _assert_matches_pair_fields(_conductivity(forceconstants, False, diffusivity_shape="lorentz", **options))

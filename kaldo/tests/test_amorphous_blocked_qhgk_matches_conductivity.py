"""The blocked Gamma QHGK reducer reproduces ``Conductivity(method='qhgk')`` per mode."""

from pathlib import Path

import numpy as np
import pytest

from kaldo.amorphous import calculate_qhgk_isotropic_blocked
from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.phonons import Phonons

# Exactly representable in float32, which Conductivity uses for scalar bandwidths.
DIFFUSIVITY_BANDWIDTH = 0.0625
TEMPERATURE = 300.


@pytest.fixture(scope="module")
def forceconstants():
    return ForceConstants.from_folder(folder=str(Path(__file__).parent / "si-amorphous"), format="eskm")


@pytest.fixture(scope="module")
def gamma_harmonic(forceconstants):
    """Eigenvectors and real derivative operators from the existing harmonic object."""
    harmonic = HarmonicWithQ(q_point=np.zeros(3), second=forceconstants.second,
                             storage="memory", is_amorphous=True)
    eigenvectors = np.asarray(harmonic._eigensystem)[1:, :]
    operators = tuple(np.asarray(derivative) for derivative in (
        harmonic._dynmat_derivatives_x, harmonic._dynmat_derivatives_y, harmonic._dynmat_derivatives_z))
    return eigenvectors, operators


@pytest.mark.parametrize("is_classic", [False, True])
@pytest.mark.parametrize("shape", ["lorentz", "gauss", "triangle"])
def test_blocked_reducer_matches_conductivity_qhgk(forceconstants, gamma_harmonic, is_classic, shape):
    phonons = Phonons(forceconstants=forceconstants, temperature=TEMPERATURE, is_classic=is_classic,
                      storage="memory")
    conductivity = Conductivity(phonons=phonons, method="qhgk", storage="memory",
                                diffusivity_bandwidth=DIFFUSIVITY_BANDWIDTH, diffusivity_shape=shape)
    expected_conductivity = np.trace(conductivity.conductivity, axis1=1, axis2=2) / 3
    expected_diffusivity = np.asarray(conductivity.diffusivity).real.reshape(-1)

    eigenvectors, operators = gamma_harmonic
    omega = phonons.omega.reshape(-1)
    actual_conductivity, actual_diffusivity = calculate_qhgk_isotropic_blocked(
        omega, DIFFUSIVITY_BANDWIDTH, phonons.physical_mode.reshape(-1), TEMPERATURE, is_classic,
        np.abs(np.linalg.det(phonons.atoms.cell)), eigenvectors, operators,
        broadening_shape=shape, block_size=100, compute_dtype=np.float64)

    # Conductivity accumulates per-mode results in float32.
    for actual, expected in ((actual_conductivity, expected_conductivity),
                             (actual_diffusivity, expected_diffusivity)):
        assert np.max(np.abs(expected)) > 0
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6 * np.max(np.abs(expected)))

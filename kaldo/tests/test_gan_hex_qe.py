"""
Regression test: hexagonal (wurtzite GaN) dispersion from Quantum ESPRESSO
force constants.

Reference frequencies were generated with QE matdyn.x from the same
espresso.ifc2 (asr="simple", q_in_cryst_coord=.true.), so any disagreement is
a bug in kaldo dynamical-matrix assembly. The 8 cm^-1 tolerance sits above
the known interpolation differences vs matdyn and far below the >40 cm^-1
signature of a replica-ordering error.
"""

import numpy as np
import pytest
from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ

THZ_TO_CM = 33.3564095198152

# matdyn.x reference frequencies in cm^-1 (q in crystal coordinates)
MATDYN_REFERENCE = {
    (0.00, 0.00, 0.00): [0.0, 0.0, 0.0, 136.0943, 136.0943, 321.1560,
                         506.5115, 529.3006, 529.3006, 536.6122, 536.6122, 659.8961],
    (0.25, 0.00, 0.00): [102.4237, 111.2168, 168.2241, 202.4963, 222.1707, 290.3288,
                         525.1332, 533.8456, 561.5943, 604.1490, 649.4710, 708.8036],
    (0.00, 0.00, 0.25): [58.9816, 58.9816, 118.1607, 128.6105, 128.6105, 294.1726,
                         529.8455, 529.8455, 535.0630, 535.0630, 672.0969, 721.0808],
}


@pytest.fixture(scope="session")
def gan_second():
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/gan",
        supercell=[5, 5, 5],
        only_second=True,
        format="qe-d3q",
    )
    return forceconstants.second


def _frequencies_cm(second, q_point):
    hwq = HarmonicWithQ(q_point=np.array(q_point), second=second, is_unfolding=False)
    return np.sort(np.array(hwq.frequency).flatten()) * THZ_TO_CM


def test_gamma_frequencies_match_matdyn(gan_second):
    actual = _frequencies_cm(gan_second, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(actual, MATDYN_REFERENCE[(0.0, 0.0, 0.0)], atol=0.5)


def test_in_plane_dispersion_matches_matdyn(gan_second):
    actual = _frequencies_cm(gan_second, (0.25, 0.0, 0.0))
    np.testing.assert_allclose(actual, MATDYN_REFERENCE[(0.25, 0.0, 0.0)], atol=8.0)


def test_out_of_plane_dispersion_matches_matdyn(gan_second):
    actual = _frequencies_cm(gan_second, (0.0, 0.0, 0.25))
    np.testing.assert_allclose(actual, MATDYN_REFERENCE[(0.0, 0.0, 0.25)], atol=8.0)


def test_gamma_A_transverse_degeneracy(gan_second):
    # Along Gamma-A the two TA branches are symmetry-required to be degenerate.
    actual = _frequencies_cm(gan_second, (0.0, 0.0, 0.25))
    assert abs(actual[0] - actual[1]) < 0.1

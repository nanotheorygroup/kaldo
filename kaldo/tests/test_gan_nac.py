"""
Regression test: non-analytic correction on a hexagonal, non-centrosymmetric, polar crystal
(wurtzite GaN).

This is the anisotropic acid test the cubic NaCl/MgO fixtures cannot provide:
wurtzite exercises an anisotropic dielectric tensor, a non-symmorphic space
group, a cell matrix that is not symmetric, and dynamical matrices that are
sensitive to phase-convention errors that centrosymmetric crystals hide.

Born charges and dielectric tensor are the AlmaBTE GaN_wurtzite reference
values (BORN file, phonopy conventions). Frequency references were generated
with phonopy 3.5.1 driven through its python API on the same force constants
(built with _build_interleaved_fc) and the same NAC parameters, with
nac_q_direction=[1, 0, 0] at Gamma. Tolerances reflect the current
kaldo-phonopy parity: a few cm^-1 away from Gamma from independent Ewald
parameter choices, essentially exact at Gamma and along Gamma-A.

The velocity test is referee-free: analytic group velocities must equal
2 pi times the slope of the code's own dispersion (dOmega/dq in A/ps).
"""

import tempfile

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ

THZ_TO_CM = 33.3564095198152

EPSILON = np.diag([5.5429220, 5.5429220, 5.8492550])
Z_GA = np.diag([2.5749225, 2.5749225, 2.7477150])
BORN = np.array([Z_GA, Z_GA, -Z_GA, -Z_GA])

# phonopy 3.5.1 reference frequencies in cm^-1 (sorted), see module docstring
PHONOPY_REFERENCE = {
    (0.0, 0.0, 0.0): [0.0000, 0.0000, 0.0000, 136.0944, 136.0944, 321.1559,
                      506.5159, 529.3052, 536.6172, 536.6172, 659.9031, 707.7290],
    (0.25, 0.0, 0.0): [101.7275, 109.3716, 165.8556, 204.4387, 220.2518, 290.7157,
                       525.2281, 538.4809, 561.8045, 601.6669, 648.9626, 689.5510],
    (0.0, 0.0, 0.25): [58.9540, 58.9540, 121.2431, 128.6243, 128.6243, 296.0839,
                       529.8501, 529.8501, 535.0679, 535.0679, 665.9864, 691.8471],
    (1.0 / 3.0, 1.0 / 3.0, 0.0): [198.9089, 201.8256, 209.2423, 241.0535, 271.5586,
                                  272.2163, 580.0541, 580.8787, 615.6957, 617.6140,
                                  621.5631, 654.0725],
    (0.1, 0.1, 0.1): [80.3810, 81.2639, 134.4827, 155.4529, 197.9083, 295.4783,
                      522.6978, 546.2738, 548.2819, 588.3804, 654.9988, 682.1429],
}


@pytest.fixture(scope="module")
def gan_second():
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/gan",
        supercell=[5, 5, 5],
        only_second=True,
        format="qe-d3q",
    )
    second = forceconstants.second
    second.atoms.info["dielectric"] = EPSILON.copy()
    second.atoms.set_array("charges", BORN.copy(), shape=(3, 3))
    second.folder = tempfile.mkdtemp(prefix="gan_nac_test_")
    return second


def _nac_frequencies_cm(second, q_point):
    hwq = HarmonicWithQ(q_point=np.array(q_point, dtype=float), second=second,
                        storage="memory")
    return np.sort(np.array(hwq.frequency).flatten()) * THZ_TO_CM


@pytest.mark.parametrize("q_point", list(PHONOPY_REFERENCE))
def test_nac_frequencies_match_phonopy(gan_second, q_point):
    actual = _nac_frequencies_cm(gan_second, q_point)
    np.testing.assert_allclose(actual, PHONOPY_REFERENCE[q_point], atol=15.0)


def test_nac_gamma_A_transverse_degeneracy(gan_second):
    actual = _nac_frequencies_cm(gan_second, (0.0, 0.0, 0.25))
    assert abs(actual[0] - actual[1]) < 0.1
    assert abs(actual[6] - actual[7]) < 0.1


def test_nac_gamma_lo_to_splitting(gan_second):
    actual = _nac_frequencies_cm(gan_second, (0.0, 0.0, 0.0))
    # highest branch is the NAC-lifted LO; without charges it sits at ~660
    assert actual[-1] > 690.0


def test_nac_velocity_matches_dispersion_slope(gan_second):
    q0 = np.array([0.1, 0.1, 0.1])
    hwq = HarmonicWithQ(q_point=q0, second=gan_second, storage="memory")
    order = np.argsort(np.array(hwq.frequency).flatten())
    velocity = np.array(hwq.velocity)[0][order]

    cell = gan_second.atoms.cell.array
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    delta = 1e-3
    step = np.linalg.solve(reciprocal, np.array([0.0, 0.0, delta]))
    plus = _nac_frequencies_cm(gan_second, q0 + step) / THZ_TO_CM
    minus = _nac_frequencies_cm(gan_second, q0 - step) / THZ_TO_CM
    slope = (plus - minus) / (2 * delta)

    usable = np.abs(slope) > 0.05
    ratio = velocity[usable, 2] / slope[usable]
    np.testing.assert_allclose(ratio, 2 * np.pi, rtol=1e-2)

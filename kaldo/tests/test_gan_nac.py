"""
Regression test: non-analytic correction on a hexagonal, non-centrosymmetric, polar crystal
(wurtzite GaN).

This is the anisotropic acid test the cubic NaCl/MgO fixtures cannot provide:
wurtzite exercises an anisotropic dielectric tensor, a non-symmorphic space
group, a cell matrix that is not symmetric, and dynamical matrices that are
sensitive to phase-convention errors that centrosymmetric crystals hide.

Born charges and dielectric tensor are the AlmaBTE GaN_wurtzite reference
values (BORN file, phonopy conventions). Frequency references were regenerated
with Phonopy 4.3.1 from the authoritative q2r header structure, the same force
constants (built with _build_interleaved_fc), and the same NAC parameters,
with nac_q_direction=[1, 0, 0] at Gamma. The former table used the auxiliary
POSCAR cell, which is not compatible with this q2r file.

The velocity test is referee-free: analytic group velocities must equal
2 pi times the slope of the code's own dispersion (dOmega/dq in A/ps).
"""

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ

THZ_TO_CM = 33.3564095198152

EPSILON = np.diag([5.5429220, 5.5429220, 5.8492550])
Z_GA = np.diag([2.5749225, 2.5749225, 2.7477150])
BORN = np.array([Z_GA, Z_GA, -Z_GA, -Z_GA])

# Phonopy 4.3.1 reference frequencies in cm^-1 (sorted), see module docstring.
PHONOPY_REFERENCE = {
    (0.0, 0.0, 0.0): [
        0.0000,
        0.0000,
        0.0000,
        136.0942,
        136.0942,
        321.1559,
        506.5114,
        529.3005,
        536.6121,
        536.6121,
        659.8960,
        703.6240,
    ],
    (0.25, 0.0, 0.0): [
        102.4734,
        110.6071,
        169.5169,
        202.0249,
        221.1020,
        290.2783,
        525.1159,
        537.1248,
        562.3188,
        603.8759,
        649.4229,
        687.8067,
    ],
    (0.0, 0.0, 0.25): [
        58.9816,
        58.9816,
        121.7420,
        128.6105,
        128.6105,
        296.2644,
        529.8455,
        529.8455,
        535.0629,
        535.0629,
        666.0976,
        692.4640,
    ],
    (1.0 / 3.0, 1.0 / 3.0, 0.0): [
        201.1314,
        201.1314,
        208.5703,
        242.9949,
        272.1075,
        272.1075,
        580.5005,
        580.5005,
        616.2490,
        618.3547,
        618.3547,
        654.9974,
    ],
    (0.1, 0.1, 0.1): [
        82.5650,
        88.3750,
        149.6019,
        156.6047,
        187.2526,
        297.5914,
        519.9670,
        539.0560,
        550.6623,
        579.4500,
        655.0061,
        692.7624,
    ],
}


@pytest.fixture(scope="module")
def gan_second(tmp_path_factory):
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/gan",
        supercell=[5, 5, 5],
        only_second=True,
        format="qe-d3q",
    )
    second = forceconstants.second
    second.atoms.info["dielectric"] = EPSILON.copy()
    second.atoms.set_array("charges", BORN.copy(), shape=(3, 3))
    second.folder = str(tmp_path_factory.mktemp("gan_nac_test"))
    return second


def _nac_frequencies_cm(second, q_point):
    hwq = HarmonicWithQ(
        q_point=np.array(q_point, dtype=np.float64),
        second=second,
        storage="memory",
    )
    return np.sort(np.array(hwq.frequency).flatten()) * THZ_TO_CM


@pytest.mark.parametrize("q_point", list(PHONOPY_REFERENCE))
def test_nac_frequencies_match_phonopy(gan_second, q_point):
    actual = _nac_frequencies_cm(gan_second, q_point)
    # The references are printed to 1e-4 cm^-1. Keep a small absolute margin
    # for the printed table without relative slack.
    np.testing.assert_allclose(actual, PHONOPY_REFERENCE[q_point], rtol=0.0, atol=0.02)


def test_nac_gamma_A_transverse_degeneracy(gan_second):
    actual = _nac_frequencies_cm(gan_second, (0.0, 0.0, 0.25))
    np.testing.assert_allclose(actual[0], actual[1], rtol=0.0, atol=0.1)
    np.testing.assert_allclose(actual[6], actual[7], rtol=0.0, atol=0.1)


def test_nac_gamma_lo_to_splitting(gan_second):
    actual = _nac_frequencies_cm(gan_second, (0.0, 0.0, 0.0))
    # highest branch is the NAC-lifted LO; without charges it sits at ~660
    assert actual[-1] > 690.0


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
def test_nac_velocity_matches_dispersion_slope(gan_second, axis):
    """The public velocity follows the dispersion along each Cartesian axis.

    The x and y cases are the regression for ASE's row-vector cell convention:
    on this non-orthogonal cell, using ``cell.T`` maps the finite-difference
    stencil onto the wrong Cartesian direction.  The z case retains the
    original velocity-versus-dispersion coverage.
    """
    q0 = np.array([0.1, 0.1, 0.1])
    hwq = HarmonicWithQ(q_point=q0, second=gan_second, storage="memory")
    order = np.argsort(np.array(hwq.frequency).flatten())
    velocity = np.array(hwq.velocity)[0][order]

    cell = gan_second.atoms.cell.array
    delta = 1e-3
    direction = np.eye(3)[axis]
    # With direct lattice vectors stored as rows, the physical wavevector is
    # k_cart = 2*pi*inv(cell) @ q_red.  Inverting that map gives this reduced
    # step along the requested Cartesian direction.
    step = cell @ direction * delta / (2 * np.pi)
    plus = _nac_frequencies_cm(gan_second, q0 + step) / THZ_TO_CM
    minus = _nac_frequencies_cm(gan_second, q0 - step) / THZ_TO_CM
    slope = (plus - minus) / (2 * delta)

    usable = np.abs(slope) > 0.05
    assert np.any(usable), f"no dispersive modes found along Cartesian axis {axis}"
    ratio = velocity[usable, axis] / slope[usable]
    np.testing.assert_allclose(ratio, 2 * np.pi, rtol=1e-2, atol=0.0)


def test_nac_sij_is_rejected_until_a_polar_flux_operator_exists(gan_second):
    # The NAC derivative is a Hermitian finite difference in the
    # integer-translation gauge; the flux projection expects the ordinary
    # kernel's pair-aware gauge. The diagonal happens to agree with the
    # velocity normalization, but QHGK consumes off-diagonals, so sij is
    # rejected on the polar path. Velocities bypass sij and stay correct.
    q0 = np.array([0.1, 0.1, 0.1])
    hwq = HarmonicWithQ(q_point=q0, second=gan_second, storage="memory")
    assert np.all(np.isfinite(np.array(hwq.velocity)))
    with pytest.raises(NotImplementedError, match="heat-flux operator"):
        hwq.calculate_sij(0)

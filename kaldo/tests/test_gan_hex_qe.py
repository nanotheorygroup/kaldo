"""
Regression test: hexagonal (wurtzite GaN) q2r interpolation against QE 7.6
``matdyn.x``. The q2r header is authoritative for atom order and lattice;
the auxiliary POSCAR is intentionally stale and must not affect these values.
"""

from pathlib import Path

import numpy as np
import pytest
from kaldo.forceconstants import ForceConstants
from kaldo.interfaces import qe_io
from kaldo.controllers import nac
from kaldo.observables.harmonic_with_q import HarmonicWithQ

THZ_TO_CM = 33.3564095198152

# matdyn.x reference frequencies in cm^-1 (q in crystal coordinates)
MATDYN_REFERENCE = {
    (0.00, 0.00, 0.00): [
        0.0,
        0.0,
        0.0,
        136.0943,
        136.0943,
        321.1560,
        506.5115,
        529.3006,
        529.3006,
        536.6122,
        536.6122,
        659.8961,
    ],
    (0.25, 0.00, 0.00): [
        102.4237,
        111.2168,
        168.2241,
        202.4963,
        222.1707,
        290.3288,
        525.1332,
        533.8456,
        561.5943,
        604.1490,
        649.4710,
        708.8036,
    ],
    (0.00, 0.00, 0.25): [
        58.9816,
        58.9816,
        118.1607,
        128.6105,
        128.6105,
        294.1726,
        529.8455,
        529.8455,
        535.0630,
        535.0630,
        672.0969,
        721.0808,
    ],
}


def _load_gan_second():
    """Load a fresh q2r IFC2 object for state-mutating provenance tests."""
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/gan",
        supercell=[5, 5, 5],
        only_second=True,
        format="qe-d3q",
    )
    return forceconstants.second


@pytest.fixture(scope="session")
def gan_second():
    return _load_gan_second()


def _frequencies_cm(second, q_point):
    hwq = HarmonicWithQ(q_point=np.array(q_point), second=second)
    assert hwq.ifc_interpolation_resolved == "wigner-seitz"
    return np.sort(np.array(hwq.frequency).flatten()) * THZ_TO_CM


def _qe76_velocity_slopes():
    """Read the independently generated QE 7.6 central-difference oracle.

    ``matdyn_velocity_reference.in`` records the six q points.  Its companion
    ``.freq`` file was generated with QE 7.6's ``matdyn.x`` as documented in
    that input file.  Each +/- pair has a Cartesian separation of
    ``2e-3 / angstrom``.
    """
    lines = (
        (Path(__file__).with_name("gan") / "matdyn_velocity_reference.freq")
        .read_text()
        .splitlines()[1:]
    )
    frequencies = []
    index = 0
    while index < len(lines):
        # Every q-point header has four floats; the following two lines hold
        # the 12 mode frequencies in cm^-1.
        [float(value) for value in lines[index].split()]
        index += 1
        modes = []
        while len(modes) < 12:
            modes.extend(float(value) for value in lines[index].split())
            index += 1
        frequencies.append(modes)
    assert len(frequencies) == 6
    frequency = np.asarray(frequencies, dtype=np.float64) / THZ_TO_CM
    return (frequency[::2] - frequency[1::2]) / (2.0e-3)


def test_gamma_frequencies_match_matdyn(gan_second):
    actual = _frequencies_cm(gan_second, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(
        actual, MATDYN_REFERENCE[(0.0, 0.0, 0.0)], rtol=0.0, atol=2.0e-3
    )


def test_in_plane_dispersion_matches_matdyn(gan_second):
    actual = _frequencies_cm(gan_second, (0.25, 0.0, 0.0))
    np.testing.assert_allclose(
        actual, MATDYN_REFERENCE[(0.25, 0.0, 0.0)], rtol=0.0, atol=2.0e-3
    )


def test_out_of_plane_dispersion_matches_matdyn(gan_second):
    actual = _frequencies_cm(gan_second, (0.0, 0.0, 0.25))
    np.testing.assert_allclose(
        actual, MATDYN_REFERENCE[(0.0, 0.0, 0.25)], rtol=0.0, atol=2.0e-3
    )


def test_gamma_A_transverse_degeneracy(gan_second):
    # Along Gamma-A the two TA branches are symmetry-required to be degenerate.
    actual = _frequencies_cm(gan_second, (0.0, 0.0, 0.25))
    assert abs(actual[0] - actual[1]) < 2.0e-3


def test_missing_born_charges_warn(caplog):
    import logging as _logging

    with caplog.at_level(_logging.WARNING):
        _, _, charges = qe_io.read_second_order_qe_matrix(
            "kaldo/tests/gan/espresso.ifc2"
        )
    assert charges is None
    assert any("non-analytic correction" in message for message in caplog.messages)


def test_nac_off_q2r_header_uses_generic_gonze_if_polar_data_is_supplied(
    monkeypatch,
):
    """A q2r header without Z* is provenance, not a QE-rigid-ion request."""
    gan_second = _load_gan_second()
    gan_second.atoms.info["dielectric"] = np.eye(3)
    gan_second.atoms.set_array(
        "charges", np.full((len(gan_second.atoms), 3, 3), 0.1), shape=(3, 3)
    )

    static_data = nac.build_static_data(gan_second)
    assert static_data["convention"] != "qe_q2r"

    monkeypatch.setattr(
        gan_second,
        "calculate_nac_short_range_force_constants",
        lambda matrix=None: "generic-gonze",
    )
    assert (
        gan_second.get_nac_short_range_force_constants(np.diag([5, 5, 5]))
        == "generic-gonze"
    )


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
def test_pair_aware_velocity_matches_dispersion_gradient(gan_second, axis):
    """The full ``R+r_j-r_i`` kernel must differentiate the same spectrum."""
    q_point = np.array([0.13, 0.07, 0.11], dtype=np.float64)
    harmonic = HarmonicWithQ(
        q_point=q_point,
        second=gan_second,
        storage="memory",
        is_nac=False,
        ifc_interpolation="wigner-seitz",
    )
    order = np.argsort(np.asarray(harmonic.frequency).reshape(-1))
    velocity = np.asarray(harmonic.velocity)[0][order, axis]

    wavevector_step = 1.0e-4  # 1/angstrom
    direction = np.eye(3)[axis]
    dq_reduced = (
        np.asarray(gan_second.atoms.cell) @ direction * wavevector_step / (2.0 * np.pi)
    )

    def frequencies(offset):
        sample = HarmonicWithQ(
            q_point=q_point + offset,
            second=gan_second,
            storage="memory",
            is_nac=False,
            ifc_interpolation="wigner-seitz",
        )
        return np.sort(np.asarray(sample.frequency).reshape(-1))

    slope = (frequencies(dq_reduced) - frequencies(-dq_reduced)) / (
        2.0 * wavevector_step
    )
    usable = np.abs(slope) > 0.05
    assert np.any(usable)
    np.testing.assert_allclose(
        velocity[usable] / slope[usable],
        2.0 * np.pi,
        rtol=2.0e-3,
        atol=0.0,
    )


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
def test_pair_aware_velocity_matches_qe76_matdyn_gradient(gan_second, axis):
    """NAC-off q2r velocity agrees with an independent QE 7.6 oracle."""
    harmonic = HarmonicWithQ(
        q_point=np.array([0.13, 0.07, 0.11], dtype=np.float64),
        second=gan_second,
        storage="memory",
        is_nac=False,
        ifc_interpolation="wigner-seitz",
    )
    order = np.argsort(np.asarray(harmonic.frequency).reshape(-1))
    actual = np.asarray(harmonic.velocity)[0][order, axis] / (2.0 * np.pi)
    expected = _qe76_velocity_slopes()[axis]
    # The reference frequencies are printed to 1e-4 cm^-1, limiting the
    # finite-difference slope to roughly 1.5e-3 THz angstrom.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.6e-3)

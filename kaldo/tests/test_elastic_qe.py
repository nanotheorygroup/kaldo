"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
import pytest
from pathlib import Path
from ase import units


@pytest.fixture(scope="session")
def forceconstants():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-crystal/qe", supercell=[3, 3, 3], format="qe-sheng")
    return forceconstants


def test_c11(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[0, 0, 0, 0], 167.2, significant=3)


def test_c12(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[0, 0, 1, 1], 88.54, significant=3)


def test_c44(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[1, 2, 1, 2], 70.22, significant=3)


QE_FIXTURE = Path(__file__).parent / "si-crystal" / "qe"


def _qe76_acoustic_elastic_constants(forceconstants):
    """Derive cubic elastic constants from retained QE acoustic slopes."""
    lines = (QE_FIXTURE / "matdyn_elastic_reference.freq").read_text().splitlines()
    mode_rows = []
    for line in lines:
        tokens = line.split()
        if len(tokens) != 6:
            continue
        try:
            mode_rows.append([float(value) for value in tokens])
        except ValueError:
            # Skip matdyn's ``&plot nbnd=..., nks=... /`` header.
            continue
    mode_rows = np.asarray(mode_rows)
    x_modes = mode_rows[0, :3]
    diagonal_modes = mode_rows[3, :3]

    # matdyn reports wavenumbers in cm^-1. The retained first points have
    # |k|=1e-3 /angstrom, so v=d(2*pi*nu)/dk. The primitive cell contains the
    # two Si atoms whose q2r masses are authoritative for this comparison.
    speed_of_light = 299_792_458.0
    wavevector_per_metre = 1.0e7
    velocity_per_wavenumber = (
        2.0 * np.pi * speed_of_light * 100.0 / wavevector_per_metre
    )
    density = (
        np.sum(forceconstants.atoms.get_masses())
        * units._amu
        / (forceconstants.atoms.get_volume() * 1.0e-30)
    )
    x_velocity = x_modes * velocity_per_wavenumber
    diagonal_velocity = diagonal_modes * velocity_per_wavenumber

    c11 = density * x_velocity[2] ** 2 / 1.0e9
    c44 = density * x_velocity[0] ** 2 / 1.0e9
    c12_from_transverse = c11 - 2.0 * density * diagonal_velocity[0] ** 2 / 1.0e9
    c12_from_longitudinal = (
        2.0 * density * diagonal_velocity[2] ** 2 / 1.0e9 - c11 - 2.0 * c44
    )
    np.testing.assert_allclose(
        c12_from_transverse, c12_from_longitudinal, rtol=0.0, atol=0.1
    )
    return c11, 0.5 * (c12_from_transverse + c12_from_longitudinal), c44


def test_elastic_constants_match_qe76_acoustic_slopes(forceconstants):
    """The q->0 IFC moments must reproduce QE 7.6's acoustic dispersion."""
    cijkl = forceconstants.elastic_prop()
    reference = _qe76_acoustic_elastic_constants(forceconstants)
    actual = (
        cijkl[0, 0, 0, 0],
        cijkl[0, 0, 1, 1],
        cijkl[1, 2, 1, 2],
    )
    np.testing.assert_allclose(actual, reference, rtol=5.0e-4, atol=0.0)

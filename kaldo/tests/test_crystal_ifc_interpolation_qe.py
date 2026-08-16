"""
Unit and regression test for the kaldo package.
Tests Wigner--Seitz interpolation of second-order force constants.
"""

# Import package, test suite, and other packages as needed
import numpy as np
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import pytest


@pytest.fixture(scope="session")
def phonons():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/si-crystal/qe", supercell=[3, 3, 3], format="qe-sheng"
    )
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[3, 3, 3],
        is_classic=False,
        temperature=300,
        ifc_interpolation="wigner-seitz",
        storage="memory",
    )
    return phonons


def test_unfolding_dispersion(phonons):
    q_point = np.array([0.3, 0, 0.3])  # chosen to check we get a degenerate pair for both acoustic and optical
    frequency_expected = np.array([4.11380807, 4.11380825, 8.44285067, 14.00947531, 14.00947536, 14.37330857])
    frequency_actual = HarmonicWithQ(
        q_point=q_point,
        second=phonons.forceconstants.second,
        ifc_interpolation="wigner-seitz",
    ).frequency
    frequency_actual = frequency_actual.flatten()  # HWQ outputs a 2d array
    np.testing.assert_array_almost_equal(frequency_expected, frequency_actual, decimal=2)


def test_zero_born_charge_q2r_auto_uses_qe_periodic_convention(phonons):
    """A present but zero Z* block is nonpolar and must not select NAC/WS.

    QE q2r files may contain the dielectric/Born section even when every Born
    tensor is zero.  Section presence is therefore not sufficient evidence of
    a polar long-range term; the ordinary q2r body must retain the direct
    periodic transform validated against ``matdyn.x``.
    """
    second = phonons.forceconstants.second

    assert "charges" in second.atoms.arrays
    np.testing.assert_allclose(second.atoms.arrays["charges"], 0.0, atol=0.0)
    assert second.ifc_interpolation_hint == "periodic"
    harmonic = HarmonicWithQ(
        q_point=np.array([0.3, 0.0, 0.3]),
        second=second,
        ifc_interpolation="auto",
        storage="memory",
    )
    assert harmonic.is_nac is False
    assert harmonic.ifc_interpolation_resolved == "periodic"

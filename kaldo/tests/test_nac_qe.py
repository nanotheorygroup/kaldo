"""
Contract tests for QE force-constant files that embed Born charges.

q2r.x writes dipole-subtracted force constants when it is given a dielectric
tensor (the flag line is T). kALDo used to pair these with a reciprocal-space
Ewald term (the legacy NAC); after its removal the non-analytic correction, which expects
total force constants, must refuse this convention loudly instead of
subtracting the dipole part twice. MgO frequencies for the legacy pairing were
validated against matdyn.x (asr=simple, q=(0.3, 0, 0.3) crystal:
239.764x2, 367.69, 422.93x2, 582.65 cm^-1); restoring physics for this
convention means bridging it to total force constants at load, tracked as
follow-up work.
"""

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ


@pytest.fixture(scope="module")
def mgo_second():
    import tempfile
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/mgo",
        supercell=[5, 5, 5],
        only_second=True,
        format="qe-d3q",
    )
    second = forceconstants.second
    second.folder = tempfile.mkdtemp(prefix="mgo_nac_test_")
    return second


def test_embedded_charges_mark_the_convention(mgo_second):
    assert mgo_second.atoms.info.get("dipole_subtracted_fc") is True
    assert "dielectric" in mgo_second.atoms.info
    assert "charges" in mgo_second.atoms.arrays


def test_dipole_subtracted_constants_refuse_nac(mgo_second):
    phonon = HarmonicWithQ(
        q_point=np.array([0.3, 0.0, 0.3]),
        second=mgo_second,
        storage="memory",
        is_unfolding=True,
    )
    with pytest.raises(NotImplementedError, match="dipole-subtracted"):
        phonon.frequency

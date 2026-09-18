"""Contract between QE NAC and IFC interpolation.

For an ordinary NAC-off force-constant model, the public selector chooses
the legacy periodic-replica Fourier sum or Wigner--Seitz shortest-vector
interpolation.  A polar QE q2r file has a stronger convention: its IFC body is
already short range, and reconstructing the QE dynamical matrix requires

    D_QE(q) = FT_WS[Phi_q2r^SR](q) + D_QE^rigid(q).

The provenance-aware QE NAC controller therefore owns the Wigner--Seitz step.
``auto`` and ``wigner-seitz`` give the same NAC-on physics; ``periodic`` is
rejected instead of being silently ignored. With NAC explicitly disabled,
``periodic`` remains available as a diagnostic model.
"""

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ


THZ_TO_CM = 33.3564095198152
Q_POINT = np.array([0.3, 0.0, 0.3], dtype=np.float64)
QE_MATDYN_FREQUENCIES_CM = np.array(
    [239.7640, 239.7640, 367.6916, 422.9322, 422.9322, 582.6500],
    dtype=np.float64,
)


@pytest.fixture(scope="module")
def qe_mgo_second(tmp_path_factory):
    """Load the polar QE 7.6 MgO q2r fixture once for this contract."""
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/mgo",
        supercell=[5, 5, 5],
        only_second=True,
        format="qe-d3q",
    )
    second = forceconstants.second
    second.folder = str(tmp_path_factory.mktemp("qe_nac_unfolding_contract"))
    return second


def _harmonic(second, *, interpolation, is_nac=None):
    return HarmonicWithQ(
        q_point=Q_POINT,
        second=second,
        storage="memory",
        ifc_interpolation=interpolation,
        is_nac=is_nac,
    )


@pytest.mark.parametrize("interpolation", ["auto", "wigner-seitz"])
def test_qe_nac_matches_matdyn_for_supported_interpolation_modes(
    qe_mgo_second, interpolation
):
    """The QE-owned WS path remains active for both supported selectors."""
    harmonic = _harmonic(qe_mgo_second, interpolation=interpolation)
    frequency_cm = np.asarray(harmonic.frequency[0]) * THZ_TO_CM

    assert harmonic.is_nac is True
    assert harmonic.ifc_interpolation_resolved == "wigner-seitz"
    np.testing.assert_allclose(
        frequency_cm,
        QE_MATDYN_FREQUENCIES_CM,
        rtol=0.0,
        atol=1.0e-2,
    )
    for left, right in ((0, 1), (3, 4)):
        np.testing.assert_allclose(
            frequency_cm[left],
            frequency_cm[right],
            rtol=0.0,
            atol=5.0e-6,
        )


def test_qe_nac_observables_are_invariant_between_auto_and_ws(qe_mgo_second):
    """Frequencies and their q-gradient must come from the same QE controller."""
    automatic = _harmonic(qe_mgo_second, interpolation="auto")
    wigner_seitz = _harmonic(qe_mgo_second, interpolation="wigner-seitz")

    np.testing.assert_allclose(
        automatic.frequency,
        wigner_seitz.frequency,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        automatic.velocity,
        wigner_seitz.velocity,
        rtol=0.0,
        atol=0.0,
    )


def test_qe_nac_off_allows_direct_periodic_diagnostic(qe_mgo_second):
    """With restoration disabled, users may request the ordinary periodic path."""
    periodic = _harmonic(qe_mgo_second, interpolation="periodic", is_nac=False)
    wigner_seitz = _harmonic(
        qe_mgo_second, interpolation="wigner-seitz", is_nac=False
    )

    assert periodic.is_nac is False
    assert wigner_seitz.is_nac is False
    assert np.max(np.abs(periodic.frequency - wigner_seitz.frequency)) > 1.0e-3


def test_qe_nac_rejects_periodic_override(qe_mgo_second):
    with pytest.raises(ValueError, match="incompatible with active NAC"):
        _harmonic(qe_mgo_second, interpolation="periodic")

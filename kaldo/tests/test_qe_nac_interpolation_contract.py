"""Contract between QE NAC and the legacy ``is_unfolding`` option.

For an ordinary NAC-off force-constant model, ``is_unfolding`` selects either
the legacy periodic-replica Fourier sum or Wigner--Seitz shortest-vector
interpolation.  A polar QE q2r file has a stronger convention: its IFC body is
already short range, and reconstructing the QE dynamical matrix requires

    D_QE(q) = FT_WS[Phi_q2r^SR](q) + D_QE^rigid(q).

The provenance-aware QE NAC controller therefore owns the Wigner--Seitz step.
The legacy flag must not disable it: doing so would restore the plain replica
transform that splits symmetry degeneracies and disagrees with ``matdyn.x``
away from Gamma.  Both flag values must consequently give the same NAC-on
physics.  When NAC is explicitly disabled, the flag controls interpolation of
the remaining short-range diagnostic model again.
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


def _harmonic(second, *, is_unfolding, is_nac=None):
    return HarmonicWithQ(
        q_point=Q_POINT,
        second=second,
        storage="memory",
        is_unfolding=is_unfolding,
        is_nac=is_nac,
    )


@pytest.mark.parametrize("is_unfolding", [False, True])
def test_qe_nac_matches_matdyn_for_both_legacy_flag_values(
    qe_mgo_second, is_unfolding
):
    """The QE-owned WS path must remain active for either legacy flag value."""
    harmonic = _harmonic(qe_mgo_second, is_unfolding=is_unfolding)
    frequency_cm = np.asarray(harmonic.frequency[0]) * THZ_TO_CM

    assert harmonic.is_nac is True
    assert harmonic.is_unfolding is is_unfolding
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


def test_qe_nac_observables_are_invariant_to_legacy_unfolding_flag(qe_mgo_second):
    """Frequencies and their q-gradient must come from the same QE controller."""
    periodic_requested = _harmonic(qe_mgo_second, is_unfolding=False)
    unfolded_requested = _harmonic(qe_mgo_second, is_unfolding=True)

    np.testing.assert_allclose(
        periodic_requested.frequency,
        unfolded_requested.frequency,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        periodic_requested.velocity,
        unfolded_requested.velocity,
        rtol=0.0,
        atol=0.0,
    )


def test_qe_nac_off_restores_legacy_unfolding_control(qe_mgo_second):
    """With restoration disabled, the flag again selects the ordinary IFC path."""
    periodic = _harmonic(qe_mgo_second, is_unfolding=False, is_nac=False)
    unfolded = _harmonic(qe_mgo_second, is_unfolding=True, is_nac=False)

    assert periodic.is_nac is False
    assert unfolded.is_nac is False
    assert np.max(np.abs(periodic.frequency - unfolded.frequency)) > 1.0e-3

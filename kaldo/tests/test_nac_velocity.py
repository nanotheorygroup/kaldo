"""Regression tests for NAC group velocities.

The removed legacy NAC implementation evaluated frequencies and velocity
derivatives with separate Ewald formulas.  On an FCC primitive cell their
Gaussian widths differed, so the reported velocity was not the gradient of
the dispersion.  The current controller finite-differences one shared polar
dynamical matrix; these tests preserve the physical invariant through the
public API rather than replaying private intermediate tensors.
"""

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ


NAC_BVK_MATRIX = np.diag([8, 8, 8])
Q_POINT = np.array([0.073, 0.041, 0.029])
WAVEVECTOR_STEP = 1.0e-3  # 1/angstrom


@pytest.fixture(scope="module")
def generic_fcc_second(tmp_path_factory):
    """Load the NaCl FCC fixture as total IFCs for the generic Gonze path."""
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    second = forceconstants.second
    # This fixture is used as total force constants, as in the generic
    # VASP/Phonopy convention.  Remove its q2r provenance deliberately so the
    # test exercises Gonze preparation rather than QE rigid-ion restoration.
    second.atoms.info.pop("dipole_subtracted_fc", None)
    second._qe_q2r_header = None
    second.folder = str(tmp_path_factory.mktemp("nac_velocity_fcc"))
    assert "dielectric" in second.atoms.info
    assert np.max(np.abs(second.atoms.get_array("charges"))) > 0
    return second


def _sorted_frequencies(second, q_point):
    harmonic = HarmonicWithQ(
        q_point=np.asarray(q_point, dtype=float),
        second=second,
        storage="memory",
        nac_bvk_supercell_matrix=NAC_BVK_MATRIX,
    )
    return np.sort(np.asarray(harmonic.frequency).reshape(-1))


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
def test_generic_fcc_nac_velocity_is_dispersion_gradient(generic_fcc_second, axis):
    """The FCC NAC velocity equals the independently sampled dispersion slope."""
    harmonic = HarmonicWithQ(
        q_point=Q_POINT,
        second=generic_fcc_second,
        storage="memory",
        nac_bvk_supercell_matrix=NAC_BVK_MATRIX,
    )
    order = np.argsort(np.asarray(harmonic.frequency).reshape(-1))
    velocity = np.asarray(harmonic.velocity)[0][order, axis]

    direction = np.eye(3)[axis]
    cell = np.asarray(generic_fcc_second.atoms.cell)
    # q_red uses cycles per reciprocal lattice vector, while the independent
    # slope below is taken with respect to the physical wavevector k in 1/A.
    dq_red = cell @ direction * WAVEVECTOR_STEP / (2 * np.pi)
    plus = _sorted_frequencies(generic_fcc_second, Q_POINT + dq_red)
    minus = _sorted_frequencies(generic_fcc_second, Q_POINT - dq_red)
    slope = (plus - minus) / (2 * WAVEVECTOR_STEP)

    usable = np.abs(slope) > 0.05
    assert np.any(usable), f"no dispersive modes found along Cartesian axis {axis}"
    # kALDo stores frequency in cycles/ps but velocity is d(2*pi*nu)/dk.
    np.testing.assert_allclose(
        velocity[usable] / slope[usable],
        2 * np.pi,
        rtol=1.0e-2,
        atol=0.0,
    )

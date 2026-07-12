"""
Wurtzite GaN second order loaded through the vasp path, validated against
phonopy on the same force constants.

The fixture derives from Togo's ab-initio GaN dataset (NIMS MDR xp68kn42d,
CC-BY-4.0): fc2 reconstructed with phonopy from the published
displacement-force data and written as a full FORCE_CONSTANTS_2ND on the
4x4x2 supercell. Reference frequencies come from phonopy on the identical
force constants (NAC off, matching the POSCAR-only kaldo path), so any
disagreement is a kaldo dynamical-matrix assembly bug. This is the
anisotropic-cell validation of the vasp replica declaration (#279 item 2);
in-plane and out-of-plane q-points would expose a wrong pairing through
broken degeneracies and shifted branches.

The out-of-plane point is incommensurate with the 2-replica c-axis, where
the folded path's half-box handling deviates (~0.3 THz, the #279 item 3
mechanism), so the phonopy comparison pins the Wigner-Seitz unfolded path.
"""
import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ

# phonopy 2.43 on the same fc2, NAC off, THz
PHONOPY_REFERENCE = {
    (0.00, 0.00, 0.00): [0.0, 0.0, 0.0, 4.1072, 4.1072, 9.8195,
                         15.5017, 16.1776, 16.1776, 16.4344, 16.4344, 20.3111],
    (0.25, 0.00, 0.00): [3.1003, 3.3685, 5.1045, 6.1886, 6.7963, 8.8917,
                         16.1010, 16.4690, 17.2397, 18.5178, 19.9445, 21.0219],
    (0.00, 0.00, 0.25): [1.7813, 1.7813, 3.8863, 3.8863, 4.1001, 8.8200,
                         16.2038, 16.2038, 16.3854, 16.3854, 17.8017, 21.1162],
}


@pytest.fixture(scope="session")
def gan_second():
    fc = ForceConstants.from_folder(folder="kaldo/tests/gan-vasp", supercell=[4, 4, 2],
                                    format="vasp-sheng", only_second=True)
    return fc.second


@pytest.mark.parametrize("q_point", list(PHONOPY_REFERENCE))
def test_unfolded_frequencies_match_phonopy(gan_second, q_point):
    f = np.sort(np.array(HarmonicWithQ(np.array(q_point), gan_second, storage="memory",
                                       is_unfolding=True).frequency).flatten())
    np.testing.assert_allclose(f[3:], PHONOPY_REFERENCE[q_point][3:], atol=2e-2)


def test_folded_in_plane_frequencies_match_phonopy(gan_second):
    """At a commensurate in-plane q the folded path must agree too; the
    out-of-plane point is left to the unfolded test above on purpose."""
    f = np.sort(np.array(HarmonicWithQ(np.array([0.25, 0.0, 0.0]), gan_second,
                                       storage="memory").frequency).flatten())
    np.testing.assert_allclose(f, PHONOPY_REFERENCE[(0.25, 0.0, 0.0)], atol=2e-2)

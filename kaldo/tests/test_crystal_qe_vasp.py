"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import pytest


@pytest.fixture(scope="session")
def phonons():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/si-crystal/qe",
        supercell=[3, 3, 3],
        third_supercell=[3, 3, 3],
        format="qe-sheng")
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[3, 3, 3],
        is_classic=False,
        temperature=300,
        storage="memory",
    )
    return phonons

def test_lagacy_format():
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/si-crystal/qe",
        supercell=[3, 3, 3],
        third_supercell=[3, 3, 3],
        format="qe-sheng")
    
    forceconstants2 = ForceConstants.from_folder(
        folder="kaldo/tests/si-crystal/qe",
        supercell=[3, 3, 3],
        third_supercell=[3, 3, 3],
        format="shengbte-qe")
    
    np.testing.assert_equal(forceconstants.second.value, forceconstants2.second.value)
    np.testing.assert_equal(forceconstants.third.value, forceconstants2.third.value)


def test_qhgk_conductivity(phonons):
    """QHGK conductivity with constant diffusivity_bandwidth.

    Why a constant bandwidth: with the default
    ``diffusivity_bandwidth=None``, the QHGK calculation uses
    ``phonons.bandwidth / 2`` per mode, which includes
    ``isotopic_bandwidth`` when ``include_isotopes=True``. The
    isotopic bandwidth involves the squared overlap
    ``|⟨e_q'μ' | e_qμ⟩|²``, which is NOT invariant under unitary
    rotations of the eigenvector basis within a degenerate subspace.
    Different LAPACK implementations (x86 vs arm64) return different
    bases — all valid solutions — so per-mode isotopic bandwidth (and
    therefore the per-mode diffusivity_bandwidth, and therefore the
    per-mode QHGK conductivity contribution) varies platform-by-platform.

    Setting ``diffusivity_bandwidth=1.0`` (THz, constant) decouples
    the conductivity from this gauge-dependent per-mode quantity. The
    SUM of conductivity contributions over a degenerate group remains
    basis-invariant (trace of the Onsager-style scattering operator
    in the degenerate subspace), but the per-mode breakdown does not.

    Same gauge-vs-observable pattern as ``test_iso_bw`` and the
    broadening-vs-symmetry trade-off documented for ``use_q_symmetry``
    (PR #253): replace the basis-dependent regularization with a
    constant to get a reproducible reference value.

    Trade-off: this test no longer exercises the default per-mode
    bandwidth path. That path's correctness is exercised by
    ``test_qhgk_conductivity`` in ``test_crystal.py`` on a
    centrosymmetric crystal where the degeneracies are sparse enough
    that the basis dependence stays below the test tolerance.
    """
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory",
                        diffusivity_bandwidth=1.0).conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_approx_equal(cond, 2.2, significant=2)


def test_rta_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_approx_equal(cond, 2.7, significant=2)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_approx_equal(cond, 2.4, significant=2)

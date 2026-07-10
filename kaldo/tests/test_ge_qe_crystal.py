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
        folder="kaldo/tests/ge-crystal",
        supercell=[10, 10, 10],
        format="qe-d3q",
        only_second=True,
    )
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[7, 7, 7],
        is_classic=False,
        temperature=300,
        include_isotopes=True,
        storage="memory",
    )
    return phonons


def test_frequency(phonons):
    freq = phonons.frequency.flatten()
    np.testing.assert_approx_equal(freq[3], 8.99, significant=3)


def test_iso_bw(phonons):
    """Isotopic bandwidth at the Γ-point optical-mode triplet.

    Physics: Ge diamond has a 3-fold degenerate optical phonon triplet at
    the Γ point (modes 3, 4, 5). Within any degenerate subspace the
    eigenvectors are defined only up to a unitary rotation U, and
    different LAPACK implementations (x86 vs arm64, vendor BLAS vs MKL)
    return different bases — all equally valid solutions of the
    eigenvalue problem.

    The Tamura formula for isotopic scattering involves the squared
    overlap |⟨e_q'μ' | e_qμ⟩|², which is NOT invariant under a basis
    rotation within the degenerate subspace. Concretely, if
    ẽ_3 = U_33 e_3 + U_34 e_4 + U_35 e_5, then
    |⟨e_μ' | ẽ_3⟩|² = |U_33 A + U_34 B + U_35 C|², which is not equal
    to |A|² alone unless U is the identity.

    The SUM (or mean) of Γ_iso(μ) over the degenerate group IS
    basis-invariant: cross-terms cancel by unitarity (∑_μ U*_μν U_μν' =
    δ_νν'), so ∑_μ |∑_ν U_μν A_ν|² = ∑_ν |A_ν|². This is the trace of
    the scattering operator restricted to the degenerate subspace — a
    true physical observable.

    This test therefore asserts the basis-invariant mean over the
    triplet, which is the trace of the scattering operator restricted
    to the degenerate subspace divided by the dimension. Asserting an
    individual triplet member would be platform-dependent.

    The per-mode SPREAD within the triplet (max - min) is itself
    basis-dependent and varies platform-by-platform — we deliberately
    do NOT bound it.

    Same gauge-vs-observable distinction as the broadening-vs-symmetry
    issue documented for ``use_q_symmetry`` (PR #253): an operation
    (rotation, broadening function evaluation) that doesn't commute
    with the gauge produces basis-dependent per-mode quantities; the
    invariant lives at the trace level.
    """
    iso_bw = phonons.isotopic_bandwidth.flatten()
    triplet_mean = np.mean(iso_bw[3:6])
    np.testing.assert_approx_equal(triplet_mean, 7.75 * 1e-3, significant=3)

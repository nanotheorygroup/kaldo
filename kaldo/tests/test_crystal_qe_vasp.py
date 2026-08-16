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
        third_bandwidth=0.5,  # fixed: gauge-invariant regression config (#290)
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
    # The q2r velocity now differentiates QE's native R-only Fourier phase;
    # the previous reference used an inconsistent R+r_j-r_i derivative.
    np.testing.assert_allclose(cond, 2.266229, rtol=5e-3, atol=0.0)


def test_rta_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    # recalibrated after the ShengBTE FC3 parser fix (mod-wrapped cell offsets); the old value baked in
    # silently dropped force constants
    np.testing.assert_allclose(cond, 4.500018, rtol=5e-3, atol=0.0)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    # recalibrated after the ShengBTE FC3 parser fix (mod-wrapped cell offsets); the old value baked in
    # silently dropped force constants
    np.testing.assert_allclose(cond, 5.049113, rtol=5e-3, atol=0.0)


def _translated_qe_si_phonons(fractional_origin_shift):
    """Load Si, move the origin, and rebuild consistent replica positions."""
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/si-crystal/qe",
        supercell=[3, 3, 3],
        third_supercell=[3, 3, 3],
        format="qe-sheng",
    )
    for observable in (forceconstants.second, forceconstants.third):
        scaled = observable.atoms.get_scaled_positions(wrap=False)
        observable.atoms.set_scaled_positions(scaled + fractional_origin_shift)
        observable.atoms.wrap()
        observable.replicated_positions = (
            observable.replica_translations[:, np.newaxis, :]
            @ np.asarray(observable.atoms.cell)
            + observable.atoms.positions[np.newaxis, :, :]
        )
    return Phonons(
        forceconstants=forceconstants,
        kpts=[3, 3, 3],
        is_classic=False,
        temperature=300,
        third_bandwidth=0.5,
        storage="memory",
    )


def test_qe_si_rta_is_invariant_to_wrapped_crystal_origin(phonons):
    """A rigid origin shift must not alter boundary IFC3 weight or RTA kappa.

    The commensurate harmonic spectrum was already origin invariant in the
    legacy implementation, so frequencies alone could not detect the bug.
    This QE fixture is intentional: unlike the ESKM fixture, its Wigner--Seitz
    IFC3 has nonzero weight on images outside the 27 stored representatives.
    Historically the wrapped origin shift below changed RTA conductivity by
    about 56 percent even though frequencies agreed to roughly 1e-14.
    """
    translated = _translated_qe_si_phonons(np.array([0.8, 0.1, 0.1]))

    interpolation = phonons.forceconstants.third.get_interpolation("auto")
    periodic = phonons.forceconstants.third.get_interpolation("periodic")
    periodic_translations = {
        tuple(translation) for translation in periodic.support.translations
    }
    outside_periodic = np.array([
        tuple(translation) not in periodic_translations
        for translation in interpolation.support.translations
    ])
    coords = np.asarray(interpolation.value.coords)
    boundary_entries = outside_periodic[coords[2]] | outside_periodic[coords[5]]
    boundary_weight = np.sum(
        np.abs(np.asarray(interpolation.value.data)[boundary_entries])
    )

    assert np.count_nonzero(boundary_entries) > 0
    assert boundary_weight > 1.0
    np.testing.assert_allclose(
        translated.frequency,
        phonons.frequency,
        rtol=0.0,
        atol=2e-13,
    )

    reference_conductivity = Conductivity(
        phonons=phonons, method="rta", storage="memory"
    ).conductivity.sum(axis=0)
    translated_conductivity = Conductivity(
        phonons=translated, method="rta", storage="memory"
    ).conductivity.sum(axis=0)
    np.testing.assert_allclose(
        translated_conductivity,
        reference_conductivity,
        rtol=1e-10,
        atol=1e-12,
    )

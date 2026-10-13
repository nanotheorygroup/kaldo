"""End-to-end calculator/MLIP-path regression for IFC interpolation.

This is the self-contained Lennard--Jones argon experiment from Issue 2 of
``chiral-Te-ace-kaldo``.  A rigid shift of every atom changes only where the
unit-cell boundary is drawn; it cannot change a phonon frequency or thermal
conductivity.  The legacy replica-only Fourier phases violated that gauge
invariance even though the frequencies on the commensurate mesh agreed.

The callable calculator factory passed to ``SecondOrder.calculate`` and
``ThirdOrder.calculate`` is the same integration boundary used by MLIP ASE
calculators.  Lennard--Jones keeps this unit test self-contained: it tests
kALDo's direct finite-displacement path without making CI depend on one MLIP
framework or model file.
"""

from functools import partial

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons


def _relaxed_argon_reference():
    """Return FCC argon in a three-atom hexagonal cell, relaxed once and pinned.

    Relaxing at test time made the structure depend on the installed ASE
    version. Band folding makes many modes (near-)degenerate. The third atom
    sits at negative fractional coordinates, which the translated copy wraps
    into the cell.
    """
    cell = np.array(
        [
            [3.719747779668492, 1.7036413759748925e-16, 5.017496633257669e-16],
            [-1.8598738898342444, 3.2213960728643447, -1.0038131725575458e-12],
            [-7.455783275549184e-14, -1.215204313724951e-12, 9.111516256278476],
        ]
    )
    positions = np.array(
        [
            [1.24000153, -0.0, 3.03717209],
            [-0.62000076, 1.07387282, 6.07434417],
            [-0.62000076, -1.07387282, 0.0],
        ]
    )
    return Atoms("Ar3", cell=cell, positions=positions, pbc=True)


def _lj_phonons(atoms, folder):
    """Generate finite-displacement IFCs and the corresponding phonons."""
    forceconstants = ForceConstants(
        atoms=atoms,
        supercell=np.array([3, 3, 3]),
        third_supercell=np.array([2, 2, 2]),
        folder=str(folder),
    )
    calculator = partial(
        LennardJones,
        sigma=3.4,
        epsilon=0.01,
        rc=9.0,
        smooth=True,
    )
    forceconstants.second.calculate(
        calculator=calculator,
        delta_shift=1e-3,
        n_workers=1,
        # IFC symmetrization is a separate projection.  Keeping it out of this
        # regression isolates the real-space interpolation gauge under test.
        symmetrize=False,
    )
    forceconstants.third.calculate(
        calculator=calculator,
        delta_shift=1e-3,
        n_workers=1,
        symmetrize=False,
    )
    # A fixed width keeps linewidths independent of the eigenvector basis
    # inside degenerate subspaces, which differs between the two runs.
    # Adaptive widths inherit that basis through per-mode velocities.
    return Phonons(
        forceconstants=forceconstants,
        kpts=[6, 6, 6],
        is_classic=False,
        temperature=300.0,
        third_bandwidth=0.05,
        folder=str(folder / "ald"),
        storage="memory",
    )


def test_calculator_factory_rta_is_invariant_to_wrapped_crystal_origin(tmp_path):
    """The direct calculator path preserves spectra and RTA after a shift."""
    reference_atoms = _relaxed_argon_reference()
    translated_atoms = reference_atoms.copy()
    translated_atoms.set_scaled_positions(
        np.mod(
            reference_atoms.get_scaled_positions(wrap=False)
            + np.array([0.5, 0.5, 0.5]),
            1.0,
        )
    )
    translated_atoms.wrap()

    reference = _lj_phonons(reference_atoms, tmp_path / "reference")
    translated = _lj_phonons(translated_atoms, tmp_path / "translated")

    # Commensurate frequencies did not expose the historical bug while RTA
    # kappa changed by factors.  Independently regenerating finite-difference
    # IFCs leaves only numerical noise in the three near-zero Gamma modes;
    # keep that separate from the much tighter non-Gamma comparison.
    np.testing.assert_allclose(
        translated.frequency[1:],
        reference.frequency[1:],
        rtol=0.0,
        atol=1e-10,
    )
    # Gamma acoustic frequencies are numerically zero; their absolute drift
    # between the gauge-equivalent runs reaches 4.7e-7 THz on macOS arm64
    # (2e-7 on linux x86). 2e-6 stays six orders below the first optical mode.
    np.testing.assert_allclose(
        translated.frequency[0],
        reference.frequency[0],
        rtol=0.0,
        atol=2e-6,
    )

    # Linewidths test the anharmonic interpolation directly; with a fixed
    # width they agree to about 4e-8.
    physical = reference.physical_mode
    np.testing.assert_allclose(
        translated.bandwidth[physical],
        reference.bandwidth[physical],
        rtol=1e-5,
    )

    reference_kappa = Conductivity(
        phonons=reference,
        method="rta",
        storage="memory",
    ).conductivity.sum(axis=0)
    translated_kappa = Conductivity(
        phonons=translated,
        method="rta",
        storage="memory",
    ).conductivity.sum(axis=0)
    assert np.isfinite(reference_kappa).all()
    assert np.isfinite(translated_kappa).all()
    assert np.linalg.norm(reference_kappa) > 0.0
    assert np.linalg.norm(translated_kappa) > 0.0
    # Per-mode velocities inside degenerate subspaces follow the eigensolver
    # basis, so kappa xx/yy differ by up to ~1e-5 between the runs (zz by
    # 1e-8, since its velocity block is diagonal there). A wrong interpolation
    # gauge changes kappa by factors.
    np.testing.assert_allclose(
        translated_kappa,
        reference_kappa,
        rtol=0.0,
        atol=1e-4 * np.abs(reference_kappa).max(),
    )

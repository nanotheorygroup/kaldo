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
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons


def _relaxed_argon_reference():
    """Return the three-atom trigonal LJ crystal used by the reproducer."""
    a, c = 4.44, 6.10
    cell = np.array(
        [
            [a, 0.0, 0.0],
            [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, c],
        ]
    )
    x = 0.2648
    atoms = Atoms(
        "Ar3",
        cell=cell,
        scaled_positions=np.array(
            [
                [x, 0.0, 1.0 / 3.0],
                [0.0, x, 2.0 / 3.0],
                [-x, -x, 0.0],
            ]
        ),
        pbc=True,
    )
    atoms.calc = LennardJones(sigma=3.4, epsilon=0.01, rc=9.0, smooth=True)
    LBFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-4, steps=300)
    atoms.calc = None
    return atoms


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
    return Phonons(
        forceconstants=forceconstants,
        kpts=[6, 6, 6],
        is_classic=False,
        temperature=300.0,
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
    np.testing.assert_allclose(
        translated_kappa,
        reference_kappa,
        rtol=1e-7,
        atol=1e-12,
    )

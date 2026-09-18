"""Regression tests for signed unstable modes after IFC interpolation."""

from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from ase import Atoms

from kaldo.grid import SupercellGrid, TranslationSupport
from kaldo.observables.harmonic_with_q import HarmonicWithQ


def _one_atom_second_with_unstable_mode():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    omega2 = (2.0 * np.pi) ** 2 * np.diag([-1.0, 9.0, 16.0])
    supercell_grid = SupercellGrid(np.eye(3, dtype=int))
    second = SimpleNamespace(
        atoms=atoms,
        supercell=np.array([1, 1, 1]),
        dynmat=tf.constant(
            omega2.reshape(1, 1, 3, 1, 1, 3), dtype=tf.complex128
        ),
        translation_support=TranslationSupport.periodic(supercell_grid),
    )
    return second


def test_wigner_seitz_interpolation_preserves_signed_modes():
    harmonic = HarmonicWithQ(
        q_point=np.zeros(3),
        second=_one_atom_second_with_unstable_mode(),
        ifc_interpolation="wigner-seitz",
        storage="memory",
    )

    eigenvalues = harmonic.calculate_eigensystem(only_eigenvals=True)
    eigensystem = harmonic.calculate_eigensystem(only_eigenvals=False)

    expected = (2.0 * np.pi) ** 2 * np.array([-1.0, 9.0, 16.0])
    np.testing.assert_allclose(eigenvalues, expected)
    np.testing.assert_allclose(eigensystem[0], eigenvalues)
    np.testing.assert_allclose(harmonic.calculate_frequency(), [-1.0, 3.0, 4.0])

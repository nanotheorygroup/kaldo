"""Regression tests for signed unstable modes in Wigner-Seitz unfolding."""

from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from ase import Atoms

from kaldo.observables.harmonic_with_q import HarmonicWithQ


def _one_atom_second_with_unstable_mode():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    omega2 = (2.0 * np.pi) ** 2 * np.diag([-1.0, 9.0, 16.0])
    return SimpleNamespace(
        atoms=atoms,
        supercell=np.array([1, 1, 1]),
        dynmat=tf.constant(omega2, dtype=tf.complex128),
        supercell_positions=np.array([[0.0, 0.0, 0.0]]),
        supercell_replicas=np.array([[0, 0, 0]]),
    )


def test_unfolding_preserves_signed_eigenvalues_and_frequency():
    harmonic = HarmonicWithQ(
        q_point=np.zeros(3),
        second=_one_atom_second_with_unstable_mode(),
        is_unfolding=True,
        storage="memory",
    )

    eigenvalues = harmonic.calculate_eigensystem_unfolded(only_eigenvals=True)
    eigensystem = harmonic.calculate_eigensystem_unfolded(only_eigenvals=False)

    expected = (2.0 * np.pi) ** 2 * np.array([-1.0, 9.0, 16.0])
    np.testing.assert_allclose(eigenvalues, expected)
    np.testing.assert_allclose(eigensystem[0], eigenvalues)
    np.testing.assert_allclose(harmonic.calculate_frequency(), [-1.0, 3.0, 4.0])

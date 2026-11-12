"""TensorFlow-free Bloch harmonics at arbitrary q.

References are the historical ``HarmonicWithQ`` / ``HarmonicWithQTemp`` objects
at the same q point (TensorFlow route) for the 216-atom amorphous Si cell.
Dynamical-matrix elements are in (rad/ps)^2, frequencies in THz.
"""

from pathlib import Path

import numpy as np
import pytest

from kaldo.controllers import _amorphous_eigensystem, _bloch_eigensystem
from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ, _HarmonicIFCInterpolation
from kaldo.observables.harmonic_with_q_temp import HarmonicWithQTemp

_FOLDER = Path(__file__).resolve().parent / "si-amorphous"
_GENERIC_Q = (0.25, 0.25, 0.25)
# Eigenvalues closer than this fraction of the spectral width are treated as degenerate.
_DEGENERACY_FRACTION = 1e-6


@pytest.fixture(scope="session")
def forceconstants() -> ForceConstants:
    """216-atom a-Si IFC2 in eV/Angstrom^2."""
    return ForceConstants.from_folder(folder=str(_FOLDER), format="eskm")


@pytest.fixture(scope="module")
def plan(forceconstants):
    return _bloch_eigensystem.harmonic_plan(forceconstants.second, "auto")


def _reference(forceconstants, q):
    return HarmonicWithQ(q_point=np.array(q), second=forceconstants.second, storage="memory", is_amorphous=False)


def _solve(plan, q):
    matrix = _bloch_eigensystem.dynamical_matrix(plan, np.array(q, dtype=np.float64))
    return _bloch_eigensystem.eigensystem(matrix, 1)


@pytest.mark.parametrize(
    "q", [(0.25, 0.25, 0.25), (0.25, -0.25, 0.25), (0.5, 0.0, 0.5), (0.0, 0.0, 0.0), (0.1, 0.2, 0.3)]
)
def test_dynamical_matrix_matches_reference(forceconstants, plan, q):
    reference = np.asarray(_reference(forceconstants, q)._dynmat_fourier)
    matrix = _bloch_eigensystem.dynamical_matrix(plan, np.array(q, dtype=np.float64))
    assert matrix.shape == reference.shape
    np.testing.assert_allclose(matrix, reference, rtol=0, atol=1e-12 * np.abs(reference).max())


@pytest.mark.parametrize("mode", ["auto", "periodic", "wigner-seitz"])
def test_plan_matches_tensorflow_build(forceconstants, mode):
    built = _HarmonicIFCInterpolation.build(forceconstants.second, mode)
    plan = _bloch_eigensystem.harmonic_plan(forceconstants.second, mode)
    assert plan.resolved_mode == built.resolved_mode
    assert plan.include_pair_displacement == built.include_pair_displacement
    np.testing.assert_allclose(plan.values, built.values, rtol=1e-15, atol=0)
    for expected, matrix in zip(built.matrices(_GENERIC_Q), plan.matrices(_GENERIC_Q)):
        np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-12 * np.abs(expected).max())


def test_trim_matrix_is_real(plan):
    matrix = _bloch_eigensystem.dynamical_matrix(plan, np.array([0.5, 0.0, 0.5]))
    assert matrix.dtype == np.float64


def test_derivative_matrix_matches_reference(forceconstants, plan):
    reference = _reference(forceconstants, _GENERIC_Q)
    kernels = (reference._dynmat_derivatives_x, reference._dynmat_derivatives_y, reference._dynmat_derivatives_z)
    for direction, kernel in enumerate(kernels):
        expected = np.asarray(kernel)
        derivative = _bloch_eigensystem.derivative_matrix(plan, np.array(_GENERIC_Q), direction)
        assert derivative.dtype == np.complex128
        np.testing.assert_allclose(derivative, expected, rtol=0, atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize("q", [(0.25, 0.25, 0.25), (0.5, 0.5, 0.0)])
def test_frequencies_match_reference(forceconstants, plan, q):
    eigenvalues, _ = _solve(plan, q)
    frequency = _bloch_eigensystem.frequency_from_eigenvalues(eigenvalues)
    np.testing.assert_allclose(frequency, _reference(forceconstants, q).frequency[0], rtol=1e-10, atol=1e-10)


def test_scipy_fallback_matches_mkl(plan, monkeypatch):
    eigenvalues, eigenvectors = _solve(plan, _GENERIC_Q)
    monkeypatch.setattr(_amorphous_eigensystem._MKL, "load", classmethod(lambda cls: None))
    fallback_values, fallback_vectors = _solve(plan, _GENERIC_Q)
    np.testing.assert_allclose(fallback_values, eigenvalues, rtol=0, atol=1e-10 * np.abs(eigenvalues).max())
    overlap = np.abs(np.einsum("im,im->m", fallback_vectors.conj(), eigenvectors))
    np.testing.assert_allclose(overlap[_nondegenerate(eigenvalues)], 1.0, rtol=0, atol=1e-8)


def _nondegenerate(eigenvalues):
    gap = _DEGENERACY_FRACTION * (eigenvalues.max() - eigenvalues.min())
    spacing = np.diff(eigenvalues)
    isolated = np.ones(eigenvalues.size, dtype=bool)
    isolated[:-1] &= spacing > gap
    isolated[1:] &= spacing > gap
    return isolated


def test_eigenvectors_span_reference(forceconstants, plan):
    eigenvalues, eigenvectors = _solve(plan, _GENERIC_Q)
    reference = np.asarray(_reference(forceconstants, _GENERIC_Q)._eigensystem)[1:]
    np.testing.assert_allclose(np.linalg.norm(eigenvectors, axis=0), 1.0, rtol=0, atol=1e-12)
    overlap = np.abs(np.einsum("im,im->m", reference.conj(), eigenvectors))
    isolated = _nondegenerate(eigenvalues)
    assert isolated.sum() > 0.9 * eigenvalues.size
    np.testing.assert_allclose(overlap[isolated], 1.0, rtol=0, atol=1e-8)


@pytest.mark.parametrize("q", [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5)])
def test_trim_eigenvectors_span_reference(forceconstants, plan, q):
    eigenvalues, eigenvectors = _solve(plan, q)
    assert eigenvectors.dtype == np.float64
    reference = np.asarray(_reference(forceconstants, q)._eigensystem)[1:]
    overlap = np.abs(np.einsum("im,im->m", reference.conj(), eigenvectors))
    isolated = _nondegenerate(eigenvalues)
    assert isolated.sum() > 0.9 * eigenvalues.size
    np.testing.assert_allclose(overlap[isolated], 1.0, rtol=0, atol=1e-8)


def test_velocity_matches_reference(forceconstants, plan):
    eigenvalues, eigenvectors = _solve(plan, _GENERIC_Q)
    frequency = _bloch_eigensystem.frequency_from_eigenvalues(eigenvalues)
    velocity = _bloch_eigensystem.diagonal_velocity(plan, np.array(_GENERIC_Q), eigenvectors, frequency, 1)
    reference = _reference(forceconstants, _GENERIC_Q).velocity[0]
    assert velocity.shape == reference.shape
    np.testing.assert_allclose(velocity, reference, rtol=0, atol=1e-10 * np.abs(reference).max())


def test_nonpositive_frequency_velocity_is_zero(plan):
    eigenvalues, eigenvectors = _solve(plan, (0.0, 0.0, 0.0))
    frequency = _bloch_eigensystem.frequency_from_eigenvalues(eigenvalues)
    frequency[:3] = (-1.0, 0.0, -0.0)
    velocity = _bloch_eigensystem.diagonal_velocity(plan, np.zeros(3), eigenvectors, frequency, 1)
    assert np.all(np.isfinite(velocity))
    np.testing.assert_array_equal(velocity[:3], 0.0)


def test_time_reversed_eigenvectors_are_conjugates(plan):
    eigenvalues, eigenvectors = _solve(plan, _GENERIC_Q)
    reversed_matrix = _bloch_eigensystem.dynamical_matrix(plan, -np.array(_GENERIC_Q))
    # The finite-difference IFCs are Hermitian only to ~1e-4 (rad/ps)^2 out of 8e3; the solver diagonalizes the
    # Hermitian completion of the C-lower triangle, so the time-reversed check applies that same operator.
    hermitian = np.tril(reversed_matrix) + np.tril(reversed_matrix, -1).conj().T
    conjugate = eigenvectors.conj()
    np.testing.assert_allclose(
        hermitian @ conjugate, conjugate * eigenvalues, rtol=0, atol=1e-10 * np.abs(reversed_matrix).max()
    )


@pytest.mark.parametrize("is_classic", [False, True])
def test_population_and_heat_capacity_match_reference(forceconstants, is_classic):
    reference = HarmonicWithQTemp(
        temperature=300, is_classic=is_classic, q_point=np.zeros(3), second=forceconstants.second,
        storage="memory", is_amorphous=False,
    )
    frequency = reference.frequency
    physical = reference.physical_mode
    population = _bloch_eigensystem.population(frequency, physical, 300, is_classic)
    heat_capacity = _bloch_eigensystem.heat_capacity(frequency, physical, 300, is_classic)
    np.testing.assert_allclose(population, reference.population, rtol=1e-14, atol=0)
    np.testing.assert_allclose(heat_capacity, reference.heat_capacity, rtol=1e-14, atol=0)

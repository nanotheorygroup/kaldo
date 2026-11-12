"""Harmonic properties on a half-step shifted q-grid, computed once without TensorFlow.

On a shifted 2x2x2 mesh the points ``(+-1/4)^3`` are complex Bloch points. Time reversal pairs ``q`` with ``-q``
on the same mesh: ``D(-q) = conj(D(q))`` exactly, so only one representative of each pair (the lower id) is
diagonalized. The partner gets the conjugated eigenvectors, the same frequencies and the opposite group velocity.
The Gamma-centred partner mesh holds time-reversal-invariant points, where ``D(q)`` and its eigenvectors are real.
"""

from dataclasses import dataclass, field

import numpy as np

from kaldo.controllers._bloch_eigensystem import (
    dynamical_matrix, diagonal_velocity, eigensystem, frequency_from_eigenvalues, harmonic_plan,
)
from kaldo.grid import QGrid


def _physical_mode(frequency, is_gamma, is_nw, min_frequency, max_frequency):
    """Mirror ``HarmonicWithQ.__init__`` (Gamma acoustic modes) and ``Phonons.physical_mode`` (frequency window)."""
    physical = np.ones(frequency.shape, dtype=bool)
    for ik, gamma in enumerate(is_gamma):
        if gamma:
            physical[ik, :4 if is_nw else 3] = False
    if min_frequency is not None:
        physical[frequency < min_frequency] = False
    if max_frequency is not None:
        physical[frequency > max_frequency] = False
    return physical


@dataclass(frozen=True)
class PartnerHarmonics:
    """Harmonics on the Gamma-centred partner mesh: ``frequency (m, n)``, real ``eigenvectors (m, n, n)``."""
    frequency: np.ndarray
    eigenvectors: np.ndarray
    physical: np.ndarray


@dataclass(frozen=True)
class ShiftedHarmonics:
    """Harmonics at every point of a shifted mesh.

    ``frequency (m, n)`` float64 THz, ``eigenvectors (m, n, n)`` complex128 with modes in columns,
    ``velocity (m, n, 3)`` float64, ``physical (m, n)`` bool, and ``representatives`` int64: the lower id of each
    time-reversal pair. ``eigenvalues (m, n)`` are the ``D(q)`` eigenvalues (rad/ps)^2, the row 0 of
    ``Phonons._eigensystem``.
    """
    frequency: np.ndarray
    eigenvectors: np.ndarray
    velocity: np.ndarray
    physical: np.ndarray
    representatives: np.ndarray
    eigenvalues: np.ndarray
    _partner_builder: object = field(repr=False, compare=False)
    _partner_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def partner(self) -> PartnerHarmonics:
        """Harmonics on the Gamma-centred partner mesh, computed on first call."""
        if "value" not in self._partner_cache:
            self._partner_cache["value"] = self._partner_builder()
        return self._partner_cache["value"]


def _solve(plan, q_point, n_workers):
    matrix = dynamical_matrix(plan, q_point, n_workers)
    w, vectors = eigensystem(matrix, n_workers)
    return np.asarray(w, dtype=np.float64), vectors


def compute_shifted_harmonics(second, grid: QGrid, requested_mode: str, min_frequency, max_frequency, n_workers,
                              is_nw: bool = False) -> ShiftedHarmonics:
    """Solve the representatives of ``grid`` and fill their time-reversal partners by conjugation."""
    plan = harmonic_plan(second, requested_mode)
    points = np.asarray(grid.fractional_points, dtype=np.float64)
    n_points = len(points)
    n_modes = 3 * len(second.atoms)
    partners = np.array([grid.partner_id(k) for k in range(n_points)], dtype=np.int64)
    representatives = np.array([k for k in range(n_points) if k < partners[k]], dtype=np.int64)
    if len(representatives) * 2 != n_points:
        raise ValueError("every point of a shifted grid must pair with a distinct time-reversed point")

    eigenvalues = np.zeros((n_points, n_modes), dtype=np.float64)
    eigenvectors = np.zeros((n_points, n_modes, n_modes), dtype=np.complex128)
    frequency = np.zeros((n_points, n_modes), dtype=np.float64)
    velocity = np.zeros((n_points, n_modes, 3), dtype=np.float64)
    for r in representatives:
        p = partners[r]
        w, vectors = _solve(plan, points[r], n_workers)
        frequency[r] = frequency[p] = frequency_from_eigenvalues(w)
        eigenvalues[r] = eigenvalues[p] = w
        eigenvectors[r] = vectors
        eigenvectors[p] = np.conj(vectors)
        velocity[r] = diagonal_velocity(plan, points[r], eigenvectors[r], frequency[r], n_workers)
        velocity[p] = -velocity[r]
    physical = _physical_mode(frequency, np.zeros(n_points, dtype=bool), is_nw, min_frequency, max_frequency)

    def build_partner():
        partner_points = np.asarray(grid.partner_grid().fractional_points, dtype=np.float64)
        partner_frequency = np.zeros((len(partner_points), n_modes), dtype=np.float64)
        partner_vectors = np.zeros((len(partner_points), n_modes, n_modes), dtype=np.float64)
        for ik, q_point in enumerate(partner_points):
            w, vectors = _solve(plan, q_point, n_workers)
            partner_frequency[ik] = frequency_from_eigenvalues(w)
            partner_vectors[ik] = vectors
        is_gamma = np.all(np.abs(partner_points) < 1e-12, axis=1)
        partner_physical = _physical_mode(partner_frequency, is_gamma, is_nw, min_frequency, max_frequency)
        return PartnerHarmonics(partner_frequency, partner_vectors, partner_physical)

    return ShiftedHarmonics(frequency, eigenvectors, velocity, physical, representatives, eigenvalues,
                            build_partner)

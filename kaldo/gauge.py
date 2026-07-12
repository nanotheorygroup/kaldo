"""Deterministic eigenvector gauge for degenerate subspaces.

Numerical diagonalizers return an arbitrary orthonormal basis within each
degenerate eigenvalue cluster, and the choice differs across BLAS/LAPACK
backends. Every observable built from mode-resolved eigenvector quantities
inside those clusters (velocities, scattering amplitudes, per-mode
bandwidths) inherits that arbitrariness, which is the root cause tracked in
issue #290.

This module removes the freedom: within each degenerate cluster the basis is
rotated to diagonalize a fixed generic Hermitian operator (a graded diagonal
in the Cartesian-atom representation), whose projected spectrum is
non-degenerate for every cluster met in practice, and each vector's phase is
fixed by making its largest-magnitude component real and positive. The result
depends only on the dynamical matrix, not on the diagonalizer, so two
machines produce identical eigenvectors.
"""
import numpy as np


def canonicalize_eigenvectors(eigenvalues, eigenvectors, degeneracy_tol=1e-6):
    """Rotate degenerate clusters into a machine-independent canonical basis.

    Parameters
    ----------
    eigenvalues : ndarray, shape (n,)
        Eigenvalues in ascending order (as returned by ``eigh``).
    eigenvectors : ndarray, shape (n, n)
        Eigenvectors as columns, ``eigenvectors[:, i]`` belonging to
        ``eigenvalues[i]``. Real or complex.
    degeneracy_tol : float
        Two adjacent eigenvalues closer than ``degeneracy_tol`` (relative to
        the spectrum's scale, with an absolute floor) belong to one cluster.

    Returns
    -------
    ndarray
        Eigenvectors with every degenerate cluster rotated to the canonical
        basis and every column's phase fixed. Non-degenerate columns only get
        the phase fix.
    """
    n = len(eigenvalues)
    out = np.array(eigenvectors, copy=True)
    scale = max(np.max(np.abs(eigenvalues)), 1.0)
    tol = degeneracy_tol * scale

    # The gauge-fixing operator: a graded diagonal. Its projection onto any
    # cluster subspace is generically non-degenerate, and it is the same on
    # every machine.
    grading = np.arange(n, dtype=np.float64)

    start = 0
    while start < n:
        end = start + 1
        while end < n and abs(eigenvalues[end] - eigenvalues[end - 1]) < tol:
            end += 1
        if end - start > 1:
            block = out[:, start:end]
            # Project the grading operator into the cluster and diagonalize:
            # projected[i, j] = <v_i| G |v_j> with G diagonal.
            projected = block.conj().T @ (grading[:, np.newaxis] * block)
            projected = 0.5 * (projected + projected.conj().T)
            _, rotation = np.linalg.eigh(projected)
            out[:, start:end] = block @ rotation
        start = end

    # Phase fix per column: largest-|component| entry real and positive.
    # Ties on magnitude are broken by the lowest index (deterministic).
    for i in range(n):
        column = out[:, i]
        j = int(np.argmax(np.abs(column)))
        pivot = column[j]
        if np.abs(pivot) > 0:
            out[:, i] = column * (np.abs(pivot) / pivot)
    return out

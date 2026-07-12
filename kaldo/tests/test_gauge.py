"""
Tests for the deterministic eigenvector gauge (issue #290).

The referee is local gauge scrambling: applying a random unitary rotation
within each degenerate cluster simulates exactly the freedom a different
BLAS backend exercises. Canonicalization must erase it.
"""
import numpy as np

from kaldo.gauge import canonicalize_eigenvectors


def _random_hermitian_with_degeneracies(n, rng, complex_valued=True):
    """Hermitian matrix with an exactly threefold and an exactly twofold
    degenerate eigenvalue, plus non-degenerate rest."""
    eigenvalues = np.sort(rng.standard_normal(n) * 10)
    eigenvalues[1:4] = eigenvalues[1]      # triplet
    eigenvalues[6:8] = eigenvalues[6]      # doublet
    if complex_valued:
        m = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    else:
        m = rng.standard_normal((n, n))
    q, _ = np.linalg.qr(m)
    return (q * eigenvalues) @ q.conj().T, eigenvalues


def _scramble_clusters(eigenvalues, eigenvectors, rng, tol=1e-6):
    """Apply a random unitary within each degenerate cluster and random
    phases everywhere: the gauge freedom a different backend exercises."""
    out = np.array(eigenvectors, copy=True)
    n = len(eigenvalues)
    start = 0
    while start < n:
        end = start + 1
        while end < n and abs(eigenvalues[end] - eigenvalues[end - 1]) < tol:
            end += 1
        size = end - start
        if size > 1:
            m = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
            u, _ = np.linalg.qr(m)
            out[:, start:end] = out[:, start:end] @ u
        start = end
    phases = np.exp(2j * np.pi * rng.random(n))
    return out * phases[np.newaxis, :]


def test_canonical_gauge_erases_cluster_scrambling():
    rng = np.random.default_rng(0)
    h, eigenvalues = _random_hermitian_with_degeneracies(12, rng)
    w, v = np.linalg.eigh(h)

    reference = canonicalize_eigenvectors(w, v)
    scrambled = _scramble_clusters(w, v, rng)
    recovered = canonicalize_eigenvectors(w, scrambled)

    np.testing.assert_allclose(recovered, reference, rtol=0.0, atol=1e-10)


def test_canonical_gauge_preserves_the_eigensystem():
    rng = np.random.default_rng(7)
    h, _ = _random_hermitian_with_degeneracies(12, rng)
    w, v = np.linalg.eigh(h)
    canonical = canonicalize_eigenvectors(w, v)
    # Still an orthonormal eigenbasis of h with the same eigenvalues
    np.testing.assert_allclose(canonical.conj().T @ canonical, np.eye(12), atol=1e-12)
    np.testing.assert_allclose(h @ canonical, canonical * w[np.newaxis, :], atol=1e-9)

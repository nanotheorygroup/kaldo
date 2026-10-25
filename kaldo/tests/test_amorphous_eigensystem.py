"""Real dense Gamma eigensolver for amorphous dynamical matrices.

References are independent NumPy diagonalizations of the dynamical matrix
assembled by the historical Fourier plan, and the historical TensorFlow route
reached by a non-binding ``distance_threshold``. Eigenvalues are in (rad/ps)^2.
"""

from pathlib import Path

import numpy as np
import pytest

from kaldo.controllers import _amorphous_eigensystem
from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.phonons import Phonons

_FIXTURE = Path(__file__).resolve().parent / "data" / "input" / "amorphous-si-tersoff-32"
# Eigenvalues reach 2e4 (rad/ps)^2; LAPACK drivers agree to ~1e-14 relative.
_EIGENVALUE_ATOL = 1e-8
_VECTOR_ATOL = 1e-9
# Far beyond every pair distance of the 10.9 Angstrom cell: no IFC is dropped,
# but the request is routed to the historical TensorFlow solver.
_NON_BINDING_THRESHOLD = 1e3


@pytest.fixture(scope="module")
def forceconstants() -> ForceConstants:
    """Load the 32-atom amorphous Si IFC2 fixture in eV/Angstrom^2."""
    return ForceConstants.from_folder(
        folder=str(_FIXTURE), format="numpy", supercell=(1, 1, 1), is_acoustic_sum=True, only_second=True,
    )


def _harmonic(forceconstants, **options):
    return HarmonicWithQ(
        q_point=np.zeros(3), second=forceconstants.second, storage="memory", is_amorphous=True, **options,
    )


@pytest.fixture(scope="module")
def reference(forceconstants):
    """C-lower symmetric legacy matrix and its NumPy eigenvalues."""
    matrix = np.asarray(_harmonic(forceconstants)._dynmat_fourier)
    symmetric = np.tril(matrix) + np.tril(matrix, -1).T
    return symmetric, np.linalg.eigvalsh(symmetric)


def _assert_eigensystem(eigensystem, symmetric, eigenvalues):
    n = symmetric.shape[0]
    assert eigensystem.shape == (n + 1, n) and eigensystem.dtype == np.float64
    assert eigensystem.flags.c_contiguous
    vectors = eigensystem[1:]
    np.testing.assert_allclose(eigensystem[0], eigenvalues, rtol=0, atol=_EIGENVALUE_ATOL)
    np.testing.assert_allclose(symmetric @ vectors, vectors * eigensystem[0], rtol=0, atol=_VECTOR_ATOL)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(n), rtol=0, atol=1e-12)


def test_assembly_matches_legacy_fourier_matrix(forceconstants, reference):
    legacy = np.asarray(_harmonic(forceconstants)._dynmat_fourier)
    n = legacy.shape[0]
    assembled = np.empty((n, n))
    _amorphous_eigensystem._assemble(forceconstants.second, assembled, 2)
    # Only Wigner--Seitz tie weights (summing to one) separate the two.
    np.testing.assert_allclose(assembled, legacy, rtol=1e-14, atol=1e-11)


@pytest.mark.parametrize("driver", ["evd", "evr"])
@pytest.mark.parametrize("n_workers", [1, 3, None])
def test_drivers_return_column_eigenvectors(forceconstants, reference, monkeypatch, driver, n_workers):
    monkeypatch.setattr(
        _amorphous_eigensystem, "_select_driver", lambda n, vectors, lp64: driver if vectors else "evd",
    )
    eigensystem = _amorphous_eigensystem.gamma_eigensystem(forceconstants.second, n_workers=n_workers)
    _assert_eigensystem(eigensystem, *reference)


def test_eigenvalues_only(forceconstants, reference):
    eigenvalues = _amorphous_eigensystem.gamma_eigenvalues(forceconstants.second, n_workers=2)
    np.testing.assert_allclose(eigenvalues, reference[1], rtol=0, atol=_EIGENVALUE_ATOL)


def test_only_c_lower_triangle_is_read(forceconstants, reference):
    """Finite-difference IFCs are not exactly symmetric; keep TensorFlow's triangle."""
    second = forceconstants.second
    original = second.value.copy()
    try:
        n = 3 * original.shape[1]
        flat = second.value.reshape(n, n)
        upper = np.triu_indices(n, 1)
        flat[upper] += np.random.default_rng(0).normal(scale=1.0, size=upper[0].size)
        eigensystem = _amorphous_eigensystem.gamma_eigensystem(second, n_workers=2)
    finally:
        second.value[...] = original
    _assert_eigensystem(eigensystem, *reference)


@pytest.mark.parametrize("n", [1, 63, 64, 65, 130])
def test_in_place_transpose_of_strided_square(n):
    buffer = np.arange((n + 1) * n, dtype=np.float64).reshape(n + 1, n)
    expected = buffer[1:].T.copy()
    head = buffer[0].copy()
    _amorphous_eigensystem._transpose_square(buffer[1:])
    np.testing.assert_array_equal(buffer[1:], expected)
    np.testing.assert_array_equal(buffer[0], head)


def test_driver_selection_bounds_workspace(monkeypatch):
    class _Memory:
        available = 0

    monkeypatch.setattr(_amorphous_eigensystem, "virtual_memory", lambda: _Memory)
    n = 1000
    _Memory.available = 3 * 8 * n * n
    assert _amorphous_eigensystem._select_driver(n, True, lp64=False) == "evd"
    _Memory.available = 8 * n * n
    assert _amorphous_eigensystem._select_driver(n, True, lp64=False) == "evr"
    assert _amorphous_eigensystem._select_driver(n, False, lp64=False) == "evd"
    # The LP64 dsyevd workspace 1+6n+2n^2 overflows int32 above n=32766.
    _Memory.available = 1 << 60
    assert _amorphous_eigensystem._select_driver(32766, True, lp64=True) == "evd"
    assert _amorphous_eigensystem._select_driver(32767, True, lp64=True) == "evr"
    assert _amorphous_eigensystem._select_driver(32767, True, lp64=False) == "evd"


def test_matches_tensorflow_route(forceconstants, reference):
    legacy = _harmonic(forceconstants, distance_threshold=_NON_BINDING_THRESHOLD)
    assert not legacy._is_real_gamma()
    expected = np.asarray(legacy._eigensystem)
    harmonic = _harmonic(forceconstants)
    assert harmonic._is_real_gamma()
    actual = harmonic._eigensystem
    assert type(actual) is np.ndarray
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=_EIGENVALUE_ATOL)
    np.testing.assert_allclose(harmonic.frequency, legacy.frequency, rtol=0, atol=1e-10)
    # Eigenvectors agree up to sign on the nondegenerate modes.
    gaps = np.diff(actual[0])
    nondegenerate = np.r_[False, (gaps[1:] > 1e-6) & (gaps[:-1] > 1e-6), False]
    overlap = np.abs(np.sum(actual[1:] * expected[1:], axis=0))
    np.testing.assert_allclose(overlap[nondegenerate], 1, rtol=0, atol=1e-10)
    np.testing.assert_allclose(harmonic.participation_ratio, legacy.participation_ratio, rtol=0, atol=1e-10)


@pytest.mark.parametrize("options", [
    dict(distance_threshold=_NON_BINDING_THRESHOLD), dict(is_amorphous=False), dict(q_point=np.full(3, 0.5)),
])
def test_unsupported_requests_keep_legacy_route(forceconstants, options):
    parameters = dict(q_point=np.zeros(3), second=forceconstants.second, storage="memory", is_amorphous=True)
    parameters.update(options)
    assert not HarmonicWithQ(**parameters)._is_real_gamma()


def test_live_eigensystem_is_shared_not_retained(forceconstants):
    import gc
    first = _harmonic(forceconstants)._eigensystem
    second = _harmonic(forceconstants)
    assert second._eigensystem is first
    eigenvalues = second.calculate_eigensystem(only_eigenvals=True)
    np.testing.assert_array_equal(eigenvalues, first[0])
    assert not np.shares_memory(eigenvalues, first)
    del first, second
    gc.collect()
    _, eigensystem_ref = forceconstants.second._gamma_eigensystem_ref
    assert eigensystem_ref() is None


def test_phonons_view_the_gamma_eigensystem(forceconstants):
    phonons = Phonons(forceconstants=forceconstants, kpts=(1, 1, 1), temperature=300.0, storage="memory", n_workers=2)
    eigensystem = phonons._eigensystem
    assert eigensystem.shape == (1, phonons.n_modes + 1, phonons.n_modes)
    harmonic = _harmonic(forceconstants)
    assert np.shares_memory(eigensystem, harmonic._eigensystem)
    np.testing.assert_array_equal(phonons.eigenvalues[0], eigensystem[0, 0])
    np.testing.assert_allclose(
        phonons.frequency[0], np.sign(eigensystem[0, 0]) * np.sqrt(np.abs(eigensystem[0, 0])) / (2 * np.pi),
        rtol=1e-14, atol=0,
    )


@pytest.mark.parametrize("block", [1, 7, 96, 500])
def test_blocked_participation_ratio(forceconstants, block):
    vectors = _amorphous_eigensystem.gamma_eigensystem(forceconstants.second, n_workers=2)[1:]
    n_atoms = vectors.shape[0] // 3
    atomic = np.sum(vectors.T.reshape(-1, n_atoms, 3) ** 2, axis=2)
    expected = 1 / (np.sum(atomic ** 2, axis=1) * n_atoms)
    actual = _amorphous_eigensystem.participation_ratio(vectors, block=block)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)

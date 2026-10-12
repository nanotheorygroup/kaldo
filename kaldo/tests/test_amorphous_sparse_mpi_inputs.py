"""Sparse harmonic, bandwidth, and exact support checks."""

from pathlib import Path

from ase.io import read
import ase.units as units
import numpy as np
import pytest

from kaldo.controllers.bandwidth import DirectBandwidth, TabulatedBandwidth
from kaldo.controllers.qhgk import calculate_qhgk_isotropic_blocked, calculate_qhgk_isotropic_dense
from kaldo.observables.sparse_harmonic import (SparseHarmonic, convert_legacy_dense, convert_legacy_dynform,
                                               from_sparse_force_constants)
from kaldo.parallel.spectrum_slicing import approximate_dos_boundaries, eigenvalue_window_intervals


@pytest.fixture
def small_harmonic():
    rng = np.random.default_rng(99)
    n = 4
    positions = rng.normal(size=(n, 3))
    masses = np.ones(n)
    atom_blocks = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(i + 1, n):
            a = rng.normal(size=(3, 3)) * 0.03
            atom_blocks[i, j] = a
            atom_blocks[j, i] = a.T
    for i in range(n):
        atom_blocks[i, i] = -sum(atom_blocks[i, j] for j in range(n) if j != i)
    # ASR holds but the onsite block need not be symmetric if offsite blocks
    # are arbitrary. Symmetrize each offsite block for this physical fixture.
    for i in range(n):
        for j in range(i + 1, n):
            a = (atom_blocks[i, j] + atom_blocks[i, j].T) / 2
            atom_blocks[i, j] = atom_blocks[j, i] = a
    for i in range(n):
        atom_blocks[i, i] = -sum(atom_blocks[i, j] for j in range(n) if j != i)
    dense = atom_blocks.transpose(0, 2, 1, 3).reshape(3*n, 3*n)
    return dense, positions, masses


def test_sparse_converter_action_validation_and_projection(small_harmonic, tmp_path):
    dense, positions, masses = small_harmonic
    np.save(tmp_path / "d.npy", dense)
    np.save(tmp_path / "positions.npy", positions)
    np.save(tmp_path / "masses.npy", masses)
    displacement = positions[:, None, :] - positions[None, :, :]
    derivative_paths = []
    for axis in range(3):
        derivative = dense * np.repeat(np.repeat(displacement[:, :, axis], 3, axis=0), 3, axis=1)
        path = tmp_path / f"derivative_{axis}.npy"
        np.save(path, derivative)
        derivative_paths.append(path)
    harmonic = convert_legacy_dense(tmp_path / "d.npy", tmp_path / "positions.npy",
                                    tmp_path / "sparse", masses_path=tmp_path / "masses.npy",
                                    derivative_paths=derivative_paths, dtype=np.float64)
    report = harmonic.validation_report(require_acoustic_sum=True)
    assert report["acoustic_sum_residual"] < 1e-13
    assert report["skew_derivative_residual"] < 1e-13
    loaded = SparseHarmonic.load(tmp_path / "sparse")
    np.testing.assert_array_equal(loaded.positions, positions)
    assert loaded.storage_bytes < dense.nbytes * 5
    rng = np.random.default_rng(4)
    panel = rng.normal(size=(12, 3))
    for axis, operator in enumerate(loaded.operators):
        expected = np.load(derivative_paths[axis]) @ panel
        np.testing.assert_allclose(operator @ panel, expected, rtol=1e-13, atol=1e-13)
        projected = panel.T @ (operator @ panel)
        np.testing.assert_allclose(projected, -projected.T, atol=1e-13)
    assert loaded.operators[0].astype(np.float32).harmonic is loaded.operators[1].astype(np.float32).harmonic
    w = np.arange(1, 13, dtype=float) * 0.2
    e = np.linalg.eigh(dense)[1]
    kwargs = dict(omega=w, diffusivity_bandwidth=0.08, physical_mode=np.ones(12, bool),
                  temperature=300., is_classic=False, volume=120., eigenvectors=e,
                  broadening_shape="triangle", compute_dtype=np.float64)
    dense_result = calculate_qhgk_isotropic_dense(
        **kwargs, harmonic_operator=tuple(np.load(path) for path in derivative_paths))
    sparse_result = calculate_qhgk_isotropic_blocked(
        **kwargs, harmonic_operator=loaded.operators, block_size=5)
    for actual, expected in zip(sparse_result, dense_result):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-13)
    mixed = calculate_qhgk_isotropic_blocked(
        **{**kwargs, "compute_dtype": np.float32},
        harmonic_operator=loaded.operators, block_size=5)
    relative_total = abs(mixed[0].sum() - dense_result[0].sum()) / abs(dense_result[0].sum())
    assert relative_total < 1e-6


def test_sparse_validation_rejects_bad_reverse(small_harmonic, tmp_path):
    dense, positions, _ = small_harmonic
    np.save(tmp_path / "d.npy", dense)
    np.save(tmp_path / "p.npy", positions)
    harmonic = convert_legacy_dense(tmp_path / "d.npy", tmp_path / "p.npy", tmp_path / "out",
                                    dtype=np.float64)
    corrupted = harmonic.displacement.copy()
    corrupted[1, 0] += 0.1
    bad = SparseHarmonic(harmonic.row_ptr, harmonic.column_atom,
                                  harmonic.blocks, corrupted)
    with pytest.raises(ValueError, match="validation failed"):
        bad.validation_report()


def test_multiple_periodic_images_share_atom_columns(tmp_path):
    identity = np.eye(3)
    blocks = np.array([np.zeros((3,3)), 2*identity, 3*identity,
                       2*identity, 3*identity, np.zeros((3,3))])
    shifts = np.array([[0.,0,0], [1.,0,0], [2.,0,0],
                       [-1.,0,0], [-2.,0,0], [0.,0,0]])
    harmonic = SparseHarmonic(
        [0,3,6], [0,1,1,0,0,1], blocks, shifts,
        metadata={"periodic_wrapping": "explicit_images"})
    harmonic.save(tmp_path / "images")
    loaded = SparseHarmonic.load(tmp_path / "images")
    assert loaded.validation_report()["skew_derivative_residual"] == 0
    panel = np.arange(12, dtype=float).reshape(6,2)
    expected = np.zeros((6,6))
    expected[:3,3:] = 8*identity
    expected[3:,:3] = -8*identity
    np.testing.assert_allclose(loaded.operators[0] @ panel, expected @ panel)


def test_symmetrize_pairs_periodic_images_by_displacement():
    identity = np.eye(3)
    skew = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]])
    # Atom pair (0, 1) is stored at images +1 and +2 along x; atom 1 sees -1 and -2.
    blocks = np.array([2 * identity, -identity + 0.1 * skew, -2 * identity,
                       -identity, -2 * identity + 0.2 * skew, 2 * identity])
    shifts = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0],
                       [-1., 0, 0], [-2., 0, 0], [0., 0, 0]])
    harmonic = from_sparse_force_constants(
        [0, 3, 6], [0, 1, 1, 0, 0, 1], blocks, shifts, [1., 1.],
        apply_acoustic_sum=False, symmetrize=True,
        force_constant_unit="kaldo_10J_per_mol")
    # Image +1 pairs with -1 (slot 3), image +2 with -2 (slot 4).
    np.testing.assert_allclose(harmonic.blocks[1], -identity + 0.05 * skew)
    np.testing.assert_allclose(harmonic.blocks[3], (-identity + 0.05 * skew).T)
    np.testing.assert_allclose(harmonic.blocks[2], -2 * identity - 0.1 * skew)
    np.testing.assert_allclose(harmonic.blocks[4], (-2 * identity - 0.1 * skew).T)
    assert harmonic.validation_report()["symmetry_residual"] < 1e-15


def test_symmetrize_rejects_missing_periodic_image():
    identity = np.eye(3)
    blocks = np.array([identity, -identity, -identity, -identity, identity])
    shifts = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0], [-1., 0, 0], [0., 0, 0]])
    with pytest.raises(ValueError, match="unmatched reverse"):
        from_sparse_force_constants([0, 3, 5], [0, 1, 1, 0, 1], blocks, shifts, [1., 1.],
                                    apply_acoustic_sum=False, symmetrize=True)


def test_streamed_dynform_and_sparse_acoustic_sum(small_harmonic, tmp_path):
    dense, positions, masses = small_harmonic
    perturbed = dense.copy()
    for atom in range(len(masses)):
        perturbed[3*atom:3*atom+3, 3*atom:3*atom+3] += np.eye(3)*0.3
    np.savetxt(tmp_path / "Dyn.form", perturbed)
    np.save(tmp_path / "positions.npy", positions)
    np.save(tmp_path / "masses.npy", masses)
    harmonic = convert_legacy_dynform(
        tmp_path / "Dyn.form", tmp_path / "positions.npy", tmp_path / "sparse",
        masses_path=tmp_path / "masses.npy", apply_acoustic_sum=True,
        dtype=np.float64)
    assert harmonic.metadata["source_format"] == "legacy-dynform-one-replica"
    assert SparseHarmonic.load(tmp_path / "sparse").validation_report(
        require_acoustic_sum=True)["acoustic_sum_residual"] < 1e-12
    panel = np.random.default_rng(5).normal(size=(len(dense), 2))
    for axis in range(3):
        displacement = np.repeat(np.repeat(
            (positions[:,None,axis]-positions[None,:,axis]), 3, axis=0), 3, axis=1)
        np.testing.assert_allclose(harmonic.operators[axis] @ panel,
                                   (dense*displacement) @ panel, atol=1e-12)


def test_sparse_ifc_mass_weight_and_asr():
    identity = np.eye(3)
    harmonic = from_sparse_force_constants(
        [0,2,4], [0,1,0,1],
        np.array([2*identity, -identity, -identity, 2*identity]),
        np.array([[0.,0,0], [1.,0,0], [-1.,0,0], [0.,0,0]]),
        [1.,4.], force_constant_unit="kaldo_10J_per_mol",
        dtype=np.float64)
    report = harmonic.validation_report(require_acoustic_sum=True)
    assert report["acoustic_sum_residual"] == 0
    np.testing.assert_allclose(harmonic.blocks[0], identity)
    np.testing.assert_allclose(harmonic.blocks[1], -identity/2)
    np.testing.assert_allclose(harmonic.blocks[3], identity/4)


def test_tracked_ifc_sparse_projection_matches_symmetrized_dense():
    root = Path(__file__).parent / "data/input/amorphous-si-tersoff-32"
    atoms = read(root / "amorphous_si_32.extxyz")
    second = np.load(root / "second.npy")[0, :, :, 0, :, :]
    n = len(atoms)
    positions = atoms.positions
    masses = atoms.get_masses()
    displacement = (positions[:, None, :] - positions[None, :, :]).reshape(-1, 3)
    force = second.transpose(0, 2, 1, 3).reshape(n*n, 3, 3)
    harmonic = from_sparse_force_constants(
        np.arange(0, n*n+1, n), np.tile(np.arange(n), n), force,
        displacement, masses, positions=positions,
        apply_acoustic_sum=False, symmetrize=True, dtype=np.float64)

    mass_scale = np.repeat(masses, 3)
    dense = second.reshape(3*n, 3*n) * (units.mol/(10*units.J))
    dense /= np.sqrt(mass_scale[:, None]*mass_scale[None, :])
    dense = (dense+dense.T)/2
    np.testing.assert_allclose(
        harmonic.blocks.reshape(n, n, 3, 3).transpose(0, 2, 1, 3).reshape(3*n, 3*n),
        dense, rtol=1e-13, atol=1e-13)
    assert harmonic.validation_report()["skew_derivative_residual"] < 1e-12

    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    panel = eigenvectors[:, 8:23]
    coordinates = np.repeat(positions, 3, axis=0)
    dense_operators = []
    for axis, operator in enumerate(harmonic.operators):
        reference = dense * (coordinates[:, None, axis]-coordinates[None, :, axis])
        dense_operators.append(reference)
        np.testing.assert_allclose(operator @ panel, reference @ panel,
                                   rtol=2e-13, atol=1e-12)
        np.testing.assert_allclose(panel.T @ (operator @ panel),
                                   panel.T @ reference @ panel,
                                   rtol=2e-13, atol=1e-11)
    omega = np.sqrt(np.maximum(eigenvalues, 0))
    physical = eigenvalues > 1e-8
    inputs = dict(omega=omega, diffusivity_bandwidth=0.05,
                  physical_mode=physical, temperature=300., is_classic=False,
                  volume=1000., eigenvectors=eigenvectors,
                  broadening_shape="triangle", compute_dtype=np.float64)
    reference = calculate_qhgk_isotropic_dense(
        **inputs, harmonic_operator=tuple(dense_operators))
    result = calculate_qhgk_isotropic_blocked(
        **inputs, harmonic_operator=harmonic.operators, block_size=17)
    for actual, expected in zip(result, reference):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
    # At this narrow width, the only supported physical pairs have zero
    # projected flux. Float32 roundoff must not turn self-pairs into transport.
    narrow_mixed = calculate_qhgk_isotropic_blocked(
        **{**inputs, "compute_dtype": np.float32},
        harmonic_operator=harmonic.operators, block_size=17)
    assert all(np.all(vector == 0) for vector in (*reference, *narrow_mixed))

    broad_inputs = {**inputs, "diffusivity_bandwidth": 1.0}
    broad_reference = calculate_qhgk_isotropic_blocked(
        **broad_inputs, harmonic_operator=harmonic.operators, block_size=17)[0]
    broad_mixed = calculate_qhgk_isotropic_blocked(
        **{**broad_inputs, "compute_dtype": np.float32},
        harmonic_operator=harmonic.operators, block_size=17)[0]
    assert abs(broad_mixed.sum()-broad_reference.sum())/broad_reference.sum() < 1e-6


def test_bandwidth_providers():
    omega = np.array([1., 2., 3., 4.])
    np.testing.assert_allclose(DirectBandwidth([1., 2., 4., 8.], unit="lifetime_ps").evaluate(omega, 300),
                               [0.5, 0.25, 0.125, 0.0625])
    table = TabulatedBandwidth([1., 3., 5.], [200., 400.],
                               [[1., 3., 5.], [3., 5., 7.]], bounds="error")
    np.testing.assert_allclose(table.evaluate(omega, 300), [2., 3., 4., 5.])
    assert table.metadata["source_sha256"]
    with pytest.raises(ValueError, match="outside table"):
        table.evaluate([0.], 300)


def test_exact_triangle_block_skip_and_bounds():
    rng = np.random.default_rng(11)
    m = 15
    omega = np.array([0., 1., 1.000001, 1.2, 1.5, 2., 3., 4., 5.,
                      6., 7., 8., 9., 10., 11.])
    a = rng.normal(size=(m, m))
    operator = a - a.T
    eigenvectors = np.linalg.qr(rng.normal(size=(m, m)))[0]
    bandwidth = np.array([0., 0.01, 0.05, 0.03] + [0.02] * 11)
    physical = np.ones(m, bool)
    physical[0] = False
    common = dict(omega=omega, diffusivity_bandwidth=bandwidth,
                  physical_mode=physical, temperature=300., is_classic=True,
                  volume=100., eigenvectors=eigenvectors,
                  harmonic_operator=(operator, operator, operator),
                  broadening_shape="triangle", compute_dtype=np.float64)
    expected = calculate_qhgk_isotropic_dense(**common)
    for block in (1, 3, 7, m):
        stats = {}
        actual = calculate_qhgk_isotropic_blocked(**common, block_size=block, stats=stats)
        for got, wanted in zip(actual, expected):
            np.testing.assert_allclose(got, wanted, rtol=1e-13, atol=1e-13)
        assert stats["evaluated_pairs"] < m*m
        assert stats["skipped_blocks"] > 0


def test_approximate_dos_boundaries_span_histogram():
    edges = approximate_dos_boundaries([0., 1., 2., 3.], [1., 3., 1.], 2)
    assert edges.shape == (3,) and edges[0] == 0 and edges[-1] == 3
    assert edges[1] == pytest.approx(1.5)


def test_zero_bandwidth_interval_halos_cover_endpoint_modes():
    left, right = eigenvalue_window_intervals([0., 1., 4.], 0.)
    assert left[2] < left[0] and left[3] > left[1]
    assert right[2] < right[0] and right[3] > right[1]
    assert left[3] > 1. and right[2] < 1.

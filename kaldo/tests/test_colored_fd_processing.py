"""Sparse IFC symmetry, acoustic sum rules, and dynamical-matrix accessors."""

import logging

import ase.units as units
import numpy as np
import pytest
from scipy.linalg import null_space
from scipy.sparse import bsr_array

from kaldo.controllers.colored_fd import (merge_colored_second_order, plan_colored_second_order,
                                          run_colored_second_order)
from kaldo.controllers.colored_fd.processing import process_sparse_ifc
from kaldo.observables.secondorder import acoustic_sum_rule
from kaldo.observables.sparse_secondorder import SparseSecondOrder
from kaldo.tests.colored_fd_helpers import ToyFiniteRange, chain as _chain


def _random_path(n_atoms, seed=4):
    rng = np.random.default_rng(seed)
    columns = []
    indptr = [0]
    for row in range(n_atoms):
        columns.extend(range(max(0, row - 1), min(n_atoms, row + 2)))
        indptr.append(len(columns))
    data = rng.normal(size=(len(columns), 3, 3))
    return bsr_array((data, np.asarray(columns, dtype=np.int32),
                      np.asarray(indptr, dtype=np.int32)), shape=(3 * n_atoms, 3 * n_atoms))


def _small_dense_projection(matrix):
    """Independent null-space projection with explicit scalar constraints."""
    n_modes = matrix.shape[0]
    n_atoms = n_modes // 3
    graph = (matrix.toarray().reshape(n_atoms, 3, n_atoms, 3) != 0).any(axis=(1, 3))
    equations = []
    for i in range(n_modes):
        for j in range(n_modes):
            if not graph[i // 3, j // 3]:
                equation = np.zeros(n_modes**2)
                equation[i * n_modes + j] = 1
                equations.append(equation)
            elif i < j:
                equation = np.zeros(n_modes**2)
                equation[i * n_modes + j] = 1
                equation[j * n_modes + i] = -1
                equations.append(equation)
    for atom in range(n_atoms):
        for beta in range(3):
            for alpha in range(3):
                equation = np.zeros(n_modes**2)
                for neighbor in range(n_atoms):
                    equation[(3 * atom + beta) * n_modes + 3 * neighbor + alpha] = 1
                equations.append(equation)
    basis = null_space(np.asarray(equations))
    flattened = matrix.toarray().ravel()
    return (basis @ (basis.T @ flattened)).reshape(matrix.shape)


def test_projection_matches_independent_minimum_change_solution():
    raw = _random_path(3)
    expected = _small_dense_projection(raw)
    processed = raw.copy()
    report = process_sparse_ifc(processed)
    np.testing.assert_allclose(processed.toarray(), expected, rtol=0, atol=2e-10)
    assert report["after"]["relative_asymmetry"] < 1e-12
    assert report["after"]["relative_row_sum"] < 1e-12
    np.testing.assert_array_equal(processed.indices, raw.indices)
    np.testing.assert_array_equal(processed.indptr, raw.indptr)


def test_match_dense_reproduces_legacy_onsite_correction():
    raw = _random_path(4)
    expected = acoustic_sum_rule(SparseSecondOrder(raw.copy()).to_dense())
    processed = raw.copy()
    report = process_sparse_ifc(processed, asr_mode="match_dense", symmetrize=False)
    np.testing.assert_allclose(SparseSecondOrder(processed).to_dense(), expected, rtol=0, atol=2e-15)
    assert report["after"]["relative_column_sum"] < 1e-14


def test_projection_supports_disconnected_and_isolated_atoms():
    raw = _random_path(3)
    data = np.concatenate((raw.data[:4], np.eye(3)[None]))
    columns = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    indptr = np.array([0, 2, 4, 5], dtype=np.int32)
    disconnected = bsr_array((data, columns, indptr), shape=(9, 9))
    process_sparse_ifc(disconnected)
    np.testing.assert_allclose(disconnected.toarray(), disconnected.toarray().T, atol=1e-12)
    np.testing.assert_allclose(disconnected.toarray().reshape(3, 3, 3, 3).sum(axis=2), 0, atol=1e-12)


def test_projection_reports_nonconvergence():
    with pytest.raises(RuntimeError, match="failed after"):
        process_sparse_ifc(_random_path(30), asr_max_iter=1)


def test_mass_weighted_bsr_and_operator_match_dense_convention():
    matrix = _random_path(4)
    sparse = SparseSecondOrder(matrix)
    masses = np.array([12.0, 16.0, 28.0, 40.0])
    scale = np.repeat(1 / np.sqrt(masses), 3)
    expected = units.mol / (10 * units.J) * scale[:, None] * matrix.toarray() * scale[None, :]
    weighted = sparse.mass_weighted_bsr(masses)
    operator = sparse.mass_weighted_operator(masses)
    np.testing.assert_allclose(weighted.toarray(), expected)
    vector = np.arange(12, dtype=np.float64)
    panel = np.arange(36, dtype=np.float64).reshape(12, 3)
    np.testing.assert_allclose(operator @ vector, expected @ vector)
    np.testing.assert_allclose(operator @ panel, expected @ panel)
    np.testing.assert_allclose(operator.rmatvec(vector), expected.T @ vector)
    np.testing.assert_array_equal(sparse.matrix.data, matrix.data)
    with pytest.raises(ValueError, match="masses"):
        sparse.mass_weighted_operator([1, 2])


def test_merge_logs_default_processing_and_invalidates_cache(tmp_path, caplog):
    atoms = _chain(9)
    plan = plan_colored_second_order(atoms, 2.5, 2.5, plan_dir=tmp_path / "plan",
                                     calculator_description="toy-processing")
    run_colored_second_order(plan.directory, ToyFiniteRange(many_body=True),
                             workdir=tmp_path / "shard", calculator_description="toy-processing")
    with caplog.at_level(logging.INFO, logger="kaldo"):
        processed = merge_colored_second_order(plan.directory, [tmp_path / "shard"],
                                               output_dir=tmp_path / "merged")
        merge_colored_second_order(plan.directory, [tmp_path / "shard"], output_dir=tmp_path / "merged")
    assert "asr_mode=projection" in caplog.text
    assert "cached=True" in caplog.text
    assert "symmetric_iterations=" in caplog.text
    assert "after residuals" in caplog.text
    assert processed.metadata["processing"] == "projection"
    raw = merge_colored_second_order(plan.directory, [tmp_path / "shard"], output_dir=tmp_path / "merged",
                                     symmetrize=False, apply_acoustic_sum=False)
    assert raw.metadata["processing"] == "raw"
    assert raw.metadata["processing_sha256"] != processed.metadata["processing_sha256"]

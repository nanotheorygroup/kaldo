"""Fixed-pattern symmetry and translational sum-rule operations on 3x3 BSR IFCs."""

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg

from kaldo.helpers.sparse_blocks import (PAIR_BATCH as _PAIR_BATCH, block_column_sums, block_row_sums,
                                         symmetrize_blocks)


ALGORITHM_VERSION = "colored-ifc-projection-v1"


def _pattern(matrix):
    """Return the atom graph and reverse slot map without transposing 3x3 data."""
    n_atoms = matrix.shape[0] // 3
    nnz = len(matrix.indices)
    if nnz > np.iinfo(np.int32).max:
        raise ValueError("BSR block count exceeds the int32 sparse-index limit")
    graph = csr_array((np.ones(nnz, dtype=np.bool_), matrix.indices, matrix.indptr),
                      shape=(n_atoms, n_atoms))
    if not graph.has_canonical_format or np.any(np.diff(graph.indptr) == 0):
        raise ValueError("IFC block graph must have sorted, unique, nonempty rows")
    indexed = csr_array((np.arange(nnz, dtype=np.int32), matrix.indices, matrix.indptr),
                        shape=graph.shape)
    reversed_graph = indexed.T.tocsr()
    if (not np.array_equal(reversed_graph.indptr, matrix.indptr) or
            not np.array_equal(reversed_graph.indices, matrix.indices)):
        raise ValueError("IFC block graph must have a symmetric fixed pattern")
    reverse = reversed_graph.data
    if not np.array_equal(reverse[reverse], np.arange(nnz)):
        raise ValueError("IFC reverse block map is inconsistent")
    rows = np.repeat(np.arange(n_atoms, dtype=np.int32), np.diff(matrix.indptr))
    onsite = np.flatnonzero(rows == matrix.indices)
    if len(onsite) != n_atoms:
        raise ValueError("IFC block graph requires one onsite block per atom")
    return graph, rows, reverse, onsite


def _asymmetry(data, reverse):
    denominator = float(np.linalg.norm(data))
    if denominator == 0:
        return 0.0
    squared = 0.0
    for start in range(0, len(reverse), _PAIR_BATCH):
        slots = np.arange(start, min(start + _PAIR_BATCH, len(reverse)))
        diff = data[slots] - data[reverse[slots]].transpose(0, 2, 1)
        squared += float(np.einsum("ijk,ijk->", diff, diff))
    return float(np.sqrt(squared) / denominator)


def _residuals(matrix, reverse):
    data = matrix.data
    denominator = max(float(np.linalg.norm(data)), np.finfo(float).tiny)
    row = block_row_sums(data, matrix.indptr)
    col = block_column_sums(data, matrix.indices, matrix.shape[0] // 3)
    return {"relative_asymmetry": _asymmetry(data, reverse),
            "relative_row_sum": float(np.linalg.norm(row) / denominator),
            "relative_column_sum": float(np.linalg.norm(col) / denominator)}


def _cg_solve(operator, preconditioner, rhs, *, tolerance, max_iter, label):
    if not np.any(rhs):
        return np.zeros_like(rhs), 0
    iterations = [0]

    def count(_):
        iterations[0] += 1

    solution, info = cg(operator, rhs, M=preconditioner, rtol=tolerance / 10,
                        atol=0.0, maxiter=max_iter, callback=count)
    if info != 0:
        raise RuntimeError(f"sparse ASR {label} solve failed after {iterations[0]} iterations (info={info})")
    return solution, iterations[0]


def _project(matrix, graph, rows, reverse, *, tolerance, max_iter):
    """Orthogonally project onto symmetric, zero-row-sum IFCs on this graph.

    The dual equations decouple into six signless-graph systems for symmetric
    Cartesian entries and three graph-Laplacian systems for skew entries.
    A root in each component fixes the Laplacian's constant gauge.
    """
    n_atoms = graph.shape[0]
    degree = np.diff(graph.indptr).astype(np.float64)
    _, component = connected_components(graph, directed=False)
    roots = np.zeros(n_atoms, dtype=np.float64)
    roots[np.unique(component, return_index=True)[1]] = 1.0

    positive = LinearOperator((n_atoms, n_atoms),
                              matvec=lambda vector: 0.5 * (degree * vector + graph @ vector),
                              dtype=np.float64)
    negative = LinearOperator((n_atoms, n_atoms),
                              matvec=lambda vector: 0.5 * (degree * vector - graph @ vector) + roots * vector,
                              dtype=np.float64)
    positive_diag = 0.5 * (degree + 1)
    negative_diag = 0.5 * (degree - 1) + roots
    positive_preconditioner = LinearOperator((n_atoms, n_atoms),
                                             matvec=lambda vector: vector / positive_diag,
                                             dtype=np.float64)
    negative_preconditioner = LinearOperator((n_atoms, n_atoms),
                                             matvec=lambda vector: vector / negative_diag,
                                             dtype=np.float64)
    residual = block_row_sums(matrix.data, matrix.indptr)
    potential = np.zeros((n_atoms, 3, 3), dtype=np.float64)
    counts = {"symmetric": [], "skew": []}
    for beta in range(3):
        for alpha in range(beta, 3):
            rhs = 0.5 * (residual[:, beta, alpha] + residual[:, alpha, beta])
            solution, iterations = _cg_solve(positive, positive_preconditioner, rhs,
                                             tolerance=tolerance, max_iter=max_iter,
                                             label=f"symmetric ({beta},{alpha})")
            potential[:, beta, alpha] = solution
            potential[:, alpha, beta] = solution
            counts["symmetric"].append(iterations)
    for beta in range(3):
        for alpha in range(beta + 1, 3):
            rhs = 0.5 * (residual[:, beta, alpha] - residual[:, alpha, beta])
            component_sums = np.bincount(component, weights=rhs)
            component_sizes = np.bincount(component)
            rhs -= component_sums[component] / component_sizes[component]
            solution, iterations = _cg_solve(negative, negative_preconditioner, rhs,
                                             tolerance=tolerance, max_iter=max_iter,
                                             label=f"skew ({beta},{alpha})")
            potential[:, beta, alpha] += solution
            potential[:, alpha, beta] -= solution
            counts["skew"].append(iterations)

    data = matrix.data
    for start in range(0, len(reverse), _PAIR_BATCH):
        slots = np.arange(start, min(start + _PAIR_BATCH, len(reverse)))
        first = slots[slots <= reverse[slots]]
        second = reverse[first]
        correction = 0.5 * (potential[rows[first]] +
                            potential[matrix.indices[first]].transpose(0, 2, 1))
        updated = data[first] - correction
        data[first] = updated
        data[second] = updated.transpose(0, 2, 1)
    return counts


def process_sparse_ifc(matrix, *, apply_acoustic_sum=True, asr_mode="projection",
                       symmetrize=True, asr_tolerance=1e-10, asr_max_iter=1000):
    """Process a canonical 3x3 BSR IFC without changing its block pattern."""
    if asr_mode not in ("projection", "match_dense"):
        raise ValueError("asr_mode must be 'projection' or 'match_dense'")
    if apply_acoustic_sum and asr_mode == "projection" and not symmetrize:
        raise ValueError("projection ASR requires symmetrize=True")
    if not np.isfinite(asr_tolerance) or not 0 < asr_tolerance < 1:
        raise ValueError("asr_tolerance must lie between zero and one")
    if not isinstance(asr_max_iter, int) or asr_max_iter < 1:
        raise ValueError("asr_max_iter must be a positive integer")
    graph, rows, reverse, onsite = _pattern(matrix)
    before = _residuals(matrix, reverse)
    if symmetrize:
        symmetrize_blocks(matrix.data, reverse)
    counts = {"symmetric": [], "skew": []}
    solver_status = "not_run"
    if apply_acoustic_sum:
        if asr_mode == "projection":
            counts = _project(matrix, graph, rows, reverse, tolerance=asr_tolerance,
                              max_iter=asr_max_iter)
            solver_status = "converged"
        else:
            sums = block_column_sums(matrix.data, matrix.indices, graph.shape[0])
            matrix.data[onsite] -= sums
            solver_status = "direct"
    after = _residuals(matrix, reverse)
    if apply_acoustic_sum and asr_mode == "projection":
        if max(after["relative_asymmetry"], after["relative_row_sum"],
               after["relative_column_sum"]) > asr_tolerance:
            raise RuntimeError(f"sparse projection ASR residual exceeds {asr_tolerance:g}: {after}")
    return {"before": before, "after": after, "solver_iterations": counts,
            "solver_status": solver_status,
            "algorithm_version": ALGORITHM_VERSION}

"""Fast checks for colored IFC2 graph construction and range planning."""

import logging

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from kaldo.controllers.colored_fd import (build_neighbor_structure, color_structure,
                                          estimate_interaction_range, greedy_cpr_groups,
                                          validate_coloring)
from kaldo.controllers.colored_fd import estimate as estimate_module
from kaldo.controllers.colored_fd import structure as structure_module
from kaldo.tests.colored_fd_helpers import ToyFiniteRange, chain as _chain


def test_structure_matches_minimum_image_on_skew_cell():
    atoms = Atoms("Si4", positions=[[0.1, 0.1, 0], [1.1, 0.1, 0],
                                     [0.2, 1.1, 0], [1.0, 1.1, 0]],
                  cell=[[4, 0, 0], [1, 4, 0], [0, 0, 8]], pbc=True)
    structure = build_neighbor_structure(atoms, 1.2)
    brute = np.array([[atoms.get_distance(i, j, mic=True) < 1.2 for j in range(len(atoms))]
                      for i in range(len(atoms))])
    np.testing.assert_array_equal(structure.matrix.toarray(), brute)
    assert structure.matrix.indices.dtype == np.int32


def test_structure_accepts_partial_periodicity_with_missing_cell_vector():
    atoms = Atoms("Si2", positions=[[0.1, 0, 0], [3.9, 0, 0]],
                  cell=[[4, 0, 0], [0, 4, 0], [0, 0, 0]], pbc=[True, True, False])
    structure = build_neighbor_structure(atoms, 0.5)
    assert structure.matrix.toarray().all()


def test_coloring_is_deterministic_orthogonal_and_compressed():
    coordinates = np.indices((6, 6, 6)).reshape(3, -1).T.astype(float)
    coordinates += np.random.default_rng(17).normal(scale=0.03, size=coordinates.shape)
    atoms = Atoms("Si" * len(coordinates), positions=coordinates, cell=[6, 6, 6], pbc=True)
    structure = build_neighbor_structure(atoms, 1.15)
    first = color_structure(structure, seeds=(0, 5, 11))
    second = color_structure(structure, seeds=(0, 5, 11))
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.n_groups < len(atoms) / 4
    assert validate_coloring(structure, first.labels)
    fallback = color_structure(structure, seeds=(0, 5, 11), use_scipy=False)
    assert validate_coloring(structure, fallback.labels)
    with pytest.raises(ValueError, match="orthogonality"):
        validate_coloring(structure, np.zeros(len(atoms), dtype=np.int32))
    with pytest.raises(ValueError, match="order"):
        greedy_cpr_groups(structure, np.zeros(len(atoms), dtype=np.int32))


@pytest.mark.parametrize("many_body,expected_radius", [(False, 1.5), (True, 2.5)])
def test_estimator_recovers_pair_and_many_body_range(tmp_path, many_body, expected_radius):
    atoms = _chain()
    calculator = ToyFiniteRange(many_body=many_body)
    checkpoint = tmp_path / "probes"
    output = tmp_path / "profile.json"
    result = estimate_interaction_range(atoms, calculator, n_probe=len(atoms),
                                        reference_radius=1.5, bin_width=0.5,
                                        tolerances=(1e-5,), checkpoint_dir=checkpoint,
                                        json_path=output, cutoff_hint=1.3,
                                        cutoff_multiplier=2 if many_body else 1)
    recommendation = result.recommendations[0]
    assert recommendation.radius == pytest.approx(expected_radius)
    assert result.structural_upper_bound == pytest.approx(2.6 if many_body else 1.3)
    assert recommendation.n_groups == color_structure(
        build_neighbor_structure(atoms, recommendation.radius)).n_groups
    assert recommendation.force_calls == 6 * recommendation.n_groups
    assert recommendation.expected_contamination < 1e-5
    assert output.exists()
    calls = calculator.calls
    estimate_interaction_range(atoms, calculator, n_probe=len(atoms), reference_radius=1.5,
                               bin_width=0.5, tolerances=(1e-5,), checkpoint_dir=checkpoint)
    assert calculator.calls == calls
    with pytest.raises(ValueError, match="incompatible"):
        estimate_interaction_range(atoms, calculator, n_probe=len(atoms), delta_shift=0.002,
                                   checkpoint_dir=checkpoint)


def test_estimator_warns_when_tail_does_not_decay():
    atoms = _chain(3)
    with pytest.warns(RuntimeWarning, match="no decay"):
        result = estimate_interaction_range(atoms, ToyFiniteRange(many_body=True),
                                            n_probe=3, reference_radius=1.5,
                                            bin_width=0.5, tolerances=(1e-8,))
    assert result.recommendations[0].radius is None


def test_estimator_accepts_calculator_factory():
    estimate = estimate_interaction_range(_chain(5), ToyFiniteRange, n_probe=3,
                                          reference_radius=1.5, tolerances=(1e-4,))
    assert estimate.recommendations[0].radius == pytest.approx(1.5)


def test_estimator_rejects_zero_norm_single_atom():
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[8, 8, 8], pbc=False)
    with pytest.raises(ValueError, match="zero"):
        estimate_interaction_range(atoms, ToyFiniteRange, n_probe=1)


def _random_atoms(n_atoms, cell, pbc, seed, *, extent=None):
    rng = np.random.default_rng(seed)
    cell = np.asarray(cell, dtype=float)
    if extent is None:
        positions = rng.uniform(-0.03, 1.03, size=(n_atoms, 3)) @ cell
    else:
        positions = rng.uniform(0, 1, size=(n_atoms, 3)) * extent
    return Atoms("Si" * n_atoms, positions=positions, cell=cell, pbc=pbc)


def _first_fit_reference(matrix, order):
    dense = matrix.toarray()
    labels = np.full(dense.shape[1], -1)
    for atom in order:
        used = {labels[column] for row in np.flatnonzero(dense[:, atom])
                for column in np.flatnonzero(dense[row]) if labels[column] >= 0}
        label = 0
        while label in used:
            label += 1
        labels[atom] = label
    return labels


@pytest.mark.parametrize("atoms,radius,n_chunks", [
    (_random_atoms(300, [[12, 0, 0], [2, 12, 0], [1, 1.5, 12]], True, 1), 3.0, 3),
    (_random_atoms(250, [[8, 0, 0], [0, 8, 0], [0, 0, 0]], [True, True, False], 2, extent=[8, 8, 25]), 2.5, 4),
    (_random_atoms(250, [[20, 0, 0], [0, 0, 0], [0, 0, 9]], [True, False, True], 3, extent=[20, 6, 9]), 2.5, 3),
    (_random_atoms(250, np.zeros((3, 3)), False, 4, extent=[14, 14, 14]), 3.0, 3),
], ids=["triclinic-periodic", "slab-nonperiodic-axis", "slab-periodic-axis", "cluster"])
def test_slab_chunked_structure_matches_single_search(atoms, radius, n_chunks):
    single = build_neighbor_structure(atoms, radius, n_chunks=1)
    chunked = build_neighbor_structure(atoms, radius, n_chunks=n_chunks)
    assert chunked.n_chunks == n_chunks
    np.testing.assert_array_equal(chunked.matrix.toarray(), single.matrix.toarray())
    np.testing.assert_allclose(chunked.distances, single.distances, rtol=1e-12, atol=1e-12)


def test_memory_limit_selects_slabs_or_refuses():
    atoms = _random_atoms(800, np.eye(3) * 20, True, 5)
    single = build_neighbor_structure(atoms, 2.5, pair_memory_limit=None)
    chunked = build_neighbor_structure(atoms, 2.5, pair_memory_limit=0.7 * single.projected_bytes)
    assert chunked.n_chunks > 1
    np.testing.assert_array_equal(chunked.matrix.toarray(), single.matrix.toarray())
    with pytest.raises(MemoryError, match="pair_memory_limit"):
        build_neighbor_structure(atoms, 2.5, pair_memory_limit=0.1 * single.projected_bytes)


def test_restricted_structure_matches_direct_search():
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=3)
    restricted = build_neighbor_structure(atoms, 5.0).restrict(3.0)
    direct = build_neighbor_structure(atoms, 3.0)
    np.testing.assert_array_equal(restricted.matrix.toarray(), direct.matrix.toarray())
    np.testing.assert_allclose(restricted.distances, direct.distances, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorized_greedy_matches_first_fit_definition(seed):
    atoms = _random_atoms(120, np.eye(3) * 9, True, seed)
    structure = build_neighbor_structure(atoms, 2.2)
    order = np.random.default_rng(seed).permutation(len(atoms))
    labels = greedy_cpr_groups(structure, order, batch_size=7)
    np.testing.assert_array_equal(labels, _first_fit_reference(structure.matrix, order))


def test_scipy_grouping_is_used_without_fallback(caplog):
    assert structure_module._scipy_group_columns is not None
    atoms = _random_atoms(200, np.eye(3) * 10, True, 6)
    with caplog.at_level(logging.WARNING, logger="kaldo"):
        coloring = color_structure(build_neighbor_structure(atoms, 2.5))
    assert "fallback" not in caplog.text
    assert coloring.n_groups < len(atoms)


def test_estimator_runs_one_neighbor_search_for_all_tolerances(monkeypatch):
    atoms = _chain()
    searched = []
    original = estimate_module.build_neighbor_structure

    def recording_build(atoms, radius, **kwargs):
        searched.append(radius)
        return original(atoms, radius, **kwargs)

    monkeypatch.setattr(estimate_module, "build_neighbor_structure", recording_build)
    result = estimate_interaction_range(atoms, ToyFiniteRange(many_body=True), n_probe=len(atoms),
                                        reference_radius=1.5, tolerances=(1e-1, 1e-3, 1e-5))
    radii = [recommendation.radius for recommendation in result.recommendations]
    assert searched == [max(radii)]
    for recommendation in result.recommendations:
        assert recommendation.n_groups == color_structure(
            original(atoms, recommendation.radius)).n_groups

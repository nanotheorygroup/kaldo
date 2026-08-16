"""Focused tests for the separated reciprocal and IFC translation grids."""

import numpy as np
import pytest
from ase import Atoms

from kaldo.grid import QGrid, SupercellGrid, TranslationSupport, WignerSeitzImages
from kaldo.observables.forceconstant import ForceConstant


def test_q_grid_order_and_exact_time_reversal_partner():
    grid = QGrid((2, 3, 4), order="C")

    np.testing.assert_array_equal(grid.addresses[1], [0, 0, 1])
    np.testing.assert_array_equal(grid.addresses[-1], [1, 2, 3])
    for point_id, address in enumerate(grid.addresses):
        partner = grid.partner_id(point_id)
        np.testing.assert_array_equal(
            (address + grid.addresses[partner]) % grid.shape,
            np.zeros(3, dtype=int),
        )
    np.testing.assert_array_equal(
        grid.momentum_partner_ids(5, is_plus=True),
        [grid.address_to_id(grid.addresses[5] + address) for address in grid.addresses],
    )


def test_q_grid_matches_legacy_enumeration_and_momentum_convention():
    """Keep historical point ids while replacing float momentum arithmetic."""
    for shape in ((1, 1, 1), (2, 3, 4), (3, 2, 5)):
        for order in ("C", "F"):
            grid = QGrid(shape, order=order)
            ids = np.arange(np.prod(shape))
            legacy_addresses = np.asarray(
                np.unravel_index(ids, shape, order=order)
            ).T
            np.testing.assert_array_equal(grid.addresses, legacy_addresses)
            np.testing.assert_allclose(
                grid.fractional_points,
                legacy_addresses / np.asarray(shape, dtype=float),
                rtol=0.0,
                atol=0.0,
            )

            for point_id in ids:
                q = legacy_addresses[point_id]
                for is_plus, sign in ((False, -1), (True, 1)):
                    expected_addresses = np.mod(q + sign * legacy_addresses, shape)
                    expected_ids = np.ravel_multi_index(
                        expected_addresses.T, shape, order=order
                    )
                    np.testing.assert_array_equal(
                        grid.momentum_partner_ids(point_id, is_plus),
                        expected_ids,
                    )


def test_supercell_diagonal_representatives_preserve_c_and_f_order():
    matrix = np.diag([2, 3, 2])

    for order in ("C", "F"):
        grid = SupercellGrid(matrix, order=order)
        expected = np.asarray(
            np.unravel_index(np.arange(12), (2, 3, 2), order=order),
        ).T
        np.testing.assert_array_equal(grid.representatives, expected)


def test_nondiagonal_supercell_quotient_is_exact_and_translation_invariant():
    matrix = np.array([[2, 1, 0], [0, 2, 0], [1, 0, 1]])
    grid = SupercellGrid(matrix)

    assert grid.size == abs(round(np.linalg.det(matrix)))
    assert len({grid.class_key(r) for r in grid.representatives}) == grid.size
    probe = np.array([17, -23, 41])
    for shift in ([0, 0, 0], [101, -77, 39], [-400, 3, 92]):
        assert grid.class_id(probe) == grid.class_id(probe + np.asarray(shift) @ matrix)


def test_translation_support_keeps_repeated_periodic_classes_distinct():
    grid = SupercellGrid(np.diag([2, 1, 1]))
    support = TranslationSupport(
        [[0, 0, 0], [2, 0, 0], [-2, 0, 0]], grid, provenance="file"
    )

    assert len(support.translations) == 3
    np.testing.assert_array_equal(support.class_ids, [0, 0, 0])
    np.testing.assert_array_equal(
        support.translations,
        [[0, 0, 0], [2, 0, 0], [-2, 0, 0]],
    )
    assert support.size == 3
    assert support.provenance == "file"
    assert support.digest == support.digest
    np.testing.assert_allclose(
        support.phases([[0.25, 0.0, 0.0]]),
        [[1.0, -1.0, -1.0]],
        atol=1e-15,
    )


def test_forceconstant_keeps_physical_replicas_separate_from_ifc_support(tmp_path):
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    grid = SupercellGrid(np.diag([2, 1, 1]))
    support = TranslationSupport(
        [[0, 0, 0], [2, 0, 0], [-2, 0, 0]], grid, provenance="file"
    )
    physical = np.array([[0, 0, 0], [-1, 0, 0]])
    positions = physical[:, None, :] + atoms.positions[None, :, :]
    forceconstant = ForceConstant(
        atoms=atoms,
        replicated_positions=positions,
        replica_translations=physical,
        supercell=(2, 1, 1),
        supercell_grid=grid,
        translation_support=support,
        folder=str(tmp_path),
    )

    assert len(forceconstant.replicated_atoms) == 2
    assert forceconstant.n_translations == 3
    np.testing.assert_array_equal(forceconstant.replicated_atoms.positions, positions.reshape(-1, 3))
    np.testing.assert_array_equal(forceconstant.list_of_replicas, support.translations)


def test_forceconstant_rejects_incomplete_physical_replica_classes(tmp_path):
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    grid = SupercellGrid(np.diag([2, 1, 1]))
    with pytest.raises(ValueError, match="each periodic class exactly once"):
        ForceConstant(
            atoms=atoms,
            replicated_positions=np.zeros((2, 1, 3)),
            replica_translations=[[0, 0, 0], [2, 0, 0]],
            supercell=(2, 1, 1),
            supercell_grid=grid,
            folder=str(tmp_path),
        )


def test_wigner_seitz_images_find_skew_cell_shortest_vector():
    cell = np.array([[2.0, 0.0, 0.0], [1.8, 0.7, 0.0], [0.1, 0.2, 2.1]])
    supercell = SupercellGrid(np.eye(3, dtype=int))
    support = TranslationSupport([[0, 0, 0]], supercell)
    positions = np.array([[0.0, 0.0, 0.0], [1.71, 0.665, 0.0]])

    images = WignerSeitzImages.build(support, positions, cell)
    _, actual, _ = images.image(0, 0, 1)
    brute_force = []
    displacement = positions[1] - positions[0]
    for a in range(-4, 5):
        for b in range(-4, 5):
            for c in range(-4, 5):
                brute_force.append(displacement + np.array([a, b, c]) @ cell)
    expected_norm = min(np.linalg.norm(vector) for vector in brute_force)

    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), expected_norm, atol=1e-12)
    # Componentwise fractional wrapping would select (-0.29, -0.035, 0),
    # while the Cartesian minimum is obtained through the skew lattice.
    assert not np.allclose(actual[0], [-0.29, -0.035, 0.0])


def test_wigner_seitz_images_retain_ties_with_normalized_weights():
    cell = np.eye(3)
    support = TranslationSupport([[0, 0, 0]], SupercellGrid(np.eye(3, dtype=int)))
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])

    images = WignerSeitzImages.build(support, positions, cell)
    translations, displacements, weights = images.image(0, 0, 1)

    assert len(translations) == 4
    np.testing.assert_allclose(np.linalg.norm(displacements, axis=1), np.sqrt(0.5))
    np.testing.assert_allclose(weights, np.full(4, 0.25), rtol=0, atol=0)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=0)


def test_periodic_amorphous_boundary_pair_uses_neighboring_cell():
    cell = np.diag([10.0, 8.0, 7.0])
    support = TranslationSupport([[0, 0, 0]], SupercellGrid(np.eye(3, dtype=int)))
    positions = np.array([[9.5, 4.0, 3.5], [0.5, 4.0, 3.5]])

    images = WignerSeitzImages.build(support, positions, cell)

    translations, displacements, weights = images.image(0, 0, 1)
    np.testing.assert_array_equal(translations, [[1, 0, 0]])
    np.testing.assert_allclose(displacements, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(weights, [1.0])


def test_wigner_seitz_images_are_lazy_and_cached():
    support = TranslationSupport.periodic(SupercellGrid(np.diag([2, 1, 1])))
    images = WignerSeitzImages.build(
        support,
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        np.eye(3),
    )

    assert images._cache == {}
    first = images.image(0, 0, 1)
    assert list(images._cache) == [(0, 0, 1)]
    assert images.image(0, 0, 1) is first

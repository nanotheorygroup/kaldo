"""Focused tests for the separated reciprocal and IFC translation grids."""

import numpy as np

from kaldo.grid import QGrid, SupercellGrid, TranslationSupport, WignerSeitzImages


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
        [grid.address_to_id(grid.addresses[5] - address) for address in grid.addresses],
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


def test_wigner_seitz_images_find_skew_cell_shortest_vector():
    cell = np.array([[2.0, 0.0, 0.0], [1.8, 0.7, 0.0], [0.1, 0.2, 2.1]])
    supercell = SupercellGrid(np.eye(3, dtype=int))
    support = TranslationSupport([[0, 0, 0]], supercell)
    positions = np.array([[0.0, 0.0, 0.0], [1.71, 0.665, 0.0]])

    images = WignerSeitzImages.build(support, positions, cell)
    actual = images.displacements[0][0][1]
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
    translations = images.translations[0][0][1]
    displacements = images.displacements[0][0][1]
    weights = images.weights[0][0][1]

    assert len(translations) == 4
    np.testing.assert_allclose(np.linalg.norm(displacements, axis=1), np.sqrt(0.5))
    np.testing.assert_allclose(weights, np.full(4, 0.25), rtol=0, atol=0)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=0)


def test_periodic_amorphous_boundary_pair_uses_neighboring_cell():
    cell = np.diag([10.0, 8.0, 7.0])
    support = TranslationSupport([[0, 0, 0]], SupercellGrid(np.eye(3, dtype=int)))
    positions = np.array([[9.5, 4.0, 3.5], [0.5, 4.0, 3.5]])

    images = WignerSeitzImages.build(support, positions, cell)

    np.testing.assert_array_equal(images.translations[0][0][1], [[1, 0, 0]])
    np.testing.assert_allclose(images.displacements[0][0][1], [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(images.weights[0][0][1], [1.0])

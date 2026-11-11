"""Tests for the Monkhorst-Pack half-step shift of the reciprocal grid."""
import numpy as np
import pytest

from kaldo.grid import QGrid

SHIFT = (0.5, 0.5, 0.5)


def test_default_grid_unchanged():
    grid = QGrid((2, 2, 2))
    expected = np.array([[i, j, k] for i in (0, 0.5) for j in (0, 0.5) for k in (0, 0.5)], dtype=np.float64)
    np.testing.assert_allclose(grid.fractional_points, expected, rtol=0, atol=1e-15)
    assert grid.shift == (0.0, 0.0, 0.0)
    assert not grid.is_shifted


def test_shifted_points_are_quarter_offsets():
    grid = QGrid((2, 2, 2), shift=SHIFT)
    np.testing.assert_allclose(grid.fractional_points[0], [0.25, 0.25, 0.25], rtol=0, atol=1e-15)
    assert grid.is_shifted


def test_shifted_time_reversal_partner():
    grid = QGrid((2, 2, 2), shift=SHIFT)
    points = grid.fractional_points
    for point_id in range(grid.size):
        total = points[point_id] + points[grid.partner_id(point_id)]
        np.testing.assert_allclose(total, np.rint(total), rtol=0, atol=1e-15)


@pytest.mark.parametrize("is_plus", [True, False])
def test_shifted_momentum_partners_land_on_gamma_grid(is_plus):
    grid = QGrid((2, 2, 2), shift=SHIFT)
    partner = grid.partner_grid()
    assert not partner.is_shifted
    assert partner.shape == grid.shape and partner.order == grid.order
    points = grid.fractional_points
    sign = 1.0 if is_plus else -1.0
    for point_id in range(grid.size):
        ids = grid.shifted_partner_ids(point_id, is_plus)
        assert ids.dtype == np.int64
        target = points[point_id] + sign * points
        got = partner.fractional_points[ids]
        diff = target - got
        np.testing.assert_allclose(diff, np.rint(diff), rtol=0, atol=1e-15)


def test_shifted_rejects_closed_partner_lookup():
    grid = QGrid((2, 2, 2), shift=SHIFT)
    with pytest.raises(ValueError):
        grid.momentum_partner_ids(0, True)


@pytest.mark.parametrize("shift", [(0.25, 0, 0), (0.5, 0.5)])
def test_shift_values_validated(shift):
    with pytest.raises(ValueError):
        QGrid((2, 2, 2), shift=shift)

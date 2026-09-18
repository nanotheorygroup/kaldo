"""Contracts for the independently implemented QE ``ibrav`` lattice table.

The expected vectors below follow the public ``INPUT_PW`` crystallographic
definitions.  They deliberately test row orientation as well as lengths and
volume because the QE rigid-ion kernel is sensitive to the lattice gauge.
"""

import numpy as np
import pytest

from kaldo.interfaces.qe_io import QELatticeError, qe_lattice_from_ibrav


ALAT = 4.0
B_OVER_A = 1.25
C_OVER_A = 1.75
COS_BC = 0.20
COS_AC = -0.10
COS_AB = 0.15
CELLDM = np.array(
    [ALAT, B_OVER_A, C_OVER_A, COS_BC, COS_AC, COS_AB],
    dtype=np.float64,
)


def _assert_documented_rows(ibrav: int, expected_in_alat: np.ndarray) -> None:
    rows, volume = qe_lattice_from_ibrav(ibrav, CELLDM)
    expected = ALAT * np.asarray(expected_in_alat, dtype=np.float64)
    np.testing.assert_allclose(rows, expected, rtol=0.0, atol=1.0e-14)
    assert volume == pytest.approx(abs(np.linalg.det(expected)), rel=1.0e-14)


@pytest.mark.parametrize(
    ("ibrav", "expected"),
    [
        (1, np.eye(3)),
        (2, 0.5 * np.array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]])),
        (3, 0.5 * np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]])),
        (-3, 0.5 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])),
        (
            4,
            np.array(
                [[1, 0, 0], [-0.5, np.sqrt(3.0) / 2.0, 0], [0, 0, C_OVER_A]]
            ),
        ),
        (6, np.diag([1.0, 1.0, C_OVER_A])),
        (
            7,
            0.5
            * np.array(
                [[1, -1, C_OVER_A], [1, 1, C_OVER_A], [-1, -1, C_OVER_A]]
            ),
        ),
        (8, np.diag([1.0, B_OVER_A, C_OVER_A])),
        (
            9,
            np.array(
                [
                    [0.5, B_OVER_A / 2.0, 0],
                    [-0.5, B_OVER_A / 2.0, 0],
                    [0, 0, C_OVER_A],
                ]
            ),
        ),
        (
            -9,
            np.array(
                [
                    [0.5, -B_OVER_A / 2.0, 0],
                    [0.5, B_OVER_A / 2.0, 0],
                    [0, 0, C_OVER_A],
                ]
            ),
        ),
        (
            91,
            np.array(
                [
                    [1, 0, 0],
                    [0, B_OVER_A / 2.0, -C_OVER_A / 2.0],
                    [0, B_OVER_A / 2.0, C_OVER_A / 2.0],
                ]
            ),
        ),
        (
            10,
            0.5
            * np.array(
                [[1, 0, C_OVER_A], [1, B_OVER_A, 0], [0, B_OVER_A, C_OVER_A]]
            ),
        ),
        (
            11,
            0.5
            * np.array(
                [
                    [1, B_OVER_A, C_OVER_A],
                    [-1, B_OVER_A, C_OVER_A],
                    [-1, -B_OVER_A, C_OVER_A],
                ]
            ),
        ),
    ],
)
def test_documented_orthogonal_and_centered_lattice_vectors(
    ibrav: int, expected: np.ndarray
) -> None:
    _assert_documented_rows(ibrav, expected)


@pytest.mark.parametrize("ibrav", [5, -5])
def test_rhombohedral_vectors_have_the_documented_metric(ibrav: int) -> None:
    rows, volume = qe_lattice_from_ibrav(ibrav, CELLDM)
    metric = rows @ rows.T
    expected = ALAT**2 * np.full((3, 3), COS_BC, dtype=np.float64)
    np.fill_diagonal(expected, ALAT**2)
    np.testing.assert_allclose(metric, expected, rtol=0.0, atol=1.0e-13)
    assert volume > 0.0
    if ibrav == 5:
        assert rows[1, 0] == pytest.approx(0.0, abs=1.0e-15)
    else:
        # The -5 choice has its threefold axis along <111>: cyclic permutation
        # of Cartesian components maps one primitive vector onto the next.
        np.testing.assert_allclose(rows[1], np.roll(rows[0], 1), rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(rows[2], np.roll(rows[1], 1), rtol=0.0, atol=1.0e-14)


@pytest.mark.parametrize(
    ("ibrav", "expected"),
    [
        (
            12,
            np.array(
                [
                    [1, 0, 0],
                    [B_OVER_A * COS_BC, B_OVER_A * np.sqrt(1.0 - COS_BC**2), 0],
                    [0, 0, C_OVER_A],
                ]
            ),
        ),
        (
            -12,
            np.array(
                [
                    [1, 0, 0],
                    [0, B_OVER_A, 0],
                    [C_OVER_A * COS_AC, 0, C_OVER_A * np.sqrt(1.0 - COS_AC**2)],
                ]
            ),
        ),
        (
            13,
            np.array(
                [
                    [0.5, 0, -C_OVER_A / 2.0],
                    [B_OVER_A * COS_BC, B_OVER_A * np.sqrt(1.0 - COS_BC**2), 0],
                    [0.5, 0, C_OVER_A / 2.0],
                ]
            ),
        ),
        (
            -13,
            np.array(
                [
                    [0.5, B_OVER_A / 2.0, 0],
                    [-0.5, B_OVER_A / 2.0, 0],
                    [C_OVER_A * COS_AC, 0, C_OVER_A * np.sqrt(1.0 - COS_AC**2)],
                ]
            ),
        ),
    ],
)
def test_documented_monoclinic_lattice_vectors(ibrav: int, expected: np.ndarray) -> None:
    _assert_documented_rows(ibrav, expected)


def test_triclinic_vectors_reproduce_the_requested_lengths_and_angles() -> None:
    rows, volume = qe_lattice_from_ibrav(14, CELLDM)
    lengths = ALAT * np.array([1.0, B_OVER_A, C_OVER_A])
    expected_metric = np.array(
        [
            [lengths[0] ** 2, lengths[0] * lengths[1] * COS_AB, lengths[0] * lengths[2] * COS_AC],
            [lengths[0] * lengths[1] * COS_AB, lengths[1] ** 2, lengths[1] * lengths[2] * COS_BC],
            [lengths[0] * lengths[2] * COS_AC, lengths[1] * lengths[2] * COS_BC, lengths[2] ** 2],
        ]
    )
    np.testing.assert_allclose(rows @ rows.T, expected_metric, rtol=0.0, atol=2.0e-13)
    assert volume == pytest.approx(np.sqrt(np.linalg.det(expected_metric)), rel=1.0e-14)


def test_ibrav_zero_transposes_q2r_columns_and_applies_alat() -> None:
    columns = np.array(
        [[1.0, 0.2, -0.1], [0.0, 0.9, 0.3], [0.0, 0.0, 1.1]],
        dtype=np.float64,
    )
    rows, volume = qe_lattice_from_ibrav(0, CELLDM, at_columns=columns)
    np.testing.assert_allclose(rows, ALAT * columns.T, rtol=0.0, atol=0.0)
    assert volume == pytest.approx(abs(np.linalg.det(ALAT * columns.T)), rel=1.0e-14)


@pytest.mark.parametrize("ibrav", [15, -1, 92])
def test_unsupported_ibrav_is_rejected(ibrav: int) -> None:
    with pytest.raises(QELatticeError, match="unsupported/nonexistent"):
        qe_lattice_from_ibrav(ibrav, CELLDM)


def test_invalid_length_angle_and_explicit_cell_are_rejected() -> None:
    invalid_length = CELLDM.copy()
    invalid_length[1] = 0.0
    with pytest.raises(QELatticeError, match=r"celldm\(2\) must be positive"):
        qe_lattice_from_ibrav(8, invalid_length)

    invalid_angle = CELLDM.copy()
    invalid_angle[5] = 1.0
    with pytest.raises(QELatticeError, match=r"celldm\(6\)"):
        qe_lattice_from_ibrav(14, invalid_angle)

    with pytest.raises(QELatticeError, match="ibrav=0.*singular"):
        qe_lattice_from_ibrav(0, CELLDM, at_columns=np.zeros((3, 3)))

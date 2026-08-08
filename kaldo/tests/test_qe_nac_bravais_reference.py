"""Compare kALDo's QE rigid-ion tensor with transparent QE 7.6 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kaldo.controllers.nac import (
    _prepare_qe_static_data,
    _qe_nonanalytic_tensor,
    _qe_rigid_ion_tensor,
    _qe_to_cartesian,
)
from kaldo.interfaces.qe_io import read_q2r_header

DATA = Path(__file__).parent / "data" / "input" / "qe76-bravais-reference"
CASES = tuple(
    case["id"] for case in json.loads((DATA / "manifest.json").read_text())["cases"]
)


def _read_matdyn_matrix(path: Path, natoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the first unmass-weighted QE fldyn matrix in Ry/bohr squared."""
    lines = path.read_text(encoding="utf-8").splitlines()
    cursor = next(i for i, line in enumerate(lines) if "q =" in line)
    q_values = lines[cursor].split("(", 1)[1].split(")", 1)[0].split()
    qpoint = np.asarray(q_values, dtype=float)
    cursor += 1
    matrix = np.zeros((3 * natoms, 3 * natoms), dtype=np.complex128)

    for atom_i in range(natoms):
        for atom_j in range(natoms):
            while not lines[cursor].split():
                cursor += 1
            indices = tuple(int(value) for value in lines[cursor].split())
            assert indices == (atom_i + 1, atom_j + 1)
            cursor += 1
            for axis_i in range(3):
                values = [
                    float(value.replace("D", "E").replace("d", "e"))
                    for value in lines[cursor].split()
                ]
                assert len(values) == 6
                row = np.asarray(values[::2]) + 1j * np.asarray(values[1::2])
                matrix[3 * atom_i + axis_i, 3 * atom_j : 3 * atom_j + 3] = row
                cursor += 1
    return qpoint, matrix


@pytest.mark.parametrize("case_id", CASES)
def test_qe_rigid_ion_matches_matdyn_for_every_bravais_class(case_id: str) -> None:
    """Match QE's directional-Gamma NAC-on/off difference for one case."""
    case = DATA / case_id
    header = read_q2r_header(case / f"{case_id}.fc")
    q_on, matrix_on = _read_matdyn_matrix(case / f"{case_id}.on.dyn", header.natoms)
    q_off, matrix_off = _read_matdyn_matrix(case / f"{case_id}.off.dyn", header.natoms)
    np.testing.assert_allclose(q_on, np.zeros(3), rtol=0, atol=1e-12)
    np.testing.assert_allclose(q_off, q_on, rtol=0, atol=1e-12)

    static_data = _prepare_qe_static_data(header)
    gamma_direction = _qe_to_cartesian(
        static_data, np.asarray([1.0, 0.0, 0.0]), "crystal"
    )
    actual = _qe_rigid_ion_tensor(static_data, np.zeros(3))
    actual += _qe_nonanalytic_tensor(static_data, gamma_direction)
    reference = (matrix_on - matrix_off).reshape(header.natoms, 3, header.natoms, 3)

    difference = np.linalg.norm(actual - reference)
    scale = np.linalg.norm(reference)
    assert difference / scale < 5e-6

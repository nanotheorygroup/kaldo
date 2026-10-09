"""Regenerate the Phonopy NAC references for the 26 Gonze validation cases.

For every case in ``manifest.json`` this loads the pinned
``<material-id>/phonopy_params.yaml`` with Phonopy, evaluates the NAC-on minus
NAC-off dynamical matrices at the test q points, and writes
``<material-id>/reference.npz`` with that difference together with the
structures and dielectric data kALDo needs to evaluate the same quantity.
The kALDo test reads only these files, so neither Phonopy nor PyYAML is a test
dependency. Run from any directory with Phonopy and PyYAML installed:

    python kaldo/tests/data/input/gonze-phonopy/generate_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import phonopy
import yaml
from ase import units

ROOT = Path(__file__).resolve().parent
QCART_ANGSTROM_INV = np.array(
    [
        [0.04, 0.02, 0.0],
        [0.2, 0.2, 1.0 / 3.0],
    ]
)


def _effective_supercell_matrix(ph) -> np.ndarray:
    """Recover the integer supercell matrix represented by a Phonopy object."""

    matrix_float = np.asarray(ph.supercell.cell) @ np.linalg.inv(np.asarray(ph.primitive.cell))
    matrix = np.rint(matrix_float).astype(int)
    np.testing.assert_allclose(matrix_float, matrix, rtol=0, atol=1e-7)
    return matrix


def _commensurate_qpoint(matrix: np.ndarray) -> np.ndarray | None:
    """Return one finite commensurate q point, if the supercell has one."""

    candidates = []
    signed_axes = np.vstack((np.eye(3, dtype=int), -np.eye(3, dtype=int)))
    for integer in signed_axes:
        qpoint = np.linalg.solve(matrix.T, integer)
        qpoint -= np.rint(qpoint)
        if np.linalg.norm(qpoint) > 1e-12:
            candidates.append(qpoint)
    if not candidates:
        assert abs(round(np.linalg.det(matrix))) == 1
        return None
    return min(candidates, key=np.linalg.norm)


def _cell_arrays(prefix: str, cell) -> dict:
    return {
        f"{prefix}_numbers": np.asarray(cell.numbers, dtype=np.int64),
        f"{prefix}_scaled_positions": np.asarray(cell.scaled_positions, dtype=np.float64),
        f"{prefix}_cell": np.asarray(cell.cell, dtype=np.float64),
        f"{prefix}_masses": np.asarray(cell.masses, dtype=np.float64),
    }


def _reference(case: dict) -> dict:
    source = ROOT / case["id"] / "phonopy_params.yaml"
    metadata = yaml.safe_load(source.read_text(encoding="utf-8"))
    load_kwargs = {"produce_fc": True, "fc_calculator": "traditional"}
    if "primitive_matrix" not in metadata:
        load_kwargs["primitive_matrix"] = "P"
    ph = phonopy.load(source, **load_kwargs)
    assert ph.nac_params is not None

    matrix = _effective_supercell_matrix(ph)
    np.testing.assert_array_equal(matrix, case["effective_supercell_matrix"])
    qpoints = QCART_ANGSTROM_INV @ np.asarray(ph.primitive.cell).T / (2 * np.pi)
    commensurate = _commensurate_qpoint(matrix)
    if commensurate is not None:
        qpoints = np.vstack((qpoints, commensurate))

    nac_params = ph.nac_params
    ph.nac_params = None
    ph.run_qpoints(qpoints, with_dynamical_matrices=True)
    nac_off = np.array(ph.qpoints.dynamical_matrices, copy=True)
    ph.nac_params = nac_params
    ph.run_qpoints(qpoints, with_dynamical_matrices=True)
    nac_on = np.array(ph.qpoints.dynamical_matrices, copy=True)

    return {
        "phonopy_version": np.array(phonopy.__version__),
        "supercell_matrix": matrix,
        "qpoints": np.asarray(qpoints, dtype=np.float64),
        "expected_delta": (nac_on - nac_off) * (units.mol / (10 * units.J)),
        "born": np.asarray(nac_params["born"], dtype=np.float64),
        "dielectric": np.asarray(nac_params["dielectric"], dtype=np.float64),
        "nac_factor": np.array(float(nac_params["factor"])),
        **_cell_arrays("primitive", ph.primitive),
        **_cell_arrays("supercell", ph.supercell),
    }


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        np.savez_compressed(ROOT / case["id"] / "reference.npz", **_reference(case))
        print(case["id"])


if __name__ == "__main__":
    main()

"""Compare kALDo's Gonze NAC kernel with 26 pinned Phonopy references.

The Phonopy NAC-on minus NAC-off dynamical matrices are stored per case in
``data/input/gonze-phonopy/<material-id>/reference.npz`` together with the
structures and dielectric data, so this test needs neither Phonopy nor PyYAML.
Regenerate the references with ``generate_reference.py`` in that directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units

from kaldo.controllers import nac

DATA = Path(__file__).parent / "data" / "input" / "gonze-phonopy"
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
CASES = tuple(MANIFEST["cases"])
DOCUMENTED_REGRESSION_BOUND = 1.0e-5
CANCELLATION_REFERENCE_NORM = 1.0e-8
CANCELLATION_ABSOLUTE_BOUND = 1.0e-9


def _atoms(reference, prefix: str) -> Atoms:
    """Rebuild the ASE cell stored for ``prefix`` (primitive or supercell)."""

    return Atoms(
        numbers=reference[f"{prefix}_numbers"],
        scaled_positions=reference[f"{prefix}_scaled_positions"],
        cell=reference[f"{prefix}_cell"],
        masses=reference[f"{prefix}_masses"],
        pbc=True,
    )


def _prepare_isolated_gonze_delta(reference, matrix: np.ndarray):
    """Build the production Gonze kernel with zero total force constants.

    kALDo stores a short-range force-constant body. For this isolated NAC
    comparison, the total body is zero, so the commensurate short-range body
    is the negative reciprocal dipole contribution. Restoring the contribution
    at arbitrary q then exposes only the NAC delta for comparison with Phonopy.
    """

    class Second:
        """Minimal SecondOrder-like container required by the NAC controller."""

    second = Second()
    second.atoms = _atoms(reference, "primitive")
    second.replicated_atoms = _atoms(reference, "supercell")
    second.supercell = (abs(round(np.linalg.det(matrix))), 1, 1)
    second.atoms.set_array("charges", np.array(reference["born"], dtype=np.float64))
    second.atoms.info["dielectric"] = np.array(reference["dielectric"], dtype=np.float64)
    second.atoms.info["nac_factor"] = float(reference["nac_factor"])

    static_data = nac.build_static_data(second, matrix)
    mapping = nac._build_supercell_matrix_mapping(
        second.atoms,
        matrix,
        replicated_atoms=second.replicated_atoms,
    )
    static_data, mapping = nac.ensure_kernel_cache(static_data, mapping)
    commensurate_qpoints = nac._commensurate_points(
        matrix, static_data["reciprocal_lattice"]
    )
    dipole_samples = np.asarray(
        [
            nac._dipole_dipole_dynamical_matrix(qpoint, static_data, mapping)
            for qpoint in commensurate_qpoints
        ]
    )
    short_range_fc = nac._inverse_transform_dynmats_to_force_constants(
        -dipole_samples,
        commensurate_qpoints,
        mapping,
        static_data["masses"],
    )
    short_range_fc *= units.mol / (10 * units.J)
    return static_data, mapping, short_range_fc


def _kaldo_isolated_delta(reference, matrix: np.ndarray, qpoints: np.ndarray) -> np.ndarray:
    """Evaluate the isolated Gonze NAC delta through the shared controller."""

    static_data, mapping, short_range_fc = _prepare_isolated_gonze_delta(reference, matrix)
    qpoint_carts = np.einsum(
        "ab,qb->qa",
        static_data["reciprocal_lattice"],
        qpoints,
        optimize=True,
    )
    return nac.dynamical_matrices(
        qpoints,
        static_data,
        mapping,
        qpoint_carts,
        fc=short_range_fc,
    )


@pytest.mark.performance
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_gonze_isolated_delta_matches_phonopy(case: dict) -> None:
    """Match Phonopy NAC-on minus NAC-off for one pinned polar material."""

    with np.load(DATA / case["id"] / "reference.npz") as stored:
        reference = {key: stored[key] for key in stored.files}
    matrix = np.asarray(reference["supercell_matrix"], dtype=int)
    np.testing.assert_array_equal(matrix, case["effective_supercell_matrix"])
    qpoints = reference["qpoints"]
    expected = reference["expected_delta"]

    actual = _kaldo_isolated_delta(reference, matrix, qpoints)
    for q_index, (candidate, target) in enumerate(zip(actual, expected)):
        absolute = float(np.linalg.norm(candidate - target))
        reference_norm = float(np.linalg.norm(target))
        if reference_norm < CANCELLATION_REFERENCE_NORM:
            assert (
                absolute <= CANCELLATION_ABSOLUTE_BOUND
            ), f"{case['id']} q[{q_index}] cancellation error: {absolute:.6e}"
        else:
            relative = absolute / reference_norm
            assert (
                relative <= DOCUMENTED_REGRESSION_BOUND
            ), f"{case['id']} q[{q_index}] relative error: {relative:.6e}"

"""Compare kALDo's Gonze NAC kernel with 26 pinned Phonopy references."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Optional cross-check dependencies: kaldo itself never imports them, and
# they are deliberately not test requirements. The 26 reference cases only
# run where phonopy happens to be installed.
phonopy = pytest.importorskip("phonopy")
yaml = pytest.importorskip("yaml")
from ase import Atoms, units

from kaldo.controllers import nac

DATA = Path(__file__).parent / "data" / "input" / "gonze-phonopy"
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
CASES = tuple(MANIFEST["cases"])
QCART_ANGSTROM_INV = np.array(
    [
        [0.04, 0.02, 0.0],
        [0.2, 0.2, 1.0 / 3.0],
    ]
)
DOCUMENTED_REGRESSION_BOUND = 1.0e-5
CANCELLATION_REFERENCE_NORM = 1.0e-8
CANCELLATION_ABSOLUTE_BOUND = 1.0e-9


def _atoms(cell) -> Atoms:
    """Convert a Phonopy cell to the ASE representation used by kALDo."""

    return Atoms(
        symbols=cell.symbols,
        scaled_positions=cell.scaled_positions,
        cell=cell.cell,
        masses=cell.masses,
        pbc=True,
    )


def _effective_supercell_matrix(ph) -> np.ndarray:
    """Recover the integer supercell matrix represented by a Phonopy object."""

    matrix_float = np.asarray(ph.supercell.cell) @ np.linalg.inv(
        np.asarray(ph.primitive.cell)
    )
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

    qpoint = min(candidates, key=np.linalg.norm)
    np.testing.assert_allclose(
        matrix.T @ qpoint,
        np.rint(matrix.T @ qpoint),
        rtol=0,
        atol=1e-12,
    )
    return qpoint


def _prepare_isolated_gonze_delta(ph, matrix: np.ndarray):
    """Build the production Gonze kernel with zero total force constants.

    kALDo stores a short-range force-constant body. For this isolated NAC
    comparison, the total body is zero, so the commensurate short-range body
    is the negative reciprocal dipole contribution. Restoring the contribution
    at arbitrary q then exposes only the NAC delta for comparison with Phonopy.
    """

    class Second:
        """Minimal SecondOrder-like container required by the NAC controller."""

    second = Second()
    second.atoms = _atoms(ph.primitive)
    second.replicated_atoms = _atoms(ph.supercell)
    second.supercell = (abs(round(np.linalg.det(matrix))), 1, 1)
    second.atoms.set_array(
        "charges", np.asarray(ph.nac_params["born"], dtype=float).copy()
    )
    second.atoms.info["dielectric"] = np.asarray(
        ph.nac_params["dielectric"], dtype=float
    ).copy()
    second.atoms.info["nac_factor"] = float(ph.nac_params["factor"])

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


def _kaldo_isolated_delta(ph, matrix: np.ndarray, qpoints: np.ndarray) -> np.ndarray:
    """Evaluate the isolated Gonze NAC delta through the shared controller."""

    static_data, mapping, short_range_fc = _prepare_isolated_gonze_delta(ph, matrix)
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

    source = DATA / case["id"] / "phonopy_params.yaml"
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

    conversion = units.mol / (10 * units.J)
    expected = (nac_on - nac_off) * conversion
    actual = _kaldo_isolated_delta(ph, matrix, qpoints)
    for q_index, (candidate, reference) in enumerate(zip(actual, expected)):
        absolute = float(np.linalg.norm(candidate - reference))
        reference_norm = float(np.linalg.norm(reference))
        if reference_norm < CANCELLATION_REFERENCE_NORM:
            assert (
                absolute <= CANCELLATION_ABSOLUTE_BOUND
            ), f"{case['id']} q[{q_index}] cancellation error: {absolute:.6e}"
        else:
            relative = absolute / reference_norm
            assert (
                relative <= DOCUMENTED_REGRESSION_BOUND
            ), f"{case['id']} q[{q_index}] relative error: {relative:.6e}"

"""Golden tests for the QE q2r functions in the common NAC controller.

The fixture was emitted by unmodified QE 7.6 rigid.f90. This module imports
only NumPy and the controller functions, avoiding the TensorFlow runtime.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np

from kaldo.interfaces.qe_io import AMU_RY, read_q2r_header
from kaldo.controllers.nac import (
    BOHR_ANGSTROM,
    EV_TO_10J_PER_MOL,
    RY_TO_EV,
    _prepare_qe_static_data,
    _qe_correction,
    _qe_nonanalytic_tensor,
    _qe_rigid_ion_tensor,
    _qe_to_cartesian,
    qe_default_alpha,
)


def _reference() -> dict[str, np.ndarray]:
    fixture = Path(__file__).parent / "data" / "qe76_rigid_f90_reference.npz"
    with np.load(fixture, allow_pickle=False) as data:
        assert str(data["qe_tag"]) == "qe-7.6"
        assert (
            str(data["rigid_f90_sha256"])
            == "c2da58dfa6849c4edb6f96ad8ad2be58834ae23142bdac2ee33eeebf1eb21e88"
        )
        return {name: np.array(data[name]) for name in data.files}


def _static_data() -> dict:
    alat = 9.7
    at = np.column_stack(([1.0, 0.0, 0.0], [0.21, 0.93, 0.0], [-0.13, 0.17, 1.11]))
    tau = np.asarray([[0.0, 0.0, 0.0], [0.19, 0.27, 0.31], [0.43, 0.11, 0.68]])
    z1 = np.asarray([2.10, 0.14, -0.09, 0.03, 1.82, 0.11, -0.07, 0.18, 2.25]).reshape(
        (3, 3), order="F"
    )
    z2 = np.asarray(
        [-1.24, 0.04, 0.16, 0.12, -0.93, -0.05, 0.08, -0.14, -1.18]
    ).reshape((3, 3), order="F")
    born = np.asarray([z1, z2])
    header = SimpleNamespace(
        has_zstar=True,
        at_columns=at,
        tau=tau,
        born=np.asarray([born[0], born[1], -born.sum(axis=0)]),
        dielectric=np.asarray(
            [[4.7, 0.12, -0.08], [0.12, 5.3, 0.21], [-0.08, 0.21, 3.9]]
        ),
        q_grid=(4, 3, 2),
        alpha=qe_default_alpha(alat),
        alat_bohr=alat,
        volume_bohr3=float(abs(np.linalg.det(at)) * alat**3),
        atom_masses_amu=np.asarray([28.085, 15.999, 15.999]),
    )
    return _prepare_qe_static_data(header)


def test_qe_rigid_ion_matches_unmodified_qe76_rigid_f90() -> None:
    reference = _reference()
    static_data = _static_data()
    np.testing.assert_allclose(
        _qe_rigid_ion_tensor(static_data, reference["q_reduced_3d"]),
        reference["finite_3d"],
        rtol=2e-13,
        atol=3e-16,
    )
    np.testing.assert_allclose(
        _qe_rigid_ion_tensor(static_data, np.zeros(3))
        + _qe_nonanalytic_tensor(
            static_data,
            _qe_to_cartesian(
                static_data, reference["gamma_direction_reduced"], "crystal"
            ),
        ),
        reference["gamma_directional_3d"],
        rtol=2e-13,
        atol=3e-16,
    )


def test_unit_bridge_is_mass_weighted_kaldo_units() -> None:
    static_data = _static_data()
    qpoint = [0.173, -0.119, 0.087]
    raw = _qe_rigid_ion_tensor(static_data, qpoint).reshape(9, 9)
    roots = np.repeat(np.sqrt(static_data["masses"]), 3)
    expected = raw * (RY_TO_EV / BOHR_ANGSTROM**2)
    expected /= roots[:, None] * roots[None, :]
    expected *= EV_TO_10J_PER_MOL
    np.testing.assert_allclose(_qe_correction(static_data, qpoint), expected)


def test_qe_preparation_uses_the_common_static_data_layout() -> None:
    static_data = _static_data()
    assert static_data["qe_g_vectors"].shape[1] == 3
    assert static_data["qe_onsite"].shape == (3, 3, 3, 3)
    np.testing.assert_allclose(static_data["masses"], [28.085, 15.999, 15.999])


def test_q2r_header_retains_alpha_grid_masses_and_born_without_asr(
    tmp_path: Path,
) -> None:
    source = tmp_path / "q2r.fc"
    source.write_text(
        "2 2 2 10.5000000 0 0 0 0 0\n"
        f"1 'Si' {28.085 * AMU_RY:.16e}\n2 'O' {15.999 * AMU_RY:.16e}\n"
        "1 1 0 0 0\n2 2 0.25 0.25 0.25\n"
        "T 2.3833275422268900\n"
        "4.7 0 0\n0 5.3 0\n0 0 3.9\n"
        "1\n2 0 0\n0 1.8 0\n0 0 2.2\n"
        "2\n-2 0 0\n0 -1.8 0\n0 0 -2.2\n"
        "4 3 2\n1 1 1 1\n",
        encoding="utf-8",
    )
    header = read_q2r_header(source)
    static_data = _prepare_qe_static_data(header)
    assert header.alpha == 2.38332754222689
    assert header.q_grid == (4, 3, 2)
    np.testing.assert_allclose(header.dielectric, np.diag([4.7, 5.3, 3.9]))
    np.testing.assert_allclose(header.born[0], np.diag([2.0, 1.8, 2.2]))
    np.testing.assert_allclose(header.born[1], np.diag([-2.0, -1.8, -2.2]))
    np.testing.assert_allclose(static_data["masses"], [28.085, 15.999])
    np.testing.assert_allclose(static_data["qe_born"], header.born)

"""Parallel F1/F2 over IBZ q1 vs the serial path.

n_workers=2 must match n_workers=1. Flattened IFC tables are attached
via SharedMemory (not pickled nested quartet/triplet lists).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.cumulant import F1_from_fc, F2_from_fc
from kaldo.cumulant.thermodynamics import _harmonic_thermo

NE_IFC = Path(__file__).parent / "cumulant_fixtures" / "LJ" / "Neon_14K_4UC"
THERMO_MESH = (5, 5, 5)
SUPERCELL = (4, 4, 4)
REF_T_K = 14.0


def _have_ne_fixture():
    return (NE_IFC / "infile.forceconstant").exists() and \
           (NE_IFC / "infile.ucposcar").exists()


def test_quartet_mode_contraction_matches_scatter():
    """Direct quartet-space Psi4 matches the original real-space scatter."""
    from kaldo.cumulant.free_energy import (
        _build_psi4_modes_quartet,
        _build_psi4_modes_quartet_batch,
        _prepare_psi4_q1,
        build_psi4_realspace_v,
    )

    rng = np.random.default_rng(123)
    n_quartets = 11
    nb = 6
    qd = {
        "a1": rng.integers(0, 2, n_quartets, dtype=np.int32),
        "a2": rng.integers(0, 2, n_quartets, dtype=np.int32),
        "a3": rng.integers(0, 2, n_quartets, dtype=np.int32),
        "a4": rng.integers(0, 2, n_quartets, dtype=np.int32),
        "lv2c": rng.standard_normal((n_quartets, 3)),
        "lv3c": rng.standard_normal((n_quartets, 3)),
        "lv4c": rng.standard_normal((n_quartets, 3)),
        "ifc": rng.standard_normal((n_quartets, 3, 3, 3, 3)),
        "nb": nb,
    }
    q1 = rng.standard_normal(3)
    q2 = rng.standard_normal(3)
    M1 = rng.standard_normal((nb, nb, nb)) + 1j * rng.standard_normal((nb, nb, nb))
    M2 = rng.standard_normal((nb, nb, nb)) + 1j * rng.standard_normal((nb, nb, nb))

    A = build_psi4_realspace_v(qd, q1, q2)
    tmp = np.einsum("kab,abcd->kcd", M1, A)
    expected = np.einsum("kcd,lcd->kl", tmp, M2)
    left_q1 = _prepare_psi4_q1(qd, M1, q1)
    actual = _build_psi4_modes_quartet(qd, left_q1, M2, q2)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    q2_batch = rng.standard_normal((4, 3))
    M2_batch = rng.standard_normal((4, nb, nb, nb)) \
        + 1j * rng.standard_normal((4, nb, nb, nb))
    expected_batch = np.stack([
        _build_psi4_modes_quartet(qd, left_q1, M2_batch[i], q2_batch[i])
        for i in range(len(q2_batch))
    ])
    actual_batch = _build_psi4_modes_quartet_batch(
        qd, left_q1, M2_batch, q2_batch,
    )
    np.testing.assert_allclose(actual_batch, expected_batch, rtol=1e-12, atol=1e-12)


def test_triplet_mode_contraction_matches_scatter():
    """Direct triplet-space Psi3 matches the original real-space scatter."""
    from kaldo.cumulant.free_energy import (
        _build_psi3_modes_triplet_batch,
        _prepare_psi3_q1,
        build_psi3_realspace,
    )

    rng = np.random.default_rng(456)
    n_triplets = 13
    nb = 6
    td = {
        "a1": rng.integers(0, 2, n_triplets, dtype=np.int32),
        "a2": rng.integers(0, 2, n_triplets, dtype=np.int32),
        "a3": rng.integers(0, 2, n_triplets, dtype=np.int32),
        "lv2c": rng.standard_normal((n_triplets, 3)),
        "lv3c": rng.standard_normal((n_triplets, 3)),
        "ifc": rng.standard_normal((n_triplets, 3, 3, 3)),
        "nb": nb,
    }
    batch = 4
    q2 = rng.standard_normal((batch, 3))
    q3 = rng.standard_normal((batch, 3))
    e1 = rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
    e2 = rng.standard_normal((batch, nb, nb)) \
        + 1j * rng.standard_normal((batch, nb, nb))
    e3 = rng.standard_normal((batch, nb, nb)) \
        + 1j * rng.standard_normal((batch, nb, nb))

    expected = []
    for ib in range(batch):
        A = build_psi3_realspace(td, q2[ib], q3[ib])
        T1 = np.einsum("abc,ak->kbc", A, np.conj(e1))
        T2 = np.einsum("kbc,bl->klc", T1, np.conj(e2[ib]))
        expected.append(np.einsum("klc,cm->klm", T2, np.conj(e3[ib])))

    left_q1 = _prepare_psi3_q1(td, e1)
    actual = _build_psi3_modes_triplet_batch(td, left_q1, e2, e3, q2, q3)
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-12, atol=1e-12)


@pytest.fixture(scope="module")
def ne_fc():
    if not _have_ne_fixture():
        pytest.skip("LJ Ne 14K_4UC fixture unavailable")
    return ForceConstants.from_folder(
        folder=str(NE_IFC), supercell=SUPERCELL, format="tdep", include_fourth=True,
    )


@pytest.mark.skipif(not _have_ne_fixture(), reason="LJ Ne 14K_4UC fixture unavailable")
def test_parallel_F1_matches_serial(ne_fc):
    masses = np.asarray(ne_fc.atoms.get_masses())
    serial = F1_from_fc(
        ne_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        use_q_symmetry=True, is_classic=False, n_workers=1,
    )
    parallel = F1_from_fc(
        ne_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        use_q_symmetry=True, is_classic=False, n_workers=2,
    )
    for key in ("F1", "S1", "Cv1", "U1"):
        np.testing.assert_allclose(
            parallel[key], serial[key], rtol=1e-7,
            err_msg=f"parallel F1 {key} drifted from serial",
        )


@pytest.mark.skipif(not _have_ne_fixture(), reason="LJ Ne 14K_4UC fixture unavailable")
def test_parallel_F2_matches_serial(ne_fc):
    masses = np.asarray(ne_fc.atoms.get_masses())
    serial = F2_from_fc(
        ne_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        sigma_THz=None, use_q_symmetry=True, is_classic=False, n_workers=1,
    )
    parallel = F2_from_fc(
        ne_fc, masses_amu=masses, kmesh=THERMO_MESH, T_K=REF_T_K,
        sigma_THz=None, use_q_symmetry=True, is_classic=False, n_workers=2,
    )
    for key in ("F2", "S2", "Cv2", "U2"):
        np.testing.assert_allclose(
            parallel[key], serial[key], rtol=1e-7,
            err_msg=f"parallel F2 {key} drifted from serial",
        )


def test_parallel_harmonic_matches_serial(ne_fc):
    """Phase-1 F_H/S_H on two processes matches the in-process Phonons loop."""
    serial = _harmonic_thermo(
        ne_fc, THERMO_MESH, T_K=REF_T_K, is_classic=False, n_workers=1,
    )
    parallel = _harmonic_thermo(
        ne_fc, THERMO_MESH, T_K=REF_T_K, is_classic=False, n_workers=2,
    )
    for key in ("F_H", "U_H", "S_H", "Cv_H"):
        np.testing.assert_allclose(
            parallel[key], serial[key], rtol=1e-7,
            err_msg=f"parallel harmonic {key} drifted from serial",
        )


def test_n_workers_zero_raises_F1(ne_fc):
    masses = np.asarray(ne_fc.atoms.get_masses())
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        F1_from_fc(
            ne_fc, masses_amu=masses, kmesh=(2, 2, 2), T_K=REF_T_K,
            use_q_symmetry=True, n_workers=0,
        )

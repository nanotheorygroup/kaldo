from pathlib import Path
import os

import numpy as np
import pytest

from kaldo.observables import harmonic_with_q as hwq


DEFAULT_NACL_DEBUG = Path(
    "/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replay/"
    "example/nacl-att2/debug"
)


def nacl_debug_dir() -> Path:
    return Path(os.environ.get("NACL_ATT2_DEBUG_DIR", DEFAULT_NACL_DEBUG))


def require_nacl_debug() -> Path:
    path = nacl_debug_dir()
    if not (path / "static" / "metadata.json").exists():
        pytest.skip(f"NaCl Gonze-Lee debug tree not found at {path}")
    return path


def test_gonze_dielectric_part_matches_quadratic_form():
    vector = np.array([1.0, 2.0, -1.0])
    dielectric = np.diag([2.0, 3.0, 4.0])
    assert hwq._gonze_dielectric_part(vector, dielectric) == pytest.approx(18.0)


def test_gonze_multiply_borns_contracts_cartesian_axes():
    dd_in = np.zeros((1, 3, 1, 3), dtype=np.complex128)
    dd_in[0, :, 0, :] = np.arange(9, dtype=float).reshape(3, 3)
    born = np.zeros((1, 3, 3), dtype=float)
    born[0] = np.diag([2.0, 3.0, 5.0])
    actual = hwq._gonze_multiply_borns(dd_in, born)
    expected = np.zeros_like(actual)
    expected[0, :, 0, :] = born[0].T @ dd_in[0, :, 0, :] @ born[0]
    np.testing.assert_allclose(actual, expected)


def test_gonze_get_g_list_matches_nacl_debug_reference():
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    reciprocal_lattice = np.load(static / "reciprocal_lattice.npy")
    g_cutoff = float(np.load(static / "G_cutoff.npy"))
    expected = np.load(static / "G_list.npy")
    actual = hwq._gonze_get_g_list(reciprocal_lattice, g_cutoff)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.0)


def test_gonze_q0_and_limiting_terms_match_nacl_debug_reference():
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    g_list = np.load(static / "G_list.npy")
    born = np.load(static / "born.npy")
    dielectric = np.load(static / "dielectric.npy")
    positions = np.load(static / "primitive_positions.npy")
    lambda_ = float(np.load(static / "Lambda.npy"))
    tolerance = 1e-5
    actual_q0 = hwq._gonze_recip_dipole_dipole_q0(
        g_list, born, dielectric, positions, lambda_, tolerance
    )
    actual_limiting = hwq._gonze_limiting_dipole_dipole(dielectric, lambda_)
    np.testing.assert_allclose(actual_q0, np.load(static / "dd_q0.npy"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        actual_limiting, np.load(static / "dd_limiting.npy"), atol=1e-14, rtol=0.0
    )


def test_gonze_recip_real_and_mass_weight_terms_match_nacl_debug_reference():
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    q_dir = debug_dir / "q-00013"
    g_list = np.load(static / "G_list.npy")
    born = np.load(static / "born.npy")
    dielectric = np.load(static / "dielectric.npy")
    positions = np.load(static / "primitive_positions.npy")
    lambda_ = float(np.load(static / "Lambda.npy"))
    tolerance = 1e-5
    nac_factor = float(np.load(static / "nac_factor.npy"))
    q_red = np.load(q_dir / "q_red.npy")
    q_cart = np.load(q_dir / "q_cart.npy")
    q_direction_cart = np.load(q_dir / "q_direction_cart.npy")
    masses = np.load(static / "masses.npy")
    svecs = np.load(static / "svecs.npy")
    multi = np.load(static / "multi.npy")
    s2pp_map = np.load(static / "s2pp_map.npy")
    supercell_cell = np.load(static / "supercell_cell.npy")

    recip_dd_q0 = np.zeros((len(masses), 3, 3), dtype=np.complex128)
    dd_recip = hwq._gonze_recip_dipole_dipole(
        recip_dd_q0,
        g_list,
        q_cart,
        q_direction_cart,
        born,
        dielectric,
        positions,
        nac_factor,
        lambda_,
        tolerance,
    )
    dd_real = hwq._gonze_real_dipole_dipole(
        q_red, svecs, multi, s2pp_map, dielectric, lambda_, supercell_cell
    )
    dd_total = np.load(q_dir / "dd_total_mass_weighted.npy").reshape(2, 3, 2, 3).copy()
    for i in range(len(masses)):
        for j in range(len(masses)):
            dd_total[i, :, j, :] *= np.sqrt(masses[i] * masses[j])
    mass_weighted = hwq._gonze_mass_weight(dd_total, masses)

    np.testing.assert_allclose(dd_recip, np.load(q_dir / "dd_recip.npy"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(dd_real, np.load(q_dir / "dd_real.npy"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        mass_weighted, np.load(q_dir / "dd_total_mass_weighted.npy"), atol=1e-12, rtol=0.0
    )


def test_gonze_short_range_dynamical_matrix_matches_debug_reference():
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    q_dir = debug_dir / "q-00013"
    actual = hwq._gonze_short_range_dynamical_matrix(
        np.load(static / "short_range_force_constants.npy"),
        np.load(q_dir / "q_red.npy"),
        np.load(static / "svecs.npy"),
        np.load(static / "multi.npy"),
        np.load(static / "masses.npy"),
        np.load(static / "s2p_map.npy"),
        np.load(static / "p2s_map.npy"),
    )
    np.testing.assert_allclose(actual, np.load(q_dir / "dm_short.npy"), atol=1e-10, rtol=0.0)

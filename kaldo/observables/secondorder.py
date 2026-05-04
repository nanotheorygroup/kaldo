from kaldo.observables.forceconstant import ForceConstant, chi
from ase import Atoms
import math
import os
import tensorflow as tf
import ase.io
import numpy as np
from numpy.typing import ArrayLike
from kaldo.interfaces.eskm_io import import_from_files
import kaldo.interfaces.shengbte_io as shengbte_io
from kaldo.interfaces.tdep_io import parse_tdep_forceconstant
from kaldo.controllers.displacement import calculate_second
from kaldo.parallel import is_parallel, validate_parallel_calculator, maybe_warn_ml_delta_shift
import ase.units as units
from kaldo.helpers.logger import get_logger, log_size
from kaldo.storable import Storable, lazy_property
from kaldo.grid import Grid
from opt_einsum import contract
import spglib

logging = get_logger()

# ---------------------------------------------------------------------------
# BZ search space for fold_points_to_first_bz
# ---------------------------------------------------------------------------

_BZ_SEARCH_SPACE = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, -1],
        [0, 1, 0],
        [0, 1, 1],
        [1, -1, -1],
        [1, -1, 0],
        [1, -1, 1],
        [1, 0, -1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, -1],
        [1, 1, 0],
        [1, 1, 1],
        [-1, -1, -1],
        [-1, -1, 0],
        [-1, -1, 1],
        [-1, 0, -1],
        [-1, 0, 0],
        [-1, 0, 1],
        [-1, 1, -1],
        [-1, 1, 0],
        [-1, 1, 1],
        [0, -1, -1],
        [0, -1, 0],
        [0, -1, 1],
        [0, 0, -1],
    ],
    dtype=np.int64,
)

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _dielectric_part(q_cart, dielectric):
    return float(np.einsum("i,ij,j->", q_cart, dielectric, q_cart))


def _get_minimum_g_rad(reciprocal_lattice, g_cutoff, g_rad=100):
    for trial_g_rad in range(g_rad, 0, -1):
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    if (a, b, c) == (0, 0, 0):
                        continue
                    norm = np.linalg.norm(
                        reciprocal_lattice @ np.array([a, b, c], dtype=float)
                    )
                    if norm * trial_g_rad < g_cutoff:
                        return trial_g_rad + 1
    return g_rad


def _get_g_vec_list(reciprocal_lattice, g_rad):
    npts = g_rad * 2 + 1
    grid = np.array(list(np.ndindex((npts, npts, npts))), dtype=np.int64) - g_rad
    return np.array(grid @ reciprocal_lattice.T, dtype="double", order="C")


def _get_g_list(reciprocal_lattice, g_cutoff):
    g_rad = _get_minimum_g_rad(reciprocal_lattice, g_cutoff)
    g_vec_list = _get_g_vec_list(reciprocal_lattice, g_rad)
    g_norm2 = (g_vec_list ** 2).sum(axis=1)
    return np.array(g_vec_list[g_norm2 < g_cutoff ** 2], dtype="double", order="C")


def _multiply_borns(dd_in, born):
    born_t = np.transpose(born, (0, 2, 1))
    # Apply the Born-charge tensors on both Cartesian sides in one contraction.
    return np.einsum("iap,ipjq,jqb->iajb", born_t, dd_in, born, optimize=True)


def _get_dd_base(
    g_list,
    q_cart,
    q_direction_cart,
    dielectric,
    positions,
    lambda_,
    tolerance,
    pair_phase=None,
):
    if pair_phase is None:
        position_deltas = positions[:, None, :] - positions[None, :, :]
        phases = 2j * np.pi * np.einsum(
            "ga,ija->gij",
            g_list,
            position_deltas,
            optimize=True,
        )
        pair_phase = np.exp(phases)

    qk = g_list + q_cart[np.newaxis, :]
    norms = np.linalg.norm(qk, axis=1)
    kk = np.zeros((len(g_list), 3, 3), dtype=np.complex128)
    l2 = 4 * lambda_ * lambda_
    active = norms >= tolerance
    if np.any(active):
        denom = np.einsum("gi,ij,gj->g", qk[active], dielectric, qk[active], optimize=True)
        scale = np.exp(-denom / l2) / denom
        # Build all reciprocal-space dyads at once before contracting onto atom pairs.
        kk[active] = np.einsum("gi,gj,g->gij", qk[active], qk[active], scale, optimize=True)
    if q_direction_cart is not None:
        inactive = ~active
        if np.any(inactive):
            direction_denom = _dielectric_part(q_direction_cart, dielectric)
            direction_kk = np.outer(q_direction_cart, q_direction_cart) / direction_denom
            kk[inactive] = direction_kk
    return np.einsum("gab,gij->iajb", kk, pair_phase, optimize=True)


def _get_dd_base_many(
    g_list,
    q_carts,
    q_direction_carts,
    dielectric,
    positions,
    lambda_,
    tolerance,
    pair_phase=None,
):
    if pair_phase is None:
        position_deltas = positions[:, None, :] - positions[None, :, :]
        phases = 2j * np.pi * np.einsum(
            "ga,ija->gij",
            g_list,
            position_deltas,
            optimize=True,
        )
        pair_phase = np.exp(phases)

    q_carts = np.asarray(q_carts, dtype=float)
    qk = g_list[np.newaxis, :, :] + q_carts[:, np.newaxis, :]
    denom = np.einsum("qgi,ij,qgj->qg", qk, dielectric, qk, optimize=True)
    norms = np.linalg.norm(qk, axis=2)
    active = norms >= tolerance
    scale = np.zeros_like(denom, dtype=np.complex128)
    scale[active] = np.exp(-denom[active] / (4.0 * lambda_ * lambda_)) / denom[active]
    # Build all reciprocal-space dyads for every q-point and reciprocal vector.
    kk = np.einsum("qgi,qgj,qg->qgij", qk, qk, scale, optimize=True)
    if q_direction_carts is not None:
        q_direction_carts = np.asarray(q_direction_carts, dtype=float)
        direction_denom = np.einsum(
            "qi,ij,qj->q",
            q_direction_carts,
            dielectric,
            q_direction_carts,
            optimize=True,
        )
        direction_kk = np.einsum(
            "qi,qj,q->qij",
            q_direction_carts,
            q_direction_carts,
            1.0 / direction_denom,
            optimize=True,
        )
        kk += (~active)[..., np.newaxis, np.newaxis] * direction_kk[:, np.newaxis, :, :]
    return np.einsum("qgab,gij->qiajb", kk, pair_phase, optimize=True)


def _recip_dipole_dipole_q0(g_list, born, dielectric, positions, lambda_, tolerance):
    zero = np.zeros(3, dtype="double")
    dd_tmp1 = _get_dd_base(g_list, zero, None, dielectric, positions, lambda_, tolerance)
    dd_tmp2 = _multiply_borns(dd_tmp1, born)
    dd_q0 = dd_tmp2.sum(axis=2)
    dd_q0 = 0.5 * (dd_q0 + np.transpose(dd_q0.conj(), (0, 2, 1)))
    return dd_q0


def _limiting_dipole_dipole(dielectric, lambda_):
    inv_eps = np.linalg.inv(dielectric)
    sqrt_det_eps = np.sqrt(np.linalg.det(dielectric))
    return -4.0 / 3 / np.sqrt(np.pi) * inv_eps / sqrt_det_eps * lambda_ ** 3


def _real_dipole_dipole(q_red, svecs, multi, s2pp_map, dielectric, lambda_, supercell_cell):
    phase_all = np.exp(2j * np.pi * (svecs @ q_red))
    h = _h_tensor(supercell_cell, svecs, dielectric, lambda_)
    vals = -(lambda_ ** 3) * h * phase_all * np.linalg.det(dielectric) ** (-0.5)
    starts = multi[:, :, 1]
    multiplicities = multi[:, :, 0].astype(np.float64)
    gathered = vals[:, :, starts]
    symmetric_blocks = 0.5 * (
        gathered + np.transpose(gathered.conj(), (1, 0, 2, 3))
    )
    contributions = np.transpose(
        symmetric_blocks / multiplicities[np.newaxis, np.newaxis, :, :],
        (2, 3, 0, 1),
    )
    num_patom = multi.shape[1]
    c_real = np.zeros((num_patom, num_patom, 3, 3), dtype=np.complex128)
    np.add.at(c_real, s2pp_map, contributions)
    return np.transpose(c_real, (0, 2, 1, 3))


def _real_dipole_dipole_many(
    q_reds,
    svecs,
    multi,
    s2pp_map,
    dielectric,
    lambda_,
    supercell_cell,
    h_tensor=None,
    det_scale=None,
):
    q_reds = np.asarray(q_reds, dtype=float)
    if h_tensor is None:
        h_tensor = _h_tensor(supercell_cell, svecs, dielectric, lambda_)
    if det_scale is None:
        det_scale = np.linalg.det(dielectric) ** (-0.5)
    phase_all = np.exp(2j * np.pi * np.einsum("qa,sa->qs", q_reds, svecs, optimize=True))
    vals = -(lambda_ ** 3) * h_tensor[np.newaxis, :, :, :] * phase_all[:, np.newaxis, np.newaxis, :]
    vals *= det_scale
    starts = multi[:, :, 1]
    multiplicities = multi[:, :, 0].astype(np.float64)
    gathered = vals[:, :, :, starts]
    symmetric_blocks = 0.5 * (
        gathered + np.transpose(gathered.conj(), (0, 2, 1, 3, 4))
    )
    contributions = np.transpose(
        symmetric_blocks / multiplicities[np.newaxis, np.newaxis, np.newaxis, :, :],
        (0, 3, 4, 1, 2),
    )
    num_q = len(q_reds)
    num_patom = multi.shape[1]
    c_real = np.zeros((num_q, num_patom, num_patom, 3, 3), dtype=np.complex128)
    for i_q in range(num_q):
        np.add.at(c_real[i_q], s2pp_map, contributions[i_q])
    return np.transpose(c_real, (0, 1, 3, 2, 4))


def _mass_weight(fc_term, masses):
    mass_matrix = np.sqrt(np.outer(masses, masses))
    out = np.array(fc_term, dtype=np.complex128, copy=True)
    out /= mass_matrix[:, np.newaxis, :, np.newaxis]
    return out.reshape(len(masses) * 3, len(masses) * 3)


def _mass_weight_many(fc_terms, masses, mass_matrix=None):
    if mass_matrix is None:
        mass_matrix = np.sqrt(np.outer(masses, masses))
    out = np.array(fc_terms, dtype=np.complex128, copy=True)
    out /= mass_matrix[np.newaxis, :, np.newaxis, :, np.newaxis]
    return out.reshape(len(out), len(masses) * 3, len(masses) * 3)


def _build_segment_phase_weights(multi, n_svec):
    n_satom, n_patom = multi.shape[:2]
    weights = np.zeros((n_satom, n_patom, n_svec), dtype=np.float64)
    for i_s in range(n_satom):
        for i_p in range(n_patom):
            multiplicity = int(multi[i_s, i_p, 0])
            start = int(multi[i_s, i_p, 1])
            weights[i_s, i_p, start : start + multiplicity] = 1.0 / multiplicity
    return weights


def _short_range_dynamical_matrix(
    fc,
    q_red,
    svecs,
    multi,
    masses,
    s2p_map,
    p2s_map,
    phase_weights=None,
    target_mask=None,
):
    num_patom = len(p2s_map)
    is_compact_fc = fc.shape[0] != fc.shape[1]
    if phase_weights is None:
        phase_weights = _build_segment_phase_weights(multi, len(svecs))
    phase_all = np.exp(2j * np.pi * (svecs @ q_red))
    phase_factors = np.einsum("spl,l->sp", phase_weights, phase_all, optimize=True)
    if target_mask is None:
        target_mask = (
            s2p_map[:, np.newaxis] == p2s_map[np.newaxis, :]
        ).astype(np.complex128)
    fc_source = fc if is_compact_fc else fc[p2s_map]
    weighted_fc = fc_source * phase_factors.T[:, :, np.newaxis, np.newaxis]
    dm_blocks = np.einsum("isab,sj->ijab", weighted_fc, target_mask, optimize=True)
    dm_blocks /= np.sqrt(np.outer(masses, masses))[:, :, np.newaxis, np.newaxis]
    dm = np.transpose(dm_blocks, (0, 2, 1, 3)).reshape(num_patom * 3, num_patom * 3)
    return 0.5 * (dm + dm.conj().T)


def _short_range_dynamical_matrix_many(
    fc,
    q_reds,
    svecs,
    multi,
    masses,
    s2p_map,
    p2s_map,
    phase_weights=None,
    target_mask=None,
    mass_matrix=None,
):
    q_reds = np.asarray(q_reds, dtype=float)
    num_patom = len(p2s_map)
    is_compact_fc = fc.shape[0] != fc.shape[1]
    if phase_weights is None:
        phase_weights = _build_segment_phase_weights(multi, len(svecs))
    phase_all = np.exp(2j * np.pi * np.einsum("qa,sa->qs", q_reds, svecs, optimize=True))
    # Average all shortest-path phases for every q-point and primitive/supercell pair.
    phase_factors = np.einsum("spl,ql->qsp", phase_weights, phase_all, optimize=True)
    if target_mask is None:
        target_mask = (
            s2p_map[:, np.newaxis] == p2s_map[np.newaxis, :]
        ).astype(np.complex128)
    if mass_matrix is None:
        mass_matrix = np.sqrt(np.outer(masses, masses))
    fc_source = fc if is_compact_fc else fc[p2s_map]
    weighted_fc = (
        fc_source[np.newaxis, :, :, :, :]
        * np.transpose(phase_factors, (0, 2, 1))[:, :, :, np.newaxis, np.newaxis]
    )
    dm_blocks = np.einsum("qisab,sj->qijab", weighted_fc, target_mask, optimize=True)
    dm_blocks /= mass_matrix[np.newaxis, :, :, np.newaxis, np.newaxis]
    dm = np.transpose(dm_blocks, (0, 1, 3, 2, 4)).reshape(
        len(q_reds), num_patom * 3, num_patom * 3
    )
    return 0.5 * (dm + np.swapaxes(dm.conj(), 1, 2))


def _h_tensor(supercell_cell, svecs, dielectric, lambda_):
    cart_vecs = svecs @ supercell_cell
    eps_inv = np.linalg.inv(dielectric)
    delta = cart_vecs @ eps_inv.T
    d_norm = np.sqrt((cart_vecs * delta).sum(axis=1))
    x = lambda_ * delta
    y = lambda_ * d_norm
    condition = y < 1e-10
    y_safe = y.copy()
    y_safe[condition] = 1.0
    y2 = y_safe ** 2
    y3 = y_safe ** 3
    exp_y2 = np.exp(-y2)
    erfc_y = np.vectorize(math.erfc)(y_safe)
    a = np.where(
        condition,
        0.0,
        (3 * erfc_y / y3 + 2 / np.sqrt(np.pi) * exp_y2 * (3 / y2 + 2)) / y2,
    )
    b = np.where(condition, 0.0, erfc_y / y3 + 2 / np.sqrt(np.pi) * exp_y2 / y2)
    h = np.einsum("si,sj,s->ijs", x, x, a, optimize=True)
    h -= eps_inv[:, :, np.newaxis] * b[np.newaxis, np.newaxis, :]
    return h

# ---------------------------------------------------------------------------
# Utility functions (public)
# ---------------------------------------------------------------------------


def normalize_bvk_supercell_matrix(nac_bvk_supercell_matrix):
    if nac_bvk_supercell_matrix is None:
        return None
    matrix = np.array(nac_bvk_supercell_matrix, dtype=int)
    if matrix.shape != (3, 3):
        raise ValueError(
            "nac_bvk_supercell_matrix must be a 3x3 integer matrix."
        )
    determinant = int(round(np.linalg.det(matrix)))
    if determinant == 0:
        raise ValueError("nac_bvk_supercell_matrix must be non-singular.")
    return matrix


def bvk_supercell_matrix_key(nac_bvk_supercell_matrix):
    matrix = normalize_bvk_supercell_matrix(nac_bvk_supercell_matrix)
    if matrix is None:
        return None
    rows = []
    for row in matrix:
        rows.append("_".join(str(int(value)).replace("-", "m") for value in row))
    return "__".join(rows)

# ---------------------------------------------------------------------------
# Workflow helpers (private module-level)
# ---------------------------------------------------------------------------


def _diagonal_supercell_sort_key(supercell_scaled, n):
    """Sort key for diagonal supercell matrices: a-axis fastest, c-axis slowest (phonopy convention)."""
    rounded = np.round(np.asarray(supercell_scaled, dtype=float) * n).astype(int) % n
    return (rounded[2], rounded[1], rounded[0])


def _unique_supercell_translations(supercell_matrix, symprec=1e-8):
    supercell_matrix = np.array(supercell_matrix, dtype=int)
    primitive_matrix = np.linalg.inv(supercell_matrix)
    target_count = int(round(abs(np.linalg.det(supercell_matrix))))
    search_radius = int(np.max(np.abs(supercell_matrix))) + 2
    translations = []
    seen = set()
    for i in range(-search_radius, search_radius + 1):
        for j in range(-search_radius, search_radius + 1):
            for k in range(-search_radius, search_radius + 1):
                shift = np.array([i, j, k], dtype=float)
                supercell_scaled = (shift @ primitive_matrix) % 1.0
                supercell_scaled[np.isclose(supercell_scaled, 1.0, atol=symprec)] = 0.0
                key = tuple(np.round(supercell_scaled, 10))
                if key not in seen:
                    seen.add(key)
                    translations.append((shift, supercell_scaled))
    if len(translations) != target_count:
        raise ValueError(
            "Could not construct the expected number of supercell translations: "
            f"expected {target_count}, found {len(translations)}."
        )
    return translations


def _nacl_phonopy_debug_sort_key(supercell_scaled):
    rounded = np.round(np.asarray(supercell_scaled, dtype=float) * 4).astype(int) % 4
    quarter_axes = tuple(axis for axis, value in enumerate(rounded) if value % 2 == 1)
    if len(quarter_axes) in (0, 2):
        if len(quarter_axes) == 0:
            group = 0
        elif quarter_axes == (1, 2):
            group = 1
        elif quarter_axes == (0, 2):
            group = 2
        elif quarter_axes == (0, 1):
            group = 3
        else:
            group = 4
    else:
        if len(quarter_axes) == 3:
            group = 0
        elif quarter_axes == (0,):
            group = 1
        elif quarter_axes == (1,):
            group = 2
        elif quarter_axes == (2,):
            group = 3
        else:
            group = 4
    return (group, rounded[2], rounded[1], rounded[0])


def _phonopy_lattice_points():
    lattice_1d = (-1, 0, 1)
    lattice_4d = np.array(
        [
            [i, j, k, ll]
            for i in lattice_1d
            for j in lattice_1d
            for k in lattice_1d
            for ll in lattice_1d
        ],
        dtype=np.int64,
    )
    bases = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]],
        dtype=np.int64,
    )
    return np.array(np.unique(lattice_4d @ bases, axis=0), dtype=np.int64)


def _fold_points_to_first_bz(qpoints, reciprocal_lattice, tolerance=0.01):
    qpoints = np.array(qpoints, dtype=float, copy=True)
    reciprocal_lattice = np.array(reciprocal_lattice, dtype=float, copy=True)
    distance_tolerance = float(min(np.sum(reciprocal_lattice ** 2, axis=0)) * tolerance)
    folded = []
    for qpoint in qpoints:
        reduced = qpoint - np.rint(qpoint)
        candidates = reduced + _BZ_SEARCH_SPACE
        distances = np.sum((candidates @ reciprocal_lattice.T) ** 2, axis=1)
        min_distance = distances.min()
        shortest_indices = np.where(distances < min_distance + distance_tolerance)[0]
        folded.append(candidates[shortest_indices[0]])
    folded = np.array(folded, dtype="double", order="C")
    folded[np.isclose(folded, 0.0, atol=1e-14)] = 0.0
    return folded


def _commensurate_points(supercell, reciprocal_lattice=None):
    supercell = np.array(supercell)
    if supercell.shape == (3,):
        grid = Grid(supercell, order="C").grid(is_wrapping=False)
        qpoints = np.array(
            grid / np.array(supercell, dtype=float), dtype="double", order="C"
        )
        if reciprocal_lattice is not None:
            return _fold_points_to_first_bz(qpoints, reciprocal_lattice)
        return qpoints
    matrix = normalize_bvk_supercell_matrix(supercell)
    translations = _unique_supercell_translations(matrix.T)
    qpoints = np.array([translation[1] for translation in translations], dtype="double")
    if reciprocal_lattice is not None:
        return _fold_points_to_first_bz(qpoints, reciprocal_lattice)
    return np.array(qpoints, dtype="double", order="C")


def _dipole_dipole_dynamical_matrix(q_red, static_data, mapping, q_direction_red=None):
    q_red = np.array(q_red, dtype=float, copy=True)
    q_cart = static_data["reciprocal_lattice"] @ q_red
    if q_direction_red is None:
        if np.linalg.norm(q_cart) < static_data["q_direction_tolerance"]:
            q_direction_cart = None
        else:
            q_direction_cart = q_cart
    else:
        q_direction_cart = static_data["reciprocal_lattice"] @ np.array(q_direction_red, dtype=float)

    recip_dd_q0 = np.zeros_like(static_data["dd_q0"])
    dd_recip = _recip_dipole_dipole(
        recip_dd_q0,
        static_data["G_list"],
        q_cart,
        q_direction_cart,
        static_data["born"],
        static_data["dielectric"],
        static_data["primitive_positions"],
        float(static_data["nac_factor"]),
        float(static_data["Lambda"]),
        float(static_data["q_direction_tolerance"]),
        pair_phase=static_data.get("pair_phase"),
    )
    dd_real = _real_dipole_dipole(
        q_red,
        mapping["svecs"],
        mapping["multi"],
        mapping["s2pp_map"],
        static_data["dielectric"],
        float(static_data["Lambda"]),
        mapping.get("svecs_cell", static_data["supercell_cell"]),
    )
    dd_total = dd_recip + static_data["dd_limiting_expanded"] + dd_real
    dd_total[:, :, :, :] -= static_data["dd_drift_blocks"]
    conversion = units.mol / (10 * units.J)
    return _mass_weight(dd_total * conversion, static_data["masses"])


def _gonze_dynamical_matrices(q_reds, static_data, mapping, q_direction_carts, fc=None):
    q_reds = np.atleast_2d(np.asarray(q_reds, dtype=float))
    q_direction_carts = np.atleast_2d(np.asarray(q_direction_carts, dtype=float))
    if fc is None:
        fc = static_data["fc_short_converted"]

    q_carts = np.einsum(
        "ab,qb->qa",
        static_data["reciprocal_lattice"],
        q_reds,
        optimize=True,
    )
    dd_base = _get_dd_base_many(
        static_data["G_list"],
        q_carts,
        q_direction_carts,
        static_data["dielectric"],
        static_data["primitive_positions"],
        float(static_data["Lambda"]),
        float(static_data["q_direction_tolerance"]),
        pair_phase=static_data.get("pair_phase"),
    )
    born_t = np.transpose(static_data["born"], (0, 2, 1))
    dd_recip = np.einsum(
        "iap,xipjq,jqb->xiajb",
        born_t,
        dd_base,
        static_data["born"],
        optimize=True,
    )
    dd_recip *= float(static_data["nac_factor"])

    dd_real = _real_dipole_dipole_many(
        q_reds,
        mapping["svecs"],
        mapping["multi"],
        mapping["s2pp_map"],
        static_data["dielectric"],
        float(static_data["Lambda"]),
        mapping.get("svecs_cell", static_data["supercell_cell"]),
        h_tensor=static_data.get("h_tensor"),
        det_scale=static_data.get("real_dipole_det_scale"),
    )
    dd_total = dd_recip + static_data["dd_limiting_expanded"][np.newaxis, :, :, :, :] + dd_real
    dd_total -= static_data["dd_drift_blocks"][np.newaxis, :, :, :, :]
    dd_total_mass_weighted = _mass_weight_many(
        dd_total * static_data["gonze_conversion"],
        static_data["masses"],
        mass_matrix=static_data.get("sqrt_mass_matrix"),
    )

    dm_short = _short_range_dynamical_matrix_many(
        fc,
        q_reds,
        mapping.get("phase_svecs", mapping["svecs"]),
        mapping["multi"],
        static_data["masses"],
        mapping["s2p_map"],
        mapping["p2s_map"],
        phase_weights=mapping.get("phase_weights"),
        target_mask=mapping.get("target_mask"),
        mass_matrix=static_data.get("sqrt_mass_matrix"),
    )
    dm_final = dm_short + dd_total_mass_weighted
    return 0.5 * (dm_final + np.swapaxes(dm_final.conj(), 1, 2))


def _recip_dipole_dipole(
    dd_q0,
    g_list,
    q_cart,
    q_direction_cart,
    born,
    dielectric,
    positions,
    factor,
    lambda_,
    tolerance,
    pair_phase=None,
):
    dd_tmp = _get_dd_base(
        g_list,
        q_cart,
        q_direction_cart,
        dielectric,
        positions,
        lambda_,
        tolerance,
        pair_phase=pair_phase,
    )
    dd = _multiply_borns(dd_tmp, born)
    diag = np.arange(positions.shape[0])
    dd[diag, :, diag, :] -= dd_q0
    return dd * factor


def _inverse_transform_dynmats_to_force_constants(dynmats, qpoints, mapping, masses):
    s2pp_map = mapping["s2pp_map"]
    svecs = mapping.get("phase_svecs", mapping["svecs"])
    phase_weights = mapping["phase_weights"]
    p2s_map = mapping["p2s_map"]
    n_p = len(p2s_map)
    n_q = len(qpoints)
    phase_all = np.exp(-2j * np.pi * np.dot(qpoints, svecs.T))
    phase_factors = np.einsum("spl,ql->qsp", phase_weights, phase_all, optimize=True)
    dyn_blocks = dynmats.reshape(n_q, n_p, 3, n_p, 3)
    gathered_blocks = dyn_blocks[:, :, :, s2pp_map, :]
    fc = np.einsum("qiasb,qsi->isab", gathered_blocks, phase_factors, optimize=True)
    coef = np.sqrt(np.outer(masses, masses[s2pp_map])) / n_q
    return (fc * coef[:, :, np.newaxis, np.newaxis]).real / (units.mol / (10 * units.J))


def _build_interleaved_fc(second_order):
    """Convert kALDo C-order FC to type-grouped compact format for phonopy-style FT.

    kALDo's ShengBTE-QE FC stores val[j, β, l_C, i, α] = force on i at cell t(l_C)
    due to displacement of j at cell 0 (R = cell of force-atom convention). The
    phonopy standard uses R = cell of displaced atom, so the translation is negated:
      k_F = (-t1 % n1) + (-t2 % n2)*n1 + (-t3 % n3)*(n1*n2)

    Returns compact FC array of shape (n_atom, n_atom*n_replicas, 3, 3) in eV/Å².
    """
    val = second_order.value[0]  # (j, β, l_C, i, α)
    n_atom = len(second_order.atoms)
    n1, n2, n3 = second_order.supercell
    n_replicas = n1 * n2 * n3
    lC = np.arange(n_replicas)
    t1 = lC // (n2 * n3)
    t2 = (lC // n3) % n2
    t3 = lC % n3
    k_F = (-t1 % n1) + (-t2 % n2) * n1 + (-t3 % n3) * (n1 * n2)
    fc = np.zeros((n_atom, n_atom * n_replicas, 3, 3), dtype=float)
    for j_type in range(n_atom):
        for l_C_idx in range(n_replicas):
            k = j_type * n_replicas + k_F[l_C_idx]
            for i_type in range(n_atom):
                fc[i_type, k, :, :] = val[j_type, :, l_C_idx, i_type, :].T
    return fc


def _fc_short_translation_data(mapping, symprec=1e-8):
    p2s_map = mapping["p2s_map"]
    n_translation = int(mapping["multi"].shape[0] // len(p2s_map))
    translations = np.array(
        mapping["primitive_shifts"][p2s_map[0] : p2s_map[0] + n_translation],
        dtype=float,
        copy=True,
    )
    primitive_matrix = np.asarray(mapping["primitive_matrix"], dtype=float)
    reduced = (translations @ primitive_matrix) % 1.0
    reduced[np.isclose(reduced, 1.0, atol=symprec)] = 0.0
    lookup = {
        tuple(np.round(reduced[i_translation], 10)): i_translation
        for i_translation in range(n_translation)
    }
    return translations, lookup


def _cartesian_rotation_from_fractional(rotation, primitive_cell):
    primitive_cell = np.asarray(primitive_cell, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    # Fractional coordinates transform as row vectors x' = x R^T, so the
    # corresponding Cartesian operator on row-vector displacements is the
    # inverse-space action used below when rotating FC blocks.
    return np.linalg.inv(primitive_cell) @ np.linalg.inv(rotation).T @ primitive_cell


def _build_fc_short_symmetry_data(atoms, mapping, symprec=1e-5):
    primitive_scaled = np.array(
        mapping["primitive_scaled_positions"],
        dtype=float,
        copy=True,
    )
    primitive_cell = np.array(atoms.cell.array, dtype=float, copy=True)
    symmetry = spglib.get_symmetry(
        (primitive_cell, primitive_scaled, atoms.numbers),
        symprec=symprec,
    )
    translations, translation_lookup = _fc_short_translation_data(mapping, symprec=symprec)
    operations = []
    for rotation, translation in zip(symmetry["rotations"], symmetry["translations"]):
        rotation = np.array(rotation, dtype=np.int64, copy=True)
        translation = np.array(translation, dtype=float, copy=True)
        cart_rotation = _cartesian_rotation_from_fractional(rotation, primitive_cell)
        atom_map = np.zeros(len(primitive_scaled), dtype=np.int64)
        atom_shifts = np.zeros((len(primitive_scaled), 3), dtype=np.int64)
        for i_atom, position in enumerate(primitive_scaled):
            transformed = position @ rotation.T + translation
            deltas = transformed - primitive_scaled
            wrapped = deltas - np.rint(deltas)
            distances = np.linalg.norm(wrapped, axis=1)
            j_atom = int(np.argmin(distances))
            if distances[j_atom] >= symprec:
                raise ValueError(
                    "Could not map primitive atom under symmetry operation while "
                    "building Gonze short-range FC symmetrizer."
                )
            atom_map[i_atom] = j_atom
            atom_shifts[i_atom] = np.rint(deltas[j_atom]).astype(np.int64)
        operations.append(
            {
                "rotation": rotation,
                "cart_rotation": cart_rotation,
                "atom_map": atom_map,
                "atom_shifts": atom_shifts,
            }
        )
    return {
        "operations": operations,
        "translations": translations,
        "translation_lookup": translation_lookup,
    }


def _symmetrize_fc_short(fc, atoms, mapping, symprec=1e-5):
    symmetry_data = mapping.get("fc_short_symmetry_data")
    if symmetry_data is None:
        symmetry_data = _build_fc_short_symmetry_data(atoms, mapping, symprec=symprec)
        mapping["fc_short_symmetry_data"] = symmetry_data

    operations = symmetry_data["operations"]
    if len(operations) <= 1:
        return np.array(fc, dtype=float, copy=True)

    fc = np.asarray(fc, dtype=float)
    n_atom = len(atoms)
    n_translation = fc.shape[1] // n_atom
    translations = np.asarray(symmetry_data["translations"], dtype=float)
    translation_lookup = symmetry_data["translation_lookup"]
    primitive_matrix = np.asarray(mapping["primitive_matrix"], dtype=float)
    symmetrized = np.zeros_like(fc, dtype=np.float64)

    for operation in operations:
        rotation = operation["rotation"]
        cart_rotation = operation["cart_rotation"]
        atom_map = operation["atom_map"]
        atom_shifts = operation["atom_shifts"]
        rotated_translations = translations @ rotation.T
        for i_atom in range(n_atom):
            target_i_atom = int(atom_map[i_atom])
            shift_i = atom_shifts[i_atom]
            for j_atom in range(n_atom):
                target_j_atom = int(atom_map[j_atom])
                transformed_shift = (
                    rotated_translations
                    + atom_shifts[j_atom][np.newaxis, :]
                    - shift_i[np.newaxis, :]
                )
                reduced = (transformed_shift @ primitive_matrix) % 1.0
                reduced[np.isclose(reduced, 1.0, atol=symprec)] = 0.0
                target_translation_indices = np.fromiter(
                    (
                        translation_lookup[tuple(np.round(reduced_row, 10))]
                        for reduced_row in reduced
                    ),
                    dtype=np.int64,
                    count=n_translation,
                )
                transformed_blocks = np.einsum(
                    "ac,tcd,bd->tab",
                    cart_rotation,
                    fc[
                        i_atom,
                        j_atom * n_translation : (j_atom + 1) * n_translation,
                    ],
                    cart_rotation,
                    optimize=True,
                )
                symmetrized[
                    target_i_atom,
                    target_j_atom * n_translation + target_translation_indices,
                ] += transformed_blocks

    return symmetrized / len(operations)


def _dynamical_matrix_from_second_order(second_order, q_red):
    """Compute the dynamical matrix at q_red directly from second_order force constants."""
    n_atom = len(second_order.atoms)
    dynmat = second_order.dynmat
    dyn_s = contract(
        "ialjb,l->iajb",
        dynmat.numpy()[0].astype(np.complex128),
        chi(np.asarray(q_red, dtype=float), second_order.list_of_replicas, second_order.cell_inv).flatten(),
        backend="numpy",
    )
    scaled_positions = second_order.atoms.get_scaled_positions(wrap=False)
    for i_atom in range(n_atom):
        for j_atom in range(n_atom):
            phase = np.exp(
                2j * np.pi * np.dot(q_red, scaled_positions[j_atom] - scaled_positions[i_atom])
            )
            dyn_s[i_atom, :, j_atom, :] *= phase
    return dyn_s.reshape(n_atom * 3, n_atom * 3)


def _build_supercell_matrix_mapping(atoms, supercell_matrix, symprec=1e-5):
    supercell_matrix = np.array(supercell_matrix, dtype=int)
    primitive_matrix = np.linalg.inv(supercell_matrix)
    primitive_cell = np.array(atoms.cell.array, dtype=float, copy=True)
    supercell_cell = supercell_matrix @ primitive_cell
    primitive_scaled = atoms.get_scaled_positions(wrap=False)
    _diag = np.diag(supercell_matrix)
    if np.all(supercell_matrix == np.diag(_diag)):
        _n = int(np.max(np.abs(_diag)))
        _max_denom = 1
        for pos in primitive_scaled:
            for coord in pos:
                frac = coord % 1.0
                if frac > 1e-10:
                    for d in range(1, 64):
                        if abs(round(frac * d) / d - frac) < 1e-8:
                            if d > _max_denom:
                                _max_denom = d
                            break
        _factor = _n * _max_denom
        _sort_key = lambda pos: _diagonal_supercell_sort_key(pos, _factor)
    else:
        _sort_key = _nacl_phonopy_debug_sort_key
    translations = _unique_supercell_translations(supercell_matrix, symprec=symprec)
    translations = sorted(translations, key=lambda item: _sort_key(item[1]))
    n_translation = len(translations)
    n_atom = len(atoms)

    supercell_scaled_positions = np.zeros((n_atom * n_translation, 3), dtype=float)
    primitive_shifts = np.zeros((n_atom * n_translation, 3), dtype=float)
    s2p_map = np.zeros(n_atom * n_translation, dtype=np.int64)
    s2pp_map = np.zeros(n_atom * n_translation, dtype=np.int64)
    p2s_map = np.arange(n_atom, dtype=np.int64) * n_translation
    p2p_map = {int(p2s_map[i]): i for i in range(n_atom)}
    for i_atom in range(n_atom):
        for i_translation, (shift, _) in enumerate(translations):
            index = i_atom * n_translation + i_translation
            supercell_scaled = ((primitive_scaled[i_atom] + shift) @ primitive_matrix) % 1.0
            supercell_scaled[np.isclose(supercell_scaled, 1.0, atol=symprec)] = 0.0
            supercell_scaled_positions[index] = supercell_scaled
            primitive_shifts[index] = shift
            s2p_map[index] = p2s_map[i_atom]
            s2pp_map[index] = i_atom
    primitive_positions_in_supercell = supercell_scaled_positions[p2s_map]
    lattice_points = _phonopy_lattice_points()
    svecs = []
    phase_svecs = []
    multi = np.zeros((len(supercell_scaled_positions), n_atom, 2), dtype=np.int64)
    for i_s, supercell_position in enumerate(supercell_scaled_positions):
        for i_p, primitive_position in enumerate(primitive_positions_in_supercell):
            candidates_supercell = supercell_position - primitive_position + lattice_points
            distances = np.linalg.norm(candidates_supercell @ supercell_cell, axis=1)
            min_distance = distances.min()
            start = len(svecs)
            for vec_supercell, distance in zip(candidates_supercell, distances):
                if abs(distance - min_distance) < symprec:
                    svecs.append(vec_supercell)
                    phase_svecs.append(vec_supercell @ supercell_matrix)
            multi[i_s, i_p, 0] = len(svecs) - start
            multi[i_s, i_p, 1] = start

    svecs = np.array(svecs, dtype=float)
    phase_svecs = np.array(phase_svecs, dtype=float)
    return {
        "supercell_matrix": supercell_matrix,
        "primitive_matrix": primitive_matrix,
        "supercell_cell": supercell_cell,
        "primitive_scaled_positions": primitive_scaled,
        "supercell_scaled_positions": supercell_scaled_positions,
        "primitive_shifts": np.array(primitive_shifts, dtype=float),
        "svecs": svecs,
        "multi": multi,
        "p2s_map": p2s_map,
        "s2p_map": s2p_map,
        "p2p_map": p2p_map,
        "s2pp_map": s2pp_map,
        "svecs_cell": supercell_cell,
        "phase_svecs": phase_svecs,
    }


def _ensure_gonze_kernel_cache(static_data, mapping):
    if "pair_phase" not in static_data:
        position_deltas = (
            static_data["primitive_positions"][:, np.newaxis, :]
            - static_data["primitive_positions"][np.newaxis, :, :]
        )
        phases = 2j * np.pi * np.einsum(
            "ga,ija->gij",
            static_data["G_list"],
            position_deltas,
            optimize=True,
        )
        static_data["pair_phase"] = np.exp(phases)
    if "dd_limiting_expanded" not in static_data:
        n_atom = len(static_data["masses"])
        dd_limiting_expanded = np.zeros((n_atom, 3, n_atom, 3), dtype=np.complex128)
        diag = np.arange(n_atom)
        dd_limiting_expanded[diag, :, diag, :] = static_data["dd_limiting"]
        static_data["dd_limiting_expanded"] = dd_limiting_expanded
    if "dd_drift_blocks" not in static_data:
        dd_drift_blocks = np.zeros_like(static_data["dd_limiting_expanded"])
        diag = np.arange(len(static_data["masses"]))
        dd_drift_blocks[diag, :, diag, :] = static_data["dd_drift"]
        static_data["dd_drift_blocks"] = dd_drift_blocks
    if "h_tensor" not in static_data:
        static_data["h_tensor"] = _h_tensor(
            mapping.get("svecs_cell", static_data["supercell_cell"]),
            mapping["svecs"],
            static_data["dielectric"],
            float(static_data["Lambda"]),
        )
    if "real_dipole_det_scale" not in static_data:
        static_data["real_dipole_det_scale"] = np.linalg.det(static_data["dielectric"]) ** (-0.5)
    if "sqrt_mass_matrix" not in static_data:
        static_data["sqrt_mass_matrix"] = np.sqrt(
            np.outer(static_data["masses"], static_data["masses"])
        )
    if "gonze_conversion" not in static_data:
        static_data["gonze_conversion"] = units.mol / (10 * units.J)
    if "phase_weights" not in mapping:
        phase_svecs = mapping.get("phase_svecs", mapping["svecs"])
        mapping["phase_weights"] = _build_segment_phase_weights(
            mapping["multi"], len(phase_svecs)
        )
    if "target_mask" not in mapping:
        mapping["target_mask"] = (
            mapping["s2p_map"][:, np.newaxis] == mapping["p2s_map"][np.newaxis, :]
        ).astype(np.complex128)
    return static_data, mapping


def acoustic_sum_rule(dynmat):
    n_unit = dynmat[0].shape[0]
    sumrulecorr = 0.0
    for i in range(n_unit):
        off_diag_sum = np.sum(dynmat[0, i, :, :, :, :], axis=(-2, -3))
        dynmat[0, i, :, 0, i, :] -= off_diag_sum
        sumrulecorr += np.sum(off_diag_sum)
    logging.info("error sum rule: " + str(sumrulecorr))
    return dynmat


class SecondOrder(ForceConstant, Storable):
    _store_formats = {
        "gonze_short_range_force_constants": "numpy",
    }

    def __init__(self, value: ArrayLike, is_acoustic_sum: bool = False, *kargs, **kwargs):
        # apply acoustic sum rule before initialize in forceconstnat
        self.is_acoustic_sum = is_acoustic_sum
        if is_acoustic_sum:
            value = acoustic_sum_rule(value)

        super().__init__(value=value, *kargs, **kwargs)

        self.n_modes = self.atoms.positions.shape[0] * 3
        self._list_of_replicas = None  # TODO: why overwrite _list_of_replicas here?
        self._gonze_nac_precomputed_cache = {}
        self._gonze_short_range_force_constants_cache = {}
        self.storage = "numpy"

    @lazy_property(label="", format="numpy")
    def gonze_short_range_force_constants(self):
        return self.calculate_gonze_short_range_force_constants()

    def get_gonze_short_range_force_constants(self, nac_bvk_supercell_matrix=None):
        matrix = normalize_bvk_supercell_matrix(nac_bvk_supercell_matrix)
        if matrix is None:
            return self.gonze_short_range_force_constants

        key = bvk_supercell_matrix_key(matrix)
        if key in self._gonze_short_range_force_constants_cache:
            return self._gonze_short_range_force_constants_cache[key]

        property_name = "gonze_short_range_force_constants_" + key
        folder = self.get_folder_from_label("")
        try:
            loaded = self._load_property(property_name, folder, format="numpy")
            logging.info("Loading " + folder + "/" + property_name)
            loaded = _symmetrize_fc_short(
                loaded,
                self.atoms,
                self.get_gonze_nac_precomputed(matrix)["mapping"],
            )
            self._gonze_short_range_force_constants_cache[key] = loaded
            return loaded
        except (FileNotFoundError, OSError, KeyError):
            logging.info(
                folder + "/" + property_name
                + " not found in numpy format, calculating "
                + property_name
            )
            force_constants = self.calculate_gonze_short_range_force_constants(matrix)
            self._save_property(property_name, folder, force_constants, format="numpy")
            self._gonze_short_range_force_constants_cache[key] = force_constants
            return force_constants

    def get_gonze_nac_precomputed(self, nac_bvk_supercell_matrix=None):
        matrix = normalize_bvk_supercell_matrix(nac_bvk_supercell_matrix)
        key = bvk_supercell_matrix_key(matrix) if matrix is not None else "default"
        if key not in self._gonze_nac_precomputed_cache:
            static_data = self._gonze_build_static_data(matrix)
            mapping = self._gonze_build_mapping(matrix)
            dd_real_q0_full = _real_dipole_dipole(
                np.zeros(3, dtype=float),
                mapping["svecs"],
                mapping["multi"],
                mapping["s2pp_map"],
                static_data["dielectric"],
                float(static_data["Lambda"]),
                mapping.get("svecs_cell", static_data["supercell_cell"]),
            )
            dd_real_q0 = dd_real_q0_full.sum(axis=2)
            static_data["dd_drift"] = (
                static_data["dd_q0"] * float(units.Rydberg / units.Bohr ** 2)
                + static_data["dd_limiting"] * len(static_data["masses"])
                + dd_real_q0
            )
            static_data, mapping = _ensure_gonze_kernel_cache(static_data, mapping)
            self._gonze_nac_precomputed_cache[key] = {
                "static_data": static_data,
                "mapping": mapping,
            }
        return self._gonze_nac_precomputed_cache[key]

    def calculate_gonze_short_range_force_constants(self, nac_bvk_supercell_matrix=None):
        if "dielectric" not in self.atoms.info or "charges" not in self.atoms.arrays:
            raise ValueError(
                "Gonze-Lee short-range force constants require atoms.info['dielectric'] "
                "and atoms.arrays['charges']."
            )
        matrix = normalize_bvk_supercell_matrix(nac_bvk_supercell_matrix)
        supercell = self.supercell if matrix is None else matrix
        bundle = self.get_gonze_nac_precomputed(matrix)
        static_data = bundle["static_data"]
        mapping = bundle["mapping"]
        qpoints = _commensurate_points(supercell, static_data["reciprocal_lattice"])
        dynmats = np.zeros(
            (len(qpoints), len(self.atoms) * 3, len(self.atoms) * 3),
            dtype=np.complex128,
        )
        logging.info(
            "Calculating Gonze-Lee short-range force constants from "
            + str(len(qpoints))
            + " commensurate q-points."
        )
        fc_full = _build_interleaved_fc(self)
        conversion = units.mol / (10 * units.J)
        svecs = mapping.get("phase_svecs", mapping["svecs"])
        fc_full_converted = fc_full * conversion
        for i_q, q_point in enumerate(qpoints):
            dynmat = _short_range_dynamical_matrix(
                fc_full_converted,
                q_point,
                svecs,
                mapping["multi"],
                static_data["masses"],
                mapping["s2p_map"],
                mapping["p2s_map"],
                phase_weights=mapping["phase_weights"],
                target_mask=mapping["target_mask"],
            )
            dynmat -= _dipole_dipole_dynamical_matrix(q_point, static_data, mapping)
            dynmats[i_q] = (dynmat + dynmat.conj().T) / 2
        force_constants = _inverse_transform_dynmats_to_force_constants(
            dynmats, qpoints, mapping, static_data["masses"]
        )
        return _symmetrize_fc_short(force_constants, self.atoms, mapping)

    def _gonze_build_static_data(self, matrix=None):
        atoms = self.atoms
        born = np.array(atoms.get_array("charges"), dtype=float, copy=True)
        dielectric = np.array(atoms.info["dielectric"], dtype=float, copy=True)
        primitive_cell = np.array(atoms.cell.array, dtype=float, copy=True)
        primitive_positions = np.array(atoms.positions, dtype=float, copy=True)
        reciprocal_lattice = np.array(atoms.cell.reciprocal(), dtype=float, copy=True)
        masses = np.array(atoms.get_masses(), dtype=float, copy=True)
        matrix = normalize_bvk_supercell_matrix(matrix)
        if matrix is None:
            supercell_cell = np.array(self.replicated_atoms.cell.array, dtype=float, copy=True)
        else:
            supercell_cell = np.array(matrix @ primitive_cell, dtype=float, copy=True)
        volume = float(abs(np.linalg.det(primitive_cell)))
        num_g_points = 300
        g_cutoff = float((3 * num_g_points / (4 * np.pi) / volume) ** (1.0 / 3))
        exp_cutoff = 1e-10
        geg = g_cutoff ** 2 * np.trace(dielectric) / 3
        lambda_ = float(np.sqrt(-geg / 4 / np.log(exp_cutoff)))
        unit_conversion_factor = 14.4
        nac_factor = float(unit_conversion_factor * 4 * np.pi / volume)
        tolerance = 1e-5
        g_list = _get_g_list(reciprocal_lattice, g_cutoff)
        dd_q0 = _recip_dipole_dipole_q0(
            g_list, born, dielectric, primitive_positions, lambda_, tolerance
        )
        dd_limiting = _limiting_dipole_dipole(dielectric, lambda_)
        return {
            "born": born,
            "dielectric": dielectric,
            "primitive_cell": primitive_cell,
            "primitive_positions": primitive_positions,
            "reciprocal_lattice": reciprocal_lattice,
            "masses": masses,
            "supercell_cell": supercell_cell,
            "volume": np.array(volume),
            "Lambda": np.array(lambda_),
            "G_cutoff": np.array(g_cutoff),
            "G_list": g_list,
            "unit_conversion_factor": np.array(unit_conversion_factor),
            "nac_factor": np.array(nac_factor),
            "q_direction_tolerance": np.array(tolerance),
            "dd_q0": dd_q0,
            "dd_limiting": dd_limiting,
        }

    def _gonze_build_mapping(self, matrix=None):
        matrix = normalize_bvk_supercell_matrix(matrix)
        if matrix is not None:
            return _build_supercell_matrix_mapping(self.atoms, matrix)
        atoms = self.atoms
        n_atom = len(atoms)
        wrapped_indices = self._direct_grid.grid(is_wrapping=True)
        s2p_map = np.tile(np.arange(n_atom, dtype=int), len(wrapped_indices))
        p2s_map = np.arange(n_atom, dtype=int)
        s2pp_map = s2p_map.copy()
        supercell = np.array(self.supercell, dtype=float)
        primitive_cell = np.array(atoms.cell.array, dtype=float, copy=True)
        supercell_cell = np.array(self.replicated_atoms.cell.array, dtype=float, copy=True)
        svecs = []
        phase_svecs = []
        multi = np.zeros((len(s2p_map), n_atom, 2), dtype=np.int64)
        primitive_scaled = atoms.get_scaled_positions(wrap=False)
        for i_s, atom_j in enumerate(s2p_map):
            wrapped_index = wrapped_indices[i_s // n_atom]
            primitive_scaled_j = primitive_scaled[atom_j] + wrapped_index
            for i_p in range(n_atom):
                primitive_scaled_i = primitive_scaled[i_p]
                candidates = []
                distances = []
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        for c in (-1, 0, 1):
                            shift = np.array([a, b, c], dtype=float)
                            phase_vec = primitive_scaled_j - primitive_scaled_i + shift * supercell
                            vec = phase_vec / supercell
                            cart = vec @ supercell_cell
                            candidates.append((vec, phase_vec))
                            distances.append(np.linalg.norm(cart))
                min_distance = min(distances)
                start = len(svecs)
                for (vec, phase_vec), distance in zip(candidates, distances):
                    if abs(distance - min_distance) < 1e-8:
                        svecs.append(vec)
                        phase_svecs.append(phase_vec)
                multi[i_s, i_p, 0] = len(svecs) - start
                multi[i_s, i_p, 1] = start
        return {
            "svecs": np.array(svecs, dtype=float),
            "phase_svecs": np.array(phase_svecs, dtype=float),
            "multi": multi,
            "s2p_map": s2p_map,
            "p2s_map": p2s_map,
            "s2pp_map": s2pp_map,
            "svecs_cell": supercell_cell,
        }

    @classmethod
    def from_supercell(cls,
                       atoms: Atoms,
                       grid_type: str,
                       supercell: tuple[int, int, int] = None,
                       value: ArrayLike | None = None,
                       is_acoustic_sum: bool = False,
                       folder: str = "kALDo"):
        # acoustic sum rule will be applied later in SecondOrder.__init__ if applicable
        ifc = super().from_supercell(
            atoms=atoms,
            supercell=supercell,
            grid_type=grid_type,
            value=value,
            is_acoustic_sum=is_acoustic_sum,
            folder=folder)
        return ifc

    @classmethod
    def load(cls,
             folder: str,
             supercell: tuple[int, int, int] = (1, 1, 1),
             format: str = "numpy",
             is_acoustic_sum: bool = False):
        """
        Load second order force constants from a folder in the given format, used for library internally.

        To load force constants data, ``ForceConstants.from_folder`` is recommended.

        Parameters
        ----------
        folder : str
            Specifies where to load the data files.
        supercell : tuple[int, int, int]
            The supercell for the third order force constant matrix.
            Default: (1, 1, 1)
        format : str
            Format of the second order force constant information being loaded into SecondOrder object.
            Default: 'sparse'
        is_acoustic_sum : bool, optional
            If true, the acoustic sum rule is applied to the dynamical matrix.
            Default: False

        Returns
        -------
        second_order : SecondOrder object
            A new instance of the SecondOrder class
        """

        match format:
            case "numpy":
                replicated_atoms_file = "replicated_atoms.xyz"
                config_file = os.path.join(folder, replicated_atoms_file)
                replicated_atoms = ase.io.read(config_file, format="extxyz")

                n_replicas = np.prod(supercell)
                n_total_atoms = replicated_atoms.positions.shape[0]
                n_unit_atoms = int(n_total_atoms / n_replicas)
                unit_symbols = []
                unit_positions = []
                for i in range(n_unit_atoms):
                    unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                    unit_positions.append(replicated_atoms.positions[i])
                unit_cell = replicated_atoms.cell / supercell

                atoms = Atoms(unit_symbols, positions=unit_positions, cell=unit_cell, pbc=[1, 1, 1])

                _second_order = np.load(os.path.join(folder, "second.npy"), allow_pickle=True)
                second_order = SecondOrder(
                    atoms=atoms,
                    replicated_positions=replicated_atoms.positions,
                    supercell=supercell,
                    value=_second_order,
                    is_acoustic_sum=is_acoustic_sum,
                    folder=folder,
                )

            case "eskm" | "lammps":
                dynmat_file = os.path.join(folder, "Dyn.form")
                if format == "eskm":
                    config_file = os.path.join(folder, "CONFIG")
                    replicated_atoms = ase.io.read(config_file, format="dlp4")
                elif format == "lammps":
                    config_file = os.path.join(folder, "replicated_atoms.xyz")
                    replicated_atoms = ase.io.read(config_file, format="extxyz")
                n_replicas = np.prod(supercell)
                n_total_atoms = replicated_atoms.positions.shape[0]
                n_unit_atoms = int(n_total_atoms / n_replicas)
                unit_symbols = []
                unit_positions = []
                for i in range(n_unit_atoms):
                    unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                    unit_positions.append(replicated_atoms.positions[i])
                unit_cell = replicated_atoms.cell / supercell

                atoms = Atoms(unit_symbols, positions=unit_positions, cell=unit_cell, pbc=[1, 1, 1])

                _second_order, _ = import_from_files(
                    replicated_atoms=replicated_atoms, dynmat_file=dynmat_file, supercell=supercell
                )
                second_order = SecondOrder(
                    atoms=atoms,
                    replicated_positions=replicated_atoms.positions,
                    supercell=supercell,
                    value=_second_order,
                    is_acoustic_sum=is_acoustic_sum,
                    folder=folder,
                )

            case ("vasp-sheng" | "shengbte") | ("qe-sheng" | "shengbte-qe") | ("qe-d3q" | "shengbte-d3q") | "vasp-d3q":
                config_file = os.path.join(folder, "CONTROL")
                try:
                    atoms, supercell, charges = shengbte_io.import_control_file(config_file)
                    if charges is not None:
                        atoms.info['dielectric'] = charges[0, :, :]
                        atoms.set_array('charges', charges[1:, :, :], shape=(3, 3))
                except FileNotFoundError:
                    config_file = os.path.join(folder, "POSCAR")
                    logging.info("Trying to open POSCAR")
                    atoms = ase.io.read(config_file)

                # Create a finite difference object
                # TODO: we need to read the grid type here
                n_replicas = np.prod(supercell)
                n_unit_atoms = atoms.positions.shape[0]
                match format:
                    case ("qe-sheng" | "shengbte-qe") | ("qe-d3q" | "shengbte-d3q"):
                        # load QE second order force constant
                        filename = os.path.join(folder, "espresso.ifc2")
                        if not os.path.isfile(filename):
                            raise FileNotFoundError(f"File {filename} not found.")
                        _second_order, supercell, charges = shengbte_io.read_second_order_qe_matrix(filename)
                        if (not charges is None):
                            atoms.info['dielectric'] = charges[0, :, :]
                            atoms.set_array('charges', charges[1:, :, :], shape=(3, 3))
                        _second_order = _second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
                        _second_order = _second_order.transpose(3, 4, 2, 0, 1)
                        grid_type = "F"
                    case _:
                        # load VASP second order force constant
                        filename = os.path.join(folder, "FORCE_CONSTANTS_2ND")
                        if not os.path.isfile(filename):
                            filename = os.path.join(folder, "FORCE_CONSTANTS")
                        if not os.path.isfile(filename):
                            raise FileNotFoundError(f"File {filename} not found.")
                        _second_order = shengbte_io.read_second_order_matrix(filename, supercell)
                        _second_order = _second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
                        grid_type = "F"
                second_order = SecondOrder.from_supercell(
                    atoms=atoms,
                    grid_type=grid_type,
                    supercell=supercell,
                    value=_second_order[np.newaxis, ...],
                    is_acoustic_sum=is_acoustic_sum,
                    folder=folder,
                )

            case "hiphive":
                filename = "atom_prim.xyz"
                # TODO: add replicated filename in example
                replicated_filename = "replicated_atoms.xyz"
                try:
                    import kaldo.interfaces.hiphive_io as hiphive_io
                except ImportError:
                    logging.error(
                        "In order to use hiphive along with kaldo, hiphive is required. \
                        Please consider installing hihphive. More info can be found at: \
                        https://hiphive.materialsmodeling.org/"
                    )

                atom_prime_file = os.path.join(folder, filename)
                replicated_atom_prime_file = os.path.join(folder, replicated_filename)
                # TODO: Make this independent of replicated file
                atoms = ase.io.read(atom_prime_file)
                try:
                    replicated_atoms = ase.io.read(replicated_atom_prime_file)
                except FileNotFoundError:
                    logging.warning(
                        "Replicated atoms file not found. Please check if the file exists. Using the unit cell atoms instead."
                    )
                    replicated_atoms = atoms * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])
                # Create a finite difference object
                if "model2.fcs" in os.listdir(folder):
                    _second_order = hiphive_io.import_second_from_hiphive(
                        folder, np.prod(supercell), atoms.positions.shape[0]
                    )
                    second_order = SecondOrder(
                        atoms=atoms,
                        replicated_positions=replicated_atoms.positions,
                        supercell=supercell,
                        value=_second_order,
                        folder=folder,
                    )

            case "tdep":
                uc_filename = "infile.ucposcar"
                replicated_filename = "infile.ssposcar"
                atom_prime_file = os.path.join(folder, uc_filename)
                replicated_atom_prime_file = os.path.join(folder, replicated_filename)
                uc = ase.io.read(atom_prime_file, format="vasp")
                sc = ase.io.read(replicated_atom_prime_file, format="vasp")
                d2 = parse_tdep_forceconstant(
                    fc_file=os.path.join(folder, "infile.forceconstant"),
                    primitive=atom_prime_file,
                    supercell=replicated_atom_prime_file,
                    reduce_fc=False,
                )
                n_unit_atoms = uc.positions.shape[0]
                n_replicas = np.prod(supercell)
                d2 = d2.reshape((n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
                d2 = d2[0, np.newaxis]
                second_order = SecondOrder(
                    atoms=uc, replicated_positions=sc.positions, supercell=supercell, value=d2, folder=folder
                )

            case _:
                raise ValueError(f"{format} is not a valid format")

        return second_order

    @property
    def supercell_replicas(self):
        try:
            return self._supercell_replicas
        except AttributeError:
            self._supercell_replicas = self.calculate_super_replicas()
            return self._supercell_replicas

    @property
    def supercell_positions(self):
        try:
            return self._supercell_positions
        except AttributeError:
            self._supercell_positions = self.calculate_supercell_positions()
            return self._supercell_positions

    @property
    def dynmat(self):
        try:
            return self._dynmat
        except AttributeError:
            self._dynmat = self.calculate_dynmat()
            return self._dynmat

    def calculate(self, calculator=None, delta_shift=1e-3, is_storing=True, is_verbose=False, n_workers=1,
                  scratch_dir=None, keep_scratch=False, use_symmetry=False, symprec=1e-5):
        """
        Calculate second-order force constants with finite differences.

        This is the method typically reached through ``fc.second.calculate(...)``.
        It can either load an existing ``second.npy`` from ``self.folder`` when
        ``is_storing`` is enabled, or compute the harmonic force constants from
        the current structure and calculator.

        See the *Parallel runs with ML calculators* section of the
        ForceConstants documentation for the recommended pattern when
        running torch-based calculators (Orb, MACE, MatterSim, CPUNEP) in
        parallel: define a no-arg factory function at module top level
        and pass it (without parentheses) as ``calculator``.

        Parameters
        ----------
        calculator : callable or ASE Calculator instance or None
            For serial runs, pass an ASE Calculator instance (the existing
            kaldo idiom). For parallel runs, pass a callable that returns
            a fresh ASE Calculator: a class with a no-arg constructor, a
            top-level factory function, ``functools.partial``, etc. Each
            worker invokes the callable once to build its own isolated
            calculator. If None, ``replicated_atoms`` must already have
            a calculator attached.
        delta_shift : float, optional
            Finite-difference displacement in Angstrom. The default
            ``1e-3`` is tuned for analytical calculators (EMT, LAMMPS).
            ML potentials in float32 (Orb, MACE, MatterSim, ...) need
            ``1e-2`` or larger because float32 force noise (~1e-7 eV/Å)
            divided by a tiny delta produces FC noise that swamps the
            physics. A warning fires when ``delta_shift < 1e-2`` and the
            calculator looks ML-based.
            Default: 1e-3
        is_storing : bool, optional
            If True, try to load an existing result from ``self.folder`` first
            and save newly computed data after the calculation.
            Default: True
        is_verbose : bool, optional
            If True, log per-atom progress information.
            Default: False
        n_workers : int or None, optional
            Number of worker processes used for the displaced-atom finite-
            difference tasks. ``1`` runs serially, ``None`` uses all available
            workers. Each worker is capped to one OpenMP / MKL / OpenBLAS
            thread so that calculators with internal multithreading (PyNEP,
            torch CPU, numpy+MKL) don't oversubscribe. Override by setting
            ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` in the environment
            before invoking.
            Default: 1
        scratch_dir : str or None, optional
            Optional scratch directory for atom-by-atom intermediate files used
            for recovery of interrupted calculations. Pass an explicit path to
            override. Pass an empty string ``''`` to disable scratch files and
            fall back to in-memory accumulation.
            Default: ``{folder}/second_order`` when ``self.folder`` is set,
            ``n_workers > 1``, and ``use_symmetry=False``. With
            ``use_symmetry=True`` the auto-default is suppressed (the two
            modes are mutually incompatible — see the ``use_symmetry``
            docstring below).
        keep_scratch : bool, optional
            If True, keep scratch files after successful assembly.
            Default: False
        use_symmetry : bool, optional
            If True, use the crystal spacegroup to reduce the number of
            atoms displaced by the FD method. Only spacegroup operations
            compatible with the supercell shape are used (e.g. an in-plane
            subgroup for slab supercells). Requires a diagonal integer
            supercell expansion. Not compatible with ``scratch_dir`` —
            pass ``scratch_dir=None`` (the default) when enabling.
            Default: False
        symprec : float, optional
            precision for symmetry using spglib.
            Default: 1e-5
        """
        if is_parallel(n_workers):
            validate_parallel_calculator(calculator, method='SecondOrder.calculate')
        maybe_warn_ml_delta_shift(calculator, delta_shift, method='SecondOrder.calculate')
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        # Attach the calculator instance to replicated_atoms once and skip the
        # per-atom rebind in _compute_iat_second. Some calculator libraries
        # require a calculator to stay bound to a single atoms object.
        if n_workers == 1 and calculator is not None and not callable(calculator):
            replicated_atoms.calc = calculator
            worker_calculator = None
        else:
            worker_calculator = calculator
        # Auto-resolve the default scratch directory only for parallel runs;
        # serial stays in memory to avoid creating unexpected directories.
        # use_symmetry is incompatible with scratch_dir (calculate_second
        # raises ValueError on the combo), so don't auto-assign in that case.
        if (scratch_dir is None and self.folder and is_parallel(n_workers)
                and not use_symmetry):
            scratch_dir = os.path.join(self.folder, 'second_order')
        elif scratch_dir == '':
            scratch_dir = None

        if is_storing:
            try:
                self.value = SecondOrder.load(
                    folder=self.folder, supercell=self.supercell, format="numpy", is_acoustic_sum=self.is_acoustic_sum
                ).value

            except FileNotFoundError:
                logging.info("Second order not found. Calculating.")
                self.value = calculate_second(
                    atoms,
                    replicated_atoms,
                    delta_shift,
                    is_verbose=is_verbose,
                    n_workers=n_workers,
                    calculator=worker_calculator,
                    scratch_dir=scratch_dir,
                    keep_scratch=keep_scratch,
                    use_symmetry=use_symmetry,
                    symprec=symprec,
                )
                self.save("second")
                if calculator is not None:
                    self.replicated_atoms.calc = calculator() if callable(calculator) else calculator
                self.replicated_atoms.get_forces()
                ase.io.write(self.folder + "/replicated_atoms.xyz", self.replicated_atoms, "extxyz")
            else:
                logging.info("Reading stored second")
        else:
            self.value = calculate_second(
                atoms,
                replicated_atoms,
                delta_shift,
                is_verbose=is_verbose,
                n_workers=n_workers,
                calculator=worker_calculator,
                scratch_dir=scratch_dir,
                keep_scratch=keep_scratch,
                use_symmetry=use_symmetry,
                symprec=symprec,
            )
        if self.is_acoustic_sum:
            self.value = acoustic_sum_rule(self.value)

    def calculate_dynmat(self):
        evtotenjovermol = units.mol / (10 * units.J)
        mass = self.atoms.get_masses()
        shape = self.value.shape
        log_size(shape, float, name="dynmat")
        dynmat = self.value * 1 / np.sqrt(mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        dynmat /= np.sqrt(mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])
        return tf.convert_to_tensor(dynmat * evtotenjovermol)

    def calculate_super_replicas(self):
        scell = self.supercell
        n_replicas = np.prod(scell)
        atoms = self.atoms
        cell = atoms.cell
        n_unit_cell = atoms.positions.shape[0]
        replicated_positions = self.replicated_atoms.positions.reshape((n_replicas, n_unit_cell, 3))

        list_of_index = np.round((replicated_positions - self.atoms.positions).dot(np.linalg.inv(atoms.cell))).astype(
            int
        )
        list_of_index = list_of_index[:, 0, :]

        tt = []
        rreplica = []
        for ix2 in [-1, 0, 1]:
            for iy2 in [-1, 0, 1]:
                for iz2 in [-1, 0, 1]:
                    for f in range(list_of_index.shape[0]):
                        scell_id = np.array([ix2 * scell[0], iy2 * scell[1], iz2 * scell[2]])
                        replica_id = list_of_index[f]
                        t = replica_id + scell_id
                        replica_position = np.tensordot(t, cell, (-1, 0))
                        tt.append(t)
                        rreplica.append(replica_position)

        tt = np.array(tt)
        return tt

    def calculate_supercell_positions(self):
        supercell = self.supercell
        atoms = self.atoms
        cell = atoms.cell
        replicated_cell = cell * supercell
        sc_r_pos = np.zeros((3**3, 3))
        ir = 0
        for ix2 in [-1, 0, 1]:
            for iy2 in [-1, 0, 1]:
                for iz2 in [-1, 0, 1]:
                    for i in np.arange(3):
                        sc_r_pos[ir, i] = np.dot(replicated_cell[:, i], np.array([ix2, iy2, iz2]))
                    ir = ir + 1
        return sc_r_pos

"""Non-analytic correction for polar materials.

Implements the mixed-space dipole-dipole treatment of Gonze and Lee,
Phys. Rev. B 55, 10355 (1997), following the conventions of phonopy: the
short-range force constants are the total force constants minus the
dipole-dipole contribution reconstructed on the commensurate grid, and the
dynamical matrix at any q is their Wigner-Seitz-weighted transform plus the
reciprocal- and real-space Ewald dipole terms.

Requires atoms.info["dielectric"] (3x3) and atoms.arrays["charges"]
(n_atoms x 3 x 3 Born effective charges); kALDo engages it automatically
whenever both are present with nonzero charges.
"""

import itertools
import math

import numpy as np
import ase.units as units
from opt_einsum import contract

from kaldo.grid import Grid
from kaldo.helpers.logger import get_logger

logging = get_logger()

NAC_VELOCITY_Q_LENGTH = 1e-5


NAC_VELOCITY_DEGENERACY_TOLERANCE = 1e-4


NAC_VELOCITY_CUTOFF_FREQUENCY = 1e-4


_PHONOPY_TO_KALDO_DM = (units.Ry / units.Bohr ** 2) * (units.mol / (10 * units.J))


NAC_VELOCITY_DIRECTIONS_CART = np.array(
    [
        np.array([1.0, 2.0, 3.0], dtype=float) / np.sqrt(14.0),
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ],
    dtype=float,
)


def degenerate_sets(frequencies, tolerance=NAC_VELOCITY_DEGENERACY_TOLERANCE):
    sets = []
    current = [0]
    for index in range(1, len(frequencies)):
        if abs(frequencies[index] - frequencies[current[-1]]) < tolerance:
            current.append(index)
        else:
            sets.append(current)
            current = [index]
    sets.append(current)
    return sets


def _to_phonopy_dm(dm):
    return np.array(dm, copy=True) / _PHONOPY_TO_KALDO_DM


def _phonopy_frequencies_from_eigenvalues(eigenvalues):
    factor = np.sqrt(_PHONOPY_TO_KALDO_DM) / (2 * np.pi)
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * factor


# All 27 nearest reciprocal-cell images, origin first so exact ties in
# the folding fall back to the unshifted point.
_BZ_SEARCH_SPACE = np.array(
    [[0, 0, 0]] + [p for p in itertools.product((-1, 0, 1), repeat=3) if any(p)],
    dtype=np.int64,
)


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


def dynamical_matrices(q_reds, static_data, mapping, q_direction_carts, fc=None):
    q_reds = np.atleast_2d(np.asarray(q_reds, dtype=float))
    q_direction_carts = np.atleast_2d(np.asarray(q_direction_carts, dtype=float))
    if fc is None:
        try:
            fc = static_data["fc_short_converted"]
        except KeyError:
            raise ValueError(
                "dynamical_matrices needs short-range force constants: pass fc "
                "explicitly or populate static_data['fc_short_converted'] first "
                "(HarmonicWithQ does this through its runtime cache)."
            ) from None

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
        dd_total * static_data["nac_conversion"],
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


def ensure_kernel_cache(static_data, mapping):
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
    if "nac_conversion" not in static_data:
        static_data["nac_conversion"] = units.mol / (10 * units.J)
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


def build_static_data(second, matrix=None):
    atoms = second.atoms
    born = np.array(atoms.get_array("charges"), dtype=float, copy=True)
    dielectric = np.array(atoms.info["dielectric"], dtype=float, copy=True)
    primitive_cell = np.array(atoms.cell.array, dtype=float, copy=True)
    primitive_positions = np.array(atoms.positions, dtype=float, copy=True)
    reciprocal_lattice = np.array(atoms.cell.reciprocal(), dtype=float, copy=True)
    masses = np.array(atoms.get_masses(), dtype=float, copy=True)
    matrix = normalize_bvk_supercell_matrix(matrix)
    if matrix is None:
        supercell_cell = np.array(second.replicated_atoms.cell.array, dtype=float, copy=True)
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


def build_mapping(second, matrix=None):
    matrix = normalize_bvk_supercell_matrix(matrix)
    if matrix is None:
        # The dedicated no-matrix builder ordered supercell atoms replica-major,
        # inconsistent with the atom-major layout _build_interleaved_fc produces.
        # A diagonal BvK matrix reproduces it through the tested code path.
        matrix = np.diag(np.asarray(second.supercell, dtype=int))
    return _build_supercell_matrix_mapping(second.atoms, matrix)

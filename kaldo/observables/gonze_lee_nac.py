import math

import numpy as np
from ase import units
from opt_einsum import contract

from kaldo.grid import Grid
from kaldo.observables.forceconstant import chi


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


def dielectric_part(q_cart, dielectric):
    return float(np.einsum("i,ij,j->", q_cart, dielectric, q_cart))


def get_minimum_g_rad(reciprocal_lattice, g_cutoff, g_rad=100):
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


def get_g_vec_list(reciprocal_lattice, g_rad):
    npts = g_rad * 2 + 1
    grid = np.array(list(np.ndindex((npts, npts, npts))), dtype=np.int64) - g_rad
    return np.array(grid @ reciprocal_lattice.T, dtype="double", order="C")


def get_g_list(reciprocal_lattice, g_cutoff):
    g_rad = get_minimum_g_rad(reciprocal_lattice, g_cutoff)
    g_vec_list = get_g_vec_list(reciprocal_lattice, g_rad)
    g_norm2 = (g_vec_list ** 2).sum(axis=1)
    return np.array(g_vec_list[g_norm2 < g_cutoff ** 2], dtype="double", order="C")


def multiply_borns(dd_in, born):
    num_atom = born.shape[0]
    dd = np.zeros((num_atom, 3, num_atom, 3), dtype=np.complex128)
    for i in range(num_atom):
        for j in range(num_atom):
            dd[i, :, j, :] = born[i].T @ dd_in[i, :, j, :] @ born[j]
    return dd


def get_dd_base(g_list, q_cart, q_direction_cart, dielectric, positions, lambda_, tolerance):
    num_atom = positions.shape[0]
    dd_part = np.zeros((num_atom, 3, num_atom, 3), dtype=np.complex128)
    l2 = 4 * lambda_ * lambda_
    for g_vec in g_list:
        q_k = g_vec + q_cart
        norm = np.linalg.norm(q_k)
        if norm < tolerance:
            if q_direction_cart is None:
                continue
            denom = dielectric_part(q_direction_cart, dielectric)
            kk = np.outer(q_direction_cart, q_direction_cart) / denom
        else:
            denom = dielectric_part(q_k, dielectric)
            kk = np.outer(q_k, q_k) / denom * np.exp(-denom / l2)
        for i in range(num_atom):
            for j in range(num_atom):
                phase = float(np.dot(positions[i] - positions[j], g_vec) * 2 * np.pi)
                dd_part[i, :, j, :] += kk * (np.cos(phase) + 1j * np.sin(phase))
    return dd_part


def recip_dipole_dipole_q0(g_list, born, dielectric, positions, lambda_, tolerance):
    zero = np.zeros(3, dtype="double")
    dd_tmp1 = get_dd_base(g_list, zero, None, dielectric, positions, lambda_, tolerance)
    dd_tmp2 = multiply_borns(dd_tmp1, born)
    num_atom = positions.shape[0]
    dd_q0 = np.zeros((num_atom, 3, 3), dtype=np.complex128)
    for i in range(num_atom):
        dd_q0[i] = dd_tmp2[i, :, :, :].sum(axis=1)
    for i in range(num_atom):
        dd_q0[i] = (dd_q0[i] + dd_q0[i].conj().T) / 2
    return dd_q0


def limiting_dipole_dipole(dielectric, lambda_):
    inv_eps = np.linalg.inv(dielectric)
    sqrt_det_eps = np.sqrt(np.linalg.det(dielectric))
    return -4.0 / 3 / np.sqrt(np.pi) * inv_eps / sqrt_det_eps * lambda_ ** 3


def recip_dipole_dipole(
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
):
    dd_tmp = get_dd_base(
        g_list, q_cart, q_direction_cart, dielectric, positions, lambda_, tolerance
    )
    dd = multiply_borns(dd_tmp, born)
    num_atom = positions.shape[0]
    for i in range(num_atom):
        dd[i, :, i, :] -= dd_q0[i]
    return dd * factor


def h_tensor(supercell_cell, svecs, dielectric, lambda_):
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
    h = np.zeros((3, 3, len(y_safe)), dtype="double", order="C")
    for i in range(3):
        for j in range(3):
            h[i, j, :] = x[:, i] * x[:, j] * a - eps_inv[i, j] * b
    return h


def real_dipole_dipole(q_red, svecs, multi, s2pp_map, dielectric, lambda_, supercell_cell):
    num_satom, num_patom = multi.shape[:2]
    phase_all = np.exp(2j * np.pi * (svecs @ q_red))
    h = h_tensor(supercell_cell, svecs, dielectric, lambda_)
    vals = -(lambda_ ** 3) * h * phase_all * np.linalg.det(dielectric) ** (-0.5)
    c_real = np.zeros((num_patom, 3, num_patom, 3), dtype=np.complex128)
    for i_s in range(num_satom):
        for i_p in range(num_patom):
            multiplicity = int(multi[i_s, i_p, 0])
            start = int(multi[i_s, i_p, 1])
            block = vals[:, :, start]
            c_real[s2pp_map[i_s], :, i_p, :] += (
                block + block.conj().T
            ) / 2 / multiplicity
    return c_real


def mass_weight(fc_term, masses):
    out = np.array(fc_term, dtype=np.complex128, copy=True)
    for i in range(len(masses)):
        for j in range(len(masses)):
            out[i, :, j, :] /= np.sqrt(masses[i] * masses[j])
    return out.reshape(len(masses) * 3, len(masses) * 3)


def short_range_dynamical_matrix(fc, q_red, svecs, multi, masses, s2p_map, p2s_map):
    num_patom = len(p2s_map)
    num_satom = len(s2p_map)
    dm = np.zeros((num_patom * 3, num_patom * 3), dtype=np.complex128)
    is_compact_fc = fc.shape[0] != fc.shape[1]
    for i in range(num_patom):
        for j in range(num_patom):
            local = np.zeros((3, 3), dtype=np.complex128)
            for k in range(num_satom):
                if s2p_map[k] != p2s_map[j]:
                    continue
                multiplicity = int(multi[k, i, 0])
                start = int(multi[k, i, 1])
                phase_factor = 0.0j
                for ll in range(multiplicity):
                    phase = float(np.dot(q_red, svecs[start + ll]) * 2 * np.pi)
                    phase_factor += np.cos(phase) + 1j * np.sin(phase)
                phase_factor /= multiplicity
                fc_i = i if is_compact_fc else p2s_map[i]
                local += fc[fc_i, k] * phase_factor
            local /= np.sqrt(masses[i] * masses[j])
            dm[i * 3 : i * 3 + 3, j * 3 : j * 3 + 3] = local
    return (dm + dm.conj().T) / 2


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


def build_static_data(second_order, nac_bvk_supercell_matrix=None):
    atoms = second_order.atoms
    born = np.array(atoms.get_array("charges"), dtype=float, copy=True)
    dielectric = np.array(atoms.info["dielectric"], dtype=float, copy=True)
    primitive_cell = np.array(atoms.cell.array, dtype=float, copy=True)
    primitive_positions = np.array(atoms.positions, dtype=float, copy=True)
    reciprocal_lattice = np.array(atoms.cell.reciprocal(), dtype=float, copy=True)
    masses = np.array(atoms.get_masses(), dtype=float, copy=True)
    matrix = normalize_bvk_supercell_matrix(nac_bvk_supercell_matrix)
    if matrix is None:
        supercell_cell = np.array(second_order.replicated_atoms.cell.array, dtype=float, copy=True)
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
    g_list = get_g_list(reciprocal_lattice, g_cutoff)
    dd_q0 = recip_dipole_dipole_q0(
        g_list, born, dielectric, primitive_positions, lambda_, tolerance
    )
    dd_limiting = limiting_dipole_dipole(dielectric, lambda_)
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


def build_short_range_inputs(second_order, nac_bvk_supercell_matrix=None):
    matrix = normalize_bvk_supercell_matrix(nac_bvk_supercell_matrix)
    if matrix is not None:
        return build_supercell_matrix_mapping(second_order.atoms, matrix)
    atoms = second_order.atoms
    n_atom = len(atoms)
    wrapped_indices = second_order._direct_grid.grid(is_wrapping=True)
    s2p_map = np.tile(np.arange(n_atom, dtype=int), len(wrapped_indices))
    p2s_map = np.arange(n_atom, dtype=int)
    s2pp_map = s2p_map.copy()
    supercell = np.array(second_order.supercell, dtype=float)
    primitive_cell = np.array(atoms.cell.array, dtype=float, copy=True)
    supercell_cell = np.array(second_order.replicated_atoms.cell.array, dtype=float, copy=True)
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


def nacl_phonopy_debug_supercell_matrix():
    return np.array(
        [
            [-2, 2, 2],
            [2, -2, 2],
            [2, 2, -2],
        ],
        dtype=int,
    )


def nacl_phonopy_debug_supercell_matrix_att3():
    return np.array(
        [
            [8, 0, 0],
            [0, 8, 0],
            [0, 0, 8],
        ],
        dtype=int,
    )


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


def build_supercell_matrix_mapping(atoms, supercell_matrix, symprec=1e-5):
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
    # Type-grouped ordering: all replicas of atom 0 first, then atom 1, etc.
    # p2s_map[i] = i * n_translation so primitive atom i maps to supercell atom i*n_translation
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


def fold_points_to_first_bz(qpoints, reciprocal_lattice, tolerance=0.01):
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


def commensurate_points(supercell, reciprocal_lattice=None):
    supercell = np.array(supercell)
    if supercell.shape == (3,):
        grid = Grid(supercell, order="C").grid(is_wrapping=False)
        qpoints = np.array(
            grid / np.array(supercell, dtype=float), dtype="double", order="C"
        )
        if reciprocal_lattice is not None:
            return fold_points_to_first_bz(qpoints, reciprocal_lattice)
        return qpoints
    matrix = normalize_bvk_supercell_matrix(supercell)
    translations = _unique_supercell_translations(matrix.T)
    qpoints = np.array([translation[1] for translation in translations], dtype="double")
    if reciprocal_lattice is not None:
        return fold_points_to_first_bz(qpoints, reciprocal_lattice)
    return np.array(qpoints, dtype="double", order="C")


def dipole_dipole_dynamical_matrix(q_red, static_data, mapping, q_direction_red=None):
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
    dd_recip = recip_dipole_dipole(
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
    )
    dd_real = real_dipole_dipole(
        q_red,
        mapping["svecs"],
        mapping["multi"],
        mapping["s2pp_map"],
        static_data["dielectric"],
        float(static_data["Lambda"]),
        mapping.get("svecs_cell", static_data["supercell_cell"]),
    )
    dd_limiting_expanded = np.zeros_like(dd_recip)
    for i in range(len(static_data["masses"])):
        dd_limiting_expanded[i, :, i, :] = static_data["dd_limiting"]
    dd_real_q0_full = real_dipole_dipole(
        np.zeros(3, dtype=float),
        mapping["svecs"],
        mapping["multi"],
        mapping["s2pp_map"],
        static_data["dielectric"],
        float(static_data["Lambda"]),
        mapping.get("svecs_cell", static_data["supercell_cell"]),
    )
    dd_real_q0 = dd_real_q0_full.sum(axis=2)
    dd_drift = (
        static_data["dd_q0"] * float(units.Rydberg / units.Bohr ** 2)
        + static_data["dd_limiting"] * len(static_data["masses"])
        + dd_real_q0
    )
    dd_total = dd_recip + dd_limiting_expanded + dd_real
    for i in range(len(static_data["masses"])):
        dd_total[i, :, i, :] -= dd_drift[i]
    conversion = units.mol / (10 * units.J)
    return mass_weight(dd_total * conversion, static_data["masses"])


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


def dynamical_matrix_from_second_order(second_order, q_red):
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


def inverse_transform_dynmats_to_force_constants(dynmats, qpoints, mapping, masses):
    s2p_map = mapping["s2p_map"]
    p2s_map = mapping["p2s_map"]
    s2pp_map = mapping["s2pp_map"]
    svecs = mapping.get("phase_svecs", mapping["svecs"])
    multi = mapping["multi"]
    n_p = len(p2s_map)
    n_s = len(s2p_map)
    n_q = len(qpoints)
    fc = np.zeros((n_p, n_s, 3, 3), dtype=float)
    for p_i in range(n_p):
        for s_j in range(n_s):
            p_j = s2pp_map[s_j]
            multiplicity = int(multi[s_j, p_i, 0])
            start = int(multi[s_j, p_i, 1])
            pos = svecs[start : start + multiplicity]
            phases = -2j * np.pi * np.dot(qpoints, pos.T)
            phase_factors = np.exp(phases).sum(axis=1) / multiplicity
            block = np.zeros((3, 3), dtype=np.complex128)
            for i_q, phase_factor in enumerate(phase_factors):
                block += (
                    dynmats[i_q, p_i * 3 : p_i * 3 + 3, p_j * 3 : p_j * 3 + 3]
                    * phase_factor
                )
            coef = np.sqrt(masses[p_i] * masses[p_j]) / n_q
            fc[p_i, s_j] = (block * coef).real
    return fc / (units.mol / (10 * units.J))

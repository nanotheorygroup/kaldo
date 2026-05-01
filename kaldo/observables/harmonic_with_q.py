from kaldo.grid import wrap_coordinates, Grid
from kaldo.observables.forceconstant import chi
from kaldo.observables.observable import Observable
import json
import math
import numpy as np
from pathlib import Path
from ase import units
import ase.io
from ase import Atoms
from opt_einsum import contract
from kaldo.storable import lazy_property, Storable
import tensorflow as tf
from scipy.linalg.lapack import zheev
from kaldo.helpers.logger import get_logger, log_size
from kaldo.observables.secondorder import normalize_bvk_supercell_matrix
# from numpy.linalg import eigh

logging = get_logger()

MIN_N_MODES_TO_STORE = 1000

GONZE_VELOCITY_Q_LENGTH = 1e-5
GONZE_VELOCITY_DEGENERACY_TOLERANCE = 1e-4
GONZE_VELOCITY_CUTOFF_FREQUENCY = 1e-4
# DM conversion: 1 Ry/bohr²/amu in (rad/ps)² = (Ry_to_eV/Å²) × eV_to_10Jmol
# = (units.Ry/units.Bohr²) × (units.mol/(10*units.J))
# Used to convert kALDo-unit DM to phonopy-unit DM for cross-validation.
_PHONOPY_TO_KALDO_DM = (units.Ry / units.Bohr ** 2) * (units.mol / (10 * units.J))
GONZE_VELOCITY_DIRECTIONS_CART = np.array(
    [
        np.array([1.0, 2.0, 3.0], dtype=float) / np.sqrt(14.0),
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ],
    dtype=float,
)


def _gonze_degenerate_sets(frequencies, tolerance=GONZE_VELOCITY_DEGENERACY_TOLERANCE):
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


def _gonze_to_phonopy_q_cart(q_cart):
    return np.array(q_cart, dtype=float, copy=True) * units.Bohr


def _gonze_to_phonopy_dm(dm):
    return np.array(dm, copy=True) / _PHONOPY_TO_KALDO_DM


def _gonze_phonopy_frequencies_from_eigenvalues(eigenvalues):
    factor = np.sqrt(_PHONOPY_TO_KALDO_DM) / (2 * np.pi)
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * factor


def _gonze_velocity_debug_nac_branch(q_red, reciprocal_lattice, tolerance):
    q_red = np.array(q_red, dtype=float, copy=True)
    direction_cart = np.array(GONZE_VELOCITY_DIRECTIONS_CART[3], dtype=float, copy=True)
    q_floor_cart = direction_cart / np.linalg.norm(direction_cart) * tolerance
    dq_red = np.linalg.solve(reciprocal_lattice, q_floor_cart)
    q_red_nac = q_red + dq_red
    q_norm = max(float(np.linalg.norm(reciprocal_lattice @ q_red_nac)), tolerance)
    return {
        "nac_applied": bool(q_norm >= tolerance),
        "q_direction_red": None,
        "q_norm": q_norm,
        "q_red": q_red_nac.tolist(),
        "tolerance": tolerance,
    }


def _gonze_dielectric_part(q_cart, dielectric):
    return float(np.einsum("i,ij,j->", q_cart, dielectric, q_cart))


def _gonze_get_minimum_g_rad(reciprocal_lattice, g_cutoff, g_rad=100):
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


def _gonze_get_g_vec_list(reciprocal_lattice, g_rad):
    npts = g_rad * 2 + 1
    grid = np.array(list(np.ndindex((npts, npts, npts))), dtype=np.int64) - g_rad
    return np.array(grid @ reciprocal_lattice.T, dtype="double", order="C")


def _gonze_get_g_list(reciprocal_lattice, g_cutoff):
    g_rad = _gonze_get_minimum_g_rad(reciprocal_lattice, g_cutoff)
    g_vec_list = _gonze_get_g_vec_list(reciprocal_lattice, g_rad)
    g_norm2 = (g_vec_list ** 2).sum(axis=1)
    return np.array(g_vec_list[g_norm2 < g_cutoff ** 2], dtype="double", order="C")


def _gonze_multiply_borns(dd_in, born):
    num_atom = born.shape[0]
    dd = np.zeros((num_atom, 3, num_atom, 3), dtype=np.complex128)
    for i in range(num_atom):
        for j in range(num_atom):
            dd[i, :, j, :] = born[i].T @ dd_in[i, :, j, :] @ born[j]
    return dd


def _gonze_get_dd_base(
    g_list, q_cart, q_direction_cart, dielectric, positions, lambda_, tolerance
):
    num_atom = positions.shape[0]
    dd_part = np.zeros((num_atom, 3, num_atom, 3), dtype=np.complex128)
    l2 = 4 * lambda_ * lambda_
    for g_vec in g_list:
        q_k = g_vec + q_cart
        norm = np.linalg.norm(q_k)
        if norm < tolerance:
            if q_direction_cart is None:
                continue
            denom = _gonze_dielectric_part(q_direction_cart, dielectric)
            kk = np.outer(q_direction_cart, q_direction_cart) / denom
        else:
            denom = _gonze_dielectric_part(q_k, dielectric)
            kk = np.outer(q_k, q_k) / denom * np.exp(-denom / l2)
        for i in range(num_atom):
            for j in range(num_atom):
                phase = float(np.dot(positions[i] - positions[j], g_vec) * 2 * np.pi)
                dd_part[i, :, j, :] += kk * (np.cos(phase) + 1j * np.sin(phase))
    return dd_part


def _gonze_recip_dipole_dipole_q0(
    g_list, born, dielectric, positions, lambda_, tolerance
):
    zero = np.zeros(3, dtype="double")
    dd_tmp1 = _gonze_get_dd_base(g_list, zero, None, dielectric, positions, lambda_, tolerance)
    dd_tmp2 = _gonze_multiply_borns(dd_tmp1, born)
    num_atom = positions.shape[0]
    dd_q0 = np.zeros((num_atom, 3, 3), dtype=np.complex128)
    for i in range(num_atom):
        dd_q0[i] = dd_tmp2[i, :, :, :].sum(axis=1)
    for i in range(num_atom):
        dd_q0[i] = (dd_q0[i] + dd_q0[i].conj().T) / 2
    return dd_q0


def _gonze_limiting_dipole_dipole(dielectric, lambda_):
    inv_eps = np.linalg.inv(dielectric)
    sqrt_det_eps = np.sqrt(np.linalg.det(dielectric))
    return -4.0 / 3 / np.sqrt(np.pi) * inv_eps / sqrt_det_eps * lambda_ ** 3


def _gonze_recip_dipole_dipole(
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
    dd_tmp = _gonze_get_dd_base(
        g_list, q_cart, q_direction_cart, dielectric, positions, lambda_, tolerance
    )
    dd = _gonze_multiply_borns(dd_tmp, born)
    num_atom = positions.shape[0]
    for i in range(num_atom):
        dd[i, :, i, :] -= dd_q0[i]
    return dd * factor


def _gonze_h_tensor(supercell_cell, svecs, dielectric, lambda_):
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


def _gonze_real_dipole_dipole(
    q_red, svecs, multi, s2pp_map, dielectric, lambda_, supercell_cell
):
    num_satom, num_patom = multi.shape[:2]
    phase_all = np.exp(2j * np.pi * (svecs @ q_red))
    h = _gonze_h_tensor(supercell_cell, svecs, dielectric, lambda_)
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


def _gonze_mass_weight(fc_term, masses):
    out = np.array(fc_term, dtype=np.complex128, copy=True)
    for i in range(len(masses)):
        for j in range(len(masses)):
            out[i, :, j, :] /= np.sqrt(masses[i] * masses[j])
    return out.reshape(len(masses) * 3, len(masses) * 3)


def _gonze_short_range_dynamical_matrix(
    fc, q_red, svecs, multi, masses, s2p_map, p2s_map
):
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


class HarmonicWithQ(Observable, Storable):
    
    # Define storage formats for harmonic properties
    _store_formats = {
        'frequency': 'formatted',
        'velocity': 'formatted',
        'participation_ratio': 'formatted',
        '_dynmat_derivatives_x': 'numpy',
        '_dynmat_derivatives_y': 'numpy', 
        '_dynmat_derivatives_z': 'numpy',
        '_dynmat_fourier': 'numpy',
        '_eigensystem': 'numpy',
        '_sij_x': 'numpy',
        '_sij_y': 'numpy',
        '_sij_z': 'numpy'
    }

    def __init__(self, q_point, second,
                 distance_threshold=None,
                 storage='numpy',
                 is_nw=False,
                 is_unfolding=False,
                 is_amorphous=False,
                 nac_method='legacy',
                 nac_debug=False,
                 nac_debug_folder='debug',
                 nac_bvk_supercell_matrix=None,
                 nac_q_direction=(1, 0, 0),
                 q_index=None,
                 gonze_nac_precomputed=None,
                 *kargs,
                 **kwargs):
        super().__init__(*kargs, **kwargs)
        # Input arguments
        self.q_point = q_point
        self.atoms = second.atoms
        self.n_modes = self.atoms.positions.shape[0] * 3
        self.supercell = second.supercell
        self.second = second
        self.distance_threshold = distance_threshold
        self.physical_mode = np.ones((1, self.n_modes), dtype=bool)
        # Arguments for specific physical assumptions
        self.is_amorphous = is_amorphous
        self.is_unfolding = is_unfolding
        self.is_nac = True if 'dielectric' in self.atoms.info else False
        supported_nac_methods = ('legacy', 'gonze')
        if nac_method not in supported_nac_methods:
            raise ValueError(
                f"Unknown nac_method {nac_method!r}. Supported values are {supported_nac_methods}."
            )
        if nac_method == 'gonze':
            if not self.is_nac or 'charges' not in self.atoms.arrays:
                raise ValueError(
                    "nac_method='gonze' requires atoms.info['dielectric'] and atoms.arrays['charges']."
                )
        self.nac_method = nac_method
        self.nac_debug = bool(nac_debug)
        self.nac_debug_folder = nac_debug_folder
        self.nac_bvk_supercell_matrix = normalize_bvk_supercell_matrix(
            nac_bvk_supercell_matrix
        )
        self.nac_q_direction = np.array(nac_q_direction, dtype=float, copy=True)
        self.q_index = q_index
        self._gonze_nac_precomputed = gonze_nac_precomputed
        self.is_nw = is_nw
        if (q_point == [0, 0, 0]).all():
            if self.is_nw:
                self.physical_mode[0, :4] = False
            else:
                self.physical_mode[0, :3] = False
        if self.n_modes > MIN_N_MODES_TO_STORE:
            self.storage = storage
        else:
            self.storage = 'memory'

    def _load_formatted_property(self, property_name, name):
        """Override formatted loading for HarmonicWithQ-specific properties"""
        if '_sij' in property_name:
            loaded = []
            for alpha in range(3):
                loaded.append(np.loadtxt(name + '_' + str(alpha) + '.dat', skiprows=1, dtype=complex))
            return np.array(loaded).transpose(1, 0)
        else:
            # Use default implementation for other properties
            return super()._load_formatted_property(property_name, name)
    
    def _save_formatted_property(self, property_name, name, data):
        """Override formatted saving for HarmonicWithQ-specific properties"""
        if '_sij' in property_name:
            fmt = '%.18e'
            for alpha in range(3):
                np.savetxt(name + '_' + str(alpha) + '.dat', data[..., alpha].flatten(), fmt=fmt, 
                          header=str(data[..., 0].shape))
        else:
            # Use default implementation for other properties
            super()._save_formatted_property(property_name, name, data)

    def _gonze_debug_static_folder(self):
        return Path(self.nac_debug_folder) / "static"

    def _gonze_debug_q_folder(self):
        if self.q_index is not None:
            return Path(self.nac_debug_folder) / f"q-{int(self.q_index):05d}"
        label_parts = []
        for value in np.asarray(self.q_point, dtype=float):
            text = f"{value:.12g}"
            if "." not in text:
                text += ".0"
            text = text.replace("-", "m").replace(".", "p")
            label_parts.append(text)
        return Path(self.nac_debug_folder) / ("q_" + "_".join(label_parts))

    def _gonze_save_debug(self, folder, arrays):
        if not self.nac_debug:
            return
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name, value in arrays.items():
            np.save(folder / f"{name}.npy", value)

    def _gonze_save_debug_json(self, folder, payloads):
        if not self.nac_debug:
            return
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            (folder / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _resolve_gonze_bvk_supercell_matrix(self):
        if self.nac_bvk_supercell_matrix is not None:
            return np.array(self.nac_bvk_supercell_matrix, dtype=int, copy=True)
        supercell = np.asarray(self.second.supercell, dtype=int)
        if supercell.shape != (3,):
            raise ValueError(
                "nac_method='gonze' requires second.supercell to be a diagonal 3-vector "
                "when nac_bvk_supercell_matrix is not provided."
            )
        return np.diag(supercell)

    def _calculate_gonze_dynamical_matrix_for_q(self, q_red):
        original_q_point = np.array(self.q_point, dtype=float, copy=True)
        original_debug = self.nac_debug
        self.q_point = np.array(q_red, dtype=float, copy=True)
        self.nac_debug = False
        try:
            return self._calculate_gonze_dynamical_matrix()
        finally:
            self.q_point = original_q_point
            self.nac_debug = original_debug

    def _calculate_gonze_velocity_direction_data(self, direction_index, static_data):
        if direction_index not in range(4):
            raise ValueError(f"direction_index must be in 0..3, got {direction_index}")
        direction_cart = np.array(
            GONZE_VELOCITY_DIRECTIONS_CART[direction_index], dtype=float, copy=True
        )
        dq_cart = direction_cart / np.linalg.norm(direction_cart) * GONZE_VELOCITY_Q_LENGTH
        dq_red = static_data["primitive_cell"] @ dq_cart / units.Bohr
        q_red = np.array(self.q_point, dtype=float, copy=True)
        dm_minus = _gonze_to_phonopy_dm(self._calculate_gonze_dynamical_matrix_for_q(q_red - dq_red))
        dm_plus = _gonze_to_phonopy_dm(self._calculate_gonze_dynamical_matrix_for_q(q_red + dq_red))
        delta_dm = dm_plus - dm_minus
        ddm_fd = delta_dm / (2 * GONZE_VELOCITY_Q_LENGTH)
        return {
            "direction_cart": direction_cart,
            "dq_cart": dq_cart,
            "dq_red": dq_red,
            "dm_minus": dm_minus,
            "dm_plus": dm_plus,
            "delta_dm": delta_dm,
            "ddm_fd": ddm_fd,
        }

    def _project_gonze_group_velocity_raw(self, ddms, eigenvectors, frequencies):
        """Project ddm_fd tensors onto eigenmodes with degenerate perturbation theory.

        ddms[0] is the d0 direction used to lift degeneracy.
        ddms[1:] are the x/y/z axes (d1, d2, d3) for the velocity components.
        Returns gv_raw of shape (n_modes, 3) in phonopy DM derivative units.
        """
        gv_raw = np.zeros((len(frequencies), 3), dtype=float)
        degenerate_sets = _gonze_degenerate_sets(frequencies)
        for indices in degenerate_sets:
            subspace = eigenvectors[:, indices]
            perturbation = subspace.conj().T @ ddms[0] @ subspace
            _, rotation = np.linalg.eigh((perturbation + perturbation.conj().T) / 2)
            rotated = subspace @ rotation
            for axis, ddm in enumerate(ddms[1:]):
                projected = rotated.conj().T @ ddm @ rotated
                gv_raw[np.array(indices), axis] = np.real(np.diag(projected))
        return gv_raw

    def _scale_gonze_group_velocity_raw(self, gv_raw, frequencies):
        """Scale raw projected derivatives to group velocity in Å×THz.

        Applies gv = (1/2ω) × dω²/dk, expressed in phonopy units as
        _PHONOPY_TO_KALDO_DM / (8π² × freq_THz).
        """
        scaling = np.zeros(len(frequencies), dtype=float)
        cutoff_mask = (np.abs(frequencies) > GONZE_VELOCITY_CUTOFF_FREQUENCY).astype(np.int64)
        active = cutoff_mask.astype(bool)
        scaling[active] = _PHONOPY_TO_KALDO_DM / (8.0 * np.pi ** 2 * frequencies[active])
        gv_scaled = gv_raw * scaling[:, np.newaxis]
        gv_scaled[~active] = 0.0
        return gv_scaled, scaling, cutoff_mask

    def _calculate_gonze_velocity_debug_data(self):
        static_data = self._build_gonze_static_data()
        q_red = np.array(self.q_point, dtype=float, copy=True)
        q_cart = static_data["reciprocal_lattice"] @ q_red
        dm_q_kaldo = self._calculate_gonze_dynamical_matrix()
        dm_q = _gonze_to_phonopy_dm(dm_q_kaldo)
        eigenvalues, eigenvectors = np.linalg.eigh(dm_q)
        eigenvalues = eigenvalues.real
        frequencies = _gonze_phonopy_frequencies_from_eigenvalues(eigenvalues)
        degenerate_sets = _gonze_degenerate_sets(frequencies)
        q_cart = _gonze_to_phonopy_q_cart(q_cart)
        nac_branch = _gonze_velocity_debug_nac_branch(
            q_red,
            _gonze_to_phonopy_q_cart(static_data["reciprocal_lattice"]),
            GONZE_VELOCITY_Q_LENGTH,
        )
        directions = {}
        for index in range(4):
            direction_name = f"d{index}"
            direction_data = self._calculate_gonze_velocity_direction_data(index, static_data)
            directions[direction_name] = direction_data
            self._gonze_save_debug(self._gonze_debug_q_folder() / direction_name, direction_data)
        ddms = [directions[f"d{i}"]["ddm_fd"] for i in range(4)]
        gv_raw = self._project_gonze_group_velocity_raw(ddms, eigenvectors, frequencies)
        gv_scaled, gv_scaling_prefactor, gv_cutoff_mask = self._scale_gonze_group_velocity_raw(
            gv_raw, frequencies
        )
        data = {
            "q_red": q_red,
            "q_cart": q_cart,
            "dm_q": dm_q,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "frequencies": frequencies,
            "degenerate_sets": {"sets": degenerate_sets},
            "nac_branch": nac_branch,
            "directions": directions,
            "gv_raw": gv_raw,
            "gv_scaling_prefactor": gv_scaling_prefactor,
            "gv_cutoff_mask": gv_cutoff_mask,
            "gv_scaled": gv_scaled,
        }
        self._gonze_save_debug(
            self._gonze_debug_q_folder(),
            {
                "q_red": q_red,
                "q_cart": q_cart,
                "dm_q": dm_q,
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "frequencies": frequencies,
                "gv_raw": gv_raw,
                "gv_scaling_prefactor": gv_scaling_prefactor,
                "gv_cutoff_mask": gv_cutoff_mask,
                "gv_scaled": gv_scaled,
            },
        )
        self._gonze_save_debug_json(
            self._gonze_debug_q_folder(),
            {
                "degenerate_sets": {"sets": degenerate_sets},
                "nac_branch": nac_branch,
            },
        )
        return data

    def _build_gonze_static_data(self):
        if self._gonze_nac_precomputed is not None:
            return dict(self._gonze_nac_precomputed["static_data"])  # shallow copy
        matrix = self._resolve_gonze_bvk_supercell_matrix()
        data = self.second._gonze_build_static_data(matrix)
        data["nac_bvk_supercell_matrix"] = np.array(matrix, dtype=int)
        self._gonze_save_debug(
            self._gonze_debug_static_folder(),
            {k: v for k, v in data.items() if k != "q_direction_tolerance"},
        )
        return data

    def _build_gonze_short_range_inputs(self, static_data):
        if self._gonze_nac_precomputed is not None:
            return self._gonze_nac_precomputed["mapping"]
        return self.second._gonze_build_mapping(self._resolve_gonze_bvk_supercell_matrix())

    def _calculate_gonze_dynamical_matrix(self):
        static_data = self._build_gonze_static_data()
        mapping = self._build_gonze_short_range_inputs(static_data)
        masses = static_data["masses"]
        q_red = np.array(self.q_point, dtype=float, copy=True)
        q_cart = static_data["reciprocal_lattice"] @ q_red
        if np.linalg.norm(q_cart) >= static_data["q_direction_tolerance"]:
            q_direction_cart = q_cart
        else:
            q_direction_cart = static_data["reciprocal_lattice"] @ self.nac_q_direction

        recip_dd_q0 = np.zeros_like(static_data["dd_q0"])
        dd_recip = _gonze_recip_dipole_dipole(
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
        dd_real = _gonze_real_dipole_dipole(
            q_red,
            mapping["svecs"],
            mapping["multi"],
            mapping["s2pp_map"],
            static_data["dielectric"],
            float(static_data["Lambda"]),
            mapping.get("svecs_cell", static_data["supercell_cell"]),
        )
        dd_limiting_expanded = np.zeros_like(dd_recip)
        for i in range(len(masses)):
            dd_limiting_expanded[i, :, i, :] = static_data["dd_limiting"]
        if "dd_drift" in static_data:
            dd_drift = static_data["dd_drift"]
            dd_real_q0 = None
        else:
            dd_real_q0_full = _gonze_real_dipole_dipole(
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
                + static_data["dd_limiting"] * len(masses)
                + dd_real_q0
            )
        dd_total = dd_recip + dd_limiting_expanded + dd_real
        for i in range(len(masses)):
            dd_total[i, :, i, :] -= dd_drift[i]
        conversion = units.mol / (10 * units.J)
        dd_total_mass_weighted = _gonze_mass_weight(dd_total * conversion, masses)

        effective_matrix = self._resolve_gonze_bvk_supercell_matrix()
        fc_short = self.second.get_gonze_short_range_force_constants(effective_matrix)
        dm_short = _gonze_short_range_dynamical_matrix(
            fc_short * conversion,
            q_red,
            mapping.get("phase_svecs", mapping["svecs"]),
            mapping["multi"],
            masses,
            mapping["s2p_map"],
            mapping["p2s_map"],
        )
        dm_final = dm_short + dd_total_mass_weighted
        dm_final = (dm_final + dm_final.conj().T) / 2
        eigvals = np.linalg.eigvalsh(dm_final).real
        frequencies = np.abs(eigvals) ** 0.5 * np.sign(eigvals) / (2 * np.pi)
        static_debug = {k: v for k, v in {
            "svecs": mapping["svecs"],
            "multi": mapping["multi"],
            "multi_counts": mapping["multi"][:, :, 0],
            "multi_offsets": mapping["multi"][:, :, 1],
            "s2p_map": mapping["s2p_map"],
            "p2s_map": mapping["p2s_map"],
            "s2pp_map": mapping["s2pp_map"],
            "dd_real_q0": dd_real_q0,
            "short_range_force_constants": fc_short,
        }.items() if v is not None}
        for optional_name in (
            "phase_svecs",
            "svecs_cell",
            "supercell_matrix",
            "primitive_matrix",
            "primitive_scaled_positions",
            "supercell_scaled_positions",
            "primitive_shifts",
        ):
            if optional_name in mapping:
                static_debug[optional_name] = mapping[optional_name]
        self._gonze_save_debug(self._gonze_debug_static_folder(), static_debug)
        self._gonze_save_debug(
            self._gonze_debug_q_folder(),
            {
                "q_red": q_red,
                "q_cart": q_cart,
                "q_direction_cart": q_direction_cart,
                "dd_recip": dd_recip,
                "dd_real": dd_real,
                "dd_limiting_expanded": dd_limiting_expanded,
                "dd_drift": dd_drift,
                "dd_total_mass_weighted": dd_total_mass_weighted,
                "dm_short": dm_short,
                "dm_final": dm_final,
                "eigenvalues": eigvals,
                "frequencies": frequencies,
            },
        )
        return dm_final

    @lazy_property(label='<q_point>')
    def frequency(self):
        frequency = self.calculate_frequency()[np.newaxis, :]
        return frequency

    @lazy_property(label='<q_point>')
    def velocity(self):
        velocity = self.calculate_velocity()
        return velocity

    @lazy_property(label='<q_point>')
    def participation_ratio(self):
        participation_ratio = self.calculate_participation_ratio()
        return participation_ratio

    @lazy_property(label='<q_point>')
    def _dynmat_derivatives_x(self):
        if self.is_unfolding:
            _dynmat_derivatives = self.calculate_dynmat_derivatives_unfolded(direction=0)
        else:
            _dynmat_derivatives = self.calculate_dynmat_derivatives(direction=0)
        return _dynmat_derivatives

    @lazy_property(label='<q_point>')
    def _dynmat_derivatives_y(self):
        if self.is_unfolding:
            _dynmat_derivatives = self.calculate_dynmat_derivatives_unfolded(direction=1)
        else:
            _dynmat_derivatives = self.calculate_dynmat_derivatives(direction=1)
        return _dynmat_derivatives

    @lazy_property(label='<q_point>')
    def _dynmat_derivatives_z(self):
        if self.is_unfolding:
            _dynmat_derivatives = self.calculate_dynmat_derivatives_unfolded(direction=2)
        else:
            _dynmat_derivatives = self.calculate_dynmat_derivatives(direction=2)
        return _dynmat_derivatives

    @lazy_property(label='<q_point>')
    def _dynmat_fourier(self):
        dynmat_fourier = self.calculate_dynmat_fourier()
        return dynmat_fourier

    @lazy_property(label='<q_point>')
    def _eigensystem(self):
        if self.is_unfolding:
            _eigensystem = self.calculate_eigensystem_unfolded(only_eigenvals=False)
        else:
            _eigensystem = self.calculate_eigensystem(only_eigenvals=False)
        return _eigensystem

    @lazy_property(label='<q_point>')
    def _sij_x(self):
        _sij = self.calculate_sij(direction=0)
        return _sij

    @lazy_property(label='<q_point>')
    def _sij_y(self):
        _sij = self.calculate_sij(direction=1)
        return _sij

    @lazy_property(label='<q_point>')
    def _sij_z(self):
        _sij = self.calculate_sij(direction=2)
        return _sij

    def calculate_frequency(self):
        # TODO: replace calculate_eigensystem() with eigensystem
        if self.is_unfolding:
            eigenvals = self.calculate_eigensystem_unfolded(only_eigenvals=True)
        else:
            eigenvals = self.calculate_eigensystem(only_eigenvals=True)
        frequency = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
        return frequency.real

    def calculate_dynmat_derivatives(self, direction):
        q_point = self.q_point
        is_amorphous = self.is_amorphous
        distance_threshold = self.distance_threshold
        atoms = self.atoms
        list_of_replicas = self.second.list_of_replicas
        replicated_cell = self.second.replicated_atoms.cell
        replicated_cell_inv = self.second._replicated_cell_inv
        cell_inv = self.second.cell_inv
        dynmat = self.second.dynmat
        positions = self.atoms.positions
        n_unit_cell = atoms.positions.shape[0]
        n_modes = n_unit_cell * 3
        n_replicas = np.prod(self.supercell)
        shape = (1, n_unit_cell * 3, n_unit_cell * 3)
        dir = ['_x', '_y', '_z']
        type = complex if (not self.is_amorphous) else float
        log_size(shape, type, name='dynamical_matrix_derivative_' + dir[direction])
        if self.is_amorphous:
            distance = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distance = wrap_coordinates(distance, replicated_cell, replicated_cell_inv)
            dynmat_derivatives = contract('ij,ibjc->ibjc',
                                          tf.convert_to_tensor(distance[..., direction]),
                                          dynmat[0, :, :, 0, :, :],
                                          backend='tensorflow')
        else:
            distance = positions[:, np.newaxis, np.newaxis, :] - (
                    positions[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])

            if distance_threshold is not None:

                distance_to_wrap = positions[:, np.newaxis, np.newaxis, :] - (
                    self.second.replicated_atoms.positions.reshape(n_replicas, n_unit_cell, 3)[
                    np.newaxis, :, :, :])

                shape = (n_unit_cell, 3, n_unit_cell, 3)
                type = complex
                dynmat_derivatives = np.zeros(shape, dtype=type)
                for l in range(n_replicas):
                    wrapped_distance = wrap_coordinates(distance_to_wrap[:, l, :, :], replicated_cell,
                                                        replicated_cell_inv)
                    mask = (np.linalg.norm(wrapped_distance, axis=-1) < distance_threshold)
                    id_i, id_j = np.argwhere(mask).T
                    dynmat_derivatives[id_i, :, id_j, :] += contract('f,fbc->fbc', distance[id_i, l, id_j, direction], \
                                                                     dynmat.numpy()[0, id_i, :, 0, id_j, :] *
                                                                     chi(q_point, list_of_replicas, cell_inv)[l])
            else:
                dynmat_derivatives = contract('ilj,ibljc,l->ibjc',
                                              tf.convert_to_tensor(distance.astype(complex)[..., direction]),
                                              tf.cast(dynmat[0], tf.complex128),
                                              tf.convert_to_tensor(
                                                  chi(q_point, list_of_replicas, cell_inv).flatten().astype(
                                                      complex)),
                                              backend='tensorflow')
        dynmat_derivatives = tf.reshape(dynmat_derivatives, (n_modes, n_modes))
        if self.is_nac:
            dynmat_derivatives += self.nac_derivatives(direction=direction)
        return dynmat_derivatives

    def calculate_sij(self, direction):
        q_point = self.q_point
        shape = (3 * self.atoms.positions.shape[0], 3 * self.atoms.positions.shape[0])
        if self.is_amorphous and (self.q_point == np.array([0, 0, 0])).all():
            type = float
        else:
            type = complex
        eigenvects = self._eigensystem[1:, :]
        if direction == 0:
            dynmat_derivatives = self._dynmat_derivatives_x
        if direction == 1:
            dynmat_derivatives = self._dynmat_derivatives_y
        if direction == 2:
            dynmat_derivatives = self._dynmat_derivatives_z
        if self.atoms.positions.shape[0] > 500:
            # We want to print only for big systems
            logging.info('Flux operators for q = ' + str(q_point) + ', direction = ' + str(direction))
            dir = ['_x', '_y', '_z']
            log_size(shape, type, name='sij' + dir[direction])
        if self.is_amorphous and (self.q_point == np.array([0, 0, 0])).all():
            sij = tf.tensordot(eigenvects, dynmat_derivatives, (0, 1))
            sij = tf.tensordot(eigenvects, sij, (0, 1))
        else:
            eigenvects = tf.cast(eigenvects, tf.complex128)
            dynmat_derivatives = tf.cast(dynmat_derivatives, tf.complex128)
            sij = tf.tensordot(eigenvects, dynmat_derivatives, (0, 1))
            sij = tf.tensordot(tf.math.conj(eigenvects), sij, (0, 1))
        return sij

    def calculate_velocity(self):
        if self.nac_method == 'gonze':
            return self._calculate_gonze_velocity_debug_data()["gv_scaled"][np.newaxis, ...]
        frequency = self.frequency[0]
        velocity = np.zeros((self.n_modes, 3))
        inverse_sqrt_freq = tf.cast(tf.convert_to_tensor(1 / np.sqrt(frequency)), tf.complex128)
        if self.is_amorphous:
            inverse_sqrt_freq = tf.cast(inverse_sqrt_freq, tf.float64)
        for alpha in range(3):
            if alpha == 0:
                sij = self._sij_x
            if alpha == 1:
                sij = self._sij_y
            if alpha == 2:
                sij = self._sij_z
            velocity_AF = 1 / (2 * np.pi) * contract('mn,m,n->mn', sij,
                                                     inverse_sqrt_freq, inverse_sqrt_freq, backend='tensorflow') / 2
            velocity_AF = tf.where(tf.math.is_nan(tf.math.real(velocity_AF)), 0., velocity_AF)
            velocity[..., alpha] = contract('mm->m', velocity_AF.numpy().imag)
        return velocity[np.newaxis, ...]

    def calculate_dynmat_fourier(self):
        q_point = self.q_point
        distance_threshold = self.distance_threshold
        atoms = self.atoms
        n_unit_cell = atoms.positions.shape[0]
        n_replicas = np.prod(self.supercell)
        dynmat = self.second.dynmat
        cell_inv = self.second.cell_inv
        replicated_cell_inv = self.second._replicated_cell_inv
        is_at_gamma = (q_point == (0, 0, 0)).all()
        list_of_replicas = self.second.list_of_replicas
        log_size((self.n_modes, self.n_modes), complex, name='dynmat_fourier')
        if distance_threshold is not None:
            shape = (n_unit_cell, 3, n_unit_cell, 3)
            type = complex
            dyn_s = np.zeros(shape, dtype=type)
            replicated_cell = self.second.replicated_atoms.cell

            for l in range(n_replicas):
                distance_to_wrap = atoms.positions[:, np.newaxis, :] - (
                    self.second.replicated_atoms.positions.reshape(n_replicas, n_unit_cell, 3)[np.newaxis, l, :, :])

                distance_to_wrap = wrap_coordinates(distance_to_wrap, replicated_cell, replicated_cell_inv)

                mask = np.linalg.norm(distance_to_wrap, axis=-1) < distance_threshold
                id_i, id_j = np.argwhere(mask).T
                dyn_s[id_i, :, id_j, :] += dynmat.numpy()[0, id_i, :, 0, id_j, :] * \
                                           chi(q_point, list_of_replicas, cell_inv)[l]
        else:
            if is_at_gamma:
                if self.is_amorphous:
                    dyn_s = dynmat[0]
                else:
                    dyn_s = contract('ialjb->iajb', dynmat[0], backend='tensorflow')
            else:
                dyn_s = contract('ialjb,l->iajb',
                                 tf.cast(dynmat[0], tf.complex128),
                                 tf.convert_to_tensor(chi(q_point, list_of_replicas, cell_inv).flatten()),
                                 backend='tensorflow')
        dyn_s = tf.reshape(dyn_s, (self.n_modes, self.n_modes))
        return dyn_s

    def calculate_eigensystem(self, only_eigenvals):
        if self.nac_method == 'gonze':
            dyn_s = self._calculate_gonze_dynamical_matrix()
            if only_eigenvals:
                return tf.convert_to_tensor(np.linalg.eigvalsh(dyn_s).real)
            log_size(dyn_s.shape, type=complex, name='eigensystem')
            eigenvals, eigenvects = np.linalg.eigh(dyn_s)
            esystem = np.vstack((eigenvals[np.newaxis, :], eigenvects))
            return tf.convert_to_tensor(esystem)
        dyn_s = self._dynmat_fourier
        if self.is_nac:
            dyn_lr = self.nac_dynmat(qpoint=None)
            dyn_lr += self.nac_dynmat(qpoint=self.q_point)
            if (self.q_point == np.array([0, 0, 0])).all():
                dyn_lr = tf.cast(dyn_lr, tf.float64)
            else:
                dyn_lr = tf.cast(dyn_lr, tf.complex128)
            dyn_s += dyn_lr

        if only_eigenvals:
            esystem = tf.linalg.eigvalsh(dyn_s)
        else:
            log_size(self._dynmat_fourier.shape, type=complex, name='eigensystem')
            esystem = tf.linalg.eigh(dyn_s)
            esystem = tf.concat(axis=0, values=(esystem[0][tf.newaxis, :], esystem[1]))
        return esystem

    def calculate_participation_ratio(self):
        n_atoms = self.n_modes // 3
        eigenvectors = self._eigensystem[1:, :]
        eigenvectors = tf.transpose(eigenvectors)
        eigenvectors = np.reshape(eigenvectors, (self.n_modes, n_atoms, 3))
        conjugate = tf.math.conj(eigenvectors)
        participation_ratio = tf.math.reduce_sum(eigenvectors*conjugate, axis=2)
        participation_ratio = tf.math.square(participation_ratio)
        participation_ratio = tf.math.reciprocal(tf.math.reduce_sum(participation_ratio, axis=1) * n_atoms)
        return participation_ratio

    def calculate_eigensystem_unfolded(self, only_eigenvals=False):
        if self.nac_method == 'gonze':
            dyn_s = self._calculate_gonze_dynamical_matrix()
            if only_eigenvals:
                return tf.convert_to_tensor(np.linalg.eigvalsh(dyn_s).real)
            eigenvals, eigenvects = np.linalg.eigh(dyn_s)
            esystem = np.vstack((eigenvals[np.newaxis, :], eigenvects))
            return tf.convert_to_tensor(esystem)
        q_point = self.q_point
        supercell = self.second.supercell
        atoms = self.second.atoms
        cell = atoms.cell
        reciprocal_n = np.round(atoms.cell.reciprocal(), 12)  # round to avoid accumulation of error
        reciprocal_n /= reciprocal_n[0, 0] # Normalized reciprocal cell
        n_unit_cell = len(atoms)
        distances = -1 * atoms.get_all_distances(vector=True, mic=False)
        # -1 in distance calc is to maintain our sign convention

        # Get Force constants
        fc_s = self.second.dynmat.numpy()
        fc_s = fc_s.reshape((n_unit_cell, 3, supercell[0], supercell[1], supercell[2], n_unit_cell, 3))
        supercell_positions = self.second.supercell_positions
        supercell_norms = 1 / 2 * np.linalg.norm(supercell_positions, axis=1) ** 2
        cell_replicas = self.second.supercell_replicas
        cell_positions = contract('ia,ab->ib', cell_replicas, cell)
        cell_plus_distance = cell_positions[:, None, None, :] + distances[None, :, :, :]
        supercell_positions = self.second.supercell_positions
        supercell_cell_distances = contract('La,inma->Linm', supercell_positions, cell_plus_distance)
        projection = supercell_cell_distances - supercell_norms[:, None, None, None]

        # Filter + Weights
        mask_distance = (projection <= 1e-6).all(axis=0)
        n_equivalent = (np.abs(projection) <= 1e-6).sum(axis=0)
        weight = 1 / n_equivalent
        coefficients = weight * mask_distance

        # Find contributing replicas
        mask_full = coefficients.any(axis=(-2, -1))
        coefficients = coefficients[mask_full]
        cell_replicas = cell_replicas[mask_full]
        cell_indices = cell_replicas % supercell

        # Calculate phase and combine with coefficient to normalize contributions from replicas
        # that may be represented more than once
        phase = np.exp(-2j * np.pi * contract('a,ia->i', q_point, cell_replicas))
        prefactors = contract('i,inm->inm', phase, coefficients)
        prefactors = prefactors.repeat(9, axis=0).reshape((-1, 3, 3, n_unit_cell, n_unit_cell))
        prefactors = prefactors.transpose((4, 2, 0, 3, 1))

        # Sum over each contribution after multiplying the force at each replica by the phase + coefficient
        dyn_s = prefactors * fc_s[:, :, cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2], :, :]
        dyn_s = np.transpose(dyn_s, axes=(3, 4, 2, 0, 1))
        dyn_s = dyn_s.sum(axis=2)
        dyn_s = dyn_s.reshape((n_unit_cell * 3, n_unit_cell * 3))

        # Apply correction for Born effective charges, if detected
        if self.is_nac:
            dyn_s += self.nac_dynmat(qpoint=None)
            dyn_s += self.nac_dynmat(qpoint=self.q_point)
        # Diagonalize
        if only_eigenvals:
            omega2, eigenvect, info = zheev(dyn_s, compute_v=False)
            frequency = np.sign(omega2) * np.sqrt(np.abs(omega2))
            frequency = frequency[:] / np.pi / 2
            esystem = (frequency[:] * np.pi * 2) ** 2
        else:
            omega2, eigenvect, info = zheev(dyn_s)
            frequency = np.sign(omega2) * np.sqrt(np.abs(omega2))
            frequency = frequency[:] / np.pi / 2
            esystem = np.vstack(((frequency[:] * np.pi * 2) ** 2, eigenvect))
        return esystem

    def calculate_dynmat_derivatives_unfolded(self, direction):
        q_point = self.q_point
        supercell = self.second.supercell
        atoms = self.second.atoms
        cell = atoms.cell
        reciprocal_n = np.round(atoms.cell.reciprocal(), 12)  # round to avoid accumulation of error
        reciprocal_n /= reciprocal_n[0, 0] # Normalized reciprocal cell
        n_unit_cell = len(atoms)
        distances = -1 * atoms.get_all_distances(vector=True, mic=False)
        # -1 in distance calc is to maintain our sign convention

        # Get Force constants
        fc_s = self.second.dynmat.numpy()
        fc_s = fc_s.reshape((n_unit_cell, 3, supercell[0], supercell[1], supercell[2], n_unit_cell, 3))
        supercell_positions = self.second.supercell_positions
        supercell_norms = (1/2) * np.linalg.norm(supercell_positions, axis=1) ** 2
        cell_replicas = self.second.supercell_replicas
        cell_positions = contract('ia,ab->ib', cell_replicas, cell)
        cell_plus_distance = cell_positions[:, None, None, :] + distances[None, :, :, :]
        supercell_cell_distances = contract('La,inma->Linm', supercell_positions, cell_plus_distance)
        projection = supercell_cell_distances - supercell_norms[:, None, None, None]

        # Filter + Weights
        mask_distance = (projection <= 1e-6).all(axis=0)
        n_equivalent = (np.abs(projection) <= 1e-6).sum(axis=0)
        weight = 1 / n_equivalent
        coefficients = weight * mask_distance

        # Find contributing replicas
        mask_full = coefficients.any(axis=(-2, -1))
        coefficients = coefficients[mask_full]
        cell_replicas = cell_replicas[mask_full]
        cell_positions = cell_positions[mask_full]
        cell_indices = cell_replicas % supercell

        # Calculate phase and combine with coefficient to normalize contributions from replicas
        # that may be represented more than once
        # NOTE: If you wanted to redo this to calculate all the directions at the same time, the first
        # prefactors line is the only place where direction is used.
        phase = np.exp(-2j * np.pi * contract('a,ia->i', q_point, cell_replicas))
        prefactors = contract('i,i,inm->inm', cell_positions[:, direction], phase, coefficients)
        prefactors = prefactors.repeat(9, axis=0).reshape((-1, 3, 3, n_unit_cell, n_unit_cell))
        prefactors = prefactors.transpose((4, 2, 0, 3, 1))

        # Sum over each contribution after multiplying the force at each replica by the phase + coefficient
        ddyn_s = prefactors * fc_s[:, :, cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2], :, :]
        ddyn_s = np.transpose(ddyn_s, axes=(3, 4, 2, 0, 1))
        ddyn_s = ddyn_s.sum(axis=2)
        ddyn_s = ddyn_s.reshape((n_unit_cell * 3, n_unit_cell * 3))

        # Apply correction for Born effective charges, if detected
        if self.is_nac:
            ddyn_s += self.nac_derivatives(direction=direction)
        return ddyn_s

    def nac_dynmat(self, qpoint=None, gmax=None, Lambda=None):
        '''
        Calculate the non-analytic correction to the dynamical matrix.

        Parameters
        ----------
        qpoint : (float, float, float)
            Vector in reciprocal space to measure at. If none, the correction is simpler, using only the second half of
            the second if block here.
        gmax : float
            Maximum g-vector to consider
        Lambda : float
            Parameter for Ewald summation. 1/(4*Lambda) is the cutoff for the

        Returns
        -------
        correction_matrix
        '''
        # Constants, and system information
        ryBr_to_eVA = units.Rydberg / (units.Bohr ** 2)  # Rydberg / Bohr^2 to eV/A^2
        eV_to_10Jmol = units.mol / (10 * units.J)
        e2 = 2.  # square of electron charge in A.U.
        atoms = self.second.atoms
        natoms = len(atoms)
        if gmax is None:
            gmax = 14  # maximum reciprocal vector (same default value in ShengBTE/QE)
        if Lambda is None:
            Lambda = 1 # (2*np.pi*units.Bohr/np.linalg.norm(atoms.cell[0,:]))**2
        geg0 = 4 * Lambda * gmax
        omega_bohr = np.linalg.det(atoms.cell.array / units.Bohr) # Vol. in Bohr^3
        positions_n = atoms.positions.copy() / atoms.cell[0, :].max()  # Normalized positions
        distances_n = positions_n[:, None, :] - positions_n[None, :, :]  # distance in crystal coordinates
        reciprocal_n = np.round(np.linalg.inv(atoms.cell), 12)  # round to avoid accumulation of error
        reciprocal_n /= np.abs(reciprocal_n[0, 0])  # Normalized reciprocal cell
        correction_matrix = tf.zeros([3, 3, natoms, natoms], dtype=tf.complex64)
        prefactor = 4 * np.pi * e2 / omega_bohr

        sqrt_mass = np.sqrt(self.atoms.get_masses().repeat(3, axis=0))
        mass_prefactor = np.reciprocal(contract('i,j->ij', sqrt_mass, sqrt_mass))

        # Charge information
        epsilon = atoms.info['dielectric']  # in e^2/Bohr
        zeff = atoms.get_array('charges')  # in e

        # Charge sum rules
        # Using the "simple" algorithm from QE, we enforce that the sum of
        # charges for each polarization (e.g. xy, or yy) is zero
        zeff -= zeff.mean(axis=0)

        # 1. Construct grid of reciprocal unit cells
        # a. Find the number of replicas to make
        n_greplicas = 2 + 2 * np.sqrt(geg0) / np.linalg.norm(reciprocal_n, axis=0)
        # b. If it's low-dimensional, don't replicate in reciprocal space along axes without replicas in real space
        n_greplicas[np.array(self.second.supercell) == 1] = 1
        # c. Generate the grid of replicas
        g_grid = Grid(n_greplicas.astype(int))
        g_replicas = g_grid.grid(is_wrapping=True)  # minimium distance replicas
        # d. Transform the raw indices, to coordinates in reciprocal space
        g_positions = contract('ib,ab->ia', g_replicas, reciprocal_n)
        if qpoint is not None:  # If we're measuring at finite q, shift the images' positions
            g_positions = g_positions + (qpoint @ reciprocal_n.T)

        # 2. Filter cells that don't meet our Ewald cutoff criteria
        # a. setup mask
        geg = contract('ia,ab,ib->i', g_positions, epsilon, g_positions)
        # change_units_gmax = 16/np.pi**2
        cells_to_include = (geg > 0) * (geg / (4 * Lambda) < gmax)
        # b. apply mask
        geg = geg[cells_to_include]
        g_positions = g_positions[cells_to_include]
        g_replicas = g_replicas[cells_to_include] # for debugging - remove in production

        # 3. Calculate for each cell
        # a. exponential decay term based on distance in reciprocal space, and dielectric tensor
        decay = prefactor * np.exp(-1 * geg / (Lambda * 4)) / geg
        # b. effective charges at each G-vector
        zag = contract('nab,ia->inb', zeff, g_positions)

        # 4. Calculate the actual correction as a product of the effective charges, exponential decay term, and phase factor
        # the phase factor is based on the distance of the G-vector and atomic positions
        # TODO: This "if-else" block could likely be replaced with the just the "if" block since the imaginary term I
        # think should be zero at Gamma, but we'd need to check that for sure.
        if qpoint is not None:
            phase = np.exp(1j * np.pi * contract('ia,nma->inm', g_positions, distances_n))

            # The long range forces are the outer product of the effective charges, scaled by the phase term. We impose
            # Hermicity on cartesian axes by taking the average of M and M^T
            lr_correction = contract('ina,inm,imb->inmab', zag, phase, zag)
            lr_correction += np.transpose(lr_correction, (0, 1, 2, 4, 3))
            lr_correction *= 0.5

            # Scale by exponential decay term, sum over G-vectors
            lr_correction = contract('i,inmab->abnm', decay, lr_correction)

            # Apply the correction to each atom pair
            correction_matrix += lr_correction

        else:  # only the real part of the phase is taken at Gamma
            phase = np.cos(np.pi * contract('ia,nma->inm', g_positions, distances_n))

            # Also, this part of the correction is only applied on "diagonal" choices of atoms. (e.g. 00, 11, 22 etc)
            # The long range forces are an outer product of the effective charges, scaled by the exponential term.
            # We impose Hermicity on cartesian axes by taking the average of M and M^T
            lr_correction = contract('ina,inm,imb->inab', zag, phase, zag)
            lr_correction += np.transpose(lr_correction, (0, 1, 3, 2))
            lr_correction *= 0.5

            # Scale by exponential decay term, sum over G-vectors
            lr_correction = contract('i,inab->abn', decay, lr_correction)

            # Apply the correction to the diagonals of the dynamical matrix
            correction_matrix = tf.linalg.set_diag(correction_matrix,
                                                   tf.linalg.diag_part(correction_matrix) - lr_correction)
        correction_matrix = tf.transpose(correction_matrix, perm=[2, 0, 3, 1])
        correction_matrix = tf.reshape(correction_matrix, shape=(natoms * 3, natoms * 3))
        correction_matrix *= mass_prefactor # 1/sqrt(mass_i * mass_j)
        correction_matrix *= ryBr_to_eVA * eV_to_10Jmol # Rydberg / Bohr^2 to 10J/mol A^2
        return correction_matrix

    def nac_derivatives(self, direction, Lambda=None, gmax=None):
        '''
        Calculate the non-analytic correction to the dynamical matrix.

        qpoint : (float, float, float)
            Vector in reciprocal space to measure at. If none, the correction is simpler, using only the second half of
            the second if block here.
        gmax : float
            Maximum g-vector to consider
        Lambda : float
            Parameter for Ewald summation. 1/(4*Lambda) is the cutoff for the
        Returns
        -------
        correction_matrix
        '''
        # Constants, and system information
        ryBr_to_eVA = units.Rydberg / (units.Bohr ** 2)  # Rydberg / Bohr^2 to eV/A^2
        eV_to_10Jmol = units.mol / (10 * units.J) # eV to 10J/mol
        atoms = self.second.atoms
        natoms = len(atoms)
        cell = atoms.cell
        e2 = 2.  # square of electron charge in A.U.

        # Begin calculated values
        if gmax==None:
            gmax = 14  # maximum reciprocal vector (same default value in ShengBTE/QE)
        if Lambda==None:
            Lambda = (2*np.pi*units.Bohr/np.linalg.norm(cell[0,:]))**2  # Ewald parameter
        geg0 = 4 * Lambda * gmax
        omega_bohr = np.linalg.det(atoms.cell.array / units.Bohr) # Vol. in Bohr^3
        positions_bohr = atoms.positions.copy() / units.Bohr
        distances_bohr = positions_bohr[:, None, :] - positions_bohr[None, :, :]
        reciprocal = 2 * np.pi * np.linalg.inv(atoms.cell / units.Bohr)
        prefactor = 4 * np.pi * e2 / omega_bohr

        sqrt_mass = np.sqrt(self.atoms.get_masses().repeat(3, axis=0))
        mass_prefactor = np.reciprocal(contract('i,j->ij', sqrt_mass, sqrt_mass))

        # Charge information
        epsilon = atoms.info['dielectric']  # in e^2/Bohr
        zeff = atoms.get_array('charges')  # in e

        # Charge sum rules
        # Using the "simple" algorithm from QE, we enforce that the sum of
        # charges for each polarization (e.g. xy, or yy) is zero
        zeff -= zeff.mean(axis=0)

        # 1. Construct grid of reciprocal unit cells
        # a. Find the number of replicas to make
        n_greplicas = 2 + 2 * np.sqrt(geg0) / np.linalg.norm(reciprocal, axis=1)
        # b. If it's low-dimensional, don't replicate in reciprocal space along axes without replicas in real space
        n_greplicas[np.array(self.second.supercell) == 1] = 1
        # c. Generate the grid of replicas
        g_grid = Grid(n_greplicas.astype(int))
        g_replicas = g_grid.grid(is_wrapping=True)  # minimium distance replicas
        # d. Transform the raw indices, to coordinates in reciprocal space
        g_positions = contract('ib,ab->ia', g_replicas, reciprocal)
        g_positions = g_positions + (self.q_point @ reciprocal.T)

        # 2. Filter cells that don't meet our Ewald cutoff criteria
        # a. setup mask
        geg = contract('ia,ab,ib->i', g_positions, epsilon, g_positions)
        cells_to_include = (geg > 0) * (geg / (4 * Lambda) < gmax)
        # b. apply mask
        geg = geg[cells_to_include]
        g_positions = g_positions[cells_to_include]

        # 3. Calculate for each cell
        # a. exponential decay term based on distance in reciprocal space, and dielectric tensor
        decay = prefactor * np.exp(-1 * geg / (Lambda * 4)) / geg
        # b. effective charges at each G-vector
        zag = contract('nab,ia->inb', zeff, g_positions)

        # 4. Calculate the actual correction as a product of the effective charges, exponential decay term, and phase factor
        # the phase factor is based on the distance of the G-vector and atomic positions
        phase = np.exp(1j * contract('ia,nma->inm', g_positions, distances_bohr))
        '''
        # All directions at once code
        # Terms 1 + 2
        zag_zeff = contract('ina,mcb->inmabc', zag, zeff)
        zbg_zeff = np.transpose(zag_zeff, (0, 2, 1, 4, 3, 5))
        # Term 3 (imaginary)
        zag_zbg_rij = 1j * contract('ina,imb,nmc->inmabc', zag, zag, distances_n)
        # Term 4 (negative)
        dgeg = contract('ab,ib->ib', epsilon + epsilon.T, g_positions)
        zag_zbg_dgeg = -1 * contract('ina,imb,ic,i->inmabc', zag, zag, dgeg, (1/(4*Lambda) + 1/geg))

        # Combine terms!
        lr_correction = zag_zeff + zbg_zeff + zag_zbg_rij + zag_zbg_dgeg

        # Scale by exponential decay term
        lr_correction = contract('i,inm,inmabc->nmabc', decay, phase, lr_correction)
        '''
        # Derivative terms in a single direction
        # Terms 1 + 2
        zag_zeff = contract('ina,mb->inmab', zag, zeff[:, direction, :])
        zbg_zeff = np.transpose(zag_zeff, (0, 2, 1, 4, 3))
        # Term 3 (imaginary)
        zag_zbg_rij = 1j * contract('ina,imb,nm->inmab', zag, zag, distances_bohr[:, :, direction])
        # Term 4 (negative)
        dgeg = contract('ab,ib->ib', epsilon + epsilon.T, g_positions)[:, direction]
        zag_zbg_dgeg = -1 * contract('ina,imb,i,i->inmab', zag, zag, dgeg,\
                                      (1/(4*Lambda) + 1/(geg)))
        # Combine terms!
        lr_correction = zag_zeff + zbg_zeff + zag_zbg_rij + zag_zbg_dgeg

        # Scale by exponential decay and phase terms, sum over G-vectors
        # Note: Einsum does not use the distributive property for complex number mult., so we have to
        # do a second multiplication operation when applying the phase factor.
        lr_correction = contract('i,inmab->inmab', decay, lr_correction)
        lr_correction *= phase[:, :, :, None, None]
        lr_correction = lr_correction.sum(axis=0)

        # Rotate, reshape, rescale, and, finally, return correction value
        correction_matrix = np.transpose(lr_correction, axes=(0, 2, 1, 3,))
        correction_matrix = np.reshape(correction_matrix, (natoms * 3, natoms * 3))
        correction_matrix *= mass_prefactor # 1/sqrt(mass_i * mass_j)
        correction_matrix *= units.Bohr * ryBr_to_eVA * eV_to_10Jmol # Rydberg / Bohr^2 to 10J/mol A^2
        correction_matrix = 1j * correction_matrix
        return correction_matrix

    def phonon_mode_frames(self, mode_index, amplitude=0.1, time_step=0.01, n_steps=100):
        """
        Generate frames animating a single phonon eigenmode over the
        replicated supercell.

        For mode (s) at wavevector (q) the displacement of atom *i* inside
        unit-cell replica (l) at time (t) is

            u_{lia}(t) = amplitude * Re[ e_{sia}(q)/sqrt(m_i) * exp(i(2*pi*q.R_l - w_s*t)) ]

        where R_l are the replica positions and w_s = 2*pi*f_s (rad/ps).
        Acoustic modes at Gamma (w_s ~ 0) use a small artificial frequency
        so they oscillate as rigid translations rather than drifting unbounded.

        Parameters
        ----------
        mode_index : int
            Phonon branch index (0-based, ascending frequency).
        amplitude : float
            Peak displacement in Angstroms.
        time_step : float
            Frame interval in picoseconds.
        n_steps : int
            Number of frames after the equilibrium frame.

        Returns
        -------
        frames : list[ase.Atoms]
        """
        if not (0 <= mode_index < self.n_modes):
            raise IndexError(
                f"mode_index {mode_index} out of range [0, {self.n_modes - 1}]."
            )

        n_atoms = len(self.atoms)
        n_replicas = int(np.prod(self.supercell))

        # Eigenvector for this mode, mass-weighted displacement pattern
        eigvec = np.array(self._eigensystem)[1:, mode_index]
        masses = np.repeat(self.atoms.get_masses(), 3)
        disp_cell = eigvec / np.sqrt(masses)
        norm = np.linalg.norm(disp_cell)
        if norm > 0:
            disp_cell /= norm
        disp_cell = (amplitude * disp_cell).reshape(n_atoms, 3)

        freq = float(self.frequency[0, mode_index])
        omega = 2.0 * np.pi * abs(freq)

        # For acoustic modes at Gamma, use the lowest optical frequency
        # so rigid translations still oscillate visually
        if omega < 1e-3:
            physical = self.physical_mode[0]
            optical_freqs = np.abs(self.frequency[0, physical])
            omega = 2.0 * np.pi * optical_freqs.min() if optical_freqs.size > 0 else 1.0

        # Replica phases: q . R_l in fractional coordinates
        rep_pos = self.second.replicated_atoms.positions.reshape(n_replicas, n_atoms, 3)
        R_l = rep_pos[:, 0, :] - self.atoms.positions[0]
        cell_inv = self.second.cell_inv
        phase_l = R_l.dot(cell_inv.dot(self.q_point))

        # Supercell geometry
        eq_positions = self.second.replicated_atoms.positions.copy()
        supercell_cell = self.second.replicated_atoms.cell
        symbols = list(self.atoms.get_chemical_symbols()) * n_replicas

        info = {
            'frequency_THz': freq,
            'q_point': list(self.q_point),
            'mode_index': mode_index,
            'amplitude_A': amplitude,
        }

        frames = []
        for step in range(n_steps + 1):
            t = step * time_step
            pf = np.exp(1j * (2.0 * np.pi * phase_l - omega * t))
            displacements = np.real(
                disp_cell[np.newaxis, :, :] * pf[:, np.newaxis, np.newaxis]
            ).reshape(n_replicas * n_atoms, 3)

            frame = Atoms(
                symbols=symbols,
                positions=eq_positions + displacements,
                cell=supercell_cell,
                pbc=True,
            )
            frame.info = {**info, 'time_ps': t}
            frames.append(frame)

        return frames

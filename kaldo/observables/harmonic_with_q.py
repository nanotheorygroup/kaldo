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
import kaldo.controllers.nac as nac
from kaldo.controllers.nac import (
    normalize_bvk_supercell_matrix,
    ensure_kernel_cache,
    dynamical_matrices,
    NAC_VELOCITY_Q_LENGTH,
    NAC_VELOCITY_CUTOFF_FREQUENCY,
    NAC_VELOCITY_DIRECTIONS_CART,
    _PHONOPY_TO_KALDO_DM,
    degenerate_sets,
    _to_phonopy_dm,
    _phonopy_frequencies_from_eigenvalues,
)
# from numpy.linalg import eigh

logging = get_logger()

MIN_N_MODES_TO_STORE = 1000
# DM conversion: 1 Ry/bohr²/amu in (rad/ps)² = (Ry_to_eV/Å²) × eV_to_10Jmol
# = (units.Ry/units.Bohr²) × (units.mol/(10*units.J))
# Used to convert kALDo-unit DM to phonopy-unit DM for cross-validation.


_warned_incommensurate = False


def _warn_incommensurate_once(q_point, supercell):
    """Warn once per process when a q-point off the supercell-commensurate grid
    is evaluated without Wigner-Seitz unfolding.

    At such q-points the periodic-replica convention used by the default
    dynamical-matrix construction is not invariant under the non-symmorphic
    spacegroup operations, which can break symmetry-protected degeneracies
    (e.g. split transverse-acoustic branches in diamond-structure crystals).
    """
    global _warned_incommensurate
    if _warned_incommensurate:
        return
    scaled = np.asarray(q_point) * np.asarray(supercell)
    if np.allclose(scaled, np.round(scaled), atol=1e-8):
        return
    _warned_incommensurate = True
    logging.warning(
        f'q-point {np.asarray(q_point)} is incommensurate with the supercell {tuple(supercell)}: '
        'the default dynamical-matrix construction can break symmetry-protected degeneracies '
        '(e.g. split transverse-acoustic branches). Consider is_unfolding=True.'
    )


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
                 nac_bvk_supercell_matrix=None,
                 nac_q_direction=(1, 0, 0),
                 nac_precomputed=None,
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
        if not is_unfolding and getattr(second, '_snf_mapping', None) is None:
            # The commensurability heuristic reads self.supercell as an
            # (nx, ny, nz) grid; SNF observables linearize it to (n_rep, 1, 1),
            # which would flag every off-axis q as incommensurate (and
            # is_unfolding is not supported on the SNF path anyway).
            if 'dielectric' not in second.atoms.info:
                _warn_incommensurate_once(q_point, self.supercell)
        has_dielectric = 'dielectric' in self.atoms.info
        if has_dielectric and 'charges' not in self.atoms.arrays:
            raise ValueError(
                "atoms.info['dielectric'] is set but atoms.arrays['charges'] is missing: "
                "the non-analytic correction needs both."
            )
        # Nonpolar QE files can carry a dielectric block with strictly zero Born
        # charges; the correction is identically zero there.
        self.is_nac = bool(
            has_dielectric
            and np.abs(self.atoms.get_array('charges')).max() > 1e-8
        )
        self.nac_bvk_supercell_matrix = normalize_bvk_supercell_matrix(
            nac_bvk_supercell_matrix
        )
        self.nac_q_direction = np.array(nac_q_direction, dtype=float, copy=True)
        self._nac_precomputed = nac_precomputed
        self._nac_runtime_cache = {}
        if self.is_nac and self._nac_precomputed is None:
            self._nac_precomputed = self.second.get_nac_precomputed(
                self._resolve_nac_bvk_supercell_matrix()
            )
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

    def _resolve_nac_bvk_supercell_matrix(self):
        if self.nac_bvk_supercell_matrix is not None:
            return np.array(self.nac_bvk_supercell_matrix, dtype=int, copy=True)
        supercell = np.asarray(self.second.supercell, dtype=int)
        if supercell.shape != (3,):
            raise ValueError(
                "The non-analytic correction requires second.supercell to be a diagonal 3-vector "
                "when nac_bvk_supercell_matrix is not provided."
            )
        return np.diag(supercell)

    def _calculate_nac_dynamical_matrix_for_q(self, q_red, _static_data=None, _mapping=None):
        return self._calculate_nac_dynamical_matrices_for_qs(
            np.array([q_red], dtype=float),
            _static_data,
            _mapping,
        )[0]

    def _calculate_nac_velocity_direction_data(self, direction_index, static_data, _mapping=None):
        if direction_index not in range(4):
            raise ValueError(f"direction_index must be in 0..3, got {direction_index}")
        direction_cart = np.array(
            NAC_VELOCITY_DIRECTIONS_CART[direction_index], dtype=float, copy=True
        )
        dq_cart = direction_cart / np.linalg.norm(direction_cart) * NAC_VELOCITY_Q_LENGTH
        dq_red = static_data["primitive_cell"].T @ dq_cart / units.Bohr
        q_red = np.array(self.q_point, dtype=float, copy=True)
        dm_minus = _to_phonopy_dm(
            self._calculate_nac_dynamical_matrix_for_q(q_red - dq_red, static_data, _mapping)
        )
        dm_plus = _to_phonopy_dm(
            self._calculate_nac_dynamical_matrix_for_q(q_red + dq_red, static_data, _mapping)
        )
        delta_dm = dm_plus - dm_minus
        ddm_fd = delta_dm / (2 * NAC_VELOCITY_Q_LENGTH)
        return {"ddm_fd": ddm_fd}

    def _ensure_nac_runtime_data(self, static_data, mapping):
        static_data, mapping = ensure_kernel_cache(static_data, mapping)
        effective_matrix = self._resolve_nac_bvk_supercell_matrix()
        current_getter = self.second.get_nac_short_range_force_constants
        getter_identity = getattr(current_getter, "__func__", current_getter)
        fc_cache = self._nac_runtime_cache.get("fc_short")
        if (
            fc_cache is None
            or fc_cache["getter_identity"] is not getter_identity
        ):
            fc_short = current_getter(effective_matrix)
            fc_cache = {
                "getter_identity": getter_identity,
                "fc_short": fc_short,
                "fc_short_converted": fc_short * static_data["nac_conversion"],
            }
            self._nac_runtime_cache["fc_short"] = fc_cache
        static_data["fc_short"] = fc_cache["fc_short"]
        static_data["fc_short_converted"] = fc_cache["fc_short_converted"]
        return static_data, mapping

    def _nac_q_direction_carts(self, q_reds, static_data):
        q_reds = np.atleast_2d(np.asarray(q_reds, dtype=float))
        reciprocal_lattice = static_data["reciprocal_lattice"]
        q_carts = np.einsum("ab,qb->qa", reciprocal_lattice, q_reds, optimize=True)
        q_direction_carts = np.array(q_carts, dtype=float, copy=True)
        inactive = np.linalg.norm(q_carts, axis=1) < static_data["q_direction_tolerance"]
        if np.any(inactive):
            nac_direction_cart = reciprocal_lattice @ self.nac_q_direction
            q_direction_carts[inactive] = nac_direction_cart
        return q_carts, q_direction_carts

    def _calculate_nac_dynamical_matrices_for_qs(
        self,
        q_reds,
        _static_data=None,
        _mapping=None,
    ):
        static_data = _static_data if _static_data is not None else self._build_nac_static_data_runtime()
        mapping = _mapping if _mapping is not None else self._build_nac_mapping_runtime(static_data)
        static_data, mapping = self._ensure_nac_runtime_data(static_data, mapping)
        q_reds = np.atleast_2d(np.asarray(q_reds, dtype=float))
        q_carts, q_direction_carts = self._nac_q_direction_carts(q_reds, static_data)
        dm_final = dynamical_matrices(
            q_reds,
            static_data,
            mapping,
            q_direction_carts=q_direction_carts,
            fc=static_data["fc_short_converted"],
        )
        return dm_final

    def _project_nac_group_velocity_raw(self, ddms, eigenvectors, frequencies):
        """Project ddm_fd tensors onto eigenmodes with degenerate perturbation theory.

        ddms[0] is the d0 direction used to lift degeneracy.
        ddms[1:] are the x/y/z axes (d1, d2, d3) for the velocity components.
        Returns gv_raw of shape (n_modes, 3) in phonopy DM derivative units.
        """
        gv_raw = np.zeros((len(frequencies), 3), dtype=float)
        sets = degenerate_sets(frequencies)
        for indices in sets:
            subspace = eigenvectors[:, indices]
            perturbation = subspace.conj().T @ ddms[0] @ subspace
            _, rotation = np.linalg.eigh((perturbation + perturbation.conj().T) / 2)
            rotated = subspace @ rotation
            for axis, ddm in enumerate(ddms[1:]):
                projected = rotated.conj().T @ ddm @ rotated
                gv_raw[np.array(indices), axis] = np.real(np.diag(projected))
        return gv_raw

    def _scale_nac_group_velocity_raw(self, gv_raw, frequencies):
        """Scale raw projected derivatives to group velocity in Å×THz.

        Applies gv = (1/2ω) × dω²/dk, expressed in phonopy units as
        _PHONOPY_TO_KALDO_DM / (8π² × freq_THz).
        """
        scaling = np.zeros(len(frequencies), dtype=float)
        cutoff_mask = (np.abs(frequencies) > NAC_VELOCITY_CUTOFF_FREQUENCY).astype(np.int64)
        active = cutoff_mask.astype(bool)
        # The finite-difference step is taken per 1/Bohr (phonopy convention),
        # so converting the projected derivative to A/ps needs the extra Bohr.
        scaling[active] = _PHONOPY_TO_KALDO_DM * units.Bohr / (8.0 * np.pi ** 2 * frequencies[active])
        gv_scaled = gv_raw * scaling[:, np.newaxis]
        gv_scaled[~active] = 0.0
        return gv_scaled, scaling, cutoff_mask

    def _calculate_nac_velocity_data(self):
        static_data = self._build_nac_static_data_runtime()
        mapping = self._build_nac_mapping_runtime(static_data)
        q_red = np.array(self.q_point, dtype=float, copy=True)
        q_samples = [q_red]
        for direction_cart in NAC_VELOCITY_DIRECTIONS_CART:
            dq_cart = direction_cart / np.linalg.norm(direction_cart) * NAC_VELOCITY_Q_LENGTH
            dq_red = static_data["primitive_cell"].T @ dq_cart / units.Bohr
            q_samples.extend((q_red - dq_red, q_red + dq_red))
        q_samples = np.array(q_samples, dtype=float)
        dm_all = self._calculate_nac_dynamical_matrices_for_qs(q_samples, static_data, mapping)
        dm_q = _to_phonopy_dm(dm_all[0])
        eigenvalues, eigenvectors = np.linalg.eigh(dm_q)
        frequencies = _phonopy_frequencies_from_eigenvalues(eigenvalues.real)
        ddms = []
        for index in range(len(NAC_VELOCITY_DIRECTIONS_CART)):
            dm_minus = _to_phonopy_dm(dm_all[1 + 2 * index])
            dm_plus = _to_phonopy_dm(dm_all[2 + 2 * index])
            ddms.append((dm_plus - dm_minus) / (2 * NAC_VELOCITY_Q_LENGTH))
        gv_raw = self._project_nac_group_velocity_raw(ddms, eigenvectors, frequencies)
        gv_scaled, _, _ = self._scale_nac_group_velocity_raw(gv_raw, frequencies)
        return {"frequencies": frequencies, "gv_scaled": gv_scaled}

    def _build_nac_static_data_runtime(self):
        if self._nac_precomputed is not None:
            return self._nac_precomputed["static_data"]
        matrix = self._resolve_nac_bvk_supercell_matrix()
        self._nac_precomputed = self.second.get_nac_precomputed(matrix)
        data = self._nac_precomputed["static_data"]
        return data

    def _build_nac_mapping_runtime(self, static_data):
        if self._nac_precomputed is not None:
            return self._nac_precomputed["mapping"]
        self._nac_precomputed = self.second.get_nac_precomputed(
            self._resolve_nac_bvk_supercell_matrix()
        )
        return self._nac_precomputed["mapping"]

    def _calculate_nac_dynamical_matrix(self, _static_data=None, _mapping=None):
        return self._calculate_nac_dynamical_matrices_for_qs(
            np.array([self.q_point], dtype=float),
            _static_data,
            _mapping,
        )[0]

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

    def _calculate_nac_dynmat_derivatives(self, direction):
        static_data = self._build_nac_static_data_runtime()
        mapping = self._build_nac_mapping_runtime(static_data)
        data = self._calculate_nac_velocity_direction_data(1 + direction, static_data, mapping)
        return data["ddm_fd"] * (_PHONOPY_TO_KALDO_DM * units.Bohr)

    def calculate_dynmat_derivatives(self, direction):
        if self.is_nac:
            return self._calculate_nac_dynmat_derivatives(direction)
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
        if self.is_nac:
            return self._calculate_nac_velocity_data()["gv_scaled"][np.newaxis, ...]
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

    def _calculate_nac_eigensystem(self, only_eigenvals=False):
        dyn_s = self._calculate_nac_dynamical_matrix()
        if only_eigenvals:
            return np.linalg.eigvalsh(dyn_s).real
        eigenvals, eigenvects = np.linalg.eigh(dyn_s)
        return np.vstack((eigenvals[np.newaxis, :], eigenvects))

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
        if self.is_nac:
            return self._calculate_nac_eigensystem(only_eigenvals=only_eigenvals)
        dyn_s = self._dynmat_fourier

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
        if self.is_nac:
            return self._calculate_nac_eigensystem(only_eigenvals=only_eigenvals)
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
        if self.is_nac:
            return self._calculate_nac_dynmat_derivatives(direction)
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
        return ddyn_s

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

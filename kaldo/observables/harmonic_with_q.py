from dataclasses import dataclass

from kaldo.grid import TranslationSupport, WignerSeitzImages
from kaldo.observables.observable import Observable
import numpy as np
from hashlib import sha256
from ase import units
from ase import Atoms
from opt_einsum import contract
from kaldo.storable import FOLDER_NAME, lazy_property, Storable
import tensorflow as tf
from kaldo.helpers.logger import get_logger, log_size
# from numpy.linalg import eigh

logging = get_logger()

MIN_N_MODES_TO_STORE = 1000
# DM conversion: 1 Ry/bohr²/amu in (rad/ps)² = (Ry_to_eV/Å²) × eV_to_10Jmol
# = (units.Ry/units.Bohr²) × (units.mol/(10*units.J))
# Used to convert kALDo-unit DM to phonopy-unit DM for cross-validation.


_IFC_INTERPOLATION_MODES = ("auto", "wigner-seitz", "periodic")


def _ifc2_source_digest(source_value):
    """Content digest of an IFC2 source tensor for plan-cache identity."""
    if hasattr(source_value, "todense"):
        source_value = source_value.todense()
    array = np.ascontiguousarray(np.asarray(source_value))
    digest = sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _validate_ifc_interpolation(mode):
    """Validate the public IFC interpolation selector without aliases."""
    if mode not in _IFC_INTERPOLATION_MODES:
        raise ValueError(
            f"ifc_interpolation={mode!r} is invalid; choose one of "
            f"{_IFC_INTERPOLATION_MODES}"
        )
    return mode


def resolve_ifc_interpolation(second, requested_mode):
    """Resolve source-aware ``auto`` for an ordinary (NAC-off) IFC2 body."""
    requested_mode = _validate_ifc_interpolation(requested_mode)
    if requested_mode != "auto":
        return requested_mode
    hint = getattr(second, "ifc_interpolation_hint", None)
    if hint is not None:
        if hint not in ("periodic", "wigner-seitz"):
            raise ValueError(f"second.ifc_interpolation_hint={hint!r} is invalid")
        return hint
    if not np.all(np.asarray(second.atoms.pbc, dtype=bool)):
        # Slabs, wires, and molecules keep the historical periodic
        # convention; the pair-image search needs full periodicity.
        return "periodic"
    if second.translation_support.provenance == "file":
        return "file"
    return "wigner-seitz"


@dataclass(frozen=True)
class _HarmonicIFCInterpolation:
    """Immutable Fourier plan for one second-order IFC representation.

    ``values`` follows ``(i, alpha, translation, j, beta)``.  A direct plan
    has one phase per stored translation.  A Wigner--Seitz plan partitions
    each periodic IFC block over every tied shortest image of the *atom pair*.
    The Fourier phase stays in kALDo's integer-translation gauge.  Pair-aware
    Wigner--Seitz plans differentiate the complete Cartesian pair vector
    ``R + r_j - r_i``.  A periodic plan whose source declares the native periodic
    gauge (``ifc_interpolation_hint == 'periodic'``) differentiates the stored
    translation ``R`` only; every other periodic plan keeps the historical
    pair-aware displacement ``R + r_j - r_i``.
    """

    values: np.ndarray
    support: TranslationSupport
    positions: np.ndarray
    cell: np.ndarray
    resolved_mode: str
    images: WignerSeitzImages | None = None
    include_pair_displacement: bool = True

    @classmethod
    def build(cls, second, requested_mode):
        """Resolve provenance, fold only on request, and prepare pair images."""
        support = second.translation_support
        resolved = resolve_ifc_interpolation(second, requested_mode)
        values = np.asarray(second.dynmat)[0]
        if values.shape[2] != support.size:
            raise ValueError(
                "second-order IFC translation axis does not match its "
                f"TranslationSupport: {values.shape[2]} != {support.size}"
            )

        if resolved in ("periodic", "wigner-seitz"):
            # Explicit periodic/WS choices intentionally forget any literal
            # file translation and first form one tensor per periodic class.
            folded = np.zeros(
                (
                    values.shape[0],
                    values.shape[1],
                    support.supercell.size,
                    values.shape[3],
                    values.shape[4],
                ),
                dtype=values.dtype,
            )
            for source_id, class_id in enumerate(support.class_ids):
                folded[:, :, class_id, :, :] += values[:, :, source_id, :, :]
            if support.provenance == "file" and support.size != support.supercell.size:
                logging.warning(
                    "ifc_interpolation=%r folds %d file-provided translations "
                    "into %d periodic classes by explicit user request.",
                    requested_mode,
                    support.size,
                    support.supercell.size,
                )
            support = TranslationSupport.periodic(
                support.supercell, order=support.supercell.order
            )
            values = folded

        positions = np.array(second.atoms.positions, dtype=float, copy=True)
        cell = np.array(second.atoms.cell, dtype=float, copy=True)
        images = (
            WignerSeitzImages.build(support, positions, cell, pbc=second.atoms.pbc)
            if resolved == "wigner-seitz"
            else None
        )
        # Translation-only derivative phases apply only when the source itself
        # declares the native periodic gauge (hint == 'periodic'). Other
        # periodic plans, including the amorphous route, keep the historical
        # pair displacement.
        include_pair_displacement = not (
            resolved == "periodic"
            and getattr(second, "ifc_interpolation_hint", None) == "periodic"
        )
        return cls(
            values,
            support,
            positions,
            cell,
            resolved,
            images,
            include_pair_displacement,
        )

    def matrices(self, q_point, distance_threshold=None):
        """Return the dynamical matrix and its three Cartesian phase kernels.

        The derivative kernels omit the explicit factor ``i``.  kALDo's flux
        projection consumes their imaginary parts, so storing ``-d`` here is
        equivalent to evaluating the real part of ``i*d*D(q)`` in the existing
        velocity convention.
        """
        q_point = np.asarray(q_point, dtype=float)
        n_atoms = len(self.positions)
        dynamical = np.zeros((n_atoms, 3, n_atoms, 3), dtype=np.complex128)
        derivatives = np.zeros((3, n_atoms, 3, n_atoms, 3), dtype=np.complex128)
        fractional_positions = self.positions @ np.linalg.inv(self.cell)

        for source_id, source_translation in enumerate(self.support.translations):
            for atom_i in range(n_atoms):
                for atom_j in range(n_atoms):
                    block = self.values[atom_i, :, source_id, atom_j, :]
                    if not np.any(block):
                        continue
                    if self.images is None:
                        translations = source_translation[np.newaxis, :]
                        pair_offset = (
                            fractional_positions[atom_j] - fractional_positions[atom_i]
                            if self.include_pair_displacement
                            else 0.0
                        )
                        displacements = (source_translation + pair_offset)[
                            np.newaxis, :
                        ] @ self.cell
                        weights = np.ones(1)
                    else:
                        translations, displacements, weights = self.images.image(
                            source_id, atom_i, atom_j
                        )

                    for translation, displacement, weight in zip(
                        translations, displacements, weights
                    ):
                        if (
                            distance_threshold is not None
                            and np.linalg.norm(displacement) >= distance_threshold
                        ):
                            continue
                        phase = np.exp(2j * np.pi * np.dot(q_point, translation))
                        contribution = weight * phase * block
                        dynamical[atom_i, :, atom_j, :] += contribution
                        for direction in range(3):
                            derivatives[direction, atom_i, :, atom_j, :] -= (
                                displacement[direction] * contribution
                            )
        n_modes = 3 * n_atoms
        return dynamical.reshape(n_modes, n_modes), derivatives.reshape(
            3, n_modes, n_modes
        )

    def real_space_moments(self, distance_threshold=None):
        """Return first and second Cartesian moments of the ordinary IFCs.

        The moments use exactly the same pair-specific shortest images and
        tie weights as the Fourier interpolation.  With
        ``d = r_i - (R + r_j)`` they are ``sum(d_x D)`` and
        ``-sum(d_x d_y D)``, matching the long-wavelength expansion consumed
        by :meth:`ForceConstants.elastic_prop`. Unlike :meth:`matrices`, the
        moments always differentiate the full pair vector, matching the
        historical elastic expansion.
        """
        n_atoms = len(self.positions)
        first = np.zeros((n_atoms, 3, n_atoms, 3, 3), dtype=np.complex128)
        second = np.zeros((n_atoms, 3, n_atoms, 3, 3, 3), dtype=np.complex128)
        fractional_positions = self.positions @ np.linalg.inv(self.cell)

        for source_id, source_translation in enumerate(self.support.translations):
            for atom_i in range(n_atoms):
                for atom_j in range(n_atoms):
                    block = self.values[atom_i, :, source_id, atom_j, :]
                    if not np.any(block):
                        continue
                    if self.images is None:
                        displacements = (
                            source_translation
                            + fractional_positions[atom_j]
                            - fractional_positions[atom_i]
                        )[np.newaxis, :] @ self.cell
                        weights = np.ones(1)
                    else:
                        _, displacements, weights = self.images.image(
                            source_id, atom_i, atom_j
                        )
                    for displacement, weight in zip(displacements, weights):
                        if (
                            distance_threshold is not None
                            and np.linalg.norm(displacement) >= distance_threshold
                        ):
                            continue
                        distance = -displacement
                        weighted_block = weight * block
                        first[atom_i, :, atom_j, :, :] += np.einsum(
                            "ab,x->abx", weighted_block, distance
                        )
                        second[atom_i, :, atom_j, :, :, :] -= np.einsum(
                            "ab,x,y->abxy", weighted_block, distance, distance
                        )
        return first, second


class HarmonicWithQ(Observable, Storable):
    """Harmonic observable at one q point."""

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

    def __init__(
        self,
        q_point,
        second,
        distance_threshold=None,
        storage="numpy",
        folder=FOLDER_NAME,
        is_nw=False,
        is_unfolding=False,
        ifc_interpolation="auto",
        is_amorphous=False,
        *kargs,
        **kwargs,
    ):
        """Initialize a q-point calculation and its optional polar correction.

        ``ifc_interpolation='auto'`` uses Wigner--Seitz interpolation for
        compact periodic IFC tensors and retains literal translations from
        formats that provide them. ``'wigner-seitz'`` and ``'periodic'`` are
        explicit overrides.

        When ``second.atoms`` carries a dielectric tensor and Born charges,
        the inline non-analytic correction is added at evaluation time on top
        of the interpolated short-range dynamical matrix.
        """
        super().__init__(folder=folder, *kargs, **kwargs)
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
        if is_unfolding:
            ifc_interpolation = "wigner-seitz"
        self.ifc_interpolation = _validate_ifc_interpolation(ifc_interpolation)
        self.ifc_interpolation_resolved = resolve_ifc_interpolation(
            self.second, self.ifc_interpolation
        )
        self.ifc_cache_key = (
            "v2_"
            + self.ifc_interpolation_resolved
            + "_"
            + self.second.translation_support.digest[:16]
        )
        self._ifc_interpolation_plan = None
        self._ifc_matrices = None
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
        """Cartesian x derivative of the dynamical matrix."""
        return self.calculate_dynmat_derivatives(direction=0)

    @lazy_property(label='<q_point>')
    def _dynmat_derivatives_y(self):
        """Cartesian y derivative of the dynamical matrix."""
        return self.calculate_dynmat_derivatives(direction=1)

    @lazy_property(label='<q_point>')
    def _dynmat_derivatives_z(self):
        """Cartesian z derivative of the dynamical matrix."""
        return self.calculate_dynmat_derivatives(direction=2)

    @lazy_property(label='<q_point>')
    def _dynmat_fourier(self):
        dynmat_fourier = self.calculate_dynmat_fourier()
        return dynmat_fourier

    @lazy_property(label='<q_point>')
    def _eigensystem(self):
        """Eigenvalues and column eigenvectors of the dynamical matrix."""
        return self.calculate_eigensystem(only_eigenvals=False)

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
        """Diagonalize the dynamical matrix and return signed THz frequencies.

        Negative dynamical-matrix eigenvalues represent imaginary modes.  kALDo
        retains their sign instead of discarding that stability information.
        """
        eigenvals = self.calculate_eigensystem(only_eigenvals=True)
        frequency = np.abs(eigenvals) ** 0.5 * np.sign(eigenvals) / (np.pi * 2.0)
        return frequency.real

    def _get_ifc_interpolation_plan(self):
        """Build the q-independent ordinary-IFC Fourier plan once."""
        if self._ifc_interpolation_plan is None:
            cache = getattr(self.second, "_ifc_interpolation_plan_cache", None)
            if cache is None:
                cache = {}
                self.second._ifc_interpolation_plan_cache = cache
            source_value = getattr(self.second, "value", None)
            if source_value is None:
                source_value = self.second.dynmat
            # The cached strong reference pins the array, so an identity hit
            # can never alias a recycled id; reassignment misses and re-digests,
            # but in-place edits of a cached source go undetected.
            cached_digest = getattr(self.second, "_ifc2_digest_cache", None)
            if cached_digest is None or cached_digest[0] is not source_value:
                cached_digest = (source_value, _ifc2_source_digest(source_value))
                self.second._ifc2_digest_cache = cached_digest
            key = (
                self.ifc_interpolation_resolved,
                self.second.translation_support.digest,
                cached_digest[1],
            )
            if key not in cache:
                cache[key] = _HarmonicIFCInterpolation.build(
                    # Pass the public request, not its internal resolution:
                    # ``file`` is provenance used by the plan builder, not a
                    # user-selectable interpolation mode.
                    self.second,
                    self.ifc_interpolation,
                )
            self._ifc_interpolation_plan = cache[key]
        return self._ifc_interpolation_plan

    def _get_ifc_matrices(self):
        """Assemble the dynamical matrix and derivative kernels once per q."""
        if self._ifc_matrices is None:
            self._ifc_matrices = self._get_ifc_interpolation_plan().matrices(
                self.q_point, self.distance_threshold
            )
        return self._ifc_matrices

    def calculate_dynmat_derivatives(self, direction):
        """Return the pair-aware Cartesian Fourier kernel for one direction."""
        dir = ['_x', '_y', '_z']
        type = complex if (not self.is_amorphous) else float
        log_size((1, self.n_modes, self.n_modes), type,
                 name='dynamical_matrix_derivative' + dir[direction])
        _, derivatives = self._get_ifc_matrices()
        derivative = derivatives[direction]
        if self.is_amorphous and (self.q_point == np.array([0, 0, 0])).all():
            # At Gamma a real IFC has a real Fourier derivative in the
            # amorphous Allen--Feldman convention.  The interpolation plan
            # stores complex arrays so it can also represent crystal q points;
            # discard that identically-zero imaginary storage component before
            # the real-valued velocity contraction.
            derivative = derivative.real
        return tf.convert_to_tensor(derivative)

    def calculate_sij(self, direction):
        """Project a dynamical-matrix derivative into the phonon eigenbasis.

        For crystal modes this computes ``e_m^dagger (dD/dq) e_n``.  The
        diagonal imaginary part, combined with ``1/sqrt(omega_m omega_n)`` in
        :meth:`calculate_velocity`, gives the group velocity.  The full matrix
        is also the heat-flux operator used by diffusivity calculations.

        Parameters
        ----------
        direction : int
            Cartesian derivative direction: 0, 1, or 2.
        """
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
            # TensorFlow's Hermitian eigensolver keeps a complex dtype because
            # the common Fourier container is complex.  In this Gamma-only
            # amorphous branch both the IFC matrix and its eigenvectors are
            # physically real; restore that contract before the real-valued
            # Allen--Feldman flux projection.
            eigenvects = tf.math.real(eigenvects)
            dynmat_derivatives = tf.math.real(dynmat_derivatives)
            sij = tf.tensordot(eigenvects, dynmat_derivatives, (0, 1))
            sij = tf.tensordot(eigenvects, sij, (0, 1))
        else:
            eigenvects = tf.cast(eigenvects, tf.complex128)
            dynmat_derivatives = tf.cast(dynmat_derivatives, tf.complex128)
            sij = tf.tensordot(eigenvects, dynmat_derivatives, (0, 1))
            sij = tf.tensordot(tf.math.conj(eigenvects), sij, (0, 1))
        return sij

    def calculate_velocity(self):
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
        """Fourier transform IFCs using the selected translation convention."""
        log_size((self.n_modes, self.n_modes), complex, name='dynmat_fourier')
        dynamical, _ = self._get_ifc_matrices()
        if (self.q_point == np.array([0, 0, 0])).all():
            # Every image phase is one at Gamma, so the matrix is exactly
            # real; keep the historical float64 artifact.
            dynamical = dynamical.real
        return tf.convert_to_tensor(dynamical)

    def calculate_eigensystem(self, only_eigenvals):
        """Return eigenvalues, optionally together with column eigenvectors."""
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
        n_replicas = self.second.n_replicas

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

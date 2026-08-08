"""Long-range dipole electrostatics for harmonic phonons in polar crystals.

Every supported path constructs the full dynamical matrix as

``D_full(q) = FT[IFC_short](q) + D_dipole(q)``.

The controller supports two input conventions, selected from force-constant
provenance rather than from the chemical species or a numerical fallback:

``gonze_total``
    VASP/Phonopy-style inputs provide total finite-supercell IFCs. The Gonze
    dipole matrix is sampled and removed on the commensurate q mesh, the
    remainder is inverse transformed to short-range IFCs, and the matching
    Gonze Ewald matrix is restored at each requested q.

``qe_q2r``
    A polar QE q2r file already contains the analytical, dipole-subtracted IFC
    body produced by ``do_q2r`` calling ``rgd_blk(..., sign=-1)``. Its native
    replica Fourier transform is therefore combined directly with the same QE
    rigid-ion term using ``sign=+1`` and, at Gamma, the directional ``nonanal``
    limit. Running the Gonze subtraction here would remove the dipole term
    twice and mix incompatible lattice gauges.

The invariant is exactly one dipole removal and one matching restoration.
Both paths require a 3x3 electronic dielectric tensor and one 3x3 Born
effective-charge tensor per primitive atom. This module implements and tests
three-dimensional periodic electrostatics only.
"""

import itertools
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import ase.units as units
from numpy.typing import NDArray
import spglib

from kaldo.grid import Grid
from kaldo.interfaces.qe_io import Q2RHeader

# ---------------------------------------------------------------------------
# Shared types and numerical conventions
# ---------------------------------------------------------------------------
_FLOAT = NDArray[np.float64]
_COMPLEX = NDArray[np.complex128]

# Finite-difference and degeneracy thresholds used by the NAC group-velocity
# calculation in HarmonicWithQ. They are unrelated to either Ewald cutoff.
NAC_VELOCITY_Q_LENGTH = 1e-5
NAC_VELOCITY_DEGENERACY_TOLERANCE = 1e-4
NAC_VELOCITY_CUTOFF_FREQUENCY = 1e-4

# Keep the numerical constants used to validate the QE 7.6 rigid-ion
# transcription together. Replacing them piecemeal with another CODATA set
# introduces small but systematic differences from rigid.f90 reference data.
PI = 3.14159265358979323846
TWO_PI, FOUR_PI, E2_RY = 2.0 * PI, 4.0 * PI, 2.0
BOHR_ANGSTROM = 0.529177210903
RY_TO_EV = (4.3597447222071e-18 / 2.0) / 1.602176634e-19
EV_TO_10J_PER_MOL = 6.02214076e23 / (10.0 * (1.0 / 1.602176634e-19))

# Phonopy-style matrices are expressed in its squared-frequency convention;
# kALDo harmonic matrices use 10 J/mol after atom-pair mass weighting.
_PHONOPY_TO_KALDO_DM = (units.Ry / units.Bohr**2) * (units.mol / (10 * units.J))

_GONZE_TOTAL = "gonze_total"
_QE_Q2R = "qe_q2r"

# Fixed Phonopy Gonze defaults reproduced by build_static_data. These values
# define the Ewald representation and are not exposed as user tuning knobs.
_GONZE_RECIPROCAL_POINT_TARGET = 300
_GONZE_EWAL_EXP_CUTOFF = 1e-10
_GONZE_Q_DIRECTION_TOLERANCE = 1e-5


# ---------------------------------------------------------------------------
# QE q2r rigid-ion convention
# ---------------------------------------------------------------------------
_QCoordinates = Literal["crystal", "qe_cartesian"]


class QENACError(ValueError):
    """QE q2r metadata or a requested rigid-ion correction is invalid."""


def qe_default_alpha(alat_bohr: float) -> float:
    """Return the default Ewald alpha used by QE ``rigid.f90``."""
    alat_bohr = float(alat_bohr)
    if not np.isfinite(alat_bohr) or alat_bohr <= 0:
        raise QENACError("alat_bohr must be finite and positive")
    return (alat_bohr / TWO_PI) ** 2


def _vector(value: Sequence[float] | _FLOAT, name: str) -> _FLOAT:
    """Validate a three-component QE vector without changing its convention."""
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise QENACError(f"{name} must contain three finite values")
    return result


QE_GMAX = 14.0
QE_GAMMA_TOLERANCE = 1.0e-6


@dataclass(slots=True)
class _QERigidIonKernel:
    """QE ``rigid.f90`` kernel for an already-short-range q2r IFC body.

    The kernel deliberately keeps QE's native gauge: direct and reciprocal
    vectors are columns in units of ``alat`` and ``2*pi/alat``, respectively,
    while ``tau`` contains Cartesian positions divided by ``alat``. Born
    charges and the electronic dielectric tensor are dimensionless. Only the
    final correction is converted from Ry/bohr² and mass weighted for kALDo.

    This is a three-dimensional transcription of QE 7.6 ``rgd_blk`` and
    ``nonanal``. Two-dimensional LO--TO electrostatics are outside the tested
    scope and are intentionally not represented by this kernel.
    """

    at_columns: _FLOAT
    tau: _FLOAT
    born: _FLOAT
    dielectric: _FLOAT
    q_grid: tuple[int, int, int]
    alpha: float
    alat_bohr: float
    volume_bohr3: float
    masses_amu: _FLOAT
    bg_columns: _FLOAT = field(init=False, repr=False)
    g_vectors: _FLOAT = field(init=False, repr=False)
    onsite: _COMPLEX = field(init=False, repr=False)

    @classmethod
    def from_header(cls, header: Q2RHeader) -> "_QERigidIonKernel":
        """Validate and preserve the rigid-ion metadata from a q2r header."""
        if not header.has_zstar:
            raise QENACError("q2r header does not contain dielectric/Born data")
        return cls(
            at_columns=np.asarray(header.at_columns, dtype=float),
            tau=np.asarray(header.tau, dtype=float),
            born=np.asarray(header.born, dtype=float),
            dielectric=np.asarray(header.dielectric, dtype=float),
            q_grid=tuple(int(value) for value in header.q_grid),
            alpha=float(header.alpha),
            alat_bohr=float(header.alat_bohr),
            volume_bohr3=float(header.volume_bohr3),
            masses_amu=np.asarray(header.atom_masses_amu, dtype=float),
        )

    def __post_init__(self) -> None:
        """Check q2r invariants and precompute the q-independent Ewald terms."""
        if self.at_columns.shape != (3, 3) or abs(np.linalg.det(self.at_columns)) < 1e-14:
            raise QENACError("q2r lattice must be a nonsingular 3x3 matrix")
        if self.tau.ndim != 2 or self.tau.shape[1] != 3:
            raise QENACError("q2r positions must have shape (natom, 3)")
        atom_count = len(self.tau)
        if self.born.shape != (atom_count, 3, 3) or self.dielectric.shape != (3, 3):
            raise QENACError("q2r Born and dielectric shapes disagree with atoms")
        if len(self.q_grid) != 3 or any(value <= 0 for value in self.q_grid):
            raise QENACError("q2r grid must contain three positive integers")
        if self.masses_amu.shape != (atom_count,) or np.any(self.masses_amu <= 0):
            raise QENACError("q2r masses must be positive for every atom")
        native_arrays = (self.at_columns, self.tau, self.born, self.dielectric)
        if not all(np.isfinite(value).all() for value in native_arrays):
            raise QENACError("q2r NAC data contain non-finite values")
        if self.alpha <= 0 or self.alat_bohr <= 0 or self.volume_bohr3 <= 0:
            raise QENACError("q2r alpha, alat, and volume must be positive")

        inferred_volume = abs(np.linalg.det(self.at_columns)) * self.alat_bohr**3
        if not np.isclose(self.volume_bohr3, inferred_volume, rtol=2e-10, atol=1e-10):
            raise QENACError("q2r volume is inconsistent with its lattice")

        self.bg_columns = np.linalg.inv(self.at_columns).T
        self.g_vectors = self._build_g_vectors()
        self.onsite = self._build_onsite_tensor()

    @property
    def atom_count(self) -> int:
        """Number of atoms in the primitive q2r cell."""
        return len(self.tau)

    @property
    def cell_rows_angstrom(self) -> _FLOAT:
        """Return the q2r primitive lattice as ASE-style rows in angstrom."""
        return self.at_columns.T * self.alat_bohr * BOHR_ANGSTROM

    def _build_g_vectors(self) -> _FLOAT:
        """Build the reciprocal box used by QE ``rgd_blk``'s Ewald sum."""
        radius = np.sqrt(4 * QE_GMAX * self.alpha)
        bounds = []
        for axis, count in enumerate(self.q_grid):
            if count == 1:
                bounds.append(0)
                continue
            norm = float(np.linalg.norm(self.bg_columns[:, axis]))
            if norm <= 0:
                raise QENACError("reciprocal lattice has a zero basis vector")
            bounds.append(int(radius / norm) + 1)

        a, b, c = bounds
        return np.asarray(
            [
                i * self.bg_columns[:, 0] + j * self.bg_columns[:, 1] + k * self.bg_columns[:, 2]
                for i in range(-a, a + 1)
                for j in range(-b, b + 1)
                for k in range(-c, c + 1)
            ],
            dtype=float,
        )

    def _screening(self, vector: _FLOAT, sign: float) -> float | None:
        """Return QE's 3D Ewald prefactor for one ``G + q`` vector.

        QE keeps vectors satisfying ``G.epsilon.G / (4*alpha) < 14``; the
        fixed value 14 is the reciprocal-space cutoff in ``rigid.f90``.
        ``None`` denotes a zero or excluded vector.
        """
        geg = float(vector @ self.dielectric @ vector)
        if geg <= 0 or geg / (4 * self.alpha) >= QE_GMAX:
            return None
        return float(
            sign * E2_RY * FOUR_PI / self.volume_bohr3 * np.exp(-geg / (4 * self.alpha)) / geg
        )

    def _charge_projection(self, vector: _FLOAT) -> _FLOAT:
        """Return ``G.Z*`` for every atom, with shape ``(n_atoms, 3)``."""
        return np.einsum("i,nij->nj", vector, self.born, optimize=True)

    def _pair_phase(self, vector: _FLOAT) -> _FLOAT:
        """Return QE's dimensionless ``2*pi*(tau_i-tau_j).vector`` phases."""
        return TWO_PI * np.einsum(
            "abi,i->ab",
            self.tau[:, None, :] - self.tau[None, :, :],
            vector,
            optimize=True,
        )

    def _build_onsite_tensor(self) -> _COMPLEX:
        """Precompute the q-independent onsite subtraction from ``rgd_blk``.

        This compensating diagonal block makes the rigid-ion contribution
        translationally consistent before the finite-q ``G+q`` sum is added.
        The returned tensor has shape ``(n_atoms, 3, n_atoms, 3)`` in
        Ry/bohr², before mass weighting.
        """
        tensor = np.zeros((self.atom_count, 3, self.atom_count, 3), dtype=np.complex128)
        for vector in self.g_vectors:
            prefactor = self._screening(vector, sign=1.0)
            if prefactor is None:
                continue
            projected_charge = self._charge_projection(vector)
            fnat = np.cos(self._pair_phase(vector)) @ projected_charge
            for atom in range(self.atom_count):
                block = np.outer(projected_charge[atom], fnat[atom])
                tensor[atom, :, atom, :] -= prefactor * 0.5 * (block + block.T)
        return tensor

    def to_cartesian(
        self, qpoint: Sequence[float], coordinates: _QCoordinates = "crystal"
    ) -> _FLOAT:
        """Convert q to QE Cartesian coordinates in units of ``2*pi/alat``."""
        qpoint = _vector(qpoint, "qpoint")
        if coordinates == "qe_cartesian":
            return np.array(qpoint, copy=True)
        if coordinates == "crystal":
            return np.asarray(qpoint @ self.bg_columns.T, dtype=float)
        raise QENACError(f"unsupported q coordinate convention {coordinates!r}")

    def is_gamma(self, qpoint: Sequence[float], coordinates: _QCoordinates = "crystal") -> bool:
        """Test for Gamma modulo a reciprocal vector, including skewed cells."""
        q_reduced = self.to_cartesian(qpoint, coordinates) @ self.at_columns
        residual = np.abs(q_reduced - np.rint(q_reduced))
        return bool(np.all(residual <= QE_GAMMA_TOLERANCE))

    def rigid_ion_tensor(self, qpoint: Sequence[float]) -> _COMPLEX:
        """Evaluate QE ``rgd_blk`` at finite q in unweighted Ry/bohr².

        The q-independent onsite subtraction is combined with the screened
        ``G+q`` sum using QE's native lattice columns, charge contraction, and
        pair phase. The result has shape ``(n_atoms, 3, n_atoms, 3)``.
        """
        q_cartesian = self.to_cartesian(qpoint)
        tensor = np.array(self.onsite, copy=True)
        for vector in self.g_vectors:
            shifted = vector + q_cartesian
            prefactor = self._screening(shifted, sign=1.0)
            if prefactor is None:
                continue
            projected_charge = self._charge_projection(shifted)
            phase = np.exp(1j * self._pair_phase(shifted))
            tensor += prefactor * np.einsum(
                "ai,bj,ab->aibj",
                projected_charge,
                projected_charge,
                phase,
                optimize=True,
            )
        return tensor

    def nonanalytic_tensor(self, direction_cartesian: Sequence[float]) -> _COMPLEX:
        """Evaluate QE ``nonanal`` for a Gamma approach direction.

        The denominator ``q.epsilon.q`` and the two ``q.Z*`` projections
        produce the directional LO--TO splitting. A zero direction, or a
        direction with vanishing screened norm, contributes a zero tensor.
        The result is unweighted and expressed in Ry/bohr².
        """
        q = self.to_cartesian(direction_cartesian, "qe_cartesian")
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return np.zeros((self.atom_count, 3, self.atom_count, 3), dtype=np.complex128)
        q /= norm
        qeq = float(q @ self.dielectric @ q)
        if qeq < 1e-8:
            return np.zeros((self.atom_count, 3, self.atom_count, 3), dtype=np.complex128)
        projected_charge = self._charge_projection(q)
        return (
            FOUR_PI
            * E2_RY
            / (qeq * self.volume_bohr3)
            * np.einsum("ai,bj->aibj", projected_charge, projected_charge, optimize=True)
        )

    def correction(
        self,
        qpoint: Sequence[float],
        gamma_direction_cartesian: Sequence[float] | None = None,
    ) -> _COMPLEX:
        """Return the mass-weighted QE restoration in kALDo matrix units.

        ``rgd_blk`` is restored at every q. At exact Gamma, ``nonanal`` is
        additionally evaluated when an approach direction is supplied. This
        is the inverse of q2r's single ``rgd_blk(..., sign=-1)`` removal; it
        must never be combined with the Gonze total-IFC subtraction path.
        """
        tensor = self.rigid_ion_tensor(qpoint)
        if gamma_direction_cartesian is not None and self.is_gamma(qpoint):
            tensor += self.nonanalytic_tensor(gamma_direction_cartesian)

        matrix = tensor.reshape(3 * self.atom_count, 3 * self.atom_count)
        roots = np.repeat(np.sqrt(self.masses_amu), 3)
        matrix *= RY_TO_EV / BOHR_ANGSTROM**2
        matrix /= roots[:, None] * roots[None, :]
        matrix *= EV_TO_10J_PER_MOL

        # Truncating the reciprocal sum leaves roundoff-level anti-Hermitian
        # components; QE's physical dynamical matrix is Hermitian.
        matrix = 0.5 * (matrix + matrix.T.conj())
        diagonal = np.diag_indices_from(matrix)
        matrix[diagonal] = matrix[diagonal].real
        return np.asarray(matrix, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Shared frequency and unit conversions
# ---------------------------------------------------------------------------
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
    """Group adjacent sorted frequencies into degenerate perturbation sets."""
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
    """Convert a kALDo dynamical matrix to Phonopy's frequency-squared units."""
    return np.array(dm, copy=True) / _PHONOPY_TO_KALDO_DM


def _phonopy_frequencies_from_eigenvalues(eigenvalues):
    """Convert signed Phonopy dynamical-matrix eigenvalues to THz."""
    factor = np.sqrt(_PHONOPY_TO_KALDO_DM) / (2 * np.pi)
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * factor


# ---------------------------------------------------------------------------
# Generic Gonze reciprocal-space Ewald kernel
# ---------------------------------------------------------------------------
# All 27 nearest reciprocal-cell images, origin first so exact ties in the
# Brillouin-zone folding fall back to the unshifted point.
_BZ_SEARCH_SPACE = np.array(
    [[0, 0, 0]] + [p for p in itertools.product((-1, 0, 1), repeat=3) if any(p)],
    dtype=np.int64,
)


def _dielectric_part(q_cart, dielectric):
    """Evaluate the screened reciprocal norm ``q.T @ dielectric @ q``."""
    return float(np.einsum("i,ij,j->", q_cart, dielectric, q_cart))


def _get_minimum_g_rad(reciprocal_lattice, g_cutoff, g_rad=100):
    """Find an integer search radius enclosing Phonopy's reciprocal cutoff."""
    for trial_g_rad in range(g_rad, 0, -1):
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    if (a, b, c) == (0, 0, 0):
                        continue
                    norm = np.linalg.norm(reciprocal_lattice @ np.array([a, b, c], dtype=float))
                    if norm * trial_g_rad < g_cutoff:
                        return trial_g_rad + 1
    return g_rad


def _get_g_vec_list(reciprocal_lattice, g_rad):
    """Enumerate reciprocal vectors in the integer cube ``[-g_rad, g_rad]``."""
    npts = g_rad * 2 + 1
    grid = np.array(list(np.ndindex((npts, npts, npts))), dtype=np.int64) - g_rad
    return np.array(grid @ reciprocal_lattice.T, dtype="double", order="C")


def _get_g_list(reciprocal_lattice, g_cutoff):
    """Return reciprocal vectors inside the spherical Gonze cutoff."""
    g_rad = _get_minimum_g_rad(reciprocal_lattice, g_cutoff)
    g_vec_list = _get_g_vec_list(reciprocal_lattice, g_rad)
    g_norm2 = (g_vec_list**2).sum(axis=1)
    return np.array(g_vec_list[g_norm2 < g_cutoff**2], dtype="double", order="C")


def _multiply_borns(dd_in, born):
    """Contract the geometric dipole kernel with Born tensors on both atoms."""
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
    """Build the reciprocal Gonze dipole kernel at one q point.

    The Ewald-screened dyad ``(G+q)(G+q)/(G+q).epsilon.(G+q)`` is summed with
    primitive-atom pair phases. When ``G+q`` vanishes, ``q_direction_cart``
    supplies the directional Gamma limit. Born charges, the physical
    prefactor, onsite subtraction, and mass weighting are applied later.
    """
    if pair_phase is None:
        position_deltas = positions[:, None, :] - positions[None, :, :]
        phases = (
            2j
            * np.pi
            * np.einsum(
                "ga,ija->gij",
                g_list,
                position_deltas,
                optimize=True,
            )
        )
        pair_phase = np.exp(phases)

    g_plus_q = g_list + q_cart[np.newaxis, :]
    norms = np.linalg.norm(g_plus_q, axis=1)
    reciprocal_dyads = np.zeros((len(g_list), 3, 3), dtype=np.complex128)
    four_lambda_squared = 4 * lambda_ * lambda_
    active = norms >= tolerance
    if np.any(active):
        screened_norms = np.einsum(
            "gi,ij,gj->g",
            g_plus_q[active],
            dielectric,
            g_plus_q[active],
            optimize=True,
        )
        ewald_weights = np.exp(-screened_norms / four_lambda_squared) / screened_norms
        # Each dyad is the geometric dipole kernel before Born-charge
        # contraction and before the primitive-atom pair phase is applied.
        reciprocal_dyads[active] = np.einsum(
            "gi,gj,g->gij",
            g_plus_q[active],
            g_plus_q[active],
            ewald_weights,
            optimize=True,
        )
    if q_direction_cart is not None:
        inactive = ~active
        if np.any(inactive):
            directional_screening = _dielectric_part(q_direction_cart, dielectric)
            directional_dyad = (
                np.outer(q_direction_cart, q_direction_cart) / directional_screening
            )
            reciprocal_dyads[inactive] = directional_dyad
    return np.einsum("gab,gij->iajb", reciprocal_dyads, pair_phase, optimize=True)


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
    """Build reciprocal Gonze dipole kernels for a batch of q points.

    ``q_carts`` and ``q_direction_carts`` have shape ``(n_q, 3)``. The
    returned geometric tensor has shape ``(n_q, n_atom, 3, n_atom, 3)``;
    Born-charge contraction, the electrostatic prefactor, onsite subtraction,
    and mass weighting are deliberately left to :func:`dynamical_matrices`.
    """
    if pair_phase is None:
        position_deltas = positions[:, None, :] - positions[None, :, :]
        phases = (
            2j
            * np.pi
            * np.einsum(
                "ga,ija->gij",
                g_list,
                position_deltas,
                optimize=True,
            )
        )
        pair_phase = np.exp(phases)

    q_carts = np.asarray(q_carts, dtype=float)
    g_plus_q = g_list[np.newaxis, :, :] + q_carts[:, np.newaxis, :]
    screened_norms = np.einsum(
        "qgi,ij,qgj->qg", g_plus_q, dielectric, g_plus_q, optimize=True
    )
    norms = np.linalg.norm(g_plus_q, axis=2)
    active = norms >= tolerance
    ewald_weights = np.zeros_like(screened_norms, dtype=np.complex128)
    ewald_weights[active] = np.exp(
        -screened_norms[active] / (4.0 * lambda_ * lambda_)
    ) / screened_norms[active]
    reciprocal_dyads = np.einsum(
        "qgi,qgj,qg->qgij", g_plus_q, g_plus_q, ewald_weights, optimize=True
    )
    if q_direction_carts is not None:
        q_direction_carts = np.asarray(q_direction_carts, dtype=float)
        directional_screening = np.einsum(
            "qi,ij,qj->q",
            q_direction_carts,
            dielectric,
            q_direction_carts,
            optimize=True,
        )
        directional_dyads = np.einsum(
            "qi,qj,q->qij",
            q_direction_carts,
            q_direction_carts,
            1.0 / directional_screening,
            optimize=True,
        )
        reciprocal_dyads += (
            (~active)[..., np.newaxis, np.newaxis]
            * directional_dyads[:, np.newaxis, :, :]
        )
    return np.einsum("qgab,gij->qiajb", reciprocal_dyads, pair_phase, optimize=True)


def _recip_dipole_dipole_q0(g_list, born, dielectric, positions, lambda_, tolerance):
    """Compute Phonopy's Hermitian q=0 onsite drift tensor before scaling."""
    zero = np.zeros(3, dtype="double")
    dd_tmp1 = _get_dd_base(g_list, zero, None, dielectric, positions, lambda_, tolerance)
    dd_tmp2 = _multiply_borns(dd_tmp1, born)
    dd_q0 = dd_tmp2.sum(axis=2)
    dd_q0 = 0.5 * (dd_q0 + np.transpose(dd_q0.conj(), (0, 2, 1)))
    return dd_q0


# ---------------------------------------------------------------------------
# Gonze short-range interpolation and mass weighting
# ---------------------------------------------------------------------------
def _mass_weight(fc_term, masses):
    """Apply ``1/sqrt(M_i M_j)`` and flatten one atom-block tensor."""
    mass_matrix = np.sqrt(np.outer(masses, masses))
    out = np.array(fc_term, dtype=np.complex128, copy=True)
    out /= mass_matrix[:, np.newaxis, :, np.newaxis]
    return out.reshape(len(masses) * 3, len(masses) * 3)


def _mass_weight_many(fc_terms, masses, mass_matrix=None):
    """Apply atom-pair mass weighting to a batch of block tensors."""
    if mass_matrix is None:
        mass_matrix = np.sqrt(np.outer(masses, masses))
    out = np.array(fc_terms, dtype=np.complex128, copy=True)
    out /= mass_matrix[np.newaxis, :, np.newaxis, :, np.newaxis]
    return out.reshape(len(out), len(masses) * 3, len(masses) * 3)


def _build_segment_phase_weights(multi, n_svec):
    """Expand shortest-vector multiplicities into normalized phase weights."""
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
    """Fourier-interpolate Gonze short-range IFCs at one reduced q.

    All equally short Wigner--Seitz vectors for an atom pair are phase averaged,
    matching Phonopy's dynamical-matrix convention. The result is mass weighted,
    flattened to ``(3*n_atoms, 3*n_atoms)``, and made Hermitian.
    """
    num_patom = len(p2s_map)
    is_compact_fc = fc.shape[0] != fc.shape[1]
    if phase_weights is None:
        phase_weights = _build_segment_phase_weights(multi, len(svecs))
    phase_all = np.exp(2j * np.pi * (svecs @ q_red))
    phase_factors = np.einsum("spl,l->sp", phase_weights, phase_all, optimize=True)
    if target_mask is None:
        target_mask = (s2p_map[:, np.newaxis] == p2s_map[np.newaxis, :]).astype(np.complex128)
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
    """Interpolate short-range IFCs at a batch of reduced q points.

    The input ``fc`` is either compact ``(n_p, n_s, 3, 3)`` or full
    ``(n_s, n_s, 3, 3)`` force constants. All shortest-vector phases for each
    atom pair are averaged before the result is mass weighted. The returned
    array has shape ``(n_q, 3*n_p, 3*n_p)`` in kALDo dynamical-matrix units.
    """
    q_reds = np.asarray(q_reds, dtype=float)
    num_patom = len(p2s_map)
    is_compact_fc = fc.shape[0] != fc.shape[1]
    if phase_weights is None:
        phase_weights = _build_segment_phase_weights(multi, len(svecs))
    phase_all = np.exp(2j * np.pi * np.einsum("qa,sa->qs", q_reds, svecs, optimize=True))
    # Average all shortest-path phases for every q-point and primitive/supercell pair.
    phase_factors = np.einsum("spl,ql->qsp", phase_weights, phase_all, optimize=True)
    if target_mask is None:
        target_mask = (s2p_map[:, np.newaxis] == p2s_map[np.newaxis, :]).astype(np.complex128)
    if mass_matrix is None:
        mass_matrix = np.sqrt(np.outer(masses, masses))
    fc_source = fc if is_compact_fc else fc[p2s_map]
    weighted_fc = (
        fc_source[np.newaxis, :, :, :, :]
        * np.transpose(phase_factors, (0, 2, 1))[:, :, :, np.newaxis, np.newaxis]
    )
    dm_blocks = np.einsum("qisab,sj->qijab", weighted_fc, target_mask, optimize=True)
    dm_blocks /= mass_matrix[np.newaxis, :, :, np.newaxis, np.newaxis]
    dm = np.transpose(dm_blocks, (0, 1, 3, 2, 4)).reshape(len(q_reds), num_patom * 3, num_patom * 3)
    return 0.5 * (dm + np.swapaxes(dm.conj(), 1, 2))


# ---------------------------------------------------------------------------
# Born--von Karman geometry and Wigner--Seitz mapping
# ---------------------------------------------------------------------------
def normalize_bvk_supercell_matrix(nac_bvk_supercell_matrix):
    """Validate an optional integer Born--von Karman supercell matrix."""
    if nac_bvk_supercell_matrix is None:
        return None
    matrix = np.array(nac_bvk_supercell_matrix, dtype=int)
    if matrix.shape != (3, 3):
        raise ValueError("nac_bvk_supercell_matrix must be a 3x3 integer matrix.")
    determinant = int(round(np.linalg.det(matrix)))
    if determinant == 0:
        raise ValueError("nac_bvk_supercell_matrix must be non-singular.")
    return matrix


def bvk_supercell_matrix_key(nac_bvk_supercell_matrix):
    """Return a filesystem-safe cache key for a BvK supercell matrix."""
    matrix = normalize_bvk_supercell_matrix(nac_bvk_supercell_matrix)
    if matrix is None:
        return None
    rows = []
    for row in matrix:
        rows.append("_".join(str(int(value)).replace("-", "m") for value in row))
    return "__".join(rows)


def _diagonal_supercell_sort_key(supercell_scaled, n):
    """Sort a diagonal supercell with a fastest and c slowest, as in phonopy."""
    rounded = np.round(np.asarray(supercell_scaled, dtype=float) * n).astype(int) % n
    return (rounded[2], rounded[1], rounded[0])


def _unique_supercell_translations(supercell_matrix, symprec=1e-8):
    """Enumerate unique primitive translations of an integer supercell."""
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


def _phonopy_lattice_points():
    """Return Phonopy's nearby lattice images for shortest-vector searches."""
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
    """Select Phonopy-compatible first-BZ images after Niggli reduction."""
    qpoints = np.array(qpoints, dtype=float, copy=True)
    reciprocal_lattice = np.array(reciprocal_lattice, dtype=float, copy=True)
    distance_tolerance = float(min(np.sum(reciprocal_lattice**2, axis=0)) * tolerance)
    reduced = spglib.niggli_reduce(reciprocal_lattice.T, eps=1.0e-5)
    if reduced is None:
        raise ValueError("Niggli reduction failed for the reciprocal lattice")
    transform = np.linalg.inv(reciprocal_lattice) @ reduced.T
    transform_inverse = np.linalg.inv(transform)
    folded = []
    for qpoint in qpoints:
        reduced_qpoint = qpoint @ transform_inverse.T
        reduced_qpoint -= np.rint(reduced_qpoint)
        candidates = reduced_qpoint + _BZ_SEARCH_SPACE
        distances = np.sum((candidates @ reduced) ** 2, axis=1)
        min_distance = distances.min()
        shortest_indices = np.where(distances < min_distance + distance_tolerance)[0]
        folded.append(candidates[shortest_indices[0]] @ transform.T)
    folded = np.array(folded, dtype="double", order="C")
    folded[np.isclose(folded, 0.0, atol=1e-14)] = 0.0
    return folded


def _commensurate_points(supercell, reciprocal_lattice=None):
    """Return q points commensurate with a diagonal or general BvK cell.

    When a reciprocal lattice is supplied, equivalent q points are folded to
    the same first-Brillouin-zone representatives used by the Gonze kernel.
    """
    supercell = np.array(supercell)
    if supercell.shape == (3,):
        grid = Grid(supercell, order="C").grid(is_wrapping=False)
        qpoints = np.array(grid / np.array(supercell, dtype=float), dtype="double", order="C")
        if reciprocal_lattice is not None:
            return _fold_points_to_first_bz(qpoints, reciprocal_lattice)
        return qpoints
    matrix = normalize_bvk_supercell_matrix(supercell)
    translations = _unique_supercell_translations(matrix.T)
    qpoints = np.array([translation[1] for translation in translations], dtype="double")
    if reciprocal_lattice is not None:
        return _fold_points_to_first_bz(qpoints, reciprocal_lattice)
    return np.array(qpoints, dtype="double", order="C")


# ---------------------------------------------------------------------------
# Full dynamical-matrix assembly
# ---------------------------------------------------------------------------
def _dipole_dipole_dynamical_matrix(q_red, static_data, mapping, q_direction_red=None):
    """Return Phonopy's reciprocal Gonze dipole matrix at one q point."""
    q_red = np.array(q_red, dtype=float, copy=True)
    q_cart = static_data["reciprocal_lattice"] @ q_red
    if q_direction_red is None:
        if np.linalg.norm(q_cart) < static_data["q_direction_tolerance"]:
            q_direction_cart = None
        else:
            q_direction_cart = q_cart
    else:
        q_direction_cart = static_data["reciprocal_lattice"] @ np.array(
            q_direction_red, dtype=float
        )

    dd_recip = _recip_dipole_dipole(
        static_data["dd_q0"],
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
    conversion = units.mol / (10 * units.J)
    return _mass_weight(dd_recip * conversion, static_data["masses"])


def dynamical_matrices(q_reds, static_data, mapping, q_direction_carts, fc=None):
    """Construct full polar dynamical matrices for a batch of reduced q points.

    For ``gonze_total``, ``fc`` contains the once-subtracted short-range IFCs;
    their Wigner--Seitz transform is combined with the Gonze dipole tensor. For
    ``qe_q2r``, the stored q2r body is already short range, so its native
    replica transform is combined with :class:`_QERigidIonKernel`. In both
    cases the returned matrices use kALDo units and are explicitly Hermitian.

    ``q_reds`` and ``q_direction_carts`` have shape ``(n_q, 3)``. ``mapping``
    and ``fc`` are required only for ``gonze_total``; the ``qe_q2r`` branch
    deliberately uses the native replica transform stored on ``SecondOrder``.
    The return shape is ``(n_q, 3*n_atom, 3*n_atom)``.
    """
    q_reds = np.atleast_2d(np.asarray(q_reds, dtype=float))
    q_direction_carts = np.atleast_2d(np.asarray(q_direction_carts, dtype=float))
    convention = static_data.get("convention")

    if convention == _QE_Q2R:
        # q2r's body is already the short-range IFC on kALDo's native replica
        # grid. Preserve the original, format-independent Fourier transform;
        # the Phonopy/Gonze Wigner-Seitz reconstruction above is specifically
        # the total-IFC subtraction strategy and is not an interchangeable
        # representation of this q2r body.
        from kaldo.observables.secondorder import _dynamical_matrix_from_second_order

        dm_short = np.asarray(
            [
                _dynamical_matrix_from_second_order(
                    static_data["qe_second_order"], q_red, include_pair_phase=False
                )
                for q_red in q_reds
            ]
        )
        corrections = np.asarray(
            [
                static_data["qe_kernel"].correction(q_red, q_direction_cart)
                for q_red, q_direction_cart in zip(q_reds, q_direction_carts)
            ]
        )
        result = dm_short + corrections
        return 0.5 * (result + np.swapaxes(result.conj(), 1, 2))

    if convention != _GONZE_TOTAL:
        raise ValueError(f"unknown NAC data convention {convention!r}")

    if fc is None:
        try:
            fc = static_data["fc_short_converted"]
        except KeyError:
            raise ValueError(
                "dynamical_matrices needs short-range force constants: pass fc "
                "explicitly or populate static_data['fc_short_converted'] first "
                "(HarmonicWithQ does this through its runtime cache)."
            ) from None

    # Reciprocal Ewald restoration. The geometric (G+q) kernel is built first,
    # then dressed by Born charges, corrected by the q=0 onsite drift, scaled
    # by the electrostatic prefactor, and finally mass weighted.
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
    diagonal = np.arange(len(static_data["masses"]))
    for atom in diagonal:
        dd_recip[:, atom, :, atom, :] -= static_data["dd_q0"][atom][np.newaxis, :, :]
    dd_recip *= float(static_data["nac_factor"])
    dd_total_mass_weighted = _mass_weight_many(
        dd_recip * static_data["nac_conversion"],
        static_data["masses"],
        mass_matrix=static_data.get("sqrt_mass_matrix"),
    )

    # Fourier-interpolate the once-subtracted IFC body on the same BvK mapping
    # used during the inverse transform. This pairing is what prevents a
    # second dipole subtraction or an incompatible shortest-vector gauge.
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
    """Assemble one scaled reciprocal Gonze dipole tensor including q=0 drift."""
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


# ---------------------------------------------------------------------------
# IFC layout conversion and supercell mapping
# ---------------------------------------------------------------------------
def _inverse_transform_dynmats_to_force_constants(dynmats, qpoints, mapping, masses):
    """Inverse-transform commensurate dynamical matrices to compact IFCs.

    This is the Gonze subtraction bridge: full dynamical matrices minus their
    sampled dipole contribution are transformed into the short-range force
    constants subsequently used for interpolation.
    """
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
    """Convert kALDo replica IFCs to the compact atom-major Gonze layout.

    kALDo stores ``value[j, beta, replica, i, alpha]``: the force on atom
    ``i`` in ``replica`` due to displacement of atom ``j`` in the origin cell.
    Gonze/Phonopy interpolation uses the opposite translation convention, so
    each replica index is mapped to its periodic negative before atom blocks
    are grouped.

    The result has shape ``(n_atom, n_atom*n_replicas, 3, 3)`` in eV/angstrom².
    """
    native_fc = second_order.value[0]
    n_atom = len(second_order.atoms)
    n1, n2, n3 = second_order.supercell
    n_replicas = n1 * n2 * n3
    replica_indices = np.arange(n_replicas)
    translation_1 = replica_indices // (n2 * n3)
    translation_2 = (replica_indices // n3) % n2
    translation_3 = replica_indices % n3
    reversed_replica_indices = (
        (-translation_1 % n1)
        + (-translation_2 % n2) * n1
        + (-translation_3 % n3) * (n1 * n2)
    )
    fc = np.zeros((n_atom, n_atom * n_replicas, 3, 3), dtype=float)
    for displaced_atom in range(n_atom):
        for replica_index in range(n_replicas):
            compact_index = (
                displaced_atom * n_replicas + reversed_replica_indices[replica_index]
            )
            for force_atom in range(n_atom):
                fc[force_atom, compact_index] = native_fc[
                    displaced_atom, :, replica_index, force_atom, :
                ].T
    return fc


def _build_supercell_matrix_mapping(
    atoms,
    supercell_matrix,
    replicated_atoms=None,
    symprec=1e-5,
):
    """Build Phonopy-compatible atom maps and shortest pair vectors.

    The returned compact ordering is atom-major, as required by kALDo's force
    constants. When an explicit replicated cell is available, its positions
    are matched to that ordering instead of being reconstructed from the
    primitive coordinates. This preserves small coordinate differences in
    externally generated Phonopy supercells without coupling the controller to
    Phonopy's own atom ordering.

    ``svecs`` are shortest vectors in supercell fractional coordinates;
    ``phase_svecs`` express the same vectors in primitive fractional
    coordinates for ``exp(2*pi*i*q.R)``. ``multi[..., 0]`` stores how many
    equally short vectors belong to an atom pair and ``multi[..., 1]`` stores
    the first vector's index.
    """

    # Establish a deterministic translation order. Diagonal cells reproduce
    # kALDo's historical C-order replica axis; general integer cells use their
    # fractional coordinates as an order independent of atom enumeration.
    supercell_matrix = np.array(supercell_matrix, dtype=int)
    primitive_matrix = np.linalg.inv(supercell_matrix)
    primitive_cell = np.array(atoms.cell.array, dtype=float, copy=True)
    supercell_cell = supercell_matrix @ primitive_cell
    primitive_scaled = atoms.get_scaled_positions(wrap=False)
    diagonal = np.diag(supercell_matrix)
    if np.all(supercell_matrix == np.diag(diagonal)):
        max_grid_extent = int(np.max(np.abs(diagonal)))
        max_basis_denominator = 1
        for position in primitive_scaled:
            for coordinate in position:
                fractional_coordinate = coordinate % 1.0
                if fractional_coordinate > 1e-10:
                    for denominator in range(1, 64):
                        rational = round(fractional_coordinate * denominator) / denominator
                        if abs(rational - fractional_coordinate) < 1e-8:
                            max_basis_denominator = max(max_basis_denominator, denominator)
                            break
        sort_factor = max_grid_extent * max_basis_denominator

        def sort_key(position):
            """Return the historical diagonal-grid replica order."""
            return _diagonal_supercell_sort_key(position, sort_factor)

    else:
        # The long-range Gonze kernel depends on a consistent translation
        # order, not on a diagonal-grid enumeration. Fractional supercell
        # coordinates provide a deterministic order for a general integer
        # matrix; callers that combine this mapping with short-range IFCs must
        # still ensure that their compact-FC replica axis uses the same order.
        def sort_key(position):
            """Return a deterministic order for a general integer cell."""
            return tuple(np.round(position, 10)[::-1])

    translations = _unique_supercell_translations(supercell_matrix, symprec=symprec)
    translations = sorted(translations, key=lambda item: sort_key(item[1]))
    n_translation = len(translations)
    n_atom = len(atoms)

    # Build the atom-major compact order consumed by _build_interleaved_fc:
    # all translations of primitive atom 0, then atom 1, and so forth.
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

    # When SecondOrder carries an explicit replicated structure, align its
    # coordinates with the compact order while matching chemical identity.
    # This preserves small input-coordinate differences in the vector search.
    if replicated_atoms is not None:
        replicated_cell = np.asarray(replicated_atoms.cell.array, dtype=float)
        if not np.allclose(replicated_cell, supercell_cell, rtol=0, atol=1.0e-6):
            raise ValueError("replicated-atom cell does not match the NAC supercell")
        if len(replicated_atoms) != len(supercell_scaled_positions):
            raise ValueError("replicated-atom count does not match the NAC supercell")

        explicit_positions = np.asarray(
            replicated_atoms.get_scaled_positions(wrap=False), dtype=float
        )
        explicit_symbols = np.asarray(replicated_atoms.get_chemical_symbols())
        primitive_symbols = np.asarray(atoms.get_chemical_symbols())
        available = np.ones(len(explicit_positions), dtype=bool)
        reordered_positions = np.empty_like(supercell_scaled_positions)
        for index, expected_position in enumerate(supercell_scaled_positions):
            delta = explicit_positions - expected_position
            delta -= np.rint(delta)
            distances = np.linalg.norm(delta @ supercell_cell, axis=1)
            compatible = available & (explicit_symbols == primitive_symbols[s2pp_map[index]])
            distances[~compatible] = np.inf
            match = int(np.argmin(distances))
            if not np.isfinite(distances[match]) or distances[match] > max(10 * symprec, 1e-6):
                raise ValueError("could not match replicated atoms to the NAC compact order")
            reordered_positions[index] = explicit_positions[match]
            available[match] = False
        supercell_scaled_positions = reordered_positions

    # Phonopy finds shortest vectors in a Niggli-reduced cell. The transform
    # must be integral so it represents an exact basis change of this lattice,
    # rather than a numerical deformation of the force-constant supercell.
    reduced_cell = spglib.niggli_reduce(supercell_cell, eps=symprec)
    if reduced_cell is None:
        raise ValueError("Niggli reduction failed for the NAC supercell")
    transform_float = supercell_cell @ np.linalg.inv(reduced_cell)
    transform = np.rint(transform_float).astype(np.int64)
    if not np.allclose(transform_float, transform, rtol=0, atol=1.0e-8):
        raise ValueError("NAC supercell Niggli transform is not integral")
    inverse_float = np.linalg.inv(transform)
    inverse_transform = np.rint(inverse_float).astype(np.int64)
    if not np.allclose(inverse_float, inverse_transform, rtol=0, atol=1.0e-8):
        raise ValueError("inverse NAC supercell Niggli transform is not integral")

    reduced_positions = supercell_scaled_positions @ transform
    reduced_positions -= np.rint(reduced_positions)
    primitive_positions_in_supercell = reduced_positions[p2s_map]
    lattice_points = _phonopy_lattice_points()

    # Retain every vector tied for the minimum distance. Their Fourier phases
    # are averaged later; choosing only one would break symmetry at BZ edges.
    svecs = []
    phase_svecs = []
    multi = np.zeros((len(supercell_scaled_positions), n_atom, 2), dtype=np.int64)
    for i_s, supercell_position in enumerate(reduced_positions):
        for i_p, primitive_position in enumerate(primitive_positions_in_supercell):
            candidates_reduced = supercell_position - primitive_position + lattice_points
            distances = np.linalg.norm(candidates_reduced @ reduced_cell, axis=1)
            min_distance = distances.min()
            start = len(svecs)
            for vec_reduced, distance in zip(candidates_reduced, distances):
                if abs(distance - min_distance) < symprec:
                    vec_supercell = vec_reduced @ inverse_transform
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


# ---------------------------------------------------------------------------
# Convention selection and reusable controller data
# ---------------------------------------------------------------------------
def ensure_kernel_cache(static_data, mapping):
    """Populate reusable arrays required by the selected NAC convention.

    The QE kernel owns its own reciprocal and onsite caches and has no Gonze
    mapping. The generic path caches atom-pair phases, mass factors, and
    Wigner--Seitz phase weights in the existing controller dictionaries. Both
    dictionaries are updated in place and returned for the caller's cache.
    """
    convention = static_data.get("convention")
    if convention == _QE_Q2R:
        # The q2r body stays on SecondOrder's native replica grid. Its QE
        # kernel owns every Ewald cache, so the Gonze Wigner--Seitz mapping is
        # intentionally absent.
        return static_data, mapping
    if convention != _GONZE_TOTAL:
        raise ValueError(f"unknown NAC data convention {convention!r}")

    if "pair_phase" not in static_data:
        position_deltas = (
            static_data["primitive_positions"][:, np.newaxis, :]
            - static_data["primitive_positions"][np.newaxis, :, :]
        )
        phases = (
            2j
            * np.pi
            * np.einsum(
                "ga,ija->gij",
                static_data["G_list"],
                position_deltas,
                optimize=True,
            )
        )
        static_data["pair_phase"] = np.exp(phases)
    if "sqrt_mass_matrix" not in static_data:
        static_data["sqrt_mass_matrix"] = np.sqrt(
            np.outer(static_data["masses"], static_data["masses"])
        )
    if "nac_conversion" not in static_data:
        static_data["nac_conversion"] = units.mol / (10 * units.J)
    if "phase_weights" not in mapping:
        phase_svecs = mapping.get("phase_svecs", mapping["svecs"])
        mapping["phase_weights"] = _build_segment_phase_weights(mapping["multi"], len(phase_svecs))
    if "target_mask" not in mapping:
        mapping["target_mask"] = (
            mapping["s2p_map"][:, np.newaxis] == mapping["p2s_map"][np.newaxis, :]
        ).astype(np.complex128)
    return static_data, mapping


def build_static_data(second, matrix=None):
    """Build convention-specific, q-independent NAC data for ``SecondOrder``.

    q2r provenance creates one :class:`_QERigidIonKernel` without altering its
    Born charges. Other polar inputs create the Phonopy-compatible Gonze data
    and enforce the Born-charge acoustic sum rule on a private copy. ``matrix``
    identifies the defining BvK lattice for the generic path; it never
    remeshes QE's native q2r IFC body.

    The returned dictionary is intentionally a controller cache rather than a
    user-facing data model. Its ``convention`` field is the sole dispatch key;
    numerical fallbacks never select between the two physical representations.
    """
    atoms = second.atoms
    qe_header = getattr(second, "_qe_q2r_header", None)
    if qe_header is not None:
        # Unlike the Gonze path below, a polar q2r body already contains
        # short-range IFCs. Prepare only the QE rigid-ion restoration data;
        # do not construct or subtract the Gonze dipole-dipole terms.
        qe_kernel = _QERigidIonKernel.from_header(qe_header)
        if len(atoms) != qe_kernel.atom_count:
            raise QENACError("q2r atom count does not match the SecondOrder primitive cell")
        qe_cell = qe_kernel.cell_rows_angstrom
        if not np.allclose(np.asarray(atoms.cell), qe_cell, rtol=2e-7, atol=2e-7):
            raise QENACError("q2r lattice does not match the SecondOrder primitive cell")
        return {
            "convention": _QE_Q2R,
            "qe_kernel": qe_kernel,
            "qe_second_order": second,
            "primitive_cell": np.array(atoms.cell.array, dtype=float, copy=True),
            # Store the column-action reciprocal transform: ``B @ q_red``.
            # ASE exposes reciprocal basis vectors as rows, whose transpose is
            # required by the controller's matrix-vector convention.
            "reciprocal_lattice": np.linalg.inv(np.array(atoms.cell.array, dtype=float, copy=True)),
            "supercell_cell": np.array(second.replicated_atoms.cell.array, dtype=float, copy=True),
            "q_direction_tolerance": np.array(QE_GAMMA_TOLERANCE),
        }

    born = np.array(atoms.get_array("charges"), dtype=float, copy=True)
    # Match Phonopy's Gonze preparation: enforce the Born acoustic sum rule on
    # a private copy so caller-owned atom data remain unchanged.
    born -= born.mean(axis=0)
    dielectric = np.array(atoms.info["dielectric"], dtype=float, copy=True)
    primitive_cell = np.array(atoms.cell.array, dtype=float, copy=True)
    primitive_positions = np.array(atoms.positions, dtype=float, copy=True)
    # Controller kernels apply this matrix to reduced-coordinate column
    # vectors; for row-vector direct cells that transform is ``inv(cell)``.
    reciprocal_lattice = np.linalg.inv(primitive_cell)
    masses = np.array(atoms.get_masses(), dtype=float, copy=True)
    matrix = normalize_bvk_supercell_matrix(matrix)
    if matrix is None:
        supercell_cell = np.array(second.replicated_atoms.cell.array, dtype=float, copy=True)
    else:
        supercell_cell = np.array(matrix @ primitive_cell, dtype=float, copy=True)
    volume = float(abs(np.linalg.det(primitive_cell)))
    # These are Phonopy's Gonze defaults. Keeping them explicit here makes the
    # reciprocal cutoff and Ewald width part of the reproduced convention,
    # rather than apparently tunable accuracy parameters.
    g_cutoff = float(
        (3 * _GONZE_RECIPROCAL_POINT_TARGET / (4 * np.pi) / volume) ** (1.0 / 3)
    )
    geg = g_cutoff**2 * np.trace(dielectric) / 3
    lambda_ = float(np.sqrt(-geg / 4 / np.log(_GONZE_EWAL_EXP_CUTOFF)))
    unit_conversion_factor = float(atoms.info.get("nac_factor", units.Hartree * units.Bohr))
    nac_factor = float(unit_conversion_factor * 4 * np.pi / volume)
    tolerance = _GONZE_Q_DIRECTION_TOLERANCE
    g_list = _get_g_list(reciprocal_lattice, g_cutoff)
    dd_q0 = _recip_dipole_dipole_q0(
        g_list, born, dielectric, primitive_positions, lambda_, tolerance
    )
    return {
        "convention": _GONZE_TOTAL,
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
    }


def build_mapping(second, matrix=None):
    """Build the Gonze Wigner--Seitz mapping for the IFC supercell.

    The current short-range reconstruction is valid only on the defining
    force-constant grid. Resampling onto a different BvK lattice is rejected
    explicitly rather than risking a misordered or silently incorrect tensor.
    QE q2r evaluation bypasses this function entirely.
    """
    matrix = normalize_bvk_supercell_matrix(matrix)
    fc_diagonal = np.diag(np.asarray(second.supercell, dtype=int))
    if matrix is None:
        # The dedicated no-matrix builder ordered supercell atoms replica-major,
        # inconsistent with the atom-major layout _build_interleaved_fc produces.
        # A diagonal BvK matrix reproduces it through the tested code path.
        matrix = fc_diagonal
    elif not np.array_equal(matrix, fc_diagonal):
        # The short-range pipeline pairs _build_interleaved_fc (whose replica
        # enumeration is a fixed formula on the force-constant supercell)
        # against this mapping's translation ordering, so the two grids must
        # be the same object. Any other BvK matrix would need the force
        # constants resampled onto its lattice, which is not implemented;
        # without this check it fails as an opaque broadcast error deep in
        # the einsum, or worse, silently mispairs blocks.
        raise NotImplementedError(
            "nac_bvk_supercell_matrix must equal diag(supercell) = "
            f"diag{tuple(int(n) for n in second.supercell)} of the force "
            f"constants; got\n{matrix}.\nReconstructing the short-range force "
            "constants on a different Born-von-Karman lattice is not "
            "implemented."
        )
    return _build_supercell_matrix_mapping(
        second.atoms,
        matrix,
        replicated_atoms=second.replicated_atoms,
    )

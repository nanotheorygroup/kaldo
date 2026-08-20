"""Exact reciprocal meshes and real-space translation topology.

The classes in this module deliberately separate three objects that older
kALDo code represented with one ambiguous grid: reciprocal q-point addresses,
periodic supercell equivalence classes, and the translations actually stored
on an IFC tensor axis.  Keeping them distinct prevents reciprocal ordering or
``|det(M)|`` from silently changing a real-space Fourier representation.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ase import Atoms


def _integer_determinant_and_adjugate(matrix):
    """Return exact 3x3 determinant and adjugate using Python integers."""
    a, b, c = ([int(value) for value in row] for row in matrix)
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    adjugate = np.array(
        [
            [
                b[1] * c[2] - b[2] * c[1],
                a[2] * c[1] - a[1] * c[2],
                a[1] * b[2] - a[2] * b[1],
            ],
            [
                b[2] * c[0] - b[0] * c[2],
                a[0] * c[2] - a[2] * c[0],
                a[2] * b[0] - a[0] * b[2],
            ],
            [
                b[0] * c[1] - b[1] * c[0],
                a[1] * c[0] - a[0] * c[1],
                a[0] * b[1] - a[1] * b[0],
            ],
        ],
        dtype=object,
    )
    return determinant, adjugate


def _readonly_integer_matrix(value, name):
    """Validate an ``(n,3)`` lattice-coordinate array and freeze a copy."""
    array = np.asarray(value)
    if array.shape[-1:] != (3,) or array.ndim != 2:
        raise ValueError(f"{name} must have shape (n, 3)")
    if not np.allclose(array, np.rint(array), rtol=0, atol=1e-12):
        raise ValueError(f"{name} must contain integer lattice coordinates")
    array = np.array(np.rint(array), dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class QGrid:
    """Reciprocal-space mesh addresses, without real-space geometry.

    ``addresses`` are integer coordinates on a regular reciprocal mesh and
    ``fractional_points`` are the corresponding reduced wavevectors.  This
    class deliberately knows nothing about atoms or minimum images: a q mesh
    labels Fourier samples, whereas real-space replicas label IFC data.
    """

    shape: tuple[int, int, int]
    order: str = "C"
    addresses: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate the mesh and materialize its deterministic addresses."""
        supplied_shape = np.asarray(self.shape)
        if supplied_shape.shape != (3,) or not np.allclose(
            supplied_shape, np.rint(supplied_shape), rtol=0, atol=1e-12
        ):
            raise ValueError("shape must contain three positive integers")
        shape = tuple(int(n) for n in np.rint(supplied_shape))
        if any(n <= 0 for n in shape):
            raise ValueError("shape must contain three positive integers")
        if self.order not in ("C", "F"):
            raise ValueError("order must be 'C' or 'F'")
        ids = np.arange(int(np.prod(shape)))
        addresses = np.asarray(np.unravel_index(ids, shape, order=self.order)).T
        addresses = np.array(addresses, dtype=np.int64, copy=True)
        addresses.setflags(write=False)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "addresses", addresses)

    @property
    def size(self):
        """Number of reciprocal mesh points."""
        return len(self.addresses)

    @property
    def fractional_points(self):
        """Reduced reciprocal coordinates in the declared storage order."""
        points = self.addresses / np.asarray(self.shape)
        points.setflags(write=False)
        return points

    def address_to_id(self, address):
        """Return the mesh id after exact componentwise reciprocal wrapping."""
        address = np.mod(np.asarray(address, dtype=np.int64), self.shape)
        return int(np.ravel_multi_index(tuple(address), self.shape, order=self.order))

    def partner_id(self, point_id):
        """Return the id of the exact time-reversed mesh point, ``-q``."""
        return self.address_to_id(-self.addresses[int(point_id)])

    def momentum_partner_ids(self, point_id, is_plus):
        """Return all third-phonon partner ids using exact mesh arithmetic.

        For every mesh address ``q'``, the returned address is ``q+q'`` when
        ``is_plus=True`` and ``q-q'`` otherwise.  This preserves
        the momentum convention used by :class:`kaldo.phonons.Phonons` without
        a floating-point fractional-coordinate round trip.
        """
        sign = 1 if is_plus else -1
        target = self.addresses[int(point_id)] + sign * self.addresses
        target = np.mod(target, np.asarray(self.shape, dtype=np.int64))
        return np.ravel_multi_index(target.T, self.shape, order=self.order)


@dataclass(frozen=True)
class SupercellGrid:
    """Exact quotient of primitive translations by a supercell lattice.

    For row-vector lattice coordinates, translations ``R`` and ``R'`` are
    equivalent precisely when ``R-R' = n M`` for an integer row vector ``n``.
    Quotient keys are evaluated with the integer adjugate of ``M``; therefore
    classification has no floating tolerance and no bounded translation
    search.  Diagonal matrices retain the historical C/F replica ordering.
    """

    matrix: NDArray[np.int64]
    order: str = "C"
    representatives: NDArray[np.int64] = field(init=False, repr=False)
    _adjugate: NDArray[np.int64] = field(init=False, repr=False)
    _determinant: int = field(init=False, repr=False)
    _key_to_id: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Build exact quotient representatives and their class lookup."""
        matrix = np.asarray(self.matrix)
        if matrix.shape != (3, 3) or not np.allclose(
            matrix, np.rint(matrix), rtol=0, atol=1e-12
        ):
            raise ValueError("matrix must be a nonsingular integer 3x3 matrix")
        matrix = np.array(np.rint(matrix), dtype=np.int64)
        determinant, adjugate = _integer_determinant_and_adjugate(matrix)
        if determinant == 0:
            raise ValueError("matrix must be nonsingular")
        if self.order not in ("C", "F"):
            raise ValueError("order must be 'C' or 'F'")
        modulus = abs(determinant)

        if np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0 and np.all(
            np.diag(matrix) > 0
        ):
            shape = tuple(int(n) for n in np.diag(matrix))
            ids = np.arange(modulus)
            representatives = np.asarray(
                np.unravel_index(ids, shape, order=self.order)
            ).T
        else:
            # Grow an enumeration region until every quotient key is present.
            # This is finite because the quotient contains exactly |det M| classes.
            zero = np.zeros(3, dtype=np.int64)
            representatives_by_key = {tuple(np.mod(zero @ adjugate, modulus)): zero}
            radius = 0
            while len(representatives_by_key) < modulus:
                radius += 1
                for a in range(-radius, radius + 1):
                    for b in range(-radius, radius + 1):
                        for c in range(-radius, radius + 1):
                            if max(abs(a), abs(b), abs(c)) != radius:
                                continue
                            vector = np.array((a, b, c), dtype=np.int64)
                            key = tuple(np.mod(vector @ adjugate, modulus))
                            representatives_by_key.setdefault(key, vector)
            representatives = np.asarray(list(representatives_by_key.values()))

        matrix.setflags(write=False)
        adjugate.setflags(write=False)
        representatives = np.array(representatives, dtype=np.int64, copy=True)
        representatives.setflags(write=False)
        key_to_id = {
            tuple(np.mod(rep @ adjugate, modulus)): index
            for index, rep in enumerate(representatives)
        }
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "_determinant", determinant)
        object.__setattr__(self, "_adjugate", adjugate)
        object.__setattr__(self, "representatives", representatives)
        object.__setattr__(self, "_key_to_id", key_to_id)

    @property
    def size(self):
        """Number of periodic translation classes, ``abs(det(M))``."""
        return abs(self._determinant)

    def class_key(self, translation):
        """Return the exact adjugate-modulo key of an integer translation."""
        translation = np.asarray(translation)
        if translation.shape != (3,) or not np.allclose(
            translation, np.rint(translation), rtol=0, atol=1e-12
        ):
            raise ValueError("translation must be an integer vector with shape (3,)")
        translation = np.asarray(np.rint(translation), dtype=np.int64)
        return tuple(np.mod(translation @ self._adjugate, self.size))

    def class_id(self, translation):
        """Return the deterministic id of a translation's periodic class."""
        return self._key_to_id[self.class_key(translation)]


@dataclass(frozen=True)
class TranslationSupport:
    """Translations actually carried by an IFC data source.

    Unlike a periodic quotient, this sequence may contain several physical
    translations in the same periodic class.  Keeping those entries distinct
    preserves file-provided translation provenance (notably for TDEP) instead
    of silently overwriting or summing them.
    """

    translations: NDArray[np.int64]
    supercell: SupercellGrid
    provenance: str = "periodic"
    class_ids: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate literal translations and classify them without folding."""
        translations = _readonly_integer_matrix(self.translations, "translations")
        if self.provenance not in ("periodic", "file", "wigner-seitz"):
            raise ValueError("provenance must be 'periodic', 'file', or 'wigner-seitz'")
        if len({tuple(vector) for vector in translations}) != len(translations):
            raise ValueError("translations must not contain exact duplicates")
        class_ids = np.asarray(
            [self.supercell.class_id(r) for r in translations], dtype=np.int64
        )
        class_ids.setflags(write=False)
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "class_ids", class_ids)

    @classmethod
    def periodic(cls, supercell, order="C"):
        """Build compact periodic support in historical centered order.

        Tensor slots follow the quotient's deterministic C/F class order,
        while their Fourier translations use the centered representative of
        each class.  This preserves the established direct-periodic gauge;
        Wigner--Seitz interpolation is invariant to which representative of a
        class supplied the starting block.
        """
        grid = (
            supercell
            if isinstance(supercell, SupercellGrid)
            else SupercellGrid(supercell, order=order)
        )
        numerators = grid.representatives @ grid._adjugate
        supercell_shifts = np.asarray(
            [
                [round(Fraction(int(value), grid._determinant)) for value in row]
                for row in numerators
            ],
            dtype=np.int64,
        )
        centered = grid.representatives - supercell_shifts @ grid.matrix
        return cls(centered, grid, provenance="periodic")

    @property
    def size(self):
        """Number of translations stored by the IFC representation."""
        return len(self.translations)

    @property
    def digest(self):
        """Stable identity for cache namespaces and interpolation plans."""
        payload = np.asarray(self.supercell.matrix, dtype="<i8").tobytes()
        payload += np.asarray(self.translations, dtype="<i8").tobytes()
        payload += self.provenance.encode("ascii")
        return sha256(payload).hexdigest()

    def phases(self, q_points):
        """Return ``exp(2*pi*i*q.R)`` in this support's storage order."""
        q_points = np.atleast_2d(np.asarray(q_points, dtype=float))
        if q_points.shape[1] != 3:
            raise ValueError("q_points must have shape (n, 3)")
        return np.exp(2j * np.pi * q_points @ self.translations.T)


@dataclass(frozen=True)
class WignerSeitzImages:
    """Lazy pair-dependent shortest images used to interpolate nonzero IFCs.

    For each stored translation ``R`` and atom pair ``(i,j)``, the physical
    vector is ``R + r_j-r_i`` in primitive fractional coordinates.  Periodic
    copies differing by a supercell translation are searched and *all* tied
    shortest vectors are retained.  Their weights are normalized to one so a
    periodic IFC block is partitioned without changing its total weight.

    Image searches are deliberately lazy.  Dense eager storage scales as
    ``n_translations * n_atoms**2`` even when nearly every IFC block is zero,
    which is prohibitive for sparse IFC3 and large amorphous cells.  The
    supercell inverse and rigorous search bound are prepared once; ``image``
    computes a nonzero pair on first use and then reuses the cached arrays.
    """

    support: TranslationSupport
    fractional_positions: NDArray[np.float64] = field(repr=False)
    cell: NDArray[np.float64] = field(repr=False)
    super_lattice: NDArray[np.float64] = field(repr=False)
    tolerance: float = 1e-5
    _inverse_super_lattice: NDArray[np.float64] = field(
        repr=False, compare=False, default=None
    )
    _smallest_singular: float = field(repr=False, compare=False, default=None)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def build(cls, support, positions, cell, tolerance=1e-5, pbc=True):
        """Prepare a lazy 3D-periodic shortest-image solver for one support."""
        if not isinstance(support, TranslationSupport):
            raise TypeError("support must be a TranslationSupport")
        cell = np.array(cell, dtype=float, copy=True)
        positions = np.array(positions, dtype=float, copy=True)
        if cell.shape != (3, 3) or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("cell and positions must have shapes (3,3) and (n,3)")
        pbc = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,))
        if not np.all(pbc):
            raise NotImplementedError(
                "Wigner-Seitz IFC interpolation currently requires periodic "
                "boundary conditions in all three directions. Partial-periodic "
                "nanowires and slabs need an axis-restricted image search, which "
                "is not implemented. ifc_interpolation='periodic' is available "
                "only as an explicit diagnostic representation."
            )
        fractional = np.array(positions @ np.linalg.inv(cell), copy=True)
        super_lattice = np.array(support.supercell.matrix @ cell, copy=True)
        inverse = np.linalg.inv(super_lattice)
        smallest_singular = float(np.linalg.svd(super_lattice, compute_uv=False)[-1])
        for array in (fractional, cell, super_lattice, inverse):
            array.setflags(write=False)
        return cls(
            support=support,
            fractional_positions=fractional,
            cell=cell,
            super_lattice=super_lattice,
            tolerance=float(tolerance),
            _inverse_super_lattice=inverse,
            _smallest_singular=smallest_singular,
        )

    def image(self, source_id, atom_i, atom_j):
        """Return translations, Cartesian displacements, and tied weights.

        The result arrays have shapes ``(multiplicity, 3)``,
        ``(multiplicity, 3)``, and ``(multiplicity,)``.  ``source_id`` indexes
        the stored translation support; atom indices select the physical
        vector ``R + r_j - r_i`` whose shortest supercell images are needed.
        """
        key = (int(source_id), int(atom_i), int(atom_j))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if not 0 <= key[0] < self.support.size:
            raise IndexError("source translation index is out of range")
        n_atoms = len(self.fractional_positions)
        if not 0 <= key[1] < n_atoms or not 0 <= key[2] < n_atoms:
            raise IndexError("atom index is out of range")

        source_r = self.support.translations[key[0]]
        base_fractional = (
            source_r
            + self.fractional_positions[key[2]]
            - self.fractional_positions[key[1]]
        )
        shifts, vectors = _all_shortest_supercell_images_prepared(
            base_fractional @ self.cell,
            self.super_lattice,
            self._inverse_super_lattice,
            self._smallest_singular,
            self.tolerance,
        )
        translations = source_r + shifts @ self.support.supercell.matrix
        weights = np.full(len(shifts), 1.0 / len(shifts))
        for array in (translations, vectors, weights):
            array.setflags(write=False)
        result = (translations, vectors, weights)
        self._cache[key] = result
        return result


def _all_shortest_supercell_images_prepared(
    displacement, lattice, inverse_lattice, smallest_singular, tolerance
):
    """Closest-image search with q-independent lattice factors precomputed."""
    center = -displacement @ inverse_lattice
    origin = np.rint(center).astype(np.int64)
    radius = 0
    best = np.inf
    candidates = []
    while True:
        candidates.clear()
        best = np.inf
        for a in range(-radius, radius + 1):
            for b in range(-radius, radius + 1):
                for c in range(-radius, radius + 1):
                    shift = origin + np.array((a, b, c), dtype=np.int64)
                    vector = displacement + shift @ lattice
                    norm = np.linalg.norm(vector)
                    if norm < best - tolerance:
                        best = norm
                        candidates = [(shift, vector)]
                    elif abs(norm - best) <= tolerance:
                        candidates.append((shift, vector))
        # Any unsearched integer point is at least radius+1/2 from the rounded
        # continuous center in max norm, hence at least this Cartesian distance.
        if smallest_singular * (radius + 0.5) > best + tolerance:
            break
        radius += 1
    shifts = np.asarray([item[0] for item in candidates], dtype=np.int64)
    vectors = np.asarray([item[1] for item in candidates], dtype=float)
    order = np.lexsort((shifts[:, 2], shifts[:, 1], shifts[:, 0]))
    return shifts[order], vectors[order]


def _all_shortest_supercell_images_batch(
    displacements, lattice, inverse_lattice, smallest_singular, tolerance
):
    """Vectorized counterpart of ``_all_shortest_supercell_images_prepared``.

    Replays the scalar search for many displacements at once with the same
    arithmetic, the same sequential tie tolerance over the same candidate
    order, and the same rigorous stopping bound, so each row reproduces the
    scalar result bit for bit.  Returns ``(pair_ids, shifts, vectors, counts)``
    flattened over every tied image; rows are grouped by input row in the
    scalar function's shift-lexicographic order and ``counts[p]`` is row
    ``p``'s multiplicity.
    """
    displacements = np.asarray(displacements, dtype=float)
    n_pairs = len(displacements)
    origins = np.rint(-displacements @ inverse_lattice).astype(np.int64)
    counts = np.zeros(n_pairs, dtype=np.int64)
    chunks = []
    active = np.arange(n_pairs)
    radius = 0
    while active.size:
        side = np.arange(-radius, radius + 1, dtype=np.int64)
        offsets = (
            np.stack(np.meshgrid(side, side, side, indexing="ij"), axis=-1)
            .reshape(-1, 3)
        )
        shifts = origins[active, None, :] + offsets[None, :, :]
        vectors = displacements[active, None, :] + shifts @ lattice
        norms = np.sqrt(np.einsum("pkc,pkc->pk", vectors, vectors))
        best = np.full(active.size, np.inf)
        selected = np.zeros(norms.shape, dtype=bool)
        for k in range(norms.shape[1]):
            column = norms[:, k]
            reset = column < best - tolerance
            tied = ~reset & (np.abs(column - best) <= tolerance)
            selected[reset] = False
            selected[reset | tied, k] = True
            best[reset] = column[reset]
        done = smallest_singular * (radius + 0.5) > best + tolerance
        chosen = selected[done]
        finished = active[done]
        counts[finished] = chosen.sum(axis=1)
        chunks.append(
            (
                np.repeat(finished, counts[finished]),
                shifts[done][chosen],
                vectors[done][chosen],
            )
        )
        active = active[~done]
        radius += 1
    pair_ids = np.concatenate([chunk[0] for chunk in chunks])
    shifts = np.concatenate([chunk[1] for chunk in chunks])
    vectors = np.concatenate([chunk[2] for chunk in chunks])
    # Stable sort restores input row order; within a row the candidate scan
    # already visits shifts in the scalar function's lexicographic order.
    order = np.argsort(pair_ids, kind="stable")
    return pair_ids[order], shifts[order], vectors[order], counts


# ---------------------------------------------------------------------------
# Legacy quotient-table grid API.
#
# The interpolation machinery above replaced these classes, but the
# cumulant module (kaldo.cumulant), ForceConstants.irred_to_full, and the
# SNF replica-table tests still consume them as a plain integer-table
# API. They are kept verbatim from the pre-refactor module and carry no
# interpolation semantics.
# ---------------------------------------------------------------------------

def wrap_coordinates(dxij, cell=None, cell_inv=None):
    # exploit periodicity to calculate the shortest distance, which may not be the one we have
    if cell is not None and cell_inv is None:
        cell_inv = np.linalg.inv(cell)
    if cell is not None:
        dxij = dxij.dot(cell_inv)
    dxij = dxij - np.round(dxij)
    if cell is not None:
        dxij = dxij.dot(cell)
    return dxij

class Grid:
    def __init__(self, grid_shape: tuple[int, int, int], order: str = 'C'):
        self.grid_shape = grid_shape
        self.grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.order = order


    def id_to_grid_index(self, id: int):
        grid_shape = self.grid_shape
        index_grid = np.array(np.unravel_index(id, grid_shape, order=self.order)).T
        return np.rint(index_grid).astype(int)


    def generate_index_grid(self):
        ids = np.arange(self.grid_size)
        grid = self.id_to_grid_index(ids)
        return grid


    def grid(self, is_wrapping: bool):
        try:
            index_grid = self._grid
        except AttributeError:
            self._grid = self.generate_index_grid()
            index_grid = self._grid
        if is_wrapping:
            index_grid = wrap_coordinates(index_grid, np.diag(self.grid_shape))
        return np.rint(index_grid).astype(int)

    def grid_index_to_id(self, cell_idx: ArrayLike, is_wrapping: bool = True):
        """Find the id of cell_idx (grid index) in the grid array. is_wrapping indicats if the given grid_index is wrapped or not.

        TODO: use dictonary to solve it, no more mask
        """
        # create mask to find the index
        list_of_index = self.grid(is_wrapping=is_wrapping)
        cell_id_mask = (list_of_index == cell_idx).prod(axis=1)
        cell_id = np.argwhere(cell_id_mask).flatten()

        return cell_id

    def cell_position_to_id(self, cell_position: NDArray, cell: ArrayLike | Atoms, is_wrapping: bool = True):
        """Find which id of grid index in the grid array that the cell position of real space (x, y, z) belongs to.         
        """

        if isinstance(cell, Atoms):
            cell = cell.cell
        
        cell_index = cell_position.dot(np.linalg.inv(cell)).round(0).astype(int)
        cell_id = self.grid_index_to_id(cell_index, is_wrapping=is_wrapping)
        
        return cell_id


class NonDiagonalGrid(Grid):
    """Grid for a non-diagonal primitive-to-supercell tiling.

    Stores an explicit ``replica_table`` of integer lattice vectors in the
    primitive basis. The table holds either one norm-minimal representative
    per Born-von-Karman congruence class (``det(M)`` rows) or, for
    second-order TDEP loads since issue #297, one row per distinct per-pair
    lattice vector found in the force-constant file (several rows of the
    same class). Mimics the :class:`Grid` public interface so downstream
    code (ForceConstant, HarmonicWithQ) can use it interchangeably.

    ``grid()`` ignores ``is_wrapping`` and always returns the stored table
    as-is. ``grid_index_to_id()`` always tries an exact row match first;
    only the congruence-wrapping fallback used on a miss respects the
    flag, returning an empty result when ``is_wrapping=False``.
    """

    def __init__(self, replica_table, M):
        self._replica_table = np.asarray(replica_table, dtype=int)
        self._M = np.asarray(M, dtype=float)
        self.grid_size = len(self._replica_table)
        # Exact-row lookup: the dominant path when parsing per-pair IFC
        # files (issue #297); misses fall back to congruence wrapping.
        self._row_index = {tuple(int(x) for x in r): i for i, r in enumerate(self._replica_table)}
        # Synthesize a shape for the Grid base-class interface (used only
        # for n_replicas bookkeeping downstream).
        self.grid_shape = (self.grid_size, 1, 1)
        self.order = "nondiag"

    def generate_index_grid(self):
        return self._replica_table.copy()

    def grid(self, is_wrapping: bool = True):
        # the stored table is returned as-is
        return self._replica_table.copy()


    def grid_index_to_id(self, cell_idx, is_wrapping: bool = True):
        cell_idx = np.asarray(cell_idx).astype(int)
        hit = self._row_index.get(tuple(int(x) for x in cell_idx))
        if hit is not None:
            return np.array([hit], dtype=int)
        if not is_wrapping:
            return np.array([], dtype=int)
        idx = wrap_lattice_vector_to_replica(
            cell_idx, self._replica_table, self._M,
        )
        if idx < 0:
            return np.array([], dtype=int)
        return np.array([idx], dtype=int)

    def id_to_grid_index(self, id: int):
        return self._replica_table[int(id)].copy()

    def cell_position_to_id(self, cell_position, cell, is_wrapping: bool = True):
        # The base-class implementation casts the position via cell_inv
        # to a 3-tuple of integer indices — meaningful only for diagonal
        # grids. Refuse on non-diagonal so callers don't get silently
        # wrong indices.
        raise NotImplementedError(
            "cell_position_to_id is not defined for NonDiagonalGrid; "
            "use grid_index_to_id with the integer primitive lattice vector."
        )


def wrap_lattice_vector_to_replica(R_prim_int, replica_table, M, tol=1e-4):
    """Find the replica index of a lattice vector R (integer primitive basis).

    ``replica_table`` may hold one row per congruence class (in
    ``[0, 1)``-fractional or Cartesian-norm-minimal form) or several
    congruent rows (per-pair tables, issue #297). An exact match on ``R``
    wins over everything else, so per-pair tables resolve each vector to
    its own row; only then the ``[0, 1)`` wrap and nearby integer shifts
    of M are tried. For deduplicated class tables the priority is
    irrelevant (at most one row is congruent to R), so their behavior is
    unchanged. Returns the index into ``replica_table`` or ``-1`` if no
    match.

    Pure SNF math — no TDEP-format dependencies. Used internally by
    :class:`NonDiagonalGrid` and by the TDEP non-diagonal IFC parsers.
    """
    R = np.asarray(R_prim_int, dtype=int)
    # rint, not a plain int cast: M arrives as float and a stored
    # 2.9999999 must round to 3, not truncate to 2.
    M_rows = np.rint(np.asarray(M)).astype(int)

    # First: sc-fractional [0, 1) wrap
    R_frac_sc = R.astype(float) @ np.linalg.inv(M)
    R_frac_sc_wrap = R_frac_sc - np.floor(R_frac_sc + tol)
    R_frac_prim_wrap = R_frac_sc_wrap @ M
    R_wrap_int = np.round(R_frac_prim_wrap).astype(int)

    # Exact match first, over the whole table: when the table stores
    # per-pair lattice vectors it may contain several rows of the same
    # congruence class, and the entry's own R must win over a same-class
    # sibling that happens to equal R's [0, 1) wrap.
    for idx, rep in enumerate(replica_table):
        if np.array_equal(rep, R):
            return idx
    for idx, rep in enumerate(replica_table):
        if np.array_equal(rep, R_wrap_int):
            return idx

    # Otherwise enumerate nearby integer shifts (replica_table may be norm-minimal)
    for a in (-2, -1, 0, 1, 2):
        for b in (-2, -1, 0, 1, 2):
            for c in (-2, -1, 0, 1, 2):
                R_shift = R - np.array([a, b, c]) @ M_rows
                for idx, rep in enumerate(replica_table):
                    if np.array_equal(rep, R_shift):
                        return idx
    return -1


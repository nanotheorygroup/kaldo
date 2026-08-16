from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np
from numpy.typing import ArrayLike, NDArray
from kaldo.helpers.logger import get_logger
from ase import Atoms
logging = get_logger()


def _integer_determinant_and_adjugate(matrix):
    """Return exact 3x3 determinant and adjugate using Python integers."""
    a, b, c = ([int(value) for value in row] for row in matrix)
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    adjugate = np.array([
        [b[1] * c[2] - b[2] * c[1], a[2] * c[1] - a[1] * c[2], a[1] * b[2] - a[2] * b[1]],
        [b[2] * c[0] - b[0] * c[2], a[0] * c[2] - a[2] * c[0], a[2] * b[0] - a[0] * b[2]],
        [b[0] * c[1] - b[1] * c[0], a[1] * c[0] - a[0] * c[1], a[0] * b[1] - a[1] * b[0]],
    ], dtype=object)
    return determinant, adjugate


def _readonly_integer_matrix(value, name):
    array = np.asarray(value)
    if array.shape[-1:] != (3,) or array.ndim != 2:
        raise ValueError(f"{name} must have shape (n, 3)")
    if not np.allclose(array, np.rint(array)):
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
        shape = tuple(int(n) for n in self.shape)
        if len(shape) != 3 or any(n <= 0 for n in shape):
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
        return len(self.addresses)

    @property
    def fractional_points(self):
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

        For every mesh address ``q'``, the returned address is ``q-q'`` for
        absorption (``is_plus=True``) and ``-q-q'`` for decay.  This preserves
        the momentum convention used by :class:`kaldo.phonons.Phonons` without
        a floating-point fractional-coordinate round trip.
        """
        sign = 1 if is_plus else -1
        target = sign * self.addresses[int(point_id)] - self.addresses
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
        matrix = np.asarray(self.matrix)
        if matrix.shape != (3, 3) or not np.allclose(matrix, np.rint(matrix)):
            raise ValueError("matrix must be a nonsingular integer 3x3 matrix")
        matrix = np.array(np.rint(matrix), dtype=np.int64)
        determinant, adjugate = _integer_determinant_and_adjugate(matrix)
        if determinant == 0:
            raise ValueError("matrix must be nonsingular")
        if self.order not in ("C", "F"):
            raise ValueError("order must be 'C' or 'F'")
        modulus = abs(determinant)

        if np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0 and np.all(np.diag(matrix) > 0):
            shape = tuple(int(n) for n in np.diag(matrix))
            ids = np.arange(modulus)
            representatives = np.asarray(np.unravel_index(ids, shape, order=self.order)).T
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
        return abs(self._determinant)

    def class_key(self, translation):
        translation = np.asarray(translation, dtype=np.int64)
        if translation.shape != (3,):
            raise ValueError("translation must have shape (3,)")
        return tuple(np.mod(translation @ self._adjugate, self.size))

    def class_id(self, translation):
        return self._key_to_id[self.class_key(translation)]

    def canonical_translation(self, class_id):
        """Return the deterministic storage representative of one class."""
        return self.representatives[int(class_id)]


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
        translations = _readonly_integer_matrix(self.translations, "translations")
        if self.provenance not in ("periodic", "file"):
            raise ValueError("provenance must be 'periodic' or 'file'")
        if len({tuple(vector) for vector in translations}) != len(translations):
            raise ValueError("translations must not contain exact duplicates")
        class_ids = np.asarray([self.supercell.class_id(r) for r in translations], dtype=np.int64)
        class_ids.setflags(write=False)
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "class_ids", class_ids)

    @classmethod
    def periodic(cls, supercell, order="C"):
        """Build the compact support used by periodic replica tensors."""
        grid = supercell if isinstance(supercell, SupercellGrid) else SupercellGrid(supercell, order=order)
        return cls(grid.representatives, grid, provenance="periodic")

    @property
    def size(self):
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
    """Pair-dependent shortest real-space images used to interpolate IFCs.

    For each stored translation ``R`` and atom pair ``(i,j)``, the physical
    vector is ``R + r_j-r_i`` in primitive fractional coordinates.  Periodic
    copies differing by a supercell translation are searched and *all* tied
    shortest vectors are retained.  Their weights are normalized to one so a
    periodic IFC block is partitioned without changing its total weight.

    ``translations``, ``displacements``, and ``weights`` are nested tuples
    indexed as ``[stored_translation][i][j]``; the final arrays have shapes
    ``(multiplicity,3)``, ``(multiplicity,3)``, and ``(multiplicity,)``.
    """

    translations: tuple
    displacements: tuple
    weights: tuple

    @property
    def multiplicities(self):
        return tuple(tuple(tuple(len(item) for item in row) for row in block)
                     for block in self.translations)

    @classmethod
    def build(cls, support, positions, cell, tolerance=1e-10):
        if not isinstance(support, TranslationSupport):
            raise TypeError("support must be a TranslationSupport")
        cell = np.asarray(cell, dtype=float)
        positions = np.asarray(positions, dtype=float)
        if cell.shape != (3, 3) or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("cell and positions must have shapes (3,3) and (n,3)")
        fractional = positions @ np.linalg.inv(cell)
        super_lattice = support.supercell.matrix @ cell

        all_t, all_d, all_w = [], [], []
        for source_r in support.translations:
            block_t, block_d, block_w = [], [], []
            for ri in fractional:
                row_t, row_d, row_w = [], [], []
                for rj in fractional:
                    base_fractional = source_r + rj - ri
                    base_cartesian = base_fractional @ cell
                    shifts, vectors = _all_shortest_supercell_images(
                        base_cartesian, super_lattice, tolerance,
                    )
                    translations = source_r + shifts @ support.supercell.matrix
                    weights = np.full(len(shifts), 1.0 / len(shifts))
                    for array in (translations, vectors, weights):
                        array.setflags(write=False)
                    row_t.append(translations)
                    row_d.append(vectors)
                    row_w.append(weights)
                block_t.append(tuple(row_t)); block_d.append(tuple(row_d)); block_w.append(tuple(row_w))
            all_t.append(tuple(block_t)); all_d.append(tuple(block_d)); all_w.append(tuple(block_w))
        return cls(tuple(all_t), tuple(all_d), tuple(all_w))


def _all_shortest_supercell_images(displacement, super_lattice, tolerance):
    """Find all closest vectors in a lattice with a rigorous stopping bound."""
    displacement = np.asarray(displacement, dtype=float)
    lattice = np.asarray(super_lattice, dtype=float)
    center = -displacement @ np.linalg.inv(lattice)
    origin = np.rint(center).astype(np.int64)
    smallest_singular = np.linalg.svd(lattice, compute_uv=False)[-1]
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


    def id_to_unitary_grid_index(self, id: int):
        q_vec = self.id_to_grid_index(id) / self.grid_shape
        return q_vec


    def generate_index_grid(self):
        ids = np.arange(self.grid_size)
        grid = self.id_to_grid_index(ids)
        return grid


    def unitary_grid(self, is_wrapping: bool):
        return self.grid(is_wrapping) / self.grid_shape


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


    @classmethod
    def recover_grid_from_array(cls,
                                replicated_positions: NDArray,
                                supercell: tuple[int, int, int],
                                atoms: Atoms):
        """Build a Grid from given grid array by guessing which type is it and recovering the Grid object. 

        """
        n_replicas, n_unit_atoms, _ = replicated_positions.shape
        detected_grid = np.round(
            (replicated_positions.reshape((n_replicas, n_unit_atoms, 3)) - atoms.positions[np.newaxis, :, :]).dot(
                np.linalg.inv(atoms.cell))[:, 0, :], 0).astype(int)

        grid_c = Grid(grid_shape=supercell, order='C')
        grid_fortran = Grid(grid_shape=supercell, order='F')
        if (grid_c.grid(is_wrapping=False) == detected_grid).all():
            logging.debug("Using C-style position grid")
            return grid_c
        elif (grid_fortran.grid(is_wrapping=False) == detected_grid).all():
            logging.debug("Using fortran-style position grid")
            return grid_fortran
        else:
            err_msg = "Unable to detect grid type"
            logging.error(err_msg)
            raise ValueError(err_msg)


class NonDiagonalGrid(Grid):
    """Grid for a non-diagonal primitive-to-supercell tiling.

    Stores an explicit ``replica_table`` (n_rep × 3) integer lattice vectors
    in the primitive basis, already minimum-image-wrapped inside the
    Wigner-Seitz cell of the supercell. Mimics the :class:`Grid` public
    interface so downstream code (ForceConstant, HarmonicWithQ) can use it
    interchangeably.

    The ``is_wrapping`` flag on ``grid()`` and ``grid_index_to_id()`` is
    accepted for API compatibility with :class:`Grid` but is **always
    treated as True**: the replica table is already minimum-image-wrapped
    by construction.
    """

    def __init__(self, replica_table, M):
        self._replica_table = np.asarray(replica_table, dtype=int)
        self._M = np.asarray(M, dtype=float)
        self.grid_size = len(self._replica_table)
        # Synthesize a shape for the Grid base-class interface (used only
        # for n_replicas bookkeeping downstream).
        self.grid_shape = (self.grid_size, 1, 1)
        self.order = "nondiag"

    def generate_index_grid(self):
        return self._replica_table.copy()

    def grid(self, is_wrapping: bool = True):
        # replica_table is already the minimum-image wrap
        return self._replica_table.copy()

    def unitary_grid(self, is_wrapping: bool = True):
        # Fractional coordinates in the supercell basis
        return self._replica_table @ np.linalg.inv(self._M)

    def grid_index_to_id(self, cell_idx, is_wrapping: bool = True):
        cell_idx = np.asarray(cell_idx).astype(int)
        if is_wrapping:
            idx = wrap_lattice_vector_to_replica(
                cell_idx, self._replica_table, self._M,
            )
            if idx < 0:
                return np.array([], dtype=int)
            return np.array([idx], dtype=int)
        mask = (self._replica_table == cell_idx).all(axis=1)
        return np.argwhere(mask).flatten()

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

    def id_to_unitary_grid_index(self, id: int):
        # Inherited unravel_index would treat (n_rep, 1, 1) as the shape,
        # producing meaningless fractional coords. Return the supercell-
        # fractional coordinate of replica `id` instead.
        return self._replica_table[int(id)] @ np.linalg.inv(self._M)


def wrap_lattice_vector_to_replica(R_prim_int, replica_table, M, tol=1e-4):
    """Find the replica index of a lattice vector R (integer primitive basis).

    Wraps R through the non-diagonal supercell PBC and looks up the unique
    replica entry. ``replica_table`` may be in either ``[0, 1)``-fractional
    form or "Cartesian-norm-minimal" form; we test both candidates via
    (a) the sc-fractional ``[0, 1)`` wrap and (b) nearby integer shifts of
    M that could produce a norm-minimal representative. Returns the index
    into ``replica_table`` or ``-1`` if no match.

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

    # Direct match (handles either-form table)
    for idx, rep in enumerate(replica_table):
        if np.array_equal(rep, R) or np.array_equal(rep, R_wrap_int):
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

"""Finite-range atom graph and structurally orthogonal CPR groups.

Each atom is one column, regardless of Cartesian direction. Atoms in one
group have disjoint structural row supports, so their central differences can
be decompressed without mixing force responses.

The column-grouping construction follows Curtis, Powell & Reid, IMA J.
Appl. Math. 13, 117-119 (1974); Coleman & More, SIAM J. Numer. Anal. 20,
187-209 (1983); and the graph-coloring review by Gebremedhin, Manne &
Pothen, SIAM Rev. 47, 629-705 (2005).
"""

from dataclasses import dataclass
import warnings

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.sparse import csr_array, issparse

from kaldo.helpers.logger import get_logger

try:
    from scipy.optimize._numdiff import group_columns as _scipy_group_columns
except ImportError:  # Private SciPy API; the greedy fallback covers its removal.
    _scipy_group_columns = None


_LOG = get_logger()
_DEFAULT_PAIR_LIMIT = 8 * 1024**3
# Peak bytes per returned pair inside ase.neighborlist.neighbor_list("ijd"),
# measured with tracemalloc on rattled diamond Si (ASE 3.25: 1.1-1.55 kB).
_ASE_BYTES_PER_PAIR = 1600
# Retained int32 pair indices and float64 distances, plus the int64 sort keys
# and permutation used to coalesce them into CSR.
_RETAINED_BYTES_PER_PAIR = 48
_GREEDY_BATCH = 4096


@dataclass(frozen=True)
class NeighborStructure:
    """Boolean atom-pair CSR graph in the calculator's physical cell.

    ``distances`` holds the shortest periodic-image distance of every stored
    pair, aligned with ``matrix.indices``, so smaller radii can be derived
    with :meth:`restrict` instead of a second neighbor search.
    """

    matrix: csr_array
    radius: float
    projected_pairs: int
    projected_bytes: int
    minimum_cell_width: float | None
    number_density: float
    distances: np.ndarray | None = None
    n_chunks: int = 1

    @property
    def n_atoms(self):
        return self.matrix.shape[0]

    @property
    def nnz_blocks(self):
        return self.matrix.nnz

    def neighbors(self, atom):
        """Sorted response atoms structurally affected by displaced ``atom``."""
        start, stop = self.matrix.indptr[atom:atom + 2]
        return self.matrix.indices[start:stop]

    def restrict(self, radius):
        """Return the subgraph with pair distance below ``radius <= self.radius``."""
        radius = float(radius)
        if self.distances is None:
            raise ValueError("structure has no stored pair distances")
        if not 0 < radius <= self.radius:
            raise ValueError("restricted radius must be positive and no larger than the built radius")
        if radius == self.radius:
            return self
        keep = self.distances < radius
        row_ids = np.repeat(np.arange(self.n_atoms, dtype=np.int32), np.diff(self.matrix.indptr))
        indptr = np.zeros(self.n_atoms + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(np.bincount(row_ids[keep], minlength=self.n_atoms))
        indices = self.matrix.indices[keep]
        matrix = csr_array((np.ones(len(indices), dtype=np.bool_), indices, indptr.astype(indices.dtype)),
                           shape=self.matrix.shape)
        matrix.has_sorted_indices = True
        projected = _projected_pairs(self.n_atoms, self.number_density, radius)
        return NeighborStructure(matrix, radius, projected, projected * _ASE_BYTES_PER_PAIR,
                                 self.minimum_cell_width, self.number_density, self.distances[keep], self.n_chunks)


@dataclass(frozen=True)
class Coloring:
    """Deterministic labels for structurally orthogonal atom groups."""

    labels: np.ndarray
    seeds: tuple[int, ...]
    n_groups: int
    geometric_lower_bound: float | None

    def members(self, group):
        return np.flatnonzero(self.labels == group)


def perpendicular_cell_widths(atoms):
    """Heights of periodic cell faces, in Angstrom; empty for nonperiodic cells."""
    periodic = np.flatnonzero(atoms.pbc)
    if not len(periodic):
        return np.empty(0, dtype=np.float64)
    cell = np.asarray(atoms.cell, dtype=np.float64)
    if np.linalg.matrix_rank(cell[periodic]) != len(periodic):
        raise ValueError("periodic calculator cell vectors must be independent")
    try:
        inverse = np.linalg.inv(np.asarray(atoms.cell.complete(), dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise ValueError("periodic calculator cell must be nonsingular") from exc
    return 1.0 / np.linalg.norm(inverse[:, periodic], axis=0)


def _density(atoms, radius):
    if np.all(atoms.pbc):
        volume = abs(float(np.linalg.det(atoms.cell)))
    else:
        span = np.ptp(np.asarray(atoms.positions), axis=0)
        volume = float(np.prod(np.maximum(span, radius)))
    if volume <= 0:
        raise ValueError("calculator cell or bounding box has zero volume")
    return len(atoms) / volume


def _projected_pairs(n_atoms, density, radius):
    pairs = int(min(n_atoms**2, np.ceil(n_atoms * density * (4 * np.pi / 3) * radius**3)))
    return max(pairs, n_atoms)


def _slab_axis(atoms):
    """Heights along the widest cell-face normal, with its extent and period vector.

    Heights are measured along the unit normal to the two other cell vectors,
    so periodic images along those vectors share a height.
    """
    cell = np.asarray(atoms.cell.complete(), dtype=np.float64)
    inverse = np.linalg.inv(cell)
    widths = 1.0 / np.linalg.norm(inverse, axis=0)
    heights = np.asarray(atoms.positions, dtype=np.float64) @ (inverse * widths)
    extents = np.where(atoms.pbc, widths, np.ptp(heights, axis=0))
    axis = int(np.argmax(extents))
    return axis, heights[:, axis], float(extents[axis]), bool(atoms.pbc[axis]), cell[axis]


def _slab_count(atoms, radius, projected_pairs, pair_memory_limit):
    """Fewest slabs whose padded neighbor search fits ``pair_memory_limit``."""
    if pair_memory_limit is None or projected_pairs * _ASE_BYTES_PER_PAIR <= pair_memory_limit:
        return 1
    retained = projected_pairs * _RETAINED_BYTES_PER_PAIR
    fraction = (pair_memory_limit - retained) / (projected_pairs * _ASE_BYTES_PER_PAIR)
    _, _, extent, _, _ = _slab_axis(atoms)
    slab = fraction * extent - 2 * radius
    if fraction <= 0 or slab <= 0:
        raise MemoryError(f"neighbor search needs about {projected_pairs * _ASE_BYTES_PER_PAIR / 1e9:.2f} GB and "
                          f"slabs padded by twice the {radius:g} Angstrom radius cannot fit pair_memory_limit; "
                          "reduce the radius or raise pair_memory_limit")
    return int(np.ceil(extent / slab))


def _slab_neighbor_pairs(atoms, radius, n_chunks):
    """Neighbor pairs for slab-interior centers, searched in padded slab windows.

    Along a periodic slab axis, window atoms are translated by whole period
    vectors so each window is contiguous, and the axis is searched as
    nonperiodic. A window shorter than the period holds each atom at most
    once, and every image within ``radius`` of an interior center lies inside
    the window, because a pair closer than ``radius`` is closer than
    ``radius`` along the face normal.
    """
    axis, heights, extent, periodic, period_vector = _slab_axis(atoms)
    origin = 0.0 if periodic else float(heights.min())
    relative = heights - origin
    if periodic:
        relative = np.mod(relative, extent)
        relative[relative >= extent] = 0.0
    slab = extent / n_chunks
    pad = radius * (1 + 1e-9) + 1e-9  # Extra window atoms are harmless; missing ones are not.
    if periodic and slab + 2 * pad >= extent:
        raise ValueError("radius is too large for periodic slab chunking; use fewer chunks")
    owner = np.minimum((relative / slab).astype(np.int64), n_chunks - 1)
    sub_pbc = np.array(atoms.pbc, dtype=np.bool_)
    sub_pbc[axis] = False
    numbers = np.asarray(atoms.numbers)
    positions = np.asarray(atoms.positions, dtype=np.float64)
    rows, columns, distances = [], [], []
    for chunk in range(n_chunks):
        start = chunk * slab - pad
        offset = relative - start
        if periodic:
            offset = np.mod(offset, extent)
        members = np.flatnonzero((offset >= 0) & (offset < slab + 2 * pad))
        window = positions[members]
        if periodic:
            images = np.rint((origin + start + offset[members] - heights[members]) / extent)
            window = window + images[:, np.newaxis] * period_vector
        sub = Atoms(numbers=numbers[members], positions=window, cell=atoms.cell, pbc=sub_pbc)
        i, j, d = neighbor_list("ijd", sub, radius, self_interaction=False)
        keep = owner[members[i]] == chunk
        rows.append(members[i[keep]].astype(np.int32))
        columns.append(members[j[keep]].astype(np.int32))
        distances.append(d[keep])
        del i, j, d, keep
        _LOG.info("Colored IFC neighbor slab %d/%d: %d window atoms", chunk + 1, n_chunks, len(members))
    return np.concatenate(rows), np.concatenate(columns), np.concatenate(distances)


def _coalesced_csr(rows, columns, distances, n_atoms):
    """CSR pattern with diagonal; duplicate image pairs keep their shortest distance."""
    diagonal = np.arange(n_atoms, dtype=np.int64)
    keys = np.concatenate((rows.astype(np.int64) * n_atoms + columns, diagonal * (n_atoms + 1)))
    distances = np.concatenate((np.asarray(distances, dtype=np.float64), np.zeros(n_atoms)))
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    distances = distances[order]
    del order
    starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    distances = np.minimum.reduceat(distances, starts)
    keys = keys[starts]
    row_ids = keys // n_atoms
    indices = (keys - row_ids * n_atoms).astype(np.int32)
    indptr = np.zeros(n_atoms + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(row_ids, minlength=n_atoms))
    matrix = csr_array((np.ones(len(indices), dtype=np.bool_), indices, indptr.astype(np.int32)),
                       shape=(n_atoms, n_atoms))
    matrix.has_sorted_indices = True
    return matrix, distances


def build_neighbor_structure(atoms, radius, *, pair_memory_limit=_DEFAULT_PAIR_LIMIT, n_chunks=None):
    """Build ``S[a,b] = (distance(a, b) < radius for some periodic image)`` with diagonal.

    ASE enumerates periodic images. Duplicate atom pairs are coalesced to the
    shortest image distance because the structure records possible dependence,
    rather than image multiplicity. ASE's search peaks near 1.6 kB per
    returned pair; when that projected peak exceeds ``pair_memory_limit``,
    centers are processed in slabs across the widest cell direction, each
    padded by ``radius``, and only slab-interior rows are kept. ``n_chunks``
    overrides the automatic slab count.
    """
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0 or len(atoms) == 0:
        raise ValueError("radius must be positive and atoms must be nonempty")
    if len(atoms) > np.iinfo(np.int32).max:
        raise ValueError("atom indices exceed int32 range")
    if pair_memory_limit is not None and pair_memory_limit <= 0:
        raise ValueError("pair_memory_limit must be positive or None")
    widths = perpendicular_cell_widths(atoms)
    minimum_width = float(widths.min()) if len(widths) else None
    if minimum_width is not None and radius > minimum_width / 2:
        warnings.warn("coloring radius exceeds half the smallest perpendicular cell width; "
                      "multiple periodic images may contribute to one atom block", RuntimeWarning, stacklevel=2)
    n_atoms = len(atoms)
    density = _density(atoms, radius)
    projected_pairs = _projected_pairs(n_atoms, density, radius)
    projected_bytes = projected_pairs * _ASE_BYTES_PER_PAIR
    if n_chunks is None:
        n_chunks = _slab_count(atoms, radius, projected_pairs, pair_memory_limit)
    elif int(n_chunks) != n_chunks or n_chunks < 1:
        raise ValueError("n_chunks must be a positive integer")
    n_chunks = int(n_chunks)
    _LOG.info("Colored IFC neighbor graph: projected %d pairs, %.3f GB unchunked search peak, %d slab(s)",
              projected_pairs, projected_bytes / 1e9, n_chunks)
    if n_chunks == 1:
        rows, columns, distances = neighbor_list("ijd", atoms, radius, self_interaction=False)
    else:
        rows, columns, distances = _slab_neighbor_pairs(atoms, radius, n_chunks)
    matrix, distances = _coalesced_csr(rows, columns, distances, n_atoms)
    return NeighborStructure(matrix, radius, projected_pairs, projected_bytes, minimum_width, density,
                             distances, n_chunks)


def _square_sparse(structure):
    matrix = structure.matrix if isinstance(structure, NeighborStructure) else structure
    if not issparse(matrix) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("structure must be a square sparse matrix")
    return matrix


def validate_coloring(structure, labels):
    """Raise if a row has two structurally nonzero columns with one label."""
    matrix = _square_sparse(structure)
    labels = np.asarray(labels)
    if (labels.shape != (matrix.shape[1],) or labels.dtype.kind not in "iu" or
            np.any(labels < 0)):
        raise ValueError("labels must be one nonnegative integer per atom")
    csr = matrix.tocsr()
    if not csr.has_canonical_format:
        csr = csr.copy()
        csr.sum_duplicates()
    n_labels = int(labels.max()) + 1
    row_ids = np.repeat(np.arange(csr.shape[0], dtype=np.int64), np.diff(csr.indptr))
    keys = row_ids * n_labels + labels[csr.indices]
    keys.sort()
    clashes = np.flatnonzero(keys[1:] == keys[:-1])
    if len(clashes):
        raise ValueError(f"CPR structural orthogonality fails in atom row {int(keys[clashes[0]] // n_labels)}")
    return True


def greedy_cpr_groups(structure, order, *, batch_size=_GREEDY_BATCH):
    """First-fit CPR fallback: each atom in ``order`` takes the smallest free label.

    A label is taken when an earlier atom with that label shares a structural
    row with the current atom. Those conflicts are rows of the distance-2
    graph ``S.T @ S``, formed ``batch_size`` atoms at a time to bound memory.
    """
    matrix = _square_sparse(structure).tocsr()
    n_atoms = matrix.shape[1]
    order = np.asarray(order, dtype=np.int64)
    if (order.shape != (n_atoms,) or np.any(order < 0) or np.any(order >= n_atoms) or
            len(np.unique(order)) != n_atoms):
        raise ValueError("order must permute all atoms")
    pattern = csr_array((np.ones(len(matrix.indices), dtype=np.bool_), matrix.indices, matrix.indptr),
                        shape=matrix.shape)
    by_column = pattern.T.tocsr()
    labels = np.full(n_atoms, -1, dtype=np.int32)
    for start in range(0, n_atoms, batch_size):
        batch = order[start:start + batch_size]
        conflicts = (by_column[batch] @ pattern).tocsr()
        for position, atom in enumerate(batch):
            used = labels[conflicts.indices[conflicts.indptr[position]:conflicts.indptr[position + 1]]]
            used = used[used >= 0]
            taken = np.zeros(len(used) + 1, dtype=np.bool_)
            taken[used[used <= len(used)]] = True
            labels[atom] = int(np.argmin(taken))
    return labels


def _scipy_groups(matrix, order):
    if _scipy_group_columns is None:
        _LOG.warning("SciPy CPR grouping is unavailable; using the greedy fallback")
        return greedy_cpr_groups(matrix, order)
    try:
        labels = np.asarray(_scipy_group_columns(matrix, order=order), dtype=np.int32)
        validate_coloring(matrix, labels)
        return labels
    except (AttributeError, TypeError, ValueError) as exc:
        _LOG.warning("SciPy CPR grouping failed or is invalid (%s); using the greedy fallback", exc)
        return greedy_cpr_groups(matrix, order)


def color_structure(structure, *, seeds=(0, 1, 2), use_scipy=True):
    """Choose the smallest valid CPR coloring from fixed and seeded orders.

    A group has disjoint row supports in the atom-level Jacobian structure;
    the sparse-Jacobian graph model is reviewed by Gebremedhin et al., SIAM
    Rev. 47, 629-705 (2005).
    """
    matrix = _square_sparse(structure)
    n_atoms = matrix.shape[0]
    if not n_atoms:
        raise ValueError("structure must be nonempty")
    seeds = tuple(int(seed) for seed in seeds)
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative")
    csr = matrix.tocsr()
    row_degree = np.diff(csr.indptr)
    weighted_degree = np.asarray(csr.T @ row_degree).ravel()
    orders = [np.argsort(-weighted_degree, kind="stable")]
    orders.extend(np.random.default_rng(seed).permutation(n_atoms) for seed in seeds)
    best = None
    best_count = n_atoms + 1
    for order in orders:
        labels = _scipy_groups(csr, order) if use_scipy else greedy_cpr_groups(csr, order)
        count = int(labels.max()) + 1
        if count < best_count:
            best, best_count = labels, count
    validate_coloring(csr, best)
    bound = None
    if isinstance(structure, NeighborStructure):
        # Close-packing estimate, not a rigorous lower bound for this graph.
        bound = structure.number_density * (2 * structure.radius)**3 / np.sqrt(2)
    _LOG.info("Colored IFC groups: %d for %d atoms (geometric heuristic %.1f, ratio %.2f)",
              best_count, n_atoms, float("nan") if bound is None else bound,
              float("nan") if bound is None or bound == 0 else best_count / bound)
    return Coloring(best, seeds, best_count, bound)

import numpy as np
from numpy.typing import ArrayLike, NDArray
from kaldo.helpers.logger import get_logger
from ase import Atoms
logging = get_logger()


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
    M_rows = np.asarray(M).astype(int)

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


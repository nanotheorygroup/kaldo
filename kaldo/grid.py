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


import numpy as np
from numpy.typing import ArrayLike
from kaldo.helpers.logger import get_logger
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

# TODO: guess which type is this grid
def guess_type_of_grid():
    pass

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

    # TODO: add a function to search index in Grid. value to index. use dictonary to solve it, no more mask
    def grid_index_to_id(self, grid_index: ArrayLike, is_wrapping: bool):
        """Find the id of grid index in the grid array. is_wrapping indicats if the given grid_index is wrapped or not.

        """
        pass
    
    # TODO: Grid constructor: build a Grid from given grid array, and guess which type is it and recover the Grid object
    @classmethod
    def recover_grid_from_array(cls, array):
        pass


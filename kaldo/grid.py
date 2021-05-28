import numpy as np
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


class Grid:
    def __init__(self, grid_shape, order='C'):
        self.grid_shape = grid_shape
        self.grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.order = order


    def id_to_grid_index(self, id):
        grid_shape = self.grid_shape
        index_grid = np.array(np.unravel_index(id, grid_shape, order=self.order)).T
        return np.rint(index_grid).astype(np.int)


    def id_to_unitary_grid_index(self, id):
        q_vec = self.id_to_grid_index(id) / self.grid_shape
        return q_vec


    def generate_index_grid(self):
        ids = np.arange(self.grid_size)
        grid = self.id_to_grid_index(ids)
        return grid

    def unitary_grid(self, is_wrapping):
        return self.grid(is_wrapping) / self.grid_shape


    def grid(self, is_wrapping):
        try:
            index_grid = self._grid
        except AttributeError:
            self._grid = self.generate_index_grid()
            index_grid = self._grid
        if is_wrapping:
            index_grid = wrap_coordinates(index_grid, np.diag(self.grid_shape))
        return np.rint(index_grid).astype(np.int)


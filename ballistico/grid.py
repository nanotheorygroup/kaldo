import numpy as np


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
    def __init__(self, grid_shape, is_centering, order='C'):
        self.grid_shape = grid_shape
        self.grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.is_centering = is_centering
        self.order = order


    def id_to_grid_index(self, id):
        grid_shape = self.grid_shape
        index_grid = np.array(np.unravel_index(id, grid_shape, order=self.order)).T
        if self.is_centering:
            index_grid = index_grid - np.rint(np.array(grid_shape)[np.newaxis, :] / 2)
        return np.rint(index_grid).astype(np.int)


    def generate_index_grid(self):
        ids = np.arange(self.grid_size)
        grid = self.id_to_grid_index(ids)
        return grid


    def unitary_grid(self, is_wrapping=False):
        return self.grid(is_wrapping) / self.grid_shape


    def grid(self, is_wrapping=False):
        try:
            index_grid = self._grid
        except AttributeError:
            self._grid = self.generate_index_grid()
            index_grid = self._grid
        if is_wrapping:
            index_grid = wrap_coordinates(index_grid, np.diag(self.grid_shape))
        return np.rint(index_grid).astype(np.int)



    def q_index_from_q_vec(self, q_vec):
        kpts = self.grid_shape
        rescaled_qpp = np.round((q_vec * kpts).T, 0).astype(np.int)
        q_index = np.ravel_multi_index(rescaled_qpp, kpts, mode='wrap')
        return q_index


    def q_vec_from_q_index(self, q_index):
        kpts = self.grid_shape
        q_vec = np.array(np.unravel_index(q_index, (kpts))).T / kpts
        wrap_coordinates(q_vec)
        return q_vec


    def allowed_index_qpp(self, index_q, is_plus):
        kpts = self.grid_shape
        n_k_points = np.prod(kpts)
        index_qp_full = np.arange(n_k_points)
        q_vec = self.q_vec_from_q_index(index_q)
        qp_vec = self.q_vec_from_q_index(index_qp_full)
        qpp_vec = q_vec[np.newaxis, :] + (int(is_plus) * 2 - 1) * qp_vec[:, :]
        index_qpp_full = self.q_index_from_q_vec(qpp_vec)
        return index_qpp_full
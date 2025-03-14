import numpy as np
from numpy.typing import NDArray, ArrayLike
from kaldo.grid import Grid
from kaldo.helpers.logger import get_logger
from kaldo.observables.observable import Observable
from ase import Atoms
logging = get_logger()


def chi(qvec, list_of_replicas, cell_inv):
    chi_k = np.exp(1j * 2 * np.pi * list_of_replicas.dot(cell_inv.dot(qvec.T)))
    return chi_k


class ForceConstant(Observable):

    def __init__(self,
                 atoms: Atoms,
                 replicated_positions: NDArray,
                 supercell: tuple[int, int, int],
                 folder: str,
                 value: ArrayLike | None = None,
                 grid: Grid | None = None,
                 **kwargs):
        super().__init__(folder=folder, **kwargs)
        self.atoms = atoms
        self.supercell = supercell
        self.value = value

        self._replicated_atoms = None
        # TODO: why replicated_positions needs a reshape?
        self.replicated_positions = replicated_positions.reshape(
            (-1, self.atoms.positions.shape[0], self.atoms.positions.shape[1]))
        self.n_replicas = np.prod(self.supercell)
        self._cell_inv = None
        self._replicated_cell_inv = None
        self._list_of_replicas = None

        # * `replicated_atoms` and `list_of_replicas` are two main variables that are widely used in other places
        # Grid type directly impact on these two variables

        if grid is not None:
            self._direct_grid = grid
        else:
            # grid info is not given, so recover it from the grid type from the supercell matrix
            self._direct_grid = Grid.recover_grid_from_array(self.replicated_positions, self.supercell, self.atoms)


    @classmethod
    def from_supercell(cls,
                       atoms: Atoms,
                       supercell: tuple[int, int, int],
                       grid_type: str,
                       value: ArrayLike | None = None,
                       folder: str = 'kALDo',
                       **kwargs):
        _direct_grid = Grid(supercell, grid_type)
        _grid_arr = _direct_grid.grid(is_wrapping=False)
        # supercell grid * cell paramemter => supercell positions
        # supercell positions + atoms in unit cell positions => atoms in supercell positions
        replicated_positions = _grid_arr.dot(atoms.cell)[:, np.newaxis, :] + atoms.positions[np.newaxis, :, :]
        inst = cls(atoms=atoms,
                   replicated_positions=replicated_positions,
                   supercell=supercell,
                   value=value,
                   folder=folder,
                   grid=_direct_grid,
                   **kwargs)
        return inst


    @property
    def positions(self):
        return self.atoms.positions


    @property
    def cell_inv(self):
        if self._cell_inv is None:
            self._cell_inv = np.linalg.inv(self.atoms.cell)
        return self._cell_inv


    @property
    def replicated_atoms(self):
        # TODO: remove this method
        # forceconstant.replicated_atoms is used
        if self._replicated_atoms is None:
            supercell = self.supercell
            atoms = self.atoms
            replicated_atoms = atoms.copy() * supercell
            replicated_positions = self._direct_grid.grid(is_wrapping=False).dot(atoms.cell)[:, np.newaxis, :] + atoms.positions[
                                                                                                np.newaxis, :, :]
            replicated_atoms.set_positions(replicated_positions.reshape(-1, 3))
            self._replicated_atoms = replicated_atoms
        return self._replicated_atoms


    @property
    def replicated_cell_inv(self):
        if self._replicated_cell_inv is None:
            self._replicated_cell_inv = np.linalg.inv(self.replicated_atoms.cell)
        return self._replicated_cell_inv


    @property
    def list_of_replicas(self):
        if self._list_of_replicas is None:
            list_of_index = self._direct_grid.grid(is_wrapping=True)
            self._list_of_replicas = list_of_index.dot(self.atoms.cell)
        return self._list_of_replicas


    def _chi_k(self, k_points):
        n_k_points = np.shape(k_points)[0]
        ch = np.zeros((n_k_points, self.n_replicas), dtype=complex)
        for index_q in range(n_k_points):
            k_point = k_points[index_q]

            list_of_replicas = self.list_of_replicas
            cell_inv = self.cell_inv
            ch[index_q] = chi(k_point, list_of_replicas, cell_inv)
        return ch


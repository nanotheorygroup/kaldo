import numpy as np
from numpy.typing import NDArray
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
                 value=None,
                 **kwargs):
        super().__init__(self, folder=folder, **kwargs)
        self.atoms = atoms
        self.supercell = supercell
        self.value = value

        self._replicated_atoms = None
        self.replicated_positions = replicated_positions.reshape(
            (-1, self.atoms.positions.shape[0], self.atoms.positions.shape[1]))
        self.n_replicas = np.prod(self.supercell)
        self._cell_inv = None
        self._replicated_cell_inv = None
        self._list_of_replicas = None

        # TODO: following code should not be triggered if it was loaded from folder, Grid info should be saved

        n_replicas, n_unit_atoms, _ = self.replicated_positions.shape
        atoms_positions = self.atoms.positions
        detected_grid = np.round(
            (replicated_positions.reshape((n_replicas, n_unit_atoms, 3)) - atoms_positions[np.newaxis, :, :]).dot(
                np.linalg.inv(self.atoms.cell))[:, 0, :], 0).astype(int)

        grid_c = Grid(grid_shape=self.supercell, order='C')
        grid_fortran = Grid(grid_shape=self.supercell, order='F')
        if (grid_c.grid(is_wrapping=False) == detected_grid).all():
            grid_type = 'C'
            logging.debug("Using C-style position grid")
        elif (grid_fortran.grid(is_wrapping=False) == detected_grid).all():
            grid_type = 'F'
            logging.debug("Using fortran-style position grid")
        else:
            err_msg = "Unable to detect grid type"
            logging.error(err_msg)
            raise TypeError(err_msg)

        self._direct_grid = Grid(self.supercell, grid_type)


    @classmethod
    def from_supercell(cls,
                       atoms: Atoms,
                       supercell: tuple[int, int, int],
                       grid_type: str,
                       value=None,
                       folder: str = 'kALDo'):
        _direct_grid = Grid(supercell, grid_type)
        _grid_arr = _direct_grid.grid(is_wrapping=False)
        # supercell grid * cell paramemter => supercell positions
        # supercell positions + atoms in unit cell positions => atoms in supercell positions
        replicated_positions = _grid_arr.dot(atoms.cell)[:, np.newaxis, :] + atoms.positions[np.newaxis, :, :]
        inst = cls(atoms=atoms,
                   replicated_positions=replicated_positions,
                   supercell=supercell,
                   value=value,
                   folder=folder)
        inst._direct_grid = _direct_grid
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


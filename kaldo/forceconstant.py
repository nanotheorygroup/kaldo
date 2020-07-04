import numpy as np
import ase.units as units
from kaldo.grid import Grid
from kaldo.controllers.harmonic import chi
from kaldo.helpers.logger import get_logger
from kaldo.observable import Observable
logging = get_logger()
EVTOTENJOVERMOL = units.mol / (10 * units.J)



class ForceConstant(Observable):
    def __init__(self, atoms, replicated_positions, supercell=None, force_constant=None):
        self.atoms = atoms
        self._replicated_atoms = None
        if force_constant is not None:
            self.value = force_constant
        self.replicated_positions = replicated_positions.reshape(
            (-1, atoms.positions.shape[0], atoms.positions.shape[1]))
        self.supercell = supercell
        self.n_replicas = np.prod(supercell)
        self._cell_inv = None
        self._replicated_cell_inv = None
        self._list_of_replicas = None
        n_replicas, n_unit_atoms, _ = self.replicated_positions.shape
        atoms_positions = atoms.positions
        detected_grid = np.round(
            (replicated_positions.reshape((n_replicas, n_unit_atoms, 3)) - atoms_positions[np.newaxis, :, :]).dot(
                np.linalg.inv(atoms.cell))[:, 0, :], 0).astype(np.int)

        grid_c = Grid(grid_shape=supercell, order='C')
        grid_fortran = Grid(grid_shape=supercell, order='F')
        if (grid_c.grid() == detected_grid).all():
            grid_type = 'C'
        elif (grid_fortran.grid() == detected_grid).all():
            grid_type = 'F'
        else:
            logging.error("Unable to detect grid type")

        if grid_type == 'C':
            logging.info("Using C-style position grid")
        else:
            logging.info("Using fortran-style position grid")
        self._direct_grid = Grid(supercell, grid_type)


    @classmethod
    def from_supercell(cls, atoms, supercell, grid_type, force_constant=None):
        _direct_grid = Grid(supercell, grid_type)
        replicated_positions = _direct_grid.grid().dot(atoms.cell)[:, np.newaxis, :] + atoms.positions[
                                                                                       np.newaxis, :, :]
        inst = cls(atoms,
                   replicated_positions,
                   supercell,
                   force_constant)
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
            replicated_positions = self._direct_grid.grid().dot(atoms.cell)[:, np.newaxis, :] + atoms.positions[
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
        ch = np.zeros((n_k_points, self.n_replicas), dtype=np.complex)
        for index_q in range(n_k_points):
            k_point = k_points[index_q]

            list_of_replicas = self.list_of_replicas
            cell_inv = self.cell_inv
            ch[index_q] = chi(k_point, list_of_replicas, cell_inv)
        return ch

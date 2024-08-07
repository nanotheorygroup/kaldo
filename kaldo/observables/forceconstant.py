import numpy as np
from kaldo.grid import Grid
from kaldo.helpers.logger import get_logger
from kaldo.observables.observable import Observable
logging = get_logger()


def chi(qvec, list_of_replicas, cell_inv):
    chi_k = np.exp(1j * 2 * np.pi * list_of_replicas.dot(cell_inv.dot(qvec.T)))
    return chi_k


class ForceConstant(Observable):

    def __init__(self, *kargs, **kwargs):
        Observable.__init__(self, *kargs, **kwargs)
        self.atoms = kwargs['atoms']
        replicated_positions = kwargs['replicated_positions']
        self.supercell = kwargs['supercell']
        try:
            self.value = kwargs['value']
        except KeyError:
            self.value = None

        self._replicated_atoms = None
        self.replicated_positions = replicated_positions.reshape(
            (-1, self.atoms.positions.shape[0], self.atoms.positions.shape[1]))
        self.n_replicas = np.prod(self.supercell)
        self._cell_inv = None
        self._replicated_cell_inv = None
        self._list_of_replicas = None
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
    def from_supercell(cls, atoms, supercell, grid_type, value=None, folder='kALDo'):
        _direct_grid = Grid(supercell, grid_type)
        replicated_positions = _direct_grid.grid(is_wrapping=False).dot(atoms.cell)[:, np.newaxis, :] + atoms.positions[
                                                                                       np.newaxis, :, :]
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




    def sigma2_td(fc_file='infile.forceconstant', primitive='infile.ucposcar', supercell='infile.ssposcar', md_run='dump.xyz'):
        from ase.io import read
        from kaldo.observables.secondorder import parse_tdep_forceconstant
        initial_structure = read(sc_file, format="vasp")
        second_order_fc_in_full_dimension     =   parse_tdep_forceconstant(
                                  fc_file     =   fc_file,
                                  primitive   =   primitive,
                                  supercell   =   supercell,
                                  symmetrize  =   True,
                                  two_dim     =   True)
        full_MD_traj = read(md_run, index=":")
        displacements = []
        for atoms in full_MD_traj:
            disp = atoms.positions - initial_structure.positions
            disp_with_mic = find_mic(disp.reshape(-1, 3), atoms.cell)[0]
            displacements.append(np.reshape(disp_with_mic,[initial_structure.positions.shape[0], initial_structure.positions.shape[1]]))

        force_harmonic = []
        for i in range(len(displacements)):
            disp = displacements[i]
            force_harmonic_vec = -1 * second_order_fc_in_full_dimension @ disp.flatten()
            force_harmonic.append(np.reshape(force_harmonic_vec, [initial_structure.positions.shape[0], initial_structure.positions.shape[1]]))

        sigma_A = []
        for i in range(len(full_MD_traj)):
            sigma_A.append(mean_squared_error(full_MD_traj[i].get_forces(), force_harmonic[i])**(0.5)/np.std(full_MD_traj[i].get_forces()))

        return sigma_A

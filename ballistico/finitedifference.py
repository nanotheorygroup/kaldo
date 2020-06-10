"""
Ballistico
Anharmonic Lattice Dynamics
"""
from ase import Atoms
import os
import ase.io
import numpy as np
from scipy.sparse import load_npz, save_npz
from sparse import COO
from ballistico.io.eskm_io import import_from_files
import ballistico.io.shengbte_io as shengbte_io
from ballistico.grid import wrap_coordinates
from ballistico.controllers.displacement import calculate_second, calculate_third
import ase.units as units
from ballistico.secondorder import SecondOrder
from ballistico.thirdorder import ThirdOrder
from ballistico.helpers.logger import get_logger
logging = get_logger()

DELTA_SHIFT = 1e-5
tenjovermoltoev = 10 * units.J / units.mol

# see bug report: https://github.com/h5py/h5py/issues/1101
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

EVTOTENJOVERMOL = units.mol / (10 * units.J)
DELTA_SHIFT = 1e-5

MAIN_FOLDER = 'displacement'
SECOND_ORDER_FILE = 'second.npy'
DYNMAT_FILE = 'dynmat.npy'
LIST_OF_REPLICAS_FILE = 'list_of_replicas.npy'
THIRD_ORDER_FILE_SPARSE = 'third.npz'
THIRD_ORDER_FILE = 'third.npy'
REPLICATED_ATOMS_FILE = 'replicated_atoms.xyz'
SECOND_ORDER_WITH_PROGRESS_FILE = 'second_order_progress.hdf5'
THIRD_ORDER_WITH_PROGRESS_FILE = 'third_order_progress'


class FiniteDifference(object):
    """ Class for constructing the finite difference object to calculate
        the second/third order force constant matrices after providing the
        unit cell geometry and calculator information.
    """

    def __init__(self,
                 atoms,
                 supercell=(1, 1, 1),
                 folder=MAIN_FOLDER,
                 is_reduced_second=True,
                 distance_threshold=None):

        """Init with an instance of constructed FiniteDifference object.

        Parameters:

        atoms: Tabulated xyz files or Atoms object
            The atoms to work on.
        supercell: tuple
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction
        folder: str
            Name str for the displacement folder
        is_reduced_second = It returns the full second order matrix if set
            to True (default).When providing a supercell != (1, 1, 1) and
            it's set to False, it provides only the reduced second order
            matrix for the elementary unit cell
        distance_threshold: float
            the maximum distance between two interacting atoms
        """

        # Store the user defined information to the object
        self.atoms = atoms
        self.supercell = supercell
        self.n_atoms = atoms.positions.shape[0]
        self.n_modes = self.n_atoms * 3
        self.n_replicas = np.prod(supercell)
        self.n_replicated_atoms = self.n_replicas * self.n_atoms
        self.cell_inv = np.linalg.inv(atoms.cell)

        self.folder = folder
        self.is_reduced_second = is_reduced_second
        self.distance_threshold = distance_threshold
        self._list_of_replicas = None


    @classmethod
    def from_files(cls, replicated_atoms, dynmat_file, third_file=None, folder=None, supercell=(1, 1, 1), third_energy_threshold=0., distance_threshold=None, is_symmetrizing=False, is_acoustic_sum_sum=False):
        """
        Create a finite difference object from files
        :param replicated_atoms:
        :param dynmat_file:
        :param third_file:
        :param folder:
        :param supercell:
        :param third_energy_threshold:
        :param distance_threshold:
        :param is_symmetrizing:
        :param is_acoustic_sum_sum:
        :return:
        """

        n_replicas = np.prod(supercell)
        n_total_atoms = replicated_atoms.positions.shape[0]
        n_unit_atoms = int(n_total_atoms / n_replicas)
        unit_symbols = []
        unit_positions = []
        for i in range(n_unit_atoms):
            unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
            unit_positions.append(replicated_atoms.positions[i])
        unit_cell = replicated_atoms.cell / supercell

        atoms = Atoms(unit_symbols,
                      positions=unit_positions,
                      cell=unit_cell,
                      pbc=[1, 1, 1])

        # Create a finite difference object
        finite_difference = {'atoms': atoms,
                             'supercell': supercell,
                             'folder': folder}
        fd = cls(**finite_difference)
        _second_order, fd.is_reduced_second, _third_order = import_from_files(replicated_atoms,
                                                                                  dynmat_file,
                                                                                  third_file,
                                                                                  supercell,
                                                                                  third_energy_threshold,
                                                                                  distance_threshold)

        fd.second_order = SecondOrder(atoms, replicated_atoms.positions, supercell, _second_order)
        fd.third_order = ThirdOrder(atoms, replicated_atoms.positions, supercell, _third_order)
        # if is_symmetrizing:
        #     fd = calculate_symmetrize_dynmat(fd)
        # if is_acoustic_sum_sum:
        #     fd = calculate_acoustic_dynmat(fd)
        fd.distance_threshold = distance_threshold
        return fd


    @classmethod
    def from_folder(cls, folder, supercell=(1, 1, 1), format='eskm', third_energy_threshold=0., distance_threshold=None,
                    third_supercell=None, is_acoustic_sum=False):
        """
        Create a finite difference object from a folder
        :param folder:
        :param supercell:
        :param format:
        :param third_energy_threshold:
        :param distance_threshold:
        :param third_supercell:
        :param is_acoustic_sum:
        :return:
        """
        if third_supercell is None:
            third_supercell = supercell

        if format == 'numpy':

            if folder[-1] != '/':
                folder = folder + '/'
            config_file = folder + REPLICATED_ATOMS_FILE
            replicated_atoms = ase.io.read(config_file, format='extxyz')

            n_replicas = np.prod(supercell)
            n_total_atoms = replicated_atoms.positions.shape[0]
            n_unit_atoms = int(n_total_atoms / n_replicas)
            unit_symbols = []
            unit_positions = []
            for i in range(n_unit_atoms):
                unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                unit_positions.append(replicated_atoms.positions[i])
            unit_cell = replicated_atoms.cell / supercell

            atoms = Atoms(unit_symbols,
                          positions=unit_positions,
                          cell=unit_cell,
                          pbc=[1, 1, 1])

            # Create a finite difference object
            finite_difference = {'atoms': atoms,
                                 'supercell': supercell,
                                 'folder': folder}
            finite_difference = cls(**finite_difference)

            n_atoms = int(atoms.positions.shape[0] / n_replicas)

            _second_order = np.load(folder + SECOND_ORDER_FILE)
            if _second_order.size == (n_replicas * n_atoms * 3) ** 2:
                finite_difference.is_reduced_second = False
            else:
                finite_difference.is_reduced_second = True
            _third_order = COO.from_scipy_sparse(load_npz(folder + THIRD_ORDER_FILE_SPARSE)) \
                .reshape((n_atoms * 3, n_replicas * n_atoms * 3, n_replicas * n_atoms * 3))
            finite_difference.second_order = SecondOrder(atoms, replicated_atoms.positions, supercell, _second_order,
                                                         is_acoustic_sum=is_acoustic_sum)
            finite_difference.third_order = ThirdOrder(atoms, replicated_atoms.positions, supercell, _third_order)

        elif format == 'eskm':
            config_file = str(folder) + "/CONFIG"
            dynmat_file = str(folder) + "/Dyn.form"
            third_file = str(folder) + "/THIRD"

            replicated_atoms = ase.io.read(config_file, format='dlp4')
            n_replicas = np.prod(supercell)
            n_total_atoms = replicated_atoms.positions.shape[0]
            n_unit_atoms = int(n_total_atoms / n_replicas)
            unit_symbols = []
            unit_positions = []
            for i in range(n_unit_atoms):
                unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                unit_positions.append(replicated_atoms.positions[i])
            unit_cell = replicated_atoms.cell / supercell

            atoms = Atoms(unit_symbols,
                          positions=unit_positions,
                          cell=unit_cell,
                          pbc=[1, 1, 1])

            # Create a finite difference object
            finite_difference = {'atoms': atoms,
                                 'supercell': supercell,
                                 'folder': folder}
            finite_difference = cls(**finite_difference)

            _second_order, finite_difference.is_reduced_second, _third_order = import_from_files(replicated_atoms,
                                                                                      dynmat_file,
                                                                                      third_file,
                                                                                      supercell,
                                                                                      third_energy_threshold=third_energy_threshold,
                                                                                      distance_threshold=distance_threshold)
            finite_difference.second_order = SecondOrder(atoms, replicated_atoms.positions, supercell, _second_order,
                                                         is_acoustic_sum=is_acoustic_sum)
            finite_difference.third_order = ThirdOrder(atoms, replicated_atoms.positions, supercell, _third_order)


        elif format == 'shengbte' or format == 'shengbte-qe':

            config_file = folder + '/' + 'CONTROL'
            try:
                atoms, supercell = shengbte_io.import_control_file(config_file)
            except FileNotFoundError as err:
                config_file = folder + '/' + 'POSCAR'
                logging.info('\nTrying to open POSCAR')
                atoms = ase.io.read(config_file)

            # Create a finite difference object
            # TODO: we need to read the grid type here
            finite_difference = cls(atoms=atoms, supercell=supercell, folder=folder)
            is_qe_input = (format == 'shengbte-qe')
            atoms = finite_difference.atoms
            supercell = finite_difference.supercell
            folder = finite_difference.folder
            n_replicas = np.prod(supercell)
            n_unit_atoms = atoms.positions.shape[0]
            if is_qe_input:
                filename = folder + '/espresso.ifc2'
                second_order, supercell = shengbte_io.read_second_order_qe_matrix(filename)
                second_order = second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
                second_order = second_order.transpose(3, 4, 2, 0, 1)
            else:
                second_order = shengbte_io.read_second_order_matrix(folder, supercell)
                second_order = second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
            is_reduced_second = True
            finite_difference.second_order = SecondOrder.from_supercell(atoms,
                                                                        grid_type='C',
                                                                        supercell=supercell,
                                                                        force_constant=second_order[np.newaxis, ...],
                                                                        is_acoustic_sum=True)
            finite_difference.is_reduced_second = is_reduced_second



            third_file = folder + '/' + 'FORCE_CONSTANTS_3RD'

            _third_order, _third_sparse, _second_list, _third_list, _third_coords  = shengbte_io.read_third_order_matrix(third_file, atoms, third_supercell, order='C')
            finite_difference.third_order = ThirdOrder.from_supercell(atoms,
                                                                      grid_type='F',
                                                                      supercell=third_supercell,
                                                                      force_constant=_third_order)
            finite_difference.third_order.second_list = _second_list
            finite_difference.third_order.third_list = _third_list
            finite_difference.third_order.third_sparse = _third_sparse
            finite_difference.third_order.third_coords = _third_coords



        elif format == 'hiphive':
            filename = 'atom_prim.xyz'
            # TODO: add replicated filename in example
            replicated_filename = 'replicated_atom_prim.xyz'
            try:
                import ballistico.io.hiphive_io as hiphive_io
            except ImportError:
                logging.error('In order to use hiphive along with ballistico, hiphive is required. \
                      Please consider installing hihphive. More info can be found at: \
                      https://hiphive.materialsmodeling.org/')

            atom_prime_file = str(folder) + '/' + filename
            replicated_atom_prime_file = str(folder) + '/' + replicated_filename
            # TODO: Make this independent of replicated file
            atoms = ase.io.read(atom_prime_file)
            replicated_atoms = ase.io.read(replicated_atom_prime_file)

            # Create a finite difference object
            finite_difference = cls(atoms=atoms, supercell=supercell, folder=folder)
            if 'model2.fcs' in os.listdir(str(folder)):
                _second_order = hiphive_io.import_second_from_hiphive(finite_difference)
                finite_difference.second_order = SecondOrder(atoms,
                                                             replicated_atoms.positions,
                                                             supercell,
                                                             _second_order)
            if 'model3.fcs' in os.listdir(str(folder)):
                # Derive constants used for third-order reshape
                supercell = np.array(supercell)
                n_prim = atoms.copy().get_masses().shape[0]
                n_sc = np.prod(supercell)
                dim = len(supercell[supercell > 1])
                _third_order = hiphive_io.import_third_from_hiphive(finite_difference)
                _third_order = _third_order[0].reshape(n_prim * dim, n_sc * n_prim * dim,
                                                                       n_sc * n_prim * dim)
                finite_difference.third_order = ThirdOrder(atoms,
                                                           replicated_atoms.positions,
                                                           supercell,
                                                           _third_order)


        else:
            raise ValueError
        return finite_difference



    def calculate_second(self, calculator, grid_type='C', delta_shift=DELTA_SHIFT):
        # TODO: move to ifc
        atoms = self.atoms
        self.second_order = SecondOrder.from_supercell(atoms,
                                                       supercell=self.supercell,
                                                       grid_type=grid_type,
                                                       is_acoustic_sum=False)
        replicated_atoms = self.second_order.replicated_atoms
        atoms.set_calculator(calculator)
        replicated_atoms.set_calculator(calculator)

        _second_order = calculate_second(atoms,
                                              replicated_atoms,
                                              delta_shift)

        self.second_order.force_constant = _second_order


    def calculate_third(self, calculator, grid_type='C', delta_shift=DELTA_SHIFT, supercell=None):
        if supercell is None:
            supercell = self.supercell
        atoms = self.atoms
        self.third_order = ThirdOrder.from_supercell(atoms,
                                                       supercell=supercell,
                                                       grid_type=grid_type)
        replicated_atoms = self.third_order.replicated_atoms
        atoms.set_calculator(calculator)
        replicated_atoms.set_calculator(calculator)

        _third_order = calculate_third(atoms,
                                       replicated_atoms,
                                       delta_shift,
                                       distance_threshold=self.distance_threshold)
        self.third_order.force_constant = _third_order



    def unfold_third_order(self, reduced_third=None, distance_threshold=None):
        logging.info('Unfolding third order matrix')
        if distance_threshold is None:
            if self.distance_threshold is not None:
                distance_threshold = self.distance_threshold
            else:
                raise ValueError('Please specify a distance threshold in Armstrong')

        logging.info('Distance threshold: ' + str(distance_threshold) + ' A')
        if (self.atoms.cell[0, 0] / 2 < distance_threshold) | \
                (self.atoms.cell[1, 1] / 2 < distance_threshold) | \
                (self.atoms.cell[2, 2] / 2 < distance_threshold):
            logging.warning('The cell size should be at least twice the distance threshold')
        if reduced_third is None:
            reduced_third = self.third_order.force_constant
        n_unit_atoms = self.n_atoms
        atoms = self.atoms
        n_replicas = self.n_replicas
        replicated_cell_inv = np.linalg.inv(self.replicated_atoms.cell)

        reduced_third = reduced_third.reshape(
            (n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
        replicated_positions = self.replicated_atoms.positions.reshape((n_replicas, n_unit_atoms, 3))
        dxij_reduced = wrap_coordinates(atoms.positions[:, np.newaxis, np.newaxis, :] - replicated_positions[np.newaxis, :, :, :], self.replicated_atoms.cell, replicated_cell_inv)
        indices = np.argwhere(np.linalg.norm(dxij_reduced, axis=-1) < distance_threshold)

        coords = []
        values = []
        for index in indices:
            for l in range(n_replicas):
                for j in range(n_unit_atoms):
                    dx2 = dxij_reduced[index[0], l, j]

                    is_storing = (np.linalg.norm(dx2) < distance_threshold)
                    if is_storing:
                        for alpha in range(3):
                            for beta in range(3):
                                for gamma in range(3):
                                    coords.append([index[0], alpha, index[1], index[2], beta, l, j, gamma])
                                    values.append(reduced_third[index[0], alpha, 0, index[2], beta, 0, j, gamma])
                                    
                                    
        logging.info('Created unfolded third order')

        shape = (n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3)
        expanded_third = COO(np.array(coords).T, np.array(values), shape)
        expanded_third = expanded_third.reshape(
            (n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
        return expanded_third


    def export_third_eskm(self, out_filename='THIRD', min_force=1e-6):
        logging.info('Exporting third in eskm format')
        n_atoms = self.n_atoms
        n_replicas = self.n_replicas
        n_replicated_atoms = n_atoms * n_replicas
        third = self.third_order.force_constant.reshape((n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3)) / tenjovermoltoev
        with open(out_filename, 'w') as out_file:
            for i in range(n_atoms):
                for alpha in range(3):
                    for j in range(n_replicated_atoms):
                        for beta in range(3):
                            value = third[i, alpha, j, beta].todense()
                            mask = np.argwhere(np.linalg.norm(value, axis=1) > min_force)
                            if mask.any():
                                for k in mask:
                                    k = k[0]
                                    out_file.write("{:5d} ".format(i + 1))
                                    out_file.write("{:5d} ".format(alpha + 1))
                                    out_file.write("{:5d} ".format(j + 1))
                                    out_file.write("{:5d} ".format(beta + 1))
                                    out_file.write("{:5d} ".format(k + 1))
                                    for gamma in range(3):
                                        out_file.write(' {:16.6f}'.format(third[i, alpha, j, beta, k, gamma]))
                                    out_file.write('\n')
        logging.info('Done exporting third.')

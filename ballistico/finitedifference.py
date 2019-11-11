"""
Ballistico
Anharmonic Lattice Dynamics
"""

import os
import ase.io
import numpy as np
from ase import Atoms
from scipy.optimize import minimize
from scipy.sparse import load_npz, save_npz
from sparse import COO
from ballistico.tools.io import import_second, import_sparse_third, import_dense_third
import h5py

# see bug report: https://github.com/h5py/h5py/issues/1101
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

DELTA_SHIFT = 1e-5

# Tolerance for symmetry search
SYMPREC_THIRD_ORDER = 1e-5
# Tolerance for geometry optimization
MAX_FORCE = 1e-20

MAIN_FOLDER = 'displacement'
SECOND_ORDER_FILE = 'second.npy'
THIRD_ORDER_FILE_SPARSE = 'third.npz'
THIRD_ORDER_FILE = 'third.npy'
REPLICATED_ATOMS_FILE = 'replicated_atoms.xyz'
SECOND_ORDER_WITH_PROGRESS_FILE = 'second_order_progress.hdf5'
THIRD_ORDER_WITH_PROGRESS_FILE = 'third_order_progress'


def import_control_file(control_file):
    positions = []
    latt_vecs = []
    lfactor = 1
    with open(control_file, "r") as fo:
        lines = fo.readlines()
    for line in lines:
        if 'lattvec' in line:
            value = line.split('=')[1]
            latt_vecs.append(np.fromstring(value, dtype=np.float, sep=' '))
        if 'elements' in line and not ('nelements' in line):
            value = line.split('=')[1]
            # TODO: only one species at the moment
            value = value.replace('"', '\'')
            value = value.replace(" ", '')
            value = value.replace("\n", '')
            value = value.replace(',', '')
            value = value.replace("''", '\t')
            value = value.replace("'", '')
            elements = value.split("\t")

        if 'types' in line:
            value = line.split('=')[1]

            types = np.fromstring(value, dtype=np.int, sep=' ')
        if 'positions' in line:
            value = line.split('=')[1]
            positions.append(np.fromstring(value, dtype=np.float, sep=' '))
        if 'lfactor' in line:
            lfactor = float(line.split('=')[1].split(',')[0])
        if 'scell' in line:
            value = line.split('=')[1]
            scell = np.fromstring(value, dtype=np.int, sep=' ')
    # l factor is in nanometer
    cell = np.array(latt_vecs) * lfactor * 10
    positions = np.array(positions).dot(cell)

    supercell = scell
    list_of_elem = []
    for i in range(len(types)):
        list_of_elem.append(elements[types[i] - 1])

    atoms = Atoms(list_of_elem,
                  positions=positions,
                  cell=cell,
                  pbc=[1, 1, 1])

    print('Atoms object created.')
    return atoms, supercell


def divmod(a, b):
    q = a / b
    r = a % b
    return q, r


def split_index(index, nx, ny, nz):
    tmp1, ix = divmod(index - 1, nx, )
    tmp2, iy = divmod(tmp1, ny)
    iatom, iz = divmod(tmp2, nz)
    ix = ix + 1
    iy = iy + 1
    iz = iz + 1
    iatom = iatom + 1
    return int(ix), int(iy), int(iz), int(iatom)


class FiniteDifference(object):
    """ Class for constructing the finite difference object to calculate
        the second/third order force constant matrices after providing the
        unit cell geometry and calculator information.
    """

    def __init__(self,
                 atoms,
                 supercell=(1, 1, 1),
                 second_order=None,
                 third_order=None,
                 calculator=None,
                 calculator_inputs=None,
                 delta_shift=DELTA_SHIFT,
                 folder=MAIN_FOLDER,
                 third_order_symmerty_inputs=None,
                 is_reduced_second=False,
                 is_optimizing=False):

        """Init with an instance of constructed FiniteDifference object.

        Parameters:

        atoms: Tabulated xyz files or Atoms object
            The atoms to work on.
        supercell: tuple
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        second_order: numpy array
            Second order force constant matrices.
        third_order:numpy array
             Second order force constant matrices.
        calculator: Calculator
            Calculator for the force constant matrices calculation.
        calculator_inputs: Calculator inputs
            Associated force field information used in the calculator.
        delta_shift: float
            Magnitude of displacement in Ang.
        folder: str
            Name str for the displacement folder
        third_order_symmerty_inputs = Third order force matrices symmetry inputs
            symmetry and nearest neighbor input for third order force matrices.
        is_reduced_second = It returns the full second order matrix if set
            to True (default).When providing a supercell != (1, 1, 1) and
            it's set to False, it provides only the reduced second order
            matrix for the elementary unit cell
        is_optimizing:boolean
            boolean flag to instruct if initial input atom geometry be optimized
        """

        # Store the user defined information to the object
        self.atoms = atoms
        self.supercell = supercell
        self.n_atoms = self.atoms.get_masses().shape[0]
        self.n_replicas = np.prod(supercell)
        self.calculator = calculator

        if calculator:
            if calculator_inputs:
                calculator_inputs['keep_alive'] = True
                self.calculator_inputs = calculator_inputs
            self.atoms.set_calculator(self.calculator(**self.calculator_inputs))
            self.second_order_delta = delta_shift
            self.third_order_delta = delta_shift
            self.third_order_symmerty_inputs = third_order_symmerty_inputs
            # Optimize the structure if optimizing flag is turned to true
            # and the calculation is set to start from the starch:
            if is_optimizing:
                self.optimize()

        self.folder_name = folder
        self.is_reduced_second = is_reduced_second
        self._replicated_atoms = None
        self._second_order = None
        self._third_order = None

        # Directly loading the second/third order force matrices
        if second_order is not None:
            self.second_order = second_order
        if third_order is not None:
            self.third_order = third_order

    @property
    def second_order(self):
        """second_order method to return the second order force matrix.
        """
        return self._second_order

    @second_order.getter
    def second_order(self):
        """Obtain the second order force constant matrix either by loading in
           or performing calculation using the provided calculator.
        """
        # Once confirming that the second order does not exist, try load in
        if self._second_order is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._second_order = np.load(folder + SECOND_ORDER_FILE)
            except FileNotFoundError as e:
                print(e)
                # After trying load in and still not exist,
                # calculate the second order
                self.second_order = self.calculate_second()
        return self._second_order

    @second_order.setter
    def second_order(self, new_second_order):
        """Save the loaded/computed second order force constant matrix
            to preset proper folders.
        """
        # make the  folder and save the second order force constant matrix into it
        folder = self.folder_name
        folder += '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(folder + SECOND_ORDER_FILE, new_second_order)
        try:
            os.remove(folder + SECOND_ORDER_WITH_PROGRESS_FILE)
        except FileNotFoundError as err:
            print(err)
        self._second_order = new_second_order

    @property
    def third_order(self):
        """third_order method to return the third order force matrix
        """
        return self._third_order

    @third_order.getter
    def third_order(self):
        """Obtain the third order force constant matrix either by loading in
           or performing calculation using the provided calculator.
        """
        # Once confirming that the third order does not exist, try load in
        if self._third_order is None:
            folder = self.folder_name
            folder += '/'
            try:
                self._third_order = COO.from_scipy_sparse(load_npz(folder + THIRD_ORDER_FILE_SPARSE)) \
                    .reshape((1 * self.n_atoms * 3, self.n_replicas * self.n_atoms * 3, self.n_replicas *
                              self.n_atoms * 3))
            except FileNotFoundError as e:
                # calculate the third order
                self.third_order = self.calculate_third()
        return self._third_order

    @third_order.setter
    def third_order(self, new_third_order):
        """Save the loaded/computed third order force constant matrix
            to preset proper folders.
        """
        # Convert third order force constant matrix from nd numpy array to
        # sparse matrix to save memory
        if type(new_third_order) == np.ndarray:
            self._third_order = COO.from_numpy(new_third_order)
        else:
            self._third_order = new_third_order

        # Make the folder and save the third order force constant matrix into it
        # Save the third order as npz to futher save memory
        folder = self.folder_name
        folder += '/'
        # Remove the partial file if exists
        try:
            os.remove(folder + THIRD_ORDER_WITH_PROGRESS_FILE)
        except FileNotFoundError as err:
            print(err)
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_npz(folder + THIRD_ORDER_FILE_SPARSE, self._third_order.reshape((self.n_atoms * 3 * self.n_replicas *
                                                                              self.n_atoms * 3, self.n_replicas *
                                                                              self.n_atoms * 3)).to_scipy_sparse())

    @property
    def replicated_atoms(self):
        """replicated method to return the duplicated atom geometry.
        """
        return self._replicated_atoms

    @replicated_atoms.getter
    def replicated_atoms(self):
        """Obtain the duplicated atom geometry either by loading in the
            tabulated  xyz file or replicate the geometry based on symmetry
        """
        # Once confirming that the duplicated atom geometry does not exist,
        # try load in from the provided xyz file
        if self._replicated_atoms is None:
            self.replicated_atoms = self.gen_supercell()
            if self.calculator:
                self.replicated_atoms.set_calculator(self.calculator(**self.calculator_inputs))
        return self._replicated_atoms

    @replicated_atoms.setter
    def replicated_atoms(self, new_replicated_atoms):
        """Save the loaded/computed duplicated atom geometry
            to preset proper folders.
        """
        # Make the folder and save the replicated atom xyz files into it
        folder = self.folder_name
        folder += '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        ase.io.write(folder + REPLICATED_ATOMS_FILE, new_replicated_atoms, format='extxyz')
        self._replicated_atoms = new_replicated_atoms

    @classmethod
    def import_from_dlpoly_folder(cls, folder, supercell=(1, 1, 1)):
        config_file = str(folder) + "/CONFIG"
        dynmat_file = str(folder) + "/Dyn.form"
        third_file = str(folder) + "/THIRD"
        atoms = ase.io.read(config_file, format='dlp4')
        return cls.import_from_dlpoly(atoms, dynmat_file, third_file, folder, supercell)

    @classmethod
    def import_from_dlpoly(cls, atoms, dynmat_file=None, third_file=None, folder=MAIN_FOLDER, supercell=(1, 1, 1)):
        n_replicas = np.prod(supercell)
        n_total_atoms = atoms.positions.shape[0]
        n_unit_atoms = int(n_total_atoms / n_replicas)
        unit_symbols = []
        unit_positions = []
        for i in range(n_unit_atoms):
            unit_symbols.append(atoms.get_chemical_symbols()[i])
            unit_positions.append(atoms.positions[i])
        unit_cell = atoms.cell / supercell

        atoms = Atoms(unit_symbols,
                      positions=unit_positions,
                      cell=unit_cell,
                      pbc=[1, 1, 1])

        # Create a finite difference object
        finite_difference = cls(atoms=atoms,
                                supercell=supercell,
                                folder=folder)

        if dynmat_file:
            second_dl = import_second(atoms, replicas=supercell, filename=dynmat_file)
            is_reduced_second = not (n_replicas ** 2 * (n_unit_atoms * 3) ** 2 == second_dl.size)
            if is_reduced_second:
                second_shape = (1, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3)
            else:
                second_shape = (n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3)

            print('Is reduced second: ', is_reduced_second)
            second_dl = second_dl.reshape(second_shape)
            finite_difference.second_order = second_dl
            finite_difference.is_reduced_second = is_reduced_second

        if third_file:
            try:
                third_dl = import_sparse_third(atoms, replicas=supercell, filename=third_file)
            except UnicodeDecodeError:
                print('Trying reading binary third')
                third_dl = import_dense_third(atoms, replicas=supercell, filename=third_file)
            third_dl = third_dl[:n_unit_atoms]
            third_shape = (
                n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3)
            third_dl = third_dl.reshape(third_shape)
            finite_difference.third_order = third_dl

        return finite_difference

    @classmethod
    def import_from_sheng_folder(cls, folder, supercell=None):

        config_file = folder + '/' + 'CONTROL'
        try:
            atoms, supercell = import_control_file(config_file)
        except FileNotFoundError as err:
            config_file = folder + '/' + 'POSCAR'
            print(err, 'Trying to open POSCAR')
            atoms = ase.io.read(config_file)

        # Create a finite difference object
        finite_difference = cls(atoms=atoms,
                                supercell=supercell,
                                is_reduced_second=False,
                                folder=folder)
        return finite_difference

    def list_of_replicas(self):
        n_replicas = np.prod(self.supercell)
        n_unit_atoms = self.atoms.positions.shape[0]
        # Create list of index
        replicated_cell = self.replicated_atoms.cell
        replicated_cell_inv = np.linalg.inv(replicated_cell)
        sxij = self.replicated_atoms.positions.dot(replicated_cell_inv)
        sxij = sxij - np.round(sxij)
        replicated_atoms_positions = sxij.dot(replicated_cell)
        list_of_replicas = (
                replicated_atoms_positions.reshape((n_replicas, n_unit_atoms, 3)) -
                self.atoms.positions[np.newaxis, :, :])
        return list_of_replicas[:, 0, :]

    def list_of_index(self):
        list_of_index = self.list_of_replicas().dot(np.linalg.inv(self.atoms.cell)).round(0).astype(int)
        return list_of_index

    def import_second_and_third_from_sheng(self):
        supercell = self.supercell
        atoms = self.atoms
        folder = self.folder_name

        n_replicas = np.prod(supercell)
        n_unit_atoms = self.atoms.positions.shape[0]

        second_file = folder + '/' + 'FORCE_CONSTANTS_2ND'
        if not os.path.isfile(second_file):
            second_file = folder + '/' + 'FORCE_CONSTANTS'

        nx, ny, nz = supercell
        n_replicas = np.prod(supercell)
        with open(second_file, 'r') as file:
            n_rows = int(file.readline())
            n_unit_atoms = int(n_rows / n_replicas)

            second_order = np.zeros((n_unit_atoms, 3, supercell[0],
                                     supercell[1], supercell[2], n_unit_atoms, 3))

            line = file.readline()
            while line:
                try:
                    i, j = np.fromstring(line, dtype=np.int, sep=' ')
                except ValueError as err:
                    print(err)
                i_ix, i_iy, i_iz, i_iatom = split_index(i, supercell[0], supercell[1], supercell[2])
                j_ix, j_iy, j_iz, j_iatom = split_index(j, supercell[0], supercell[1], supercell[2])
                for alpha in range(3):
                    if (i_ix == 1) and (i_iy == 1) and (i_iz == 1):
                        second_order[i_iatom - 1, alpha, j_iz - 1, j_iy - 1, j_ix - 1, j_iatom - 1, :] = \
                            np.fromstring(file.readline(), dtype=np.float, sep=' ')
                    else:
                        file.readline()
                line = file.readline()
        self.is_reduced_second = True
        third_order = np.zeros((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))

        third_file = folder + '/' + 'FORCE_CONSTANTS_3RD'
        second_cell_list = []
        third_cell_list = []
        with open(third_file, 'r') as file:
            line = file.readline()
            n_third = int(line)
            list_1 = np.zeros((n_third, 3))
            list_2 = np.zeros((n_third, 3))
            for i in range(n_third):
                file.readline()
                file.readline()
                second_cell_position = np.fromstring(file.readline(), dtype=np.float, sep=' ')
                second_cell_index = second_cell_position.dot(np.linalg.inv(atoms.cell)).round(0).astype(int)
                second_cell_list.append(second_cell_index)

                # create mask to find the index
                second_cell_id = (self.list_of_index()[:] == second_cell_index).prod(axis=1)
                second_cell_id = np.argwhere(second_cell_id).flatten()

                third_cell_position = np.fromstring(file.readline(), dtype=np.float, sep=' ')
                third_cell_index = third_cell_position.dot(np.linalg.inv(atoms.cell)).round(0).astype(int)
                third_cell_list.append(third_cell_index)

                # create mask to find the index
                third_cell_id = (self.list_of_index()[:] == third_cell_index).prod(axis=1)
                third_cell_id = np.argwhere(third_cell_id).flatten()

                atom_i, atom_j, atom_k = np.fromstring(file.readline(), dtype=np.int, sep=' ') - 1
                interactions = np.zeros((3, 3, 3))
                for _ in range(27):
                    values = np.fromstring(file.readline(), dtype=np.float, sep=' ')
                    coords = values[:3].round(0).astype(int) - 1
                    alpha, beta, gamma = values[:3].round(0).astype(int) - 1
                    try:
                        third_order[atom_i, alpha, second_cell_id, atom_j, beta, third_cell_id, atom_k, gamma] = values[
                            3]
                    except TypeError as err:
                        print(err)
                    except IndexError as err:
                        print(err)
        second_cell_list = np.array(second_cell_list)
        third_cell_list = np.array(third_cell_list)
        second_order = second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3), order='C')
        third_order = third_order.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas *
                                           n_unit_atoms * 3), order='C')
        self.second_order = second_order
        self.third_order = third_order

    def __convert_to_poscar(self, atoms, supercell=None):
        list_of_types = []
        for symbol in atoms.get_chemical_symbols():
            for i in range(np.unique(atoms.get_chemical_symbols()).shape[0]):
                if np.unique(atoms.get_chemical_symbols())[i] == symbol:
                    list_of_types.append(str(i))

        poscar = {'lattvec': atoms.cell / 10,
                  'positions': (atoms.positions.dot(np.linalg.inv(atoms.cell))).T,
                  'elements': atoms.get_chemical_symbols(),
                  'types': list_of_types}
        if supercell is not None:
            poscar['na'] = supercell[0]
            poscar['nb'] = supercell[1]
            poscar['nc'] = supercell[2]
        return poscar

    def gen_supercell(self):
        """Generate the geometry based on symmetry
        """
        supercell = self.supercell
        atoms = self.atoms
        replicated_atoms = atoms.copy() * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])
        return replicated_atoms

    def optimize(self, method='CG', tol=MAX_FORCE):
        """Execute the geometry optimization by minimizing
           the maximum force component
        """
        # Compute the maximum force component based on initial geometry
        # and specified method
        print('Initial max force: ' + "{0:.4e}".format(self.max_force(self.atoms.positions, self.atoms)))
        print('Optimization method ' + method)

        # Execute the minimization and display the
        # optimized the maximum force component
        result = minimize(self.max_force, self.atoms.positions, args=self.atoms, jac=self.gradient, method=method,
                          tol=tol)
        print(result.message)
        # Rewrite the atomic position based on the optimized geometry
        self.atoms.positions = result.x.reshape((int(result.x.size / 3), 3))
        ase.io.write('minimized_' + str(self.atoms.get_chemical_formula()) + '.xyz', self.atoms, format='extxyz')
        print('Final max force: ' + "{0:.4e}".format(self.max_force(self.atoms.positions, self.atoms)))
        return self.max_force(self.atoms.positions, self.atoms)

    def max_force(self, x, atoms):
        """Construct the maximum force component for a given structure
        """
        # Define the gradient based on atomic position and atom object
        grad = self.gradient(x, atoms)
        # Maximum force component is set as the 2 norm of the gradient
        return np.linalg.norm(grad, 2)

    def gradient(self, x, input_atoms):
        """Construct the gradient based on the given structure and atom object
        """
        # Set a copy for the atom object so that
        # the progress of the optimization is traceable
        atoms = input_atoms.copy()
        input_atoms.positions = np.reshape(x, (int(x.size / 3.), 3), order='C')
        # Force is the negative of the gradient
        gr = -1. * input_atoms.get_forces()
        grad = np.reshape(gr, gr.size, order='C')
        input_atoms.positions = atoms.positions
        return grad

    def calculate_single_second(self, atom_id):
        replicated_atoms = self.replicated_atoms
        n_replicated_atoms = len(replicated_atoms.numbers)
        dx = self.second_order_delta
        second_per_atom = np.zeros((3, n_replicated_atoms * 3), order='C')
        for alpha in range(3):
            for move in (-1, 1):
                shift = np.zeros((n_replicated_atoms, 3), order='C')
                shift[atom_id, alpha] += move * dx

                # Compute the numerator of the approximated second matrices
                # (approximated force from forward difference -
                #  approximated force from backward difference )
                #  based on the atom move
                second_per_atom[alpha, :] += move * self.gradient(replicated_atoms.positions + shift,
                                                                  replicated_atoms)
        return second_per_atom

    def calculate_second(self):
        """Core method to compute second order force constant matrices
        """
        atoms = self.atoms
        print('Calculating second order potential derivatives')
        n_unit_cell_atoms = len(atoms.numbers)
        replicated_atoms = self.replicated_atoms
        n_replicated_atoms = len(replicated_atoms.numbers)
        dx = self.second_order_delta
        if self.is_reduced_second:
            n_atoms = n_unit_cell_atoms
        else:
            n_atoms = n_replicated_atoms
        second = np.zeros((n_atoms, 3, n_replicated_atoms * 3), order='C')

        # Shift the atom back and forth (-1 and +1) after specifying
        # the atom and direction to shift
        # Try to read the partial file if any
        filename = self.folder_name + '/' + SECOND_ORDER_WITH_PROGRESS_FILE

        for i in range(n_atoms):
            with h5py.File(filename, 'a') as partial_second:
                i_force_atom_exists = str(i) in partial_second
                if not i_force_atom_exists:
                    print('Moving atom ' + str(i))
                    partial_second.create_dataset(str(i), data=self.calculate_single_second(i), chunks=True)
        with h5py.File(filename, 'r') as partial_second:
            for i in range(n_atoms):
                second[i] = partial_second[str(i)]

        n_supercell = np.prod(self.supercell)
        if self.is_reduced_second:
            second = second.reshape((n_unit_cell_atoms, 3, n_supercell, n_unit_cell_atoms, 3), order='C')
        else:
            second = second.reshape((n_supercell, n_unit_cell_atoms, 3, n_supercell, n_unit_cell_atoms, 3), order="C")

        # Approximate the second order force constant matrices
        # using central difference formula
        second = second / (2. * dx)
        return second

    def calculate_third(self):
        """Core method to compute third order force constant matrices
        """
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms

        # TODO: Here we should create it sparse
        n_in_unit_cell = len(atoms.numbers)
        replicated_atoms = replicated_atoms
        n_replicated_atoms = len(replicated_atoms.numbers)
        n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
        dx = self.third_order_delta
        print('Calculating third order potential derivatives')

        # Exploit the geometry symmetry prior to compute
        # third order force constant matrices
        is_symmetry_enabled = (self.third_order_symmerty_inputs is not None)

        if is_symmetry_enabled:
            from thirdorder_core import SymmetryOperations, Wedge, reconstruct_ifcs
            from thirdorder_espresso import calc_frange, calc_dists
            if self.third_order_symmerty_inputs['SYMPREC']:
                symprec = self.third_order_symmerty_inputs['SYMPREC']
            else:
                symprec = SYMPREC_THIRD_ORDER

            if self.third_order_symmerty_inputs['NNEIGH']:
                nneigh = self.third_order_symmerty_inputs['NNEIGH']
            else:
                # default on full calculation
                self.third_order_symmerty_inputs = None

            # Compute third order force constant matrices
            # by utilizing the geometry symmetry
            poscar = self.__convert_to_poscar(self.atoms)
            f_range = None
            print("Analyzing symmetries")
            symops = SymmetryOperations(
                poscar["lattvec"], poscar["types"], poscar["positions"].T, symprec)
            print("- Symmetry group {0} detected".format(symops.symbol))
            print("- {0} symmetry operations".format(symops.translations.shape[0]))
            print("Creating the supercell")
            sposcar = self.__convert_to_poscar(self.replicated_atoms, self.supercell)
            print("Computing all distances in the supercell")
            dmin, nequi, shifts = calc_dists(sposcar)
            if nneigh != None:
                frange = calc_frange(poscar, sposcar, nneigh, dmin)
                print("- Automatic cutoff: {0} nm".format(frange))
            else:
                frange = f_range
                print("- User-defined cutoff: {0} nm".format(frange))
            print("Looking for an irreducible set of third-order IFCs")
            wedge = Wedge(poscar, sposcar, symops, dmin, nequi, shifts,
                          frange)
            self.wedge = wedge
            print("- {0} triplet equivalence classes found".format(wedge.nlist))
            two_atoms_mesh = wedge.build_list4()
            two_atoms_mesh = np.array(two_atoms_mesh)
            print('object created')

            k_coord_sparse = []
            mesh_index_sparse = []
            k_at_sparse = []


        else:
            # Compute third order force constant matrices by using the central
            # difference formula for the approximation for third order derivatives
            i_at_sparse = []
            i_coord_sparse = []
            jat_sparse = []
            j_coord_sparse = []
            k_sparse = []


        value_sparse = []
        two_atoms_mesh_index = -1
        file = None
        progress_filename = self.folder_name + '/' + THIRD_ORDER_WITH_PROGRESS_FILE
        try:
            file = open(progress_filename, 'r+')
        except FileNotFoundError as err:
            print(err)
        else:
            for line in file:
                iat, icoord, jat, jcoord, k, value = np.fromstring(line, dtype=np.float, sep=' ')
                read_iat = int(iat)
                read_icoord = int(icoord)
                read_jat = int(jat)
                read_jcoord = int(jcoord)
                read_k = int(k)
                value_sparse.append(value)
                two_atoms_mesh_index = np.ravel_multi_index((read_icoord, read_jcoord, read_iat, read_jat),
                                                            (3, 3, n_in_unit_cell, n_supercell * n_in_unit_cell),
                                                            order='C')

                if is_symmetry_enabled:
                    # Here we need some index manipulation to use the results produced by the third order library
                    mask = (two_atoms_mesh[:] == np.array([jat, iat, jcoord, icoord]).astype(int))
                    two_atoms_mesh_index = np.argwhere(mask[:, 0] * mask[:, 1] * mask[:, 2] * mask[:, 3])[0, 0]

                    mesh_index_sparse.append(two_atoms_mesh_index)
                    read_kat, read_kcoord = np.unravel_index(read_k, (n_supercell * n_in_unit_cell, 3), order='C')
                    k_at_sparse.append(read_kat)
                    k_coord_sparse.append(read_kcoord)
                else:
                    i_at_sparse.append(read_iat)
                    i_coord_sparse.append(read_icoord)
                    jat_sparse.append(read_jat)
                    j_coord_sparse.append(read_jcoord)
                    k_sparse.append(read_k)

        two_atoms_mesh_index = two_atoms_mesh_index + 1
        if is_symmetry_enabled:
            n_tot_phonons = two_atoms_mesh.shape[0]
        else:
            n_tot_phonons = n_supercell * (n_in_unit_cell * 3) ** 2

        for mesh_index_counter in range(two_atoms_mesh_index, n_tot_phonons):
            if not file:
                file = open(progress_filename, 'a+')

            if is_symmetry_enabled:
                jat, iat, jcoord, icoord = two_atoms_mesh[mesh_index_counter]
            else:
                icoord, jcoord, iat, jat = np.unravel_index(mesh_index_counter,
                                                            (3, 3, n_in_unit_cell, n_supercell * n_in_unit_cell),
                                                            order='C')
                # print(jat, iat, jcoord, icoord, two_atoms_mesh[mesh_index_counter])
            value = self.calculate_single_third(iat, icoord, jat, jcoord)

            sensitiviry = 1e-20
            if (np.abs(value) > sensitiviry).any():
                for k in np.argwhere(np.abs(value) > sensitiviry):
                    k = k[0]
                    file.write('%i %i %i %i %i %.8e\n' % (iat, icoord, jat, jcoord, k, value[k]))
                    if is_symmetry_enabled:
                        kat, kcoord = np.unravel_index(k, (n_supercell * n_in_unit_cell, 3), order='C')
                        k_at_sparse.append(kat)
                        k_coord_sparse.append(kcoord)
                        mesh_index_sparse.append(mesh_index_counter)
                    else:
                        i_at_sparse.append(iat)
                        i_coord_sparse.append(icoord)
                        jat_sparse.append(jat)
                        j_coord_sparse.append(jcoord)
                        k_sparse.append(k)
                    value_sparse.append(value[k])
            if (mesh_index_counter % 500) == 0:
                print('Calculate third ', mesh_index_counter / n_tot_phonons * 100, '%')

        file.close()

        if is_symmetry_enabled:
            coords = np.array([k_coord_sparse, mesh_index_sparse, k_at_sparse])
            shape = (3, two_atoms_mesh.shape[0], n_replicated_atoms)
            phipart = COO(coords, value_sparse, shape).todense()
            phifull = np.array(reconstruct_ifcs(phipart, self.wedge, two_atoms_mesh,
                                                poscar, sposcar))
            phifull = phifull.swapaxes(3, 2).swapaxes(1, 2).swapaxes(0, 1)
            phifull = phifull.swapaxes(4, 3).swapaxes(3, 2)
            phifull = phifull.swapaxes(5, 4)
            phifull = COO.from_numpy(phifull)

        else:
            coords = np.array([i_at_sparse, i_coord_sparse, jat_sparse, j_coord_sparse, k_sparse])
            shape = (n_in_unit_cell, 3, n_supercell * n_in_unit_cell, 3, n_supercell * n_in_unit_cell * 3)
            phifull = COO(coords, np.array(value_sparse), shape)

        # phifull = phifull.reshape(
        #     (1, n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3))
        phifull = phifull / (4. * dx * dx)
        phifull = phifull.reshape((self.n_atoms * 3, self.n_replicas * self.n_atoms * 3, self.n_replicas *
                                   self.n_atoms * 3))
        return phifull

    def calculate_single_third(self, iat, icoord, jat, jcoord):
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        dx = self.third_order_delta
        n_in_unit_cell = len(atoms.numbers)
        replicated_atoms = replicated_atoms
        n_replicated_atoms = len(replicated_atoms.numbers)
        n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)

        phi_partial = np.zeros((n_supercell * n_in_unit_cell * 3))

        for n in range(4):
            shift = np.zeros((n_replicated_atoms, 3))
            # TODO: change this silly way to do isign and jsign
            isign = (-1) ** (n // 2)
            jsign = -(-1) ** (n % 2)
            delta = np.zeros(3)
            delta[icoord] = isign * dx
            shift[iat, :] += delta
            delta = np.zeros(3)
            delta[jcoord] = jsign * dx
            shift[jat, :] += delta
            phi_partial[:] += isign * jsign * (
                    -1. * self.gradient(replicated_atoms.positions + shift, replicated_atoms))
        return phi_partial

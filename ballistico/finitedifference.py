"""
Ballistico
Anharmonic Lattice Dynamics
"""
from ase import Atoms
import os
import ase.io
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import load_npz, save_npz
from sparse import COO
import ballistico.io.eskm_io as io
import ballistico.io.shengbte_io as shengbte_io
from ballistico.grid import wrap_coordinates
import h5py
import ase.units as units
from ballistico.grid import Grid
from ballistico.helpers.logger import get_logger
logging = get_logger()


tenjovermoltoev = 10 * units.J / units.mol

# see bug report: https://github.com/h5py/h5py/issues/1101
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

EVTOTENJOVERMOL = units.mol / (10 * units.J)
DELTA_SHIFT = 1e-5

# Tolerance for symmetry search
SYMPREC_THIRD_ORDER = 1e-5
# Tolerance for geometry optimization
MAX_FORCE = 1e-4

MAIN_FOLDER = 'displacement'
SECOND_ORDER_FILE = 'second.npy'
DYNMAT_FILE = 'dynmat.npy'
LIST_OF_REPLICAS_FILE = 'list_of_replicas.npy'
THIRD_ORDER_FILE_SPARSE = 'third.npz'
THIRD_ORDER_FILE = 'third.npy'
REPLICATED_ATOMS_FILE = 'replicated_atoms.xyz'
SECOND_ORDER_WITH_PROGRESS_FILE = 'second_order_progress.hdf5'
THIRD_ORDER_WITH_PROGRESS_FILE = 'third_order_progress'


def convert_to_poscar(atoms, supercell=None):
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


def calculate_symmetrize_dynmat(finite_difference):
    if not finite_difference.is_reduced_second:
        second_transpose = finite_difference.second_order.transpose((3, 4, 5, 0, 1, 2))
        delta_symm = np.sum(np.abs(finite_difference.second_order - second_transpose))
        logging.info('asymmetry of dynmat: ' + str(delta_symm))
        finite_difference.second_order = 0.5 * (finite_difference.second_order + second_transpose)
    else:
        logging.info('cannot symmetrize a reduced dynamical matrix')
    return finite_difference


def calculate_acoustic_dynmat(finite_difference):
    dynmat = finite_difference.second_order
    n_unit = finite_difference.second_order[0].shape[0]
    sumrulecorr = 0.
    for i in range(n_unit):
        off_diag_sum = np.sum(dynmat[0, i, :, :, :, :], axis=(-2, -3))
        dynmat[0, i, :, 0, i, :] -= off_diag_sum
        sumrulecorr += np.sum(off_diag_sum)
    logging.info('error sum rule: ' + str(sumrulecorr))
    finite_difference.second_order = dynmat
    return finite_difference


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
                 is_reduced_second=True,
                 optimization_method=None,
                 distance_threshold=None):

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
        optimization_method:string, default is None
            optional optimization method to instruct if initial input atom geometry be optimized. Some options are
            ‘CG’, ‘BFGS’, ‘Newton-CG’, and ‘L-BFGS-B’. For the list of all available optimization methods see:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        distance_threshold: float
            the maximum distance between two interacting atoms
        """

        # Store the user defined information to the object
        self.atoms = atoms
        self.supercell = supercell
        self.n_atoms = self.atoms.get_masses().shape[0]
        self._direct_grid = Grid(supercell, order='F')
        self.n_modes = self.n_atoms * 3
        self.n_replicas = np.prod(supercell)
        self.n_replicated_atoms = self.n_replicas * self.n_atoms
        self.cell_inv = np.linalg.inv(self.atoms.cell)

        self.folder = folder
        self.is_reduced_second = is_reduced_second
        self.distance_threshold = distance_threshold
        self._replicated_atoms = None
        self._list_of_replicas = None
        self._second_order = None
        self._third_order = None
        self._dynmat = None
        self._replicated_cell_inv = None

        # Directly loading the second/third order force matrices
        if second_order is not None:
            self.second_order = second_order
        if third_order is not None:
            self.third_order = third_order

        # Initialize calculator
        if calculator:
            self._is_able_to_calculate = True
            self.calculator = calculator
            self.second_order_delta = delta_shift
            self.third_order_delta = delta_shift
            self.optimization_method = optimization_method
            if calculator_inputs is not None:
                calculator_inputs['keep_alive'] = True
                self.calculator_inputs = calculator_inputs
                self.atoms.set_calculator(self.calculator(**self.calculator_inputs))
            else:
                self.calculator_inputs = None
                self.atoms.set_calculator(self.calculator())
            if third_order_symmerty_inputs is not None:
                self.third_order_symmerty_inputs = third_order_symmerty_inputs.copy()
                for k, v in third_order_symmerty_inputs.items():
                    self.third_order_symmerty_inputs[k.upper()] = v
            else:
                self.third_order_symmerty_inputs = None

            # Optimize the structure if optimizing flag is turned to true
            # and the calculation is set to start from the starch:
            if self.optimization_method is not None:
                self.optimize()



    @classmethod
    def from_files(cls, replicated_atoms, dynmat_file, third_file=None, folder=None, supercell=(1, 1, 1), third_energy_threshold=0., distance_threshold=None, is_symmetrizing=False, is_acoustic_sum=False):
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
        :param is_acoustic_sum:
        :return:
        """
        fd = cls._import_from_files(replicated_atoms, dynmat_file, third_file, folder, supercell, third_energy_threshold, distance_threshold)
        if is_symmetrizing:
            fd = calculate_symmetrize_dynmat(fd)
        if is_acoustic_sum:
            fd = calculate_acoustic_dynmat(fd)
        fd.distance_threshold = distance_threshold
        return fd


    @classmethod
    def from_folder(cls, folder, supercell=(1, 1, 1), format='eskm', third_energy_threshold=0., distance_threshold=None, is_symmetrizing=False, is_acoustic_sum=False):
        """
        Create a finite difference object from a folder
        :param folder:
        :param supercell:
        :param format:
        :param third_energy_threshold:
        :param distance_threshold:
        :param is_symmetrizing:
        :param is_acoustic_sum:
        :return:
        """
        if format == 'numpy':
            fd = cls.__from_numpy(folder, supercell, distance_threshold=distance_threshold)
        elif format == 'eskm':
            fd = cls.__from_eskm(folder, supercell,
                          third_energy_threshold=third_energy_threshold, distance_threshold=distance_threshold)
        elif format == 'shengbte':
            fd = cls.__from_shengbte(folder, supercell)

        elif format == 'hiphive':
            fd = cls.__from_hiphive(folder, supercell)    
        else:
            raise ValueError
        if is_symmetrizing:
            fd = calculate_symmetrize_dynmat(fd)
        if is_acoustic_sum:
            fd = calculate_acoustic_dynmat(fd)
        return fd

    @classmethod
    def _import_from_files(cls, atoms, dynmat_file=None, third_file=None, folder=None, supercell=(1, 1, 1),
                           third_energy_threshold=0., distance_threshold=None):
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
        finite_difference = {'atoms': atoms,
                             'supercell': supercell,
                             'folder': folder}
        fd = cls(**finite_difference)

        if dynmat_file:
            logging.info('Reading dynamical matrix')
            second_dl = io.import_second(atoms, replicas=supercell, filename=dynmat_file)
            is_reduced_second = not (n_replicas ** 2 * (n_unit_atoms * 3) ** 2 == second_dl.size)
            logging.info('Is reduced second: ' + str(is_reduced_second))
            fd.second_order = second_dl
            fd.is_reduced_second = is_reduced_second
            logging.info('Dynamical matrix stored.')

        if third_file:
            try:
                logging.info('Reading sparse third order')
                third_dl = io.import_sparse_third(atoms=atoms,
                                                  supercell=supercell,
                                                  filename=third_file,
                                                  third_energy_threshold=third_energy_threshold,
                                                  distance_threshold=distance_threshold,
                                                  replicated_atoms=fd.replicated_atoms)
                logging.info('Third order matrix stored.')

            except UnicodeDecodeError:
                if third_energy_threshold != 0:
                    raise ValueError('Third threshold not supported for dense third')
                logging.info('Reading dense third order')
                third_dl = io.import_dense_third(atoms, supercell=supercell, filename=third_file)
                logging.info('Third order matrix stored.')
            third_dl = third_dl[:n_unit_atoms]
            third_shape = (
                n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3)
            third_dl = third_dl.reshape(third_shape)
            fd.third_order = third_dl

        return fd


    @classmethod
    def __from_numpy(cls, folder, supercell=(1, 1, 1), distance_threshold=None):
        if folder[-1] != '/':
            folder = folder + '/'
        config_file = folder + REPLICATED_ATOMS_FILE
        n_replicas = np.prod(supercell)
        atoms = ase.io.read(config_file, format='extxyz')
        n_atoms = int(atoms.positions.shape[0] / n_replicas)
        fd = cls._import_from_files(atoms=atoms,
                                    folder=folder,
                                    supercell=supercell)
        second_order = np.load(folder + SECOND_ORDER_FILE)
        if second_order.size == (n_replicas * n_atoms * 3) ** 2:
            fd.is_reduced_second = False
        else:
            fd.is_reduced_second = True
        third_order = COO.from_scipy_sparse(load_npz(folder + THIRD_ORDER_FILE_SPARSE)) \
            .reshape((n_atoms * 3, n_replicas * n_atoms * 3, n_replicas * n_atoms * 3))
        fd.second_order = second_order
        fd.third_order = third_order
        if distance_threshold is not None:
            fd.third_order = fd.prune_sparse_third_with_distance_threshold(distance_threshold)
        return fd


    @classmethod
    def __from_eskm(cls, folder, supercell=(1, 1, 1),
                          third_energy_threshold=0., distance_threshold=None):
        config_file = str(folder) + "/CONFIG"
        dynmat_file = str(folder) + "/Dyn.form"
        third_file = str(folder) + "/THIRD"
        atoms = ase.io.read(config_file, format='dlp4')
        fd = cls._import_from_files(atoms, dynmat_file, third_file, folder, supercell,
                                    third_energy_threshold=third_energy_threshold, distance_threshold=distance_threshold)
        return fd


    @classmethod
    def __from_shengbte(cls, folder, supercell=None):
        config_file = folder + '/' + 'CONTROL'
        try:
            atoms, supercell = shengbte_io.import_control_file(config_file)
        except FileNotFoundError as err:
            config_file = folder + '/' + 'POSCAR'
            logging.info('\nTrying to open POSCAR')
            atoms = ase.io.read(config_file)
        # Create a finite difference object
        finite_difference = cls(atoms=atoms, supercell=supercell, folder=folder)
        second_order, is_reduced_second, third_order = shengbte_io.import_second_and_third_from_sheng(finite_difference)
        finite_difference.second_order = second_order
        finite_difference.third_order = third_order
        finite_difference.is_reduced_second = is_reduced_second
        return finite_difference

    @classmethod
    def __from_hiphive(cls, folder, supercell=None):
      try:
        import ballistico.io.hiphive_io as hiphive_io
      except ImportError as err:
        logging.info(err)
        logging.error('In order to use hiphive along with ballistico, hiphive is required. \
                Please consider installing hihphive. More info can be found at: \
                https://hiphive.materialsmodeling.org/')
        import sys
        sys.exit()
      atom_prime_file = str(folder) + '/atom_prim.xyz'
      atoms = ase.io.read(atom_prime_file)
      # Create a finite difference object
      finite_difference = cls(atoms=atoms, supercell=supercell, folder=folder)
      if 'model2.fcs' in os.listdir(str(folder)):
        second_order = hiphive_io.import_second_from_hiphive(finite_difference)
        finite_difference.second_order = second_order
      if 'model3.fcs' in os.listdir(str(folder)):
        # Derive constants used for third-order reshape
        supercell = np.array(supercell)
        n_prim = atoms.copy().get_masses().shape[0]
        n_sc = np.prod(supercell)
        dim = len(supercell[supercell > 1])
        third_order = hiphive_io.import_third_from_hiphive(finite_difference)
        finite_difference.third_order = third_order[0].reshape(n_prim*dim, n_sc*n_prim*dim,
            n_sc*n_prim*dim)
      return finite_difference
		

    @property
    def atoms(self):
        """Atoms object in ase format"""
        return self._atoms


    @atoms.getter
    def atoms(self):
        return self._atoms


    @atoms.setter
    def atoms(self, new_atoms):
        self._atoms = new_atoms


    @property
    def dynmat(self):
        """Dynamical matrix calculated from the derivative of the input forcefield. Ouput in THz^2.

        Returns
        -------
        np.array
            (n_particles, 3, n_replicas, n_particles, 3) tensor containing the second order derivative of the dynamical matrix rescaled by the masses

        """
        return self._dynmat


    @dynmat.getter
    def dynmat(self):
        """Obtain the second order force constant matrix either by loading in
           or performing calculation using the provided calculator.
        """
        # Once confirming that the second order does not exist, try load in
        if self._dynmat is None:
            self._dynmat = self.calculate_dynamical_matrix()
        return self._dynmat


    @dynmat.setter
    def dynmat(self, new_dynmat):
        """Save the loaded/computed second order force constant matrix
            to preset proper folders.
        """
        # make the  folder and save the second order force constant matrix into it
        self._dynmat = new_dynmat


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
                folder = self.folder
                folder += '/'
                self._second_order = np.load(folder + SECOND_ORDER_FILE)
            except FileNotFoundError as e:
                logging.info(e)
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
        folder = self.folder
        folder += '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(folder + SECOND_ORDER_FILE, new_second_order)
        try:
            os.remove(folder + SECOND_ORDER_WITH_PROGRESS_FILE)
        except FileNotFoundError as err:
            logging.info('No second order progress file found.')
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
            folder = self.folder
            folder += '/'
            try:
                self._third_order = COO.from_scipy_sparse(load_npz(folder + THIRD_ORDER_FILE_SPARSE)) \
                    .reshape((1 * self.n_atoms * 3, self.n_replicas * self.n_atoms * 3, self.n_replicas *
                              self.n_atoms * 3))
            except FileNotFoundError as e:
                # calculate the third order
                self.third_order = self.calculate_third(self.distance_threshold)
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
        folder = self.folder
        folder += '/'
        # Remove the partial file if exists
        try:
            os.remove(folder + THIRD_ORDER_WITH_PROGRESS_FILE)
        except FileNotFoundError as err:
            logging.info('No third order progress file found')
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
            try:
                if self.calculator_inputs is not None:
                    self.replicated_atoms.set_calculator(self.calculator(**self.calculator_inputs))
                else:
                    self.replicated_atoms.set_calculator(self.calculator())
            except AttributeError:
                logging.info('No calculator found')
        return self._replicated_atoms


    @replicated_atoms.setter
    def replicated_atoms(self, new_replicated_atoms):
        """Save the loaded/computed duplicated atom geometry
            to preset proper folders.
        """
        # Make the folder and save the replicated atom xyz files into it
        folder = self.folder
        folder += '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        ase.io.write(folder + REPLICATED_ATOMS_FILE, new_replicated_atoms, format='extxyz')
        self._replicated_atoms = new_replicated_atoms


    @property
    def list_of_replicas(self):
        return self._list_of_replicas


    @list_of_replicas.getter
    def list_of_replicas(self):
        if self._list_of_replicas is None:
            self.list_of_replicas = self.calculate_list_of_replicas()
        return self._list_of_replicas


    @list_of_replicas.setter
    def list_of_replicas(self, new_list_of_replicas):
        self._list_of_replicas = new_list_of_replicas


    @property
    def replicated_cell_inv(self):
        return self._replicated_cell_inv


    @replicated_cell_inv.getter
    def replicated_cell_inv(self):
        if self._replicated_cell_inv is None:
            self.replicated_cell_inv = np.linalg.inv(self.replicated_atoms.cell)
        return self._replicated_cell_inv


    @replicated_cell_inv.setter
    def replicated_cell_inv(self, new_replicated_cell_inv):
        self._replicated_cell_inv = new_replicated_cell_inv


    def calculate_list_of_replicas(self):
        list_of_index = self._direct_grid.grid(is_wrapping=True)
        return list_of_index.dot(self.atoms.cell)


    def gen_supercell(self):
        """Generate the geometry based on symmetry
        """
        supercell = self.supercell
        atoms = self.atoms
        replicated_atoms = atoms.copy() * (supercell[0], supercell[1], supercell[2])
        replicated_positions = self._direct_grid.grid().dot(atoms.cell)[:, np.newaxis, :] + atoms.positions[np.newaxis, :, :]
        replicated_atoms.set_positions(replicated_positions.reshape(self.n_replicas * self.n_atoms, 3))
        return replicated_atoms


    def optimize(self, tol=MAX_FORCE):
        """Execute the geometry optimization by minimizing
           the maximum force component
        """
        # Compute the maximum force component based on initial geometry
        # and specified method
        method = self.optimization_method
        logging.info('Initial max force: ' + "{0:.4e}".format(self.max_force(self.atoms.positions, self.atoms)))
        logging.info('Optimization method ' + method)

        # Execute the minimization and display the
        # optimized the maximum force component
        result = minimize(self.max_force, self.atoms.positions, args=self.atoms, jac=self.gradient, method=method,
                          tol=tol)
        logging.info(result.message)
        # Rewrite the atomic position based on the optimized geometry
        self.atoms.positions = result.x.reshape((int(result.x.size / 3), 3))
        ase.io.write('minimized_' + str(self.atoms.get_chemical_formula()) + '.xyz', self.atoms, format='extxyz')
        logging.info('Final max force: ' + "{0:.4e}".format(self.max_force(self.atoms.positions, self.atoms)))
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
        input_atoms.positions = np.reshape(x, (int(x.size / 3.), 3))
        # Force is the negative of the gradient

        gr = -1. * input_atoms.get_forces()
        grad = np.reshape(gr, gr.size)
        input_atoms.positions = atoms.positions
        return grad


    def calculate_single_second(self, atom_id):
        replicated_atoms = self.replicated_atoms
        n_replicated_atoms = len(replicated_atoms.numbers)
        dx = self.second_order_delta
        second_per_atom = np.zeros((3, n_replicated_atoms * 3))
        for alpha in range(3):
            for move in (-1, 1):
                shift = np.zeros((n_replicated_atoms, 3))
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
        logging.info('Calculating second order potential derivatives')
        n_unit_cell_atoms = len(atoms.numbers)
        replicated_atoms = self.replicated_atoms
        n_replicated_atoms = len(replicated_atoms.numbers)
        dx = self.second_order_delta
        if self.is_reduced_second:
            n_atoms = n_unit_cell_atoms
        else:
            n_atoms = n_replicated_atoms
        second = np.zeros((n_atoms, 3, n_replicated_atoms * 3))

        # Shift the atom back and forth (-1 and +1) after specifying
        # the atom and direction to shift
        # Try to read the partial file if any
        filename = self.folder + '/' + SECOND_ORDER_WITH_PROGRESS_FILE

        for i in range(n_atoms):
            with h5py.File(filename, 'a') as partial_second:
                i_force_atom_exists = str(i) in partial_second
                if not i_force_atom_exists:
                    logging.info('Moving atom ' + str(i))
                    partial_second.create_dataset(str(i), data=self.calculate_single_second(i), chunks=True)
        with h5py.File(filename, 'r') as partial_second:
            for i in range(n_atoms):
                second[i] = partial_second[str(i)]

        n_supercell = np.prod(self.supercell)
        if self.is_reduced_second:
            second = second.reshape((1, n_unit_cell_atoms, 3, n_supercell, n_unit_cell_atoms, 3))
        else:
            second = second.reshape((n_supercell, n_unit_cell_atoms, 3, n_supercell, n_unit_cell_atoms, 3), order="C")

        # Approximate the second order force constant matrices
        # using central difference formula
        second = second / (2. * dx)
        return second

    def calculate_third(self, distance_threshold=None):
        """Core method to compute third order force constant matrices
        """
        logging.info('Calculating third order potential derivatives')
        is_symmetry_enabled = (self.third_order_symmerty_inputs is not None)
        if is_symmetry_enabled:
            if distance_threshold is not None:
                raise TypeError('If symmetry is enabled, no distance_threshold is allowed')
            # Exploit the geometry symmetry prior to compute
            # third order force constant matrices
            phifull = self.calculate_single_third_with_symmetry()
        else:
            phifull = self.calculate_single_third_without_symmetry(distance_threshold=distance_threshold)
        if distance_threshold:
            phifull = self.prune_sparse_third_with_distance_threshold(phifull)
        self.distance_threshold = distance_threshold
        return phifull


    def calculate_single_third_with_symmetry(self):
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        # TODO: Here we should create it sparse
        n_in_unit_cell = len(atoms.numbers)
        replicated_atoms = replicated_atoms
        n_replicated_atoms = len(replicated_atoms.numbers)
        n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
        dx = self.third_order_delta

        # Exploit the geometry symmetry prior to compute
        # third order force constant matrices
        logging.info('Calculating third order potential derivatives')
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
        poscar = convert_to_poscar(self.atoms)
        f_range = None
        logging.info("Analyzing symmetries")
        symops = SymmetryOperations(
            poscar["lattvec"], poscar["types"], poscar["positions"].T, symprec)
        logging.info("- Symmetry group {0} detected".format(symops.symbol))
        logging.info("- {0} symmetry operations".format(symops.translations.shape[0]))
        logging.info("Creating the supercell")
        sposcar = convert_to_poscar(self.replicated_atoms, self.supercell)
        logging.info("Computing all distances in the supercell")
        dmin, nequi, shifts = calc_dists(sposcar)
        if nneigh != None:
            frange = calc_frange(poscar, sposcar, nneigh, dmin)
            logging.info("- Automatic cutoff: {0} nm".format(frange))
        else:
            frange = f_range
            logging.info("- User-defined cutoff: {0} nm".format(frange))
        logging.info("Looking for an irreducible set of third-order IFCs")
        wedge = Wedge(poscar, sposcar, symops, dmin, nequi, shifts,
                      frange)
        self.wedge = wedge
        logging.info("- {0} triplet equivalence classes found".format(wedge.nlist))
        two_atoms_mesh = wedge.build_list4()
        two_atoms_mesh = np.array(two_atoms_mesh)
        logging.info('object created')
        k_coord_sparse = []
        mesh_index_sparse = []
        k_at_sparse = []
        value_sparse = []
        two_atoms_mesh_index = -1
        file = None
        progress_filename = self.folder + '/' + THIRD_ORDER_WITH_PROGRESS_FILE
        try:
            file = open(progress_filename, 'r+')
        except FileNotFoundError as err:
            logging.info('No third order progress file found')
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
                # Here we need some index manipulation to use the results produced by the third order library
                mask = (two_atoms_mesh[:] == np.array([jat, iat, jcoord, icoord]).astype(int))
                two_atoms_mesh_index = np.argwhere(mask[:, 0] * mask[:, 1] * mask[:, 2] * mask[:, 3])[0, 0]

                mesh_index_sparse.append(two_atoms_mesh_index)
                read_kat, read_kcoord = np.unravel_index(read_k, (n_supercell * n_in_unit_cell, 3))
                k_at_sparse.append(read_kat)
                k_coord_sparse.append(read_kcoord)

        two_atoms_mesh_index = two_atoms_mesh_index + 1
        n_tot_phonons = two_atoms_mesh.shape[0]

        for mesh_index_counter in range(two_atoms_mesh_index, n_tot_phonons):
            if not file:
                file = open(progress_filename, 'a+')

            jat, iat, jcoord, icoord = two_atoms_mesh[mesh_index_counter]

            value = self.calculate_single_third(iat, icoord, jat, jcoord)

            sensitiviry = 1e-20
            if (np.abs(value) > sensitiviry).any():
                for k in np.argwhere(np.abs(value) > sensitiviry):
                    k = k[0]
                    file.write('%i %i %i %i %i %.8e\n' % (iat, icoord, jat, jcoord, k, value[k]))
                    kat, kcoord = np.unravel_index(k, (n_supercell * n_in_unit_cell, 3))
                    k_at_sparse.append(kat)
                    k_coord_sparse.append(kcoord)
                    mesh_index_sparse.append(mesh_index_counter)
                    value_sparse.append(value[k])
            if (mesh_index_counter % 500) == 0:
                logging.info('Calculate third ' + str(mesh_index_counter / n_tot_phonons * 100) + '%')

        # TODO: remove this file.close and use with file instead
        file.close()

        coords = np.array([k_coord_sparse, mesh_index_sparse, k_at_sparse])
        shape = (3, two_atoms_mesh.shape[0], n_replicated_atoms)
        phipart = COO(coords, value_sparse, shape).todense()
        phifull = np.array(reconstruct_ifcs(phipart, self.wedge, two_atoms_mesh,
                                            poscar, sposcar))
        #TODO: use transpose here
        phifull = phifull.swapaxes(3, 2).swapaxes(1, 2).swapaxes(0, 1)
        phifull = phifull.swapaxes(4, 3).swapaxes(3, 2)
        phifull = phifull.swapaxes(5, 4)
        phifull = COO.from_numpy(phifull)

        # phifull = phifull.reshape(
        #     (1, n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3))
        phifull = phifull.reshape((self.n_atoms * 3, self.n_replicas * self.n_atoms * 3, self.n_replicas *
                                   self.n_atoms * 3))
        return phifull


    def calculate_single_third_without_symmetry(self, distance_threshold=None):
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        n_in_unit_cell = len(atoms.numbers)
        replicated_atoms = replicated_atoms
        n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)

        # Compute third order force constant matrices by using the central
        # difference formula for the approximation for third order derivatives
        i_at_sparse = []
        i_coord_sparse = []
        jat_sparse = []
        j_coord_sparse = []
        k_sparse = []
        value_sparse = []
        n_forces_to_calculate = n_supercell * (n_in_unit_cell * 3) ** 2
        n_forces_done = 0
        n_forces_skipped = 0
        for iat in range(n_in_unit_cell):
            for jat in range(n_supercell * n_in_unit_cell):
                is_computing = True
                m, j_small = np.unravel_index(jat, (n_supercell, n_in_unit_cell))
                if (distance_threshold is not None):
                    dxij = atoms.positions[iat] - (self.list_of_replicas[m] + atoms.positions[j_small])
                    if (np.linalg.norm(dxij) > distance_threshold):
                        is_computing = False
                        n_forces_skipped += 9
                if is_computing:
                    for icoord in range(3):
                        for jcoord in range(3):
                            value = self.calculate_single_third(iat, icoord, jat, jcoord)
                            # mask = np.linalg.norm(value.reshape(n_supercell * n_in_unit_cell, 3), axis=1) > 1e-6
                            for id in range(value.shape[0]):
                                i_at_sparse.append(iat)
                                i_coord_sparse.append(icoord)
                                jat_sparse.append(jat)
                                j_coord_sparse.append(jcoord)
                                k_sparse.append(id)
                                value_sparse.append(value[id])
                    n_forces_done += 9
                if (n_forces_done + n_forces_skipped % 300) == 0:
                    logging.info('Calculate third derivatives ' + str(int((n_forces_done + n_forces_skipped) / n_forces_to_calculate * 100)) + '%')

        logging.info('total forces to calculate : ' + str(n_forces_to_calculate))
        logging.info('forces calculated : ' + str(n_forces_done))
        logging.info('forces skipped (outside distance threshold) : ' + str(n_forces_skipped))
        coords = np.array([i_at_sparse, i_coord_sparse, jat_sparse, j_coord_sparse, k_sparse])
        shape = (n_in_unit_cell, 3, n_supercell * n_in_unit_cell, 3, n_supercell * n_in_unit_cell * 3)
        phifull = COO(coords, np.array(value_sparse), shape)
        phifull = phifull.reshape((self.n_atoms * 3, self.n_replicas * self.n_atoms * 3, self.n_replicas * self.n_atoms * 3))
        return phifull


    def calculate_single_third(self, iat, icoord, jat, jcoord, dx=None):
        if dx is None:
            dx = self.third_order_delta
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        n_in_unit_cell = len(atoms.numbers)
        n_replicated_atoms = len(replicated_atoms.numbers)
        n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
        phi_partial = np.zeros((n_supercell * n_in_unit_cell * 3))
        for isign in (1, -1):
            for jsign in (1, -1):
                shift = np.zeros((n_replicated_atoms, 3))
                shift[iat, icoord] += isign * dx
                shift[jat, jcoord] += jsign * dx
                phi_partial[:] += isign * jsign * self.calculate_single_third_with_shift(shift)
        return phi_partial / (4. * dx * dx)


    def calculate_single_third_on_phonons(self, k_0, m_0, k_2, m_2, evect, chi):
        #TODO: use a different dx value for the reciprocal space
        #TODO: we probably need to rescale by the mass
        dx = self.third_order_delta
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms

        n_in_unit_cell = len(atoms.numbers)
        replicated_atoms = replicated_atoms
        n_replicated_atoms = len(replicated_atoms.numbers)
        n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
        phi_partial = np.zeros((n_supercell * n_in_unit_cell * 3))
        for sign_1 in (1, -1):
            for sign_2 in (1, -1):
                shift_1 = sign_1 * dx * (evect[k_0, :, m_0] * chi[k_0, 0]).reshape((1, n_in_unit_cell, 3))
                shift_2 = sign_2 * dx * (np.conj(evect[np.newaxis, k_2, :, m_2]) * np.conj(chi[k_2, :, np.newaxis])).reshape((self.n_replicas, n_in_unit_cell, 3))
                shift = (shift_1 + shift_2).reshape((n_replicated_atoms, 3))
                phi_partial += sign_1 * sign_2 * self.calculate_single_third_with_shift(shift)
        return phi_partial / (4. * dx * dx)


    def calculate_single_third_with_shift(self, shift):
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        n_in_unit_cell = len(atoms.numbers)
        n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
        phi_partial = np.zeros((n_supercell * n_in_unit_cell * 3))
        phi_partial[:] = (-1. * self.gradient(replicated_atoms.positions + shift, replicated_atoms))
        return phi_partial


    def calculate_dynamical_matrix(self):
        atoms = self.atoms
        dynmat = self.second_order.copy()
        mass = np.sqrt(atoms.get_masses())
        dynmat /= mass[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        dynmat /= mass[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        dynmat *= EVTOTENJOVERMOL
        return dynmat


    def prune_sparse_third_with_distance_threshold(self, third_order=None, distance_threshold=None):
        if third_order is None:
            third_order = self.third_order
        if distance_threshold is None:
            distance_threshold = self.distance_threshold
        n_replicated_atoms = self.n_atoms * self.n_replicas
        shape = (self.n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3)
        third_order = third_order.reshape(shape)
        coords = third_order.coords
        atoms = self.atoms
        c2_m, c2_at = np.unravel_index(coords[2], (self.n_replicas, self.n_atoms))
        dxij = atoms.positions[coords[0]] - (self.list_of_replicas[c2_m] + atoms.positions[c2_at])
        mask = (np.linalg.norm(dxij, axis=1) < distance_threshold)

        c4_m, c4_at = np.unravel_index(coords[4], (self.n_replicas, self.n_atoms))
        dxij = self.list_of_replicas[c2_m] + atoms.positions[c2_at] - (
                    self.list_of_replicas[c4_m] + atoms.positions[c4_at])
        mask *= (np.linalg.norm(dxij, axis=1) < distance_threshold)

        dxij = (self.list_of_replicas[c4_m] + atoms.positions[c4_at]) - atoms.positions[coords[0]]
        mask *= (np.linalg.norm(dxij, axis=1) < distance_threshold)

        coords = np.vstack(
            [coords[0][mask], coords[1][mask], coords[2][mask], coords[3][mask], coords[4][mask], coords[5][mask]])
        data = third_order.data[mask]
        shape = (self.n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3)
        third_order = COO(coords, data, shape)
        return third_order.reshape((self.n_atoms * 3, n_replicated_atoms * 3, n_replicated_atoms * 3))


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
            reduced_third = self.third_order
        n_unit_atoms = self.n_atoms
        atoms = self.atoms
        n_replicas = self.n_replicas
        reduced_third = reduced_third.reshape(
            (n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
        replicated_positions = self.replicated_atoms.positions.reshape((n_replicas, n_unit_atoms, 3))
        dxij_reduced = atoms.positions[:, np.newaxis, np.newaxis, :] - replicated_positions[np.newaxis, :, :, :]
        dxij_reduced = wrap_coordinates(dxij_reduced, self.replicated_atoms.cell, self.replicated_cell_inv)
        dxij_full = replicated_positions[:, :, np.newaxis, np.newaxis, :] - replicated_positions[np.newaxis, np.newaxis, :, :, :]
        dxij_full = wrap_coordinates(dxij_full, self.replicated_atoms.cell, self.replicated_cell_inv)
        indices = np.argwhere(np.linalg.norm(dxij_reduced, axis=-1) < distance_threshold)

        coords = []
        values = []
        for index in indices:
            for l in range(n_replicas):
                for j in range(n_unit_atoms):
                    dx2 = dxij_reduced[index[0], l, j]
                    is_storing = (np.linalg.norm(dx2) < distance_threshold)
                    dx3 = dxij_full[index[1], index[2], l, j]
                    is_storing *= (np.linalg.norm(dx3) < distance_threshold)
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
        third = self.third_order.reshape((n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3)) / tenjovermoltoev
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

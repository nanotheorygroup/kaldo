import numpy as np
from ballistico.logger import Logger
from ballistico.constants import *
from ballistico.atoms_helper import replicate_atoms
from scipy.optimize import minimize
import ase.io as io
import os
import ballistico.atoms_helper as atom_helper
import numpy as np
import ase.io
import os
import ballistico.constants as constants
from ase import Atoms
from ballistico.logger import Logger


SECOND_ORDER_FILE = 'second.npy'
THIRD_ORDER_FILE = 'third.npy'

REPLICATED_ATOMS_FILE = 'replicated_atoms.xyz'
LIST_OF_INDEX_FILE = 'list_of_index.npy'


class FiniteDifference (object):
    def __init__(self, atoms, supercell=(1,1,1), second_order=None, third_order=None, calculator=None, calculator_inputs=None, is_persistency_enabled=True, folder='displacement'):
        self.atoms = atoms
        self.supercell = supercell

        self._replicated_atoms = None
        self._list_of_index = None
        list_of_types = []
        for symbol in atoms.get_chemical_symbols ():
            for i in range (np.unique (atoms.get_chemical_symbols ()).shape[0]):
                if np.unique (atoms.get_chemical_symbols ())[i] == symbol:
                    list_of_types.append(str (i + 1))

        self.poscar = {'lattvec': atoms.cell/10,
                       'positions': atoms.positions.T,
                       'elements': atoms.get_chemical_symbols(),
                       'types': list_of_types}
        self.calculator = calculator
        self.calculator_inputs = calculator_inputs
        self._second_order = None
        self._third_order = None
        self.is_persistency_enabled = is_persistency_enabled
        self.folder_name = folder
        if self.is_persistency_enabled:
            if not os.path.exists (folder):
                os.makedirs (folder)
        if second_order is not None:
            self.second_order = second_order
        if third_order is not None:
            self.third_order = third_order
                
    @property
    def second_order(self):
        return self._second_order

    @second_order.getter
    def second_order(self):
        if self._second_order is None:
            if self.is_persistency_enabled:
                try:
                    folder = self.folder_name
                    folder += '/'
                    self._second_order = np.load (folder + SECOND_ORDER_FILE)
                except FileNotFoundError as e:
                    print (e)
            if self._second_order is None:
                self.second_order = self.calculate_second ()
        return self._second_order

    @second_order.setter
    def second_order(self, new_second_order):
        if self.is_persistency_enabled:
            folder = self.folder_name
            folder += '/'
            np.save (folder + SECOND_ORDER_FILE, new_second_order)
        self._second_order = new_second_order

    @property
    def third_order(self):
        return self._third_order

    @third_order.getter
    def third_order(self):
        if self._third_order is None:
            if self.is_persistency_enabled:
                try:
                    folder = self.folder_name
                    folder += '/'
                    self._third_order = np.load (folder + THIRD_ORDER_FILE)
                except FileNotFoundError as e:
                    print (e)
            if self._third_order is None:
                self.third_order = self.calculate_third ()
        return self._third_order

    @third_order.setter
    def third_order(self, new_third_order):
        if self.is_persistency_enabled:
            folder = self.folder_name
            folder += '/'
            np.save (folder + THIRD_ORDER_FILE, new_third_order)
        self._third_order = new_third_order

    @property
    def replicated_atoms(self):
        return self._replicated_atoms

    @replicated_atoms.getter
    def replicated_atoms(self):
        if self._replicated_atoms is None:
            if self.is_persistency_enabled:
                try:
                    folder = self.folder_name
                    folder += '/'
                    self._replicated_atoms = ase.io.read (folder + REPLICATED_ATOMS_FILE, format='extxyz')
                except FileNotFoundError as e:
                    print (e)
            if self._replicated_atoms is None:
                atoms = self.atoms
                supercell = self.supercell
                list_of_index = self.list_of_index
                replicated_symbols = []
                n_replicas = list_of_index.shape[0]
                n_unit_atoms = len (atoms.numbers)
                replicated_geometry = np.zeros ((n_replicas, n_unit_atoms, 3))
            
                for i in range (n_replicas):
                    vector = list_of_index[i]
                    replicated_symbols.extend (atoms.get_chemical_symbols ())
                    replicated_geometry[i, :, :] = atoms.positions + vector
                replicated_geometry = replicated_geometry.reshape ((n_replicas * n_unit_atoms, 3))
                replicated_cell = atoms.cell * supercell
                replicated_atoms = Atoms (positions=replicated_geometry,
                                          symbols=replicated_symbols, cell=replicated_cell, pbc=[1, 1, 1])
                self.replicated_atoms = replicated_atoms
        return self._replicated_atoms

    @replicated_atoms.setter
    def replicated_atoms(self, new_replicated_atoms):
        if self.is_persistency_enabled:
            folder = self.folder_name
            folder += '/'
            ase.io.write (folder + REPLICATED_ATOMS_FILE, new_replicated_atoms, format='extxyz')
        self._replicated_atoms = new_replicated_atoms

    @property
    def list_of_index(self):
        return self._list_of_index

    @list_of_index.getter
    def list_of_index(self):
        if self._list_of_index is None:
            if self.is_persistency_enabled:
                try:
                    folder = self.folder_name
                    folder += '/'
                    self._list_of_index = np.load (folder + LIST_OF_INDEX_FILE)
                except FileNotFoundError as e:
                    print (e)
            if self._list_of_index is None:
                self.list_of_index = atom_helper.create_list_of_index (
                    self.atoms,
                    self.supercell)
                self.list_of_index = self.list_of_index.dot (self.atoms.cell)
        return self._list_of_index

    @list_of_index.setter
    def list_of_index(self, new_list_of_index):
        if self.is_persistency_enabled:
            folder = self.folder_name
            folder += '/'
            np.save (folder + LIST_OF_INDEX_FILE, new_list_of_index)
        self._list_of_index = new_list_of_index

    def optimize(self, method):
        # Mimimization of the sructure
        print('Initial max force: ', self.max_force(self.atoms.positions, self.atoms))
        if not ((method == 'none') or (method == False)):
            print ('Optimization method ' + method)
            result = minimize (self.max_force, self.atoms.positions, args=self.atoms, jac=self.gradient, method=method)
            print (result.message)
            self.atoms.positions = result.x.reshape((int(result.x.size / 3), 3))
            io.write ('mInimized_' + str(self) + '.xyz', self.atoms, format='extxyz')
        return self.atoms, self.max_force(self.atoms.positions)

    def max_force(self, x, atoms):
        grad = self.gradient (x, atoms)
        return np.linalg.norm (grad, 2)

    def gradient(self, x, input_atoms):
        atoms = input_atoms.copy ()
        atoms.positions = np.reshape (x, (int (x.size / 3.), 3))

        calc = self.calculator(lmpcmds=self.calculator_inputs, log_file='log_lammps.out')
        atoms.set_calculator (calc)
        gr = -1. * atoms.get_forces ()
        grad = np.reshape (gr, gr.size)
        return grad
    
        
    def calculate_second(self):
        '''
        Calculate the second order derivative of the force using finite differences
        :return:
        tensor with second derivative in eV/A^2
        '''
        supercell = self.supercell
        atoms = self.atoms
        Logger().info ('Calculating second order potential derivatives')
        n_in_unit_cell = len (atoms.numbers)
        replicated_atoms, _ = replicate_atoms(atoms, supercell)
    
        atoms = replicated_atoms
        n_atoms = len (atoms.numbers)
        dx = 1e-5
        second = np.zeros ((n_atoms * 3, n_atoms * 3))
        for alpha in range (3):
            for i in range (n_atoms):
                for move in (-1, 1):
                    shift = np.zeros ((n_atoms, 3))
                    shift[i, alpha] += move * dx
                    second[i * 3 + alpha, :] += move * self.gradient (atoms.positions + shift, atoms)
                    
        n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
        second = second.reshape ((n_supercell, n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3))
        second = second / (2. * dx)
        second *= evoverdlpoly
        np.save(SECOND_ORDER_FILE, second)
        return second
    
    def calculate_third(self):
        atoms = self.atoms
        supercell = self.supercell
        replicated_atoms, _ = replicate_atoms(atoms, supercell)
        
        # TODO: Here we should create it sparse
        Logger().info ('Calculating third order')
        n_in_unit_cell = len (atoms.numbers)
        atoms = replicated_atoms
        n_atoms = len (atoms.numbers)
        n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
        dx = 1e-6 * evoverdlpoly
        third = np.zeros ((n_in_unit_cell, 3, n_supercell * n_in_unit_cell, 3, n_supercell * n_in_unit_cell * 3))
        for alpha in range (3):
            for i in range (n_in_unit_cell):
                for beta in range (3):
                    for j in range (n_supercell * n_in_unit_cell):
                        for move_1 in (-1, 1):
                            for move_2 in (-1, 1):
                                shift = np.zeros ((n_atoms, 3))
                                shift[i, alpha] += move_1 * dx
                                
                                shift[j, beta] += move_2 * dx
                                third[i, alpha, j, beta, :] += move_1 * move_2 * (
                                    -1. * self.gradient (atoms.positions + shift, atoms))
        third = third.reshape ((1, n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3))
        third /= (4. * dx * dx)
        np.save(THIRD_ORDER_FILE, third)
        return third
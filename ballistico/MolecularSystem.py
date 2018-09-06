from ballistico.io_helper import save_eigensystem
from ballistico.constants import *
from scipy.optimize import minimize
import ballistico.atoms_helper as ath
import ballistico.constants as const
from ase.calculators.lammpslib import LAMMPSlib
import ase.io as io
import numpy as np
from sparse import tensordot,COO
from scipy.sparse import save_npz, load_npz
import os

SECOND_ORDER_FILE = 'second.npy'
THIRD_ORDER_FILE = 'third.npy'
THIRD_ORDER_SPARSE_FILE = 'third.npz'

class MolecularSystem (object):
    def __init__(self, configuration, replicas, temperature, lammps_cmd, optimize=False):
        self.replicas = replicas
        self.temperature = temperature
        self.configuration = configuration
        self.lammps_cmd = lammps_cmd
        self.folder = str (self) + '/'
        if not os.path.exists (self.folder):
            os.makedirs (self.folder)

        if optimize != 'none' and optimize!=False:
            if optimize == True:
                optimize = 'BFGS'
            self.optimize (optimize)

        self.replicated_configuration, self.list_of_replicas, self.list_of_indices= ath.replicate_configuration(configuration, replicas)
        
        self._second_order = None
        self._dynamical_matrix = None
        self._third_order = None
        
        

        self.index_first_cell = 0
        for replica_id in range (self.list_of_replicas.shape[0]):
            replica = self.list_of_replicas[replica_id]
            if (replica == np.zeros (3)).all ():
                self.index_first_cell = replica_id
    
    @classmethod
    def from_geometry(self, filename):
        data = open (filename).readlines ()
        return self (data)
    
    @property
    def second_order(self):
        return self._second_order
    
    @second_order.getter
    def second_order(self):
        if self._second_order is None:
            if self._dynamical_matrix is None:
                self._second_order = self.calculate_second ()
            else:
                self._second_order = self._dynamical_matrix.copy()
                size = self._second_order.shape[0] * self._second_order.shape[1]
                for i in range (size):
                    atom_i_index, replica_i_index = self.atom_and_replica_index (i)
                    mass_i = self.configuration.get_masses ()[atom_i_index]
                    for j in range (size):
                        atom_j_index, replica_j_index = self.atom_and_replica_index (j)
                        mass_j = self.configuration.get_masses ()[atom_j_index]
                        mass = np.sqrt (mass_i * mass_j)
                        self._second_order[replica_i_index, atom_i_index, :, replica_j_index, atom_j_index, :] *= mass
        return self._second_order
    
    @second_order.setter
    def second_order(self, new_second_order):
        self._second_order = new_second_order

    @property
    def dynamical_matrix(self):
        return self._dynamical_matrix

    @dynamical_matrix.getter
    def dynamical_matrix(self):
        if self._dynamical_matrix is None:
            if self._second_order is None:
                self._second_order = self.calculate_second ()
                self._dynamical_matrix = self.dynamical_matrix
            else:
                second = self._second_order
                self._dynamical_matrix = np.empty_like(second)
                size = second.shape[0] * second.shape[1]
                for i in range (size):
                    atom_i_index, replica_i_index = self.atom_and_replica_index (i)
                    mass_i = self.configuration.get_masses ()[atom_i_index]
                    for j in range (size):
                        atom_j_index, replica_j_index = self.atom_and_replica_index (j)
                        mass_j = self.configuration.get_masses ()[atom_j_index]
                        mass = np.sqrt (mass_i * mass_j)
                        self._dynamical_matrix[replica_i_index, atom_i_index, :, replica_j_index, atom_j_index,:] = second[replica_i_index, atom_i_index, :, replica_j_index, atom_j_index, :] / mass
        return self._dynamical_matrix

    @dynamical_matrix.setter
    def dynamical_matrix(self, new_dynamical_matrix):
        self._dynamical_matrix = new_dynamical_matrix

    @property
    def third_order(self):
        return self._third_order
    
    @third_order.getter
    def third_order(self):
        if self._third_order is None:
            self._third_order = self.calculate_third()
        return self._third_order
    
    @third_order.setter
    def third_order(self, new_third_order):
        self._third_order = new_third_order

    def max_force(self, x, atoms):
        grad = self.gradient (x, atoms)
        return np.linalg.norm (grad, 2)
    
    def gradient(self, x, input_atoms):
        '''
        Lammps wrapper to calculate the gradient
        :param x:
        :param input_atoms:
        :return: gradient (negative of the force) in  lammps metal units (eV/A)
        '''
        atoms = input_atoms.copy ()
        atoms.positions = np.reshape (x, (int (x.size / 3.), 3))
        calc = LAMMPSlib (lmpcmds=self.lammps_cmd, log_file=self.folder + 'log_lammps.out')
        atoms.set_calculator (calc)
        gr = -1. * atoms.get_forces ()
        grad = np.reshape (gr, gr.size)
        return grad
    
    def atom_and_replica_index(self, absolute_index):
        n_atoms = self.configuration.positions.shape[0]
        id_replica = absolute_index / n_atoms
        id_atom = absolute_index % n_atoms
        return int(id_atom), int(id_replica)
    
    def absolute_index(self, atom_index, replica_index):
        n_atoms = self.configuration.positions.shape[0]
        return int(replica_index * n_atoms + atom_index)

    def optimize(self, method):
        # Mimimization of the sructure
        print('Initial max force: ', self.max_force(self.configuration.positions, self.configuration))
        if not ((method == 'none') or (method == False)):
            print ('Optimization method ' + method)
            result = minimize (self.max_force, self.configuration.positions, args=self.configuration, jac=self.gradient, method=method)
            print (result.message)
            self.configuration.positions = result.x.reshape((int(result.x.size / 3), 3))
            io.write (self.folder + 'mInimized_' + str(self) + '.xyz', self.configuration, format='extxyz')
        print ('Final max force: ', self.max_force(self.configuration.positions, self.configuration))
        self.replicated_configuration, self.list_of_replicas, self.list_of_indices= ath.replicate_configuration(self.configuration, self.replicas)

    
    def calculate_second(self):
        '''
        Calculate the second order derivative of the force using finite differences
        :return:
        tensor with second derivative in eV/A^2
        '''
        try:
            second = np.load(self.folder + SECOND_ORDER_FILE)
        except IOError as err:
            print(err)
            print('Calculating second order potential derivatives')
            n_in_unit_cell = len (self.configuration.numbers)
            atoms = self.replicated_configuration
            n_atoms = len (atoms.numbers)
            dx = 1e-5
            second = np.zeros ((n_atoms * 3, n_atoms * 3))
            for alpha in range (3):
                for i in range (n_atoms):
                    for move in (-1, 1):
                        shift = np.zeros ((n_atoms, 3))
                        shift[i, alpha] += move * dx
                        second[i * 3 + alpha, :] += move * self.gradient (atoms.positions + shift, atoms)
            n_replicas = self.list_of_replicas.shape[0]
            second = second.reshape ((n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3))
            second = second / (2. * dx)
            second *= evoverdlpoly
            np.save(self.folder + SECOND_ORDER_FILE, second)
        return second
    
    def calculate_third(self):
        try:
            third = np.load(self.folder + THIRD_ORDER_FILE)
        except IOError as err:
            print(err)
            # TODO: Here we should create it sparse
            print('Calculating third order')
            n_in_unit_cell = len (self.configuration.numbers)
            atoms = self.replicated_configuration
            n_atoms = len (atoms.numbers)
            n_replicas = self.list_of_replicas.shape[0]
            dx = 1e-6 * evoverdlpoly
            third = np.zeros ((n_in_unit_cell, 3, n_replicas * n_in_unit_cell, 3, n_replicas * n_in_unit_cell * 3))
            for alpha in range (3):
                for i in range (n_in_unit_cell):
                    for beta in range (3):
                        for j in range (n_replicas * n_in_unit_cell):
                            for move_1 in (-1, 1):
                                for move_2 in (-1, 1):
                                    shift = np.zeros ((n_atoms, 3))
                                    shift[self.index_first_cell * n_in_unit_cell + i, alpha] += move_1 * dx
                                    
                                    shift[j, beta] += move_2 * dx
                                    third[i, alpha, j, beta, :] += move_1 * move_2 * (
                                        -1. * self.gradient (atoms.positions + shift, atoms))
            third = third.reshape ((1, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3))
            third /= (4. * dx * dx)
            np.save(self.folder + THIRD_ORDER_FILE, third)
        return third

    def __str__(self):
        atoms = self.configuration
        string = ''
        unique_elements = np.unique (atoms.get_chemical_symbols ())
        for element in unique_elements:
            string += element
        volume = np.linalg.det (atoms.cell) / 1000.
        string += '_a' + str (int (volume * 1000.))
        string += '_r' + str (self.replicas).replace (" ", "").replace ('[', "").replace (']', "")
        string += '_T' + str (int (self.temperature))
        return string

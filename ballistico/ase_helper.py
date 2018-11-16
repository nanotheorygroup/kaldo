import numpy as np
from ballistico.logger import Logger
from ballistico.constants import *
from ballistico.atoms_helper import replicate_atoms
from scipy.optimize import minimize
from ase.calculators.lammpslib import LAMMPSlib
import ase.io as io
from ase.calculators.tip3p import TIP3P


SECOND_ORDER_FILE = 'second.npy'
THIRD_ORDER_FILE = 'third.npy'

# TODO: this should be an imput somehow
LAMMPS_CMD = ["pair_style tersoff", "pair_coeff * * forcefields/Si.tersoff Si"]
CALCULATOR = LAMMPSlib (lmpcmds=LAMMPS_CMD, log_file='log_lammps.out')
# CALCULATOR = TIP3P(rc=7.)


class Displacement (object):
    def __init__(self, atoms, supercell):
        self.atoms = atoms
        self.supercell = supercell
        self.lammps_cmd = LAMMPS_CMD

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
        calc = LAMMPSlib (lmpcmds=self.lammps_cmd, log_file='log_lammps.out')
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
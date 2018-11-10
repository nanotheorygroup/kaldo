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
# LAMMPS_CMD = ["pair_style tersoff", "pair_coeff * * forcefields/Si.tersoff Si"]
# CALCULATOR = LAMMPSlib (lmpcmds=LAMMPS_CMD, log_file='log_lammps.out')
CALCULATOR = TIP3P(rc=7.)


def max_force(x, atoms):
    grad = gradient (x, atoms)
    return np.linalg.norm (grad, 2)

def gradient(x, input_atoms):
    '''
    Lammps wrapper to calculate the gradient
    :param x:
    :param input_atoms:
    :return: gradient (negative of the force) in  lammps metal units (eV/A)
    '''
    atoms = input_atoms.copy ()
    atoms.positions = np.reshape (x, (int (x.size / 3.), 3))
    atoms.set_calculator (CALCULATOR)
    gr = -1. * atoms.get_forces ()
    grad = np.reshape (gr, gr.size)
    return grad


def optimize(atoms, method='BFGS'):
    # Mimimization of the sructure
    Logger().info ('Initial max force: ', max_force(atoms.positions, atoms))
    if not ((method == 'none') or (method == False)):
        Logger().info ('Optimization method ' + method)
        result = minimize (max_force, atoms.positions, args=atoms, jac=gradient, method=method)
        Logger().info (result.message)
        atoms.positions = result.x.reshape((int(result.x.size / 3), 3))
        io.write ('minimized_' + '.xyz', atoms, format='extxyz')
    Logger().info ('Final max force: ', max_force(atoms.positions, atoms))
    return atoms


def calculate_second(atoms, replicas):
    '''
    Calculate the second order derivative of the force using finite differences
    :return:
    tensor with second derivative in eV/A^2
    '''
    Logger().info ('Calculating second order potential derivatives')
    n_in_unit_cell = len (atoms.numbers)
    replicated_atoms, list_of_replicas = replicate_atoms(atoms, replicas)

    atoms = replicated_atoms
    n_atoms = len (atoms.numbers)
    dx = 1e-5
    second = np.zeros ((n_atoms * 3, n_atoms * 3))
    for alpha in range (3):
        for i in range (n_atoms):
            for move in (-1, 1):
                shift = np.zeros ((n_atoms, 3))
                shift[i, alpha] += move * dx
                second[i * 3 + alpha, :] += move * gradient (atoms.positions + shift, atoms)
    n_replicas = list_of_replicas.shape[0]
    second = second.reshape ((n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3))
    second = second / (2. * dx)
    second *= evoverdlpoly
    np.save(SECOND_ORDER_FILE, second)
    return second

def calculate_third(atoms, replicas):
    replicated_atoms, list_of_replicas = replicate_atoms(atoms, replicas)

            
    # TODO: Here we should create it sparse
    Logger().info ('Calculating third order')
    n_in_unit_cell = len (atoms.numbers)
    atoms = replicated_atoms
    n_atoms = len (atoms.numbers)
    n_replicas = list_of_replicas.shape[0]
    dx = 1e-6 * evoverdlpoly
    third = np.zeros ((n_in_unit_cell, 3, n_replicas * n_in_unit_cell, 3, n_replicas * n_in_unit_cell * 3))
    for alpha in range (3):
        for i in range (n_in_unit_cell):
            for beta in range (3):
                for j in range (n_replicas * n_in_unit_cell):
                    for move_1 in (-1, 1):
                        for move_2 in (-1, 1):
                            shift = np.zeros ((n_atoms, 3))
                            shift[i, alpha] += move_1 * dx
                            
                            shift[j, beta] += move_2 * dx
                            third[i, alpha, j, beta, :] += move_1 * move_2 * (
                                -1. * gradient (atoms.positions + shift, atoms))
    third = third.reshape ((1, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3))
    third /= (4. * dx * dx)
    np.save(THIRD_ORDER_FILE, third)
    return third
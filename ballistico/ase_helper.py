import numpy as np

from ballistico.constants import *
from ballistico.atoms_helper import replicate_configuration
from scipy.optimize import minimize
from ase.calculators.lammpslib import LAMMPSlib
import ase.io as io

SECOND_ORDER_FILE = 'second.npy'
THIRD_ORDER_FILE = 'third.npy'

# TODO: this should be an imput somehow
LAMMPS_CMD = ["pair_style tersoff", "pair_coeff * * forcefields/Si.tersoff Si"]


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
    calc = LAMMPSlib (lmpcmds=LAMMPS_CMD, log_file='log_lammps.out')
    atoms.set_calculator (calc)
    gr = -1. * atoms.get_forces ()
    grad = np.reshape (gr, gr.size)
    return grad


def optimize(configuration, method='BFGS'):
    # Mimimization of the sructure
    print('Initial max force: ', max_force(configuration.positions, configuration))
    if not ((method == 'none') or (method == False)):
        print ('Optimization method ' + method)
        result = minimize (max_force, configuration.positions, args=configuration, jac=gradient, method=method)
        print (result.message)
        configuration.positions = result.x.reshape((int(result.x.size / 3), 3))
        io.write ('minimized_' + '.xyz', configuration, format='extxyz')
    print ('Final max force: ', max_force(configuration.positions, configuration))
    return configuration


def calculate_second(configuration, replicas):
    '''
    Calculate the second order derivative of the force using finite differences
    :return:
    tensor with second derivative in eV/A^2
    '''
    print('Calculating second order potential derivatives')
    n_in_unit_cell = len (configuration.numbers)
    replicated_configuration, list_of_replicas = replicate_configuration(configuration, replicas)

    atoms = replicated_configuration
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

def calculate_third(configuration, replicas):
    replicated_configuration, list_of_replicas = replicate_configuration(configuration, replicas)

            
    # TODO: Here we should create it sparse
    print('Calculating third order')
    n_in_unit_cell = len (configuration.numbers)
    atoms = replicated_configuration
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
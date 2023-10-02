"""
kaldo
Anharmonic Lattice Dynamics
"""
import numpy as np
from sparse import COO
import pandas as pd
import ase.units as units
from kaldo.helpers.tools import count_rows
from ase import Atoms
import re
from kaldo.helpers.logger import get_logger
logging = get_logger()


def import_from_files(replicated_atoms, dynmat_file=None, third_file=None, supercell=(1, 1, 1),
                      third_energy_threshold=0.):
    # TODO: split this method into two pieces
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

    second_order = None
    third_order = None

    if dynmat_file:
        logging.info('Reading dynamical matrix')
        second_dl = import_second(atoms, replicas=supercell, filename=dynmat_file)
        second_order = second_dl

    if third_file:
        try:
            logging.info('Reading sparse third order')
            third_dl = import_sparse_third(atoms=atoms,
                                              supercell=supercell,
                                              filename=third_file,
                                              third_energy_threshold=third_energy_threshold)

        except UnicodeDecodeError:
            if third_energy_threshold != 0:
                raise ValueError('Third threshold not supported for dense third')
            logging.info('Reading dense third order')
            third_dl = import_dense_third(atoms, supercell=supercell, filename=third_file)
            logging.info('Third order matrix stored.')
        third_dl = third_dl[:n_unit_atoms]
        third_shape = (
            n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3)
        third_dl = third_dl.reshape(third_shape)
        third_order = third_dl

    return second_order, third_order



def import_second(atoms, replicas=(1, 1, 1), filename='Dyn.form'):
    replicas = np.array(replicas)
    n_unit_cell = atoms.positions.shape[0]
    dyn_mat = import_dynamical_matrix(n_unit_cell, replicas, filename)
    mass = np.sqrt (atoms.get_masses ())
    dyn_mat = dyn_mat * mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    dyn_mat = dyn_mat * mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    return dyn_mat


def import_dynamical_matrix(n_atoms, supercell=(1, 1, 1), filename='Dyn.form'):
    supercell = np.array(supercell)
    dynamical_matrix_frame = pd.read_csv(filename, header=None, delim_whitespace=True)
    dynamical_matrix = dynamical_matrix_frame.values
    n_replicas = np.prod(supercell)
    if dynamical_matrix.size == n_replicas * (n_atoms * 3) ** 2:
        dynamical_matrix = dynamical_matrix.reshape((n_atoms, 3, n_replicas, n_atoms, 3))
    elif dynamical_matrix.size == (n_replicas * n_atoms * 3) ** 2:
        dynamical_matrix = dynamical_matrix.reshape((n_replicas, n_atoms, 3, n_replicas, n_atoms, 3))[0]
    elif dynamical_matrix.size == (n_atoms * 3) ** 2:
        dynamical_matrix = dynamical_matrix.reshape((n_atoms, 3, 1, n_atoms, 3))
    else:
        logging.error('Impossible to read calculate_dynmat with size ' + str(dynamical_matrix.size))
    tenjovermoltoev = 10 * units.J / units.mol
    return dynamical_matrix * tenjovermoltoev


def import_sparse_third(atoms, supercell=(1, 1, 1), filename='THIRD', third_energy_threshold=0.):
    supercell = np.array(supercell)
    n_replicas = np.prod(supercell)
    n_atoms = atoms.get_positions().shape[0]
    n_replicated_atoms = n_atoms * n_replicas
    n_rows = count_rows(filename)
    array_size = min(n_rows * 3, n_atoms * 3 * (n_replicated_atoms * 3) ** 2)
    coords = np.zeros((array_size, 6), dtype=int)
    values = np.zeros((array_size))
    index_in_unit_cell = 0
    tenjovermoltoev = 10 * units.J / units.mol
    with open(filename) as f:
        for i, line in enumerate(f):
            l_split = re.split('\s+', line.strip())
            coords_to_write = np.array(l_split[0:-3], dtype=int) - 1
            values_to_write = np.array(l_split[-3:], dtype=float)
            #TODO: add 'if' third_energy_threshold before calculating the mask
            mask_to_write = np.abs(values_to_write) > third_energy_threshold

            if mask_to_write.any() and coords_to_write[0] < n_atoms:
                for alpha in np.arange(3)[mask_to_write]:
                    coords[index_in_unit_cell, :-1] = coords_to_write[np.newaxis, :]
                    coords[index_in_unit_cell, -1] = alpha
                    values[index_in_unit_cell] = values_to_write[alpha] * tenjovermoltoev
                    index_in_unit_cell = index_in_unit_cell + 1
            if i % 1000000 == 0:
                logging.info('reading third order: ' + str(np.round(i / n_rows, 2) * 100) + '%')
    logging.info('read ' + str(3 * i) + ' interactions')
    coords = coords[:index_in_unit_cell].T
    values = values[:index_in_unit_cell]
    sparse_third = COO (coords, values, shape=(n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3))
    return sparse_third


def import_dense_third(atoms, supercell, filename, is_reduced=True):
    supercell = np.array(supercell)
    n_replicas = np.prod(supercell)
    n_atoms = atoms.get_positions().shape[0]
    if is_reduced:
        total_rows = (n_atoms *  3) * (n_atoms * n_replicas * 3) ** 2
        third = np.fromfile(filename, dtype=float, count=total_rows)
        third = third.reshape((n_atoms, 3, n_atoms * n_replicas, 3, n_atoms * n_replicas, 3))
    else:
        total_rows = (n_atoms * n_replicas * 3) ** 3
        third = np.fromfile(filename, dtype=float, count=total_rows)
        third = third.reshape((n_atoms * n_replicas, 3, n_atoms * n_replicas, 3, n_atoms * n_replicas, 3))
    return third

"""
Ballistico
Anharmonic Lattice Dynamics
"""
import numpy as np
from sparse import COO
import pandas as pd
import ase.units as units
from ballistico.helpers.tools import count_rows, apply_boundary_with_cell
from ase import Atoms
import re
from ballistico.helpers.logger import get_logger
logging = get_logger()

tenjovermoltoev = 10 * units.J / units.mol


def import_second(atoms, replicas=(1, 1, 1), filename='Dyn.form'):
    replicas = np.array(replicas)
    n_unit_cell = atoms.positions.shape[0]
    dyn_mat = import_dynamical_matrix(n_unit_cell, replicas, filename)
    mass = np.sqrt (atoms.get_masses ())
    mass = mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    return dyn_mat * mass


def import_dynamical_matrix(n_atoms, supercell=(1, 1, 1), filename='Dyn.form'):
    supercell = np.array(supercell)
    dynamical_matrix_frame = pd.read_csv(filename, header=None, delim_whitespace=True)
    dynamical_matrix = dynamical_matrix_frame.values
    n_replicas = np.prod(supercell)
    if dynamical_matrix.size == n_replicas * (n_atoms * 3) ** 2:
        dynamical_matrix = dynamical_matrix.reshape((1, n_atoms, 3, n_replicas, n_atoms, 3))
    else:
        dynamical_matrix = dynamical_matrix.reshape((n_replicas, n_atoms, 3, n_replicas, n_atoms, 3))
    return dynamical_matrix * tenjovermoltoev


def import_sparse_third(atoms, replicated_atoms=None, supercell=(1, 1, 1), filename='THIRD', third_energy_threshold=0., distance_threshold=None):
    supercell = np.array(supercell)
    n_replicas = np.prod(supercell)
    n_atoms = atoms.get_positions().shape[0]
    n_replicated_atoms = n_atoms * n_replicas
    n_rows = count_rows(filename)
    array_size = min(n_rows * 3, n_atoms * 3 * (n_replicated_atoms * 3) ** 2)
    coords = np.zeros((array_size, 6), dtype=np.int16)
    values = np.zeros((array_size))
    index_in_unit_cell = 0

    # Create list of index
    replicated_cell = replicated_atoms.cell
    replicated_cell_inv = np.linalg.inv(replicated_cell)
    replicated_atoms_positions = replicated_atoms.positions.reshape(
        (n_replicas, n_atoms, 3)) - atoms.positions[np.newaxis, :, :]
    replicated_atoms_positions = apply_boundary_with_cell(replicated_atoms_positions, replicated_cell,
                                                          replicated_cell_inv)
    list_of_replicas = replicated_atoms_positions[:, 0, :]


    with open(filename) as f:
        for i, line in enumerate(f):
            l_split = re.split('\s+', line.strip())
            coords_to_write = np.array(l_split[0:-3], dtype=int) - 1
            values_to_write = np.array(l_split[-3:], dtype=np.float)
            #TODO: add 'if' third_energy_threshold before calculating the mask
            mask_to_write = np.abs(values_to_write) > third_energy_threshold
            if mask_to_write.any() and coords_to_write[0] < n_atoms:
                iat = coords_to_write[0]
                jat = coords_to_write[2]
                is_storing = False
                if (distance_threshold is None):
                    is_storing = True
                else:

                    l, jsmall = np.unravel_index(jat, (n_replicas, n_atoms))
                    dxij = atoms.positions[iat] - (list_of_replicas[l] + atoms.positions[jsmall])
                    is_interacting = (np.linalg.norm(dxij) <= distance_threshold)
                    if is_interacting:
                        is_storing = True
                if is_storing:
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
        third = np.fromfile(filename, dtype=np.float, count=total_rows)
        third = third.reshape((n_atoms, 3, n_atoms * n_replicas, 3, n_atoms * n_replicas, 3))
    else:
        total_rows = (n_atoms * n_replicas * 3) ** 3
        third = np.fromfile(filename, dtype=np.float, count=total_rows)
        third = third.reshape((n_atoms * n_replicas, 3, n_atoms * n_replicas, 3, n_atoms * n_replicas, 3))
    return third


def import_from_files(atoms, dynmat_file=None, third_file=None, folder=None, supercell=(1, 1, 1), third_energy_threshold=0., distance_threshold=None):
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
    finite_difference = {'atoms' : atoms,
                         'supercell' : supercell,
                         'folder' : folder}

    if dynmat_file:
        second_dl = import_second(atoms, replicas=supercell, filename=dynmat_file)
        is_reduced_second = not (n_replicas ** 2 * (n_unit_atoms * 3) ** 2 == second_dl.size)
        if is_reduced_second:
            second_shape = (1, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3)
        else:
            second_shape = (n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3)

        logging.info('Is reduced second: ' + str(is_reduced_second))
        second_dl = second_dl.reshape(second_shape)
        finite_difference['second_order'] = second_dl
        finite_difference['is_reduced_second'] = is_reduced_second

    if third_file:
        try:
            logging.info('Reading sparse third')
            third_dl = import_sparse_third(atoms=atoms,
                                           supercell=supercell,
                                           filename=third_file,
                                           third_energy_threshold=third_energy_threshold,
                                           distance_threshold=distance_threshold)
        except UnicodeDecodeError:
            if third_energy_threshold != 0:
                raise ValueError('Third threshold not supported for dense third')
            logging.info('Trying reading binary third')
            third_dl = import_dense_third(atoms, supercell=supercell, filename=third_file)
        third_dl = third_dl[:n_unit_atoms]
        third_shape = (
            n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3)
        third_dl = third_dl.reshape(third_shape)
        finite_difference['third_order'] = third_dl

    return finite_difference


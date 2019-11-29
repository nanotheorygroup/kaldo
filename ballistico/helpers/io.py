"""
Ballistico
Anharmonic Lattice Dynamics
"""
import numpy as np
from sparse import COO
import pandas as pd
import ase.units as units
import re
from ballistico.helpers.tools import count_rows, split_index
from ase import Atoms
import os
import re

tenjovermoltoev = 10 * units.J / units.mol


def import_second(atoms, replicas=(1, 1, 1), filename='dlpoly_files/Dyn.form'):
    replicas = np.array(replicas)
    n_unit_cell = atoms.positions.shape[0]
    dyn_mat = import_dynamical_matrix(n_unit_cell, replicas, filename)
    mass = np.sqrt (atoms.get_masses ())
    mass = mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    return dyn_mat * mass


def import_dynamical_matrix(n_particles, replicas=(1, 1, 1), filename='dlpoly_files/Dyn.form'):
    replicas = np.array(replicas)
    dynamical_matrix_frame = pd.read_csv(filename, header=None, delim_whitespace=True)
    dynamical_matrix = dynamical_matrix_frame.values
    n_replicas = replicas[0] * replicas[1] * replicas[2]
    if dynamical_matrix.size == n_replicas * (n_particles * 3) ** 2:
        dynamical_matrix = dynamical_matrix.reshape(1, n_particles, 3, n_replicas, n_particles, 3)
    else:
        dynamical_matrix = dynamical_matrix.reshape(n_replicas, n_particles, 3, n_replicas, n_particles, 3)
    return dynamical_matrix * tenjovermoltoev


def import_sparse_third(atoms, replicas=(1, 1, 1), filename='THIRD'):
    replicas = np.array(replicas)
    n_replicas = np.prod(replicas)
    n_unit_cell = atoms.get_positions().shape[0]
    n_particles = n_unit_cell * n_replicas
    n_rows = count_rows(filename)
    array_size = min(n_rows * 3, n_unit_cell * 3 * (n_particles * 3) ** 2)
    coords = np.zeros((array_size, 6), dtype=np.int16)
    values = np.zeros((array_size))
    index_in_unit_cell = 0
    with open(filename) as f:
        for i, line in enumerate(f):
            l_split = re.split('\s+', line.strip())
            coords_to_write = np.array(l_split[0:-3], dtype=int) - 1
            if coords_to_write[0] < n_unit_cell:
                coords[3 * index_in_unit_cell:3 * (index_in_unit_cell + 1), :-1] = coords_to_write[np.newaxis, :]
                coords[3 * index_in_unit_cell:3 * (index_in_unit_cell + 1), -1] = [0, 1, 2]
                values[3 * index_in_unit_cell:3 * (index_in_unit_cell + 1)] = np.array(l_split[-3:], dtype=np.float) * tenjovermoltoev
                index_in_unit_cell = index_in_unit_cell + 1
            if i % 1000000 == 0:
                print('reading third order: ', np.round(i / n_rows, 2) * 100, '%')
    print('read', 3 * i, 'interactions')
    coords = coords[:3 * index_in_unit_cell].T
    values = values[:3 * index_in_unit_cell]
    sparse_third = COO (coords, values, shape=(n_unit_cell, 3, n_particles, 3, n_particles, 3))
    return sparse_third


def import_dense_third(atoms, replicas, filename, is_reduced=True):
    replicas = np.array(replicas)
    n_replicas = np.prod(replicas)
    n_particles = atoms.get_positions().shape[0]
    if is_reduced:
        total_rows = (n_particles *  3) * (n_particles * n_replicas * 3) ** 2
        third = np.fromfile(filename, dtype=np.float, count=total_rows)
        third = third.reshape((n_particles, 3, n_particles * n_replicas, 3, n_particles * n_replicas, 3))
    else:
        total_rows = (n_particles * n_replicas * 3) ** 3
        third = np.fromfile(filename, dtype=np.float, count=total_rows)
        third = third.reshape((n_particles * n_replicas, 3, n_particles * n_replicas, 3, n_particles * n_replicas, 3))
    return third


def import_from_files(atoms, dynmat_file=None, third_file=None, folder=None, supercell=(1, 1, 1)):
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

        print('Is reduced second: ', is_reduced_second)
        second_dl = second_dl.reshape(second_shape)
        finite_difference['second_order'] = second_dl
        finite_difference['is_reduced_second'] = is_reduced_second

    if third_file:
        try:
            print('Reading sparse third')
            third_dl = import_sparse_third(atoms, replicas=supercell, filename=third_file)
        except UnicodeDecodeError:
            print('Trying reading binary third')
            third_dl = import_dense_third(atoms, replicas=supercell, filename=third_file)
        third_dl = third_dl[:n_unit_atoms]
        third_shape = (
            n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3)
        third_dl = third_dl.reshape(third_shape)
        finite_difference['third_order'] = third_dl

    return finite_difference


def import_second_and_third_from_sheng(finite_difference):
    atoms = finite_difference.atoms
    supercell = finite_difference.supercell
    folder = finite_difference.folder
    second_file = folder + '/' + 'FORCE_CONSTANTS_2ND'
    if not os.path.isfile(second_file):
        second_file = folder + '/' + 'FORCE_CONSTANTS'
    n_replicas = np.prod(supercell)
    with open(second_file, 'r') as file:
        first_row = file.readline()
        first_row_split = re.findall(r'\d+', first_row)
        n_rows = int(list(map(int, first_row_split))[0])
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
    is_reduced_second = True
    third_order = np.zeros((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
    third_file = folder + '/' + 'FORCE_CONSTANTS_3RD'
    second_cell_list = []
    third_cell_list = []
    with open(third_file, 'r') as file:
        line = file.readline()
        n_third = int(line)
        for i in range(n_third):
            file.readline()
            file.readline()
            second_cell_position = np.fromstring(file.readline(), dtype=np.float, sep=' ')
            second_cell_index = second_cell_position.dot(np.linalg.inv(atoms.cell)).round(0).astype(int)
            second_cell_list.append(second_cell_index)

            # create mask to find the index
            second_cell_id = (finite_difference.list_of_index()[:] == second_cell_index).prod(axis=1)
            second_cell_id = np.argwhere(second_cell_id).flatten()

            third_cell_position = np.fromstring(file.readline(), dtype=np.float, sep=' ')
            third_cell_index = third_cell_position.dot(np.linalg.inv(atoms.cell)).round(0).astype(int)
            third_cell_list.append(third_cell_index)

            # create mask to find the index
            third_cell_id = (finite_difference.list_of_index()[:] == third_cell_index).prod(axis=1)
            third_cell_id = np.argwhere(third_cell_id).flatten()

            atom_i, atom_j, atom_k = np.fromstring(file.readline(), dtype=np.int, sep=' ') - 1
            for _ in range(27):
                values = np.fromstring(file.readline(), dtype=np.float, sep=' ')
                alpha, beta, gamma = values[:3].round(0).astype(int) - 1
                try:
                    third_order[atom_i, alpha, second_cell_id, atom_j, beta, third_cell_id, atom_k, gamma] = values[
                        3]
                except TypeError as err:
                    print(err)
                except IndexError as err:
                    print(err)
    second_order = second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3), order='C')
    third_order = third_order.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas *
                                       n_unit_atoms * 3), order='C')
    return second_order, is_reduced_second, third_order


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
            supercell = np.fromstring(value, dtype=np.int, sep=' ')
    # l factor is in nanometer
    cell = np.array(latt_vecs) * lfactor * 10
    positions = np.array(positions).dot(cell)
    list_of_elem = []
    for i in range(len(types)):
        list_of_elem.append(elements[types[i] - 1])

    atoms = Atoms(list_of_elem,
                  positions=positions,
                  cell=cell,
                  pbc=[1, 1, 1])

    print('Atoms object created.')
    return atoms, supercell

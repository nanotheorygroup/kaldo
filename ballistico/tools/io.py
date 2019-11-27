"""
Ballistico
Anharmonic Lattice Dynamics
"""
import numpy as np
from sparse import COO
import pandas as pd
import ase.units as units
from itertools import takewhile, repeat
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

def count_rows(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
    return sum(buf.count(b'\n') for buf in bufgen if buf)


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
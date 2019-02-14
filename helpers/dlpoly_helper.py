import numpy as np
from sparse import COO
import pandas as pd
import ase.units as units
tenjovermol = 10 * units.J / units.mol


def import_second_dlpoly(atoms, replicas=(1, 1, 1), filename='dlpoly_files/Dyn.form'):
    replicas = np.array(replicas)
    dyn_mat = import_dynamical_matrix_dlpoly(replicas, filename)
    mass = np.sqrt (atoms.get_masses ())
    mass = mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    return dyn_mat * mass


def import_dynamical_matrix_dlpoly(replicas=(1, 1, 1), filename='dlpoly_files/Dyn.form'):
    replicas = np.array(replicas)
    dynamical_matrix_frame = pd.read_csv(filename, header=None, delim_whitespace=True)
    dynamical_matrix_vector = dynamical_matrix_frame.values
    n_replicas = replicas[0] * replicas[1] * replicas[2]
    n_particles = int((dynamical_matrix_vector.size / (3. ** 2.)) ** (1. / 2.)/n_replicas)
    return dynamical_matrix_vector.reshape((n_replicas, n_particles, 3, n_replicas, n_particles, 3), order='C') \
           * tenjovermol


def import_third_order_dlpoly(atoms, replicas=(1, 1, 1), filename='dlpoly_files/THIRD'):
    replicas = np.array(replicas)
    n_replicas = np.prod(replicas)
    n_particles = atoms.get_positions().shape[0] * n_replicas
    third_order_frame = pd.read_csv (filename, header=None, delim_whitespace=True)
    third_order = third_order_frame.values.T
    v3ijk = third_order[5:8].T
    coords = np.vstack ((third_order[0:5] - 1, 0 * np.ones ((third_order.shape[1]))))
    sparse_x = COO (coords, v3ijk[:, 0], shape=(n_particles, 3, n_particles, 3, n_particles, 3))
    coords = np.vstack ((third_order[0:5] - 1, 1 * np.ones ((third_order.shape[1]))))
    sparse_y = COO (coords, v3ijk[:, 1], shape=(n_particles, 3, n_particles, 3, n_particles, 3))
    coords = np.vstack ((third_order[0:5] - 1, 2 * np.ones ((third_order.shape[1]))))
    sparse_z = COO (coords, v3ijk[:, 2], shape=(n_particles, 3, n_particles, 3, n_particles, 3))
    sparse = sparse_x + sparse_y + sparse_z
    n_particles_small = int(n_particles / n_replicas)
    sparse = sparse.reshape ((n_replicas * n_particles_small * 3, n_replicas * n_particles_small * \
                              3, n_replicas * n_particles_small * 3), order='C')
    return sparse * tenjovermol

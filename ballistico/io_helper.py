import subprocess
import numpy as np
from sparse import COO
import pandas as pd
import ballistico.atoms_helper as ath
from ballistico.constants import evoverdlpoly


def import_dynamical_matrix_charlie(dynamical_matrix_file, replicas):
    # dynamical_matrix_file = '/Users/giuseppe/Development/research-dev/charlie-lammps/dynmat/dynmat.dat'
    dynamical_matrix_frame = pd.read_csv(dynamical_matrix_file, header=None, skiprows=1, delim_whitespace=True)
    dynamical_matrix_vector = dynamical_matrix_frame.values
    n_replicas = replicas[0] * replicas[1] * replicas[2]
    n_particles = int((dynamical_matrix_vector.size / (3. ** 2.)) ** (1. / 2.)/n_replicas)
    return dynamical_matrix_vector.reshape(n_replicas, n_particles, 3, n_replicas, n_particles, 3)

def import_dynamical_matrix_dlpoly(dynamical_matrix_file, replicas):
    
    # dynamical_matrix_file = '/Users/giuseppe/Development/research-dev/PhononRelax/test-Si-54/Dyn.form'
    dynamical_matrix_frame = pd.read_csv(dynamical_matrix_file, header=None, delim_whitespace=True)
    dynamical_matrix_vector = dynamical_matrix_frame.values
    n_replicas = replicas[0] * replicas[1] * replicas[2]
    n_particles = int((dynamical_matrix_vector.size / (3. ** 2.)) ** (1. / 2.)/n_replicas)
    return dynamical_matrix_vector.reshape(n_replicas, n_particles, 3, n_replicas, n_particles, 3)


def import_third_order_dlpoly(file, configuration, replicas):
    replicated_configuration, list_of_replicas, list_of_indices = ath.replicate_configuration (
        configuration, replicas)
    n_particles = replicated_configuration.get_positions().shape[0]
    
            
    third_order_frame = pd.read_csv (file, header=None, delim_whitespace=True)
    third_order = third_order_frame.values.T
    v3ijk = third_order[5:8].T
    coords = np.vstack ((third_order[0:5] - 1, 0 * np.ones ((third_order.shape[1]))))
    sparse_x = COO (coords, v3ijk[:, 0], shape=(n_particles, 3, n_particles, 3, n_particles, 3))
    coords = np.vstack ((third_order[0:5] - 1, 1 * np.ones ((third_order.shape[1]))))
    sparse_y = COO (coords, v3ijk[:, 1], shape=(n_particles, 3, n_particles, 3, n_particles, 3))
    coords = np.vstack ((third_order[0:5] - 1, 2 * np.ones ((third_order.shape[1]))))
    sparse_z = COO (coords, v3ijk[:, 2], shape=(n_particles, 3, n_particles, 3, n_particles, 3))
    sparse = sparse_x + sparse_y + sparse_z
    n_replicas = np.prod(replicas)
    n_particles_small = int(n_particles / n_replicas)
    sparse = sparse.reshape ((n_replicas, n_particles_small, 3, n_replicas, n_particles_small, 3, n_replicas, n_particles_small, 3,))
    sparse = sparse.todense() / evoverdlpoly
    return sparse[0].reshape ((1, n_particles_small, 3, n_replicas, n_particles_small, 3, n_replicas, n_particles_small, 3))


def save_fourier_second_order(harmonic_system, second_order, k_list):
    n_k_points = k_list.shape[0]
    
    # WRITE ON FILE
    filename = 'EIGENVALUES_WITH_MOMENTA'
    cmd = ['rm', filename]
    subprocess.Popen (cmd, stdout=subprocess.PIPE).communicate ()[0]
    file = open ('%s' % filename, 'a+')
    for i_k in range (n_k_points):
        evals, evects = harmonic_system.diagonalize_second_order_k (second_order, k_list[i_k])
        file.write (str (i_k) + ' ' +
                    str (evals[0]) + ' ' +
                    str (evals[1]) + ' ' +
                    str (evals[2]) + ' ' +
                    str (evals[3]) + ' ' +
                    str (evals[4]) + ' ' +
                    str (evals[5]) + ' ' + '\n')
    
    file.close ()


def save_eigensystem(eigenvals, eigenvect):
    eigenvect = eigenvect.reshape ((int (eigenvect.size / 3), 3))
    subprocess.Popen (['rm', 'EIGENVALUES'], stdout=subprocess.PIPE).communicate ()
    subprocess.Popen (['rm', 'EIGVEC'], stdout=subprocess.PIPE).communicate ()
    file = open ('EIGENVALUES', 'a+')
    file2 = open ('EIGVEC', 'a+')
    for i in range (eigenvect.shape[0]):
        eigenvector = eigenvect[i]
        for value in eigenvector:
            file2.write (" \t " + str (value))
        file2.write (' \n')
    for i in range (eigenvals.shape[0]):
        eigenvalue = eigenvals[i]
        # TODO: here we are going to cm^(-1) probably better to make the conversion factor constants, or to use a library for units
        file.write (
            str (i + 1) + '  ' + str (np.real (eigenvalue * 33.356)) + '  ' + str (np.real (eigenvalue)) + '\n')
    file.close ()
    file2.close ()

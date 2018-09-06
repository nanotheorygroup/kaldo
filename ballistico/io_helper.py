import subprocess
import numpy as np

from ballistico.geometry_helper import normalize_geometry


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

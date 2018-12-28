import numpy as np
from ase.dft.kpoints import ibz_points, bandpath

k_space_file = 'QPLIST'


def create_k_and_symmetry_space(cell, symmetry='fcc', n_k_points=50):

    # TODO: implement symmetry here
    # import spglib as spg
    # spacegroup = spg.get_spacegroup (atoms, symprec=1e-5)

    # High-symmetry points in the Brillouin zone
    points = ibz_points[symmetry]
    G = points['Gamma']
    X = points['X']
    W = points['W']
    K = points['K']
    L = points['L']
    U = points['U']

    point_names = ['$\Gamma$', 'X', 'U', 'L', '$\Gamma$', 'K']
    path = [G, X, U, L, G, K]

    # Band structure in meV
    path_kc, q, Q = bandpath (path, cell, n_k_points)
    return path_kc, q, Q, point_names


def filter_modes_and_k_points(k_list, modes):
    k_list -= np.round (k_list)
    k_list *= 2.
    k_x = k_list[:, 0]
    k_y = k_list[:, 1]
    k_z = k_list[:, 2]
    # gamma to x
    first_filt = (k_x >= 0.) & ((k_y == 0.) & (k_z == k_x))
    return k_list[first_filt], modes[first_filt]


def filter_k_points(k_list):
    k_list -= np.round (k_list)
    k_x = k_list[:, 0]
    k_y = k_list[:, 1]
    k_z = k_list[:, 2]
    # gamma to x
    first_filt = (k_x >= 0.) & ((k_y == 0.) & (k_z == k_x)) & (k_x <= 0.5)
    second_filt = (k_x >= 0.5) & ((k_y > 0.) & (k_z == k_x))
    return k_list[first_filt]

import numpy as np
from ballistico.logger import Logger
import seekpath
from ase.dft.kpoints import ibz_points, bandpath

k_space_file = 'QPLIST'


def normalize_geometry(geometry, sizes):
	inverted_matrix = np.linalg.inv (sizes)
	return np.dot (inverted_matrix, geometry.T).T


def create_k_space_with_path(structure, resolution, time_refersal=True):
	numbers = structure.get_atomic_numbers ()
	inp = (structure.cell, structure.get_scaled_positions (), numbers)
	explicit_data = seekpath.get_explicit_k_path (inp, reference_distance=resolution, with_time_reversal=time_refersal)
	
	kpath = explicit_data['explicit_kpoints_rel']
	new_data = seekpath.get_path(inp)
	Logger().info ('Spacegroup: {} ({})'.format(new_data['spacegroup_international'], new_data['spacegroup_number']))
	Logger().info ('Inversion symmetry?: {}'.format(new_data['has_inversion_symmetry']))
	
	explicit_segments = np.array (explicit_data['explicit_segments'])[:, 0]
	to_append = np.array([np.array (explicit_data['explicit_segments'])[:, 1][-1]])
	explicit_segments = np.concatenate([explicit_segments, to_append])
	
	point_names = np.array (explicit_data['path'], dtype='S10')[:, 0]
	to_append = np.array([np.array (explicit_data['path'], dtype='S10')[:, 1][-1]])
	point_names = np.concatenate([point_names, to_append])
	
	for i in range (point_names.size):
		point_names[i] = point_names[i].replace('GAMMA',"$\Gamma$")

	dx = explicit_segments.max() / (1. * kpath.shape[0])
	return kpath, np.arange(0., explicit_segments.max(), dx), 1. * explicit_segments, point_names


def create_k_and_symmetry_space(structure, symmetry='fcc', n_k_points=50):
	
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
	path_kc, q, Q = bandpath (path, structure.cell, n_k_points)
	return path_kc, q, Q, point_names

def save_k_space(k_list):
	np.savetxt (k_space_file, k_list, header=str (k_list.shape[0]), comments='')


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

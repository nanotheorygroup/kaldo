import ballistico.geometry_helper as geh
import ase.io as io
from ase import Atoms
from ase.build import bulk
import numpy as np


def replicate_configuration(atoms, replicas):
	replicas = np.array(replicas)
	list_of_replicas, list_of_indices = create_list_of_replicas (atoms, replicas)
	replicated_symbols = []
	n_replicas = list_of_replicas.shape[0]
	n_unit_atoms = len(atoms.numbers)
	replicated_geometry = np.zeros((n_replicas, n_unit_atoms, 3))

	for i in range(n_replicas):
		vector = list_of_replicas[i]
		replicated_symbols.extend(atoms.get_chemical_symbols ())
		replicated_geometry[i, :,:] = atoms.positions + vector
	replicated_geometry = replicated_geometry.reshape((n_replicas * n_unit_atoms, 3))
	replicated_cell = atoms.cell * replicas
	replicated_configuration = Atoms(positions=replicated_geometry, symbols=replicated_symbols, cell=replicated_cell, pbc=[1, 1, 1])
	
	# Apply pbc to all elements in new positions
	# for i in range(replicated_configuration.positions.shape[0]):
		# replicated_configuration.positions[i] = apply_boundary(replicated_configuration, replicated_configuration.positions[i])
	# for i in range(list_of_replicas.shape[0]):
		# list_of_replicas[i] = apply_boundary(replicated_configuration, list_of_replicas[i])
	
	return replicated_configuration, list_of_replicas, list_of_indices

def create_list_of_replicas(atoms, replicas):
	# TODO: replicas[i] needs to be odd, throw an exception otherwise
	n_replicas = replicas[0] * replicas[1] * replicas[2]
	replica_id = 0
	list_of_replicas = np.zeros ((n_replicas, 3))
	list_of_indices = np.zeros ((n_replicas, 3))
	
	range_0 = np.linspace(-int(replicas[0]/2),int(replicas[0]/2),int(replicas[0]))
	# range_0[range_0 > replicas[0] / 2] = range_0[range_0 > replicas[0] / 2] - replicas[0]
	
	range_1 = np.linspace(-int(replicas[1]/2),int(replicas[1]/2),int(replicas[1]))
	# range_1[range_1 > replicas[1] / 2] = range_1[range_1 > replicas[1] / 2] - replicas[1]
	
	range_2 = np.linspace(-int(replicas[2]/2),int(replicas[2]/2),int(replicas[2]))
	# range_2[range_2 > replicas[2] / 2] = range_2[range_2 > replicas[2] / 2] - replicas[2]
	
	for lx in range_0:
		for ly in range_1:
			for lz in range_2:
				index = np.array ([lx, ly, lz])#%replicas
				list_of_replicas[replica_id] = index.dot(atoms.cell)
				list_of_indices[replica_id] = index
				replica_id += 1
	return list_of_replicas, list_of_indices

def apply_boundary(atoms, dxij):
	cellinv = np.linalg.inv (atoms.cell)
	# exploit periodicity to calculate the shortest distance, which may not be the one we have
	sxij = cellinv.dot (dxij)
	sxij = sxij - np.round (sxij)
	dxij = atoms.cell.dot (sxij)
	return dxij

def type_element_id(atoms, element_name):
	unique_elements = np.unique (atoms.get_chemical_symbols ())
	for i in range(len(unique_elements)):
		element = unique_elements[i]
		if element == element_name:
			return i

def export_xyz(atoms, filename):
	save_file(atoms, filename, 'xyz')
	
def export_config(atoms, filename):
	save_file(atoms, filename, 'config')

def save_file(atoms, filename, type='xyz'):
	save_header (atoms, filename, type)
	save_coordinates (atoms, filename, type)

def save_header(atoms, filename, type):
	positions = atoms.positions
	fullsize = atoms.cell
	separator = '\t'
	if type == 'config':
		separator = '\n'
	xyz_file = open ('%s' % filename, 'w')
	if type == 'config':
		# 0, only coordinates in config file (no velocities, forces)
		# 1, cubic, 2 orthorombic, 3 parallelepiped
		xyz_file.write ('CONFIG FILE %i\n 0   3 \n ' % positions.shape[0])
	else:
		xyz_file.write ('%i \n' % positions.shape[0])
	for i in range (fullsize.shape[0]):
		if i == (fullsize.shape[0] - 1):
			separator = '\n'
		xyz_file.write ('%10.6f    %10.6f    %10.6f' % (fullsize[i][0], fullsize[i][1], fullsize[i][2]))
		xyz_file.write (separator)
	xyz_file.close ()

def save_coordinates(atoms, filename, type):
	positions = atoms.positions
	xyz_file = open (filename, 'a+')
	for i in range (positions.shape[0]):
		element_name = atoms.get_chemical_symbols ()[i]
		if type == 'xyz':
			xyz_file.write (
				element_name + '\t' + '' + str (positions[i][0]) + '\t' + str (
					positions[i][1]) + '\t' + str (
					positions[i][2]) + '\n')
		else:
			xyz_file.write (
				element_name + '\n')
			xyz_file.write ('%10.6f    %10.6f    %10.6f' % (positions[i][0], positions[i][1], positions[i][2]))
			xyz_file.write ('\n')
	xyz_file.close ()


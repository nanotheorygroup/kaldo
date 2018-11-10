import numpy as np
from ase import Atoms


def replicate_atoms(atoms, replicas):
	"""

	:rtype: Atoms,
	"""
	replicas = np.array(replicas)
	list_of_replicas = create_list_of_replicas (atoms, replicas)
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
	replicated_atoms = Atoms(positions=replicated_geometry - np.min(list_of_replicas, axis=0), symbols=replicated_symbols, cell=replicated_cell, pbc=[1, 1, 1])
	
	return replicated_atoms, list_of_replicas

def create_list_of_replicas(atoms, replicas):
	# TODO: supercell[i] needs to be odd, throw an exception otherwise
	n_replicas = replicas[0] * replicas[1] * replicas[2]
	replica_id = 0
	list_of_replicas = np.zeros ((n_replicas, 3))
	
	# range_0 = np.linspace(-int(supercell[0]/2),int(supercell[0]/2),int(supercell[0]))
	# range_0[range_0 > supercell[0] / 2] = range_0[range_0 > supercell[0] / 2] - supercell[0]
	#
	# range_1 = np.linspace(-int(supercell[1]/2),int(supercell[1]/2),int(supercell[1]))
	# range_1[range_1 > supercell[1] / 2] = range_1[range_1 > supercell[1] / 2] - supercell[1]
	#
	# range_2 = np.linspace(-int(supercell[2]/2),int(supercell[2]/2),int(supercell[2]))
	# range_2[range_2 > supercell[2] / 2] = range_2[range_2 > supercell[2] / 2] - supercell[2]

	range_0 = np.arange(int(replicas[0]))
	range_0[range_0 > replicas[0] / 2] = range_0[range_0 > replicas[0] / 2] - replicas[0]
	range_1 = np.arange(int(replicas[1]))
	range_1[range_1 > replicas[1] / 2] = range_1[range_1 > replicas[1] / 2] - replicas[1]
	range_2 = np.arange(int(replicas[2]))
	range_2[range_2 > replicas[2] / 2] = range_2[range_2 > replicas[2] / 2] - replicas[2]
	
	for lx in range_0:
		for ly in range_1:
			for lz in range_2:
				index = np.array ([lx, ly, lz])#%supercell
				list_of_replicas[replica_id] = index.dot(atoms.cell)
				replica_id += 1
	return list_of_replicas

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

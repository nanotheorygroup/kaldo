import numpy as np
from ase import Atoms

def convert_to_poscar(atoms, supercell=None):
	list_of_types = []
	for symbol in atoms.get_chemical_symbols ():
		for i in range (np.unique (atoms.get_chemical_symbols ()).shape[0]):
			if np.unique (atoms.get_chemical_symbols ())[i] == symbol:
				list_of_types.append (str (i))
	
	poscar = {'lattvec': atoms.cell / 10,
	               'positions': atoms.positions.T,
	               'elements': atoms.get_chemical_symbols (),
	               'types': list_of_types}
	if supercell is not None:
		poscar['na'] = supercell[0]
		poscar['nb'] = supercell[1]
		poscar['nc'] = supercell[2]
	return poscar

def replicate_atoms(atoms, replicas):
	replicas = np.array(replicas)
	list_of_replicas = create_list_of_index (atoms, replicas).dot(atoms.cell)
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

def create_list_of_index(atoms, replicas):
	# TODO: refactor removing atoms object
	n_replicas = replicas[0] * replicas[1] * replicas[2]
	replica_id = 0
	list_of_index = np.zeros ((n_replicas, 3))

	range_0 = np.arange(int(replicas[0]))
	range_0[range_0 > replicas[0] / 2] = range_0[range_0 > replicas[0] / 2] - replicas[0]
	range_1 = np.arange(int(replicas[1]))
	range_1[range_1 > replicas[1] / 2] = range_1[range_1 > replicas[1] / 2] - replicas[1]
	range_2 = np.arange(int(replicas[2]))
	range_2[range_2 > replicas[2] / 2] = range_2[range_2 > replicas[2] / 2] - replicas[2]
	
	for lx in range_0:
		for ly in range_1:
			for lz in range_2:
				index = np.array ([lx, ly, lz])
				list_of_index[replica_id] = index
				replica_id += 1
				
	return list_of_index

def apply_boundary(atoms, dxij):
	# exploit periodicity to calculate the shortest distance, which may not be the one we have
	cellinv = np.linalg.inv (atoms.cell)
	sxij = dxij.dot(cellinv)
	sxij = sxij - np.round (sxij)
	dxij = sxij.dot(atoms.cell)
	return dxij

def type_element_id(atoms, element_name):
	# TODO: remove this method
	unique_elements = np.unique (atoms.get_chemical_symbols ())
	for i in range(len(unique_elements)):
		element = unique_elements[i]
		if element == element_name:
			return i

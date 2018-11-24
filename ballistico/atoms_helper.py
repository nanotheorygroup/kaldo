import numpy as np
from ase import Atoms

def convert_to_poscar(atoms, supercell=None):
	list_of_types = []
	for symbol in atoms.get_chemical_symbols ():
		for i in range (np.unique (atoms.get_chemical_symbols ()).shape[0]):
			if np.unique (atoms.get_chemical_symbols ())[i] == symbol:
				list_of_types.append (str (i))
	
	poscar = {'lattvec': atoms.cell / 10,
	               'positions': (atoms.positions.dot(np.linalg.inv(atoms.cell))).T,
	               'elements': atoms.get_chemical_symbols (),
	               'types': list_of_types}
	if supercell is not None:
		poscar['na'] = supercell[0]
		poscar['nb'] = supercell[1]
		poscar['nc'] = supercell[2]
	return poscar

def convert_to_atoms_and_super_cell(poscar):
	cell = poscar['lattvec'] * 10
	atoms = Atoms(symbols=poscar['elements'],
	              positions=poscar['positions'].T.dot(cell),
	              cell=cell,
	              pbc=(1,1,1)
	              )
	supercell = np.ones(3)
	if poscar['na'] is not None:
		supercell[0] = poscar['na']
		supercell[1] = poscar['nb']
		supercell[2] = poscar['nc']
	return atoms, supercell



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

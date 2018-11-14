import numpy as np
import ballistico.atoms_helper as atom_helper
import ase.io
import os
import ballistico.constants as constants
from ase import Atoms
from ballistico.logger import Logger



REPLICATED_ATOMS_FILE = 'replicated_atoms.xyz'
LIST_OF_INDEX_FILE = 'list_of_index.npy'
FREQUENCIES_FILE = 'frequencies.npy'
EIGENVALUES_FILE = 'eigenvalues.npy'
EIGENVECTORS_FILE = 'eigenvectors.npy'
VELOCITIES_FILE = 'velocities.npy'
GAMMA_FILE = 'gamma.npy'
DOS_FILE = 'dos.npy'
OCCUPATIONS_FILE = 'occupations.npy'
K_POINTS_FILE = 'k_points.npy'
C_V_FILE = 'c_v.npy'


class Phonons (object):
	def __init__(self, atoms, folder_name, supercell = (1, 1, 1), kpts = (1, 1, 1), is_classic = False, temperature = 300, is_persistency_enabled = True, sigma_in=None):
		self.atoms = atoms
		self.supercell = np.array (supercell)
		self.kpts = np.array (kpts)
		self.is_classic = is_classic
		
		self.n_k_points = np.prod (self.kpts)
		self.n_modes = self.atoms.get_masses ().shape[0] * 3
		self.n_phonons = self.n_k_points * self.n_modes
		self.temperature = temperature
		self.is_persistency_enabled = is_persistency_enabled
		self._replicated_atoms = None
		self._list_of_index = None
		self._frequencies = None
		self._velocities = None
		self._eigenvalues = None
		self._eigenvectors = None
		self._dos = None
		self._occupations = None
		self._gamma = None
		self._n_k_points = None
		self._n_modes = None
		self._n_phonons = None
		self._k_points = None
		self.folder_name = folder_name
		self.sigma_in = sigma_in
		self._c_v = None
		if self.is_classic:
			classic_string = 'classic'
		else:
			classic_string = 'quantum'
		folder = self.folder_name + '/' + str (self.temperature) + '/' + classic_string + '/'
		if self.sigma_in is not None:
			folder += 'sigma_in_' + str (self.sigma_in).replace ('.', '_') + '/'
		folders = [self.folder_name, folder]
		if self.is_persistency_enabled:
			for folder in folders:
				if not os.path.exists (folder):
					os.makedirs (folder)
	
	@property
	def replicated_atoms(self):
		return self._replicated_atoms
	
	@replicated_atoms.getter
	def replicated_atoms(self):
		if self._replicated_atoms is None :
			if self.is_persistency_enabled:
				try:
					folder = self.folder_name
					folder += '/'
					self._replicated_atoms = ase.io.read (folder + REPLICATED_ATOMS_FILE, format='extxyz')
				except FileNotFoundError as e:
					print (e)
			if self._replicated_atoms is None:
				atoms = self.atoms
				supercell = self.supercell
				list_of_index = self.list_of_index
				replicated_symbols = []
				n_replicas = list_of_index.shape[0]
				n_unit_atoms = len (atoms.numbers)
				replicated_geometry = np.zeros ((n_replicas, n_unit_atoms, 3))
				
				for i in range (n_replicas):
					vector = list_of_index[i]
					replicated_symbols.extend (atoms.get_chemical_symbols ())
					replicated_geometry[i, :, :] = atoms.positions + vector
				replicated_geometry = replicated_geometry.reshape ((n_replicas * n_unit_atoms, 3))
				replicated_cell = atoms.cell * supercell
				replicated_atoms = Atoms (positions=replicated_geometry,
				                          symbols=replicated_symbols, cell=replicated_cell, pbc=[1, 1, 1])
				self.replicated_atoms = replicated_atoms
		return self._replicated_atoms
	
	@replicated_atoms.setter
	def replicated_atoms(self, new_replicated_atoms):
		if self.is_persistency_enabled:
			folder = self.folder_name
			folder += '/'
			ase.io.write (folder + REPLICATED_ATOMS_FILE, new_replicated_atoms, format='extxyz')
		self._replicated_atoms = new_replicated_atoms
	
	@property
	def list_of_index(self):
		return self._list_of_index
	
	@list_of_index.getter
	def list_of_index(self):
		if self._list_of_index is None:
			if self.is_persistency_enabled:
				try:
					folder = self.folder_name
					folder += '/'
					self._list_of_index= np.load (folder + LIST_OF_INDEX_FILE)
				except FileNotFoundError as e:
					print (e)
			if self._list_of_index is None:
				self.list_of_index = atom_helper.create_list_of_index (
					self.atoms,
					self.supercell)
				self.list_of_index = self.list_of_index.dot(self.atoms.cell)
		return self._list_of_index
	
	@list_of_index.setter
	def list_of_index(self, new_list_of_index):
		if self.is_persistency_enabled:
			folder = self.folder_name
			folder += '/'
			np.save (folder + LIST_OF_INDEX_FILE, new_list_of_index)
		self._list_of_index = new_list_of_index
	
	@property
	def frequencies(self):
		return self._frequencies
	
	@frequencies.getter
	def frequencies(self):
		if self._frequencies is None and self.is_persistency_enabled:
			try:
				folder = self.folder_name
				folder += '/'
				self._frequencies = np.load (folder + FREQUENCIES_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._frequencies
	

	@frequencies.setter
	def frequencies(self, new_frequencies):
		if self.is_persistency_enabled:
			folder = self.folder_name
			folder += '/'
			np.save (folder + FREQUENCIES_FILE, new_frequencies)
		self._frequencies = new_frequencies
	
	@property
	def velocities(self):
		return self._velocities
	
	@velocities.getter
	def velocities(self):
		if self._velocities is None and self.is_persistency_enabled:
			try:
				folder = self.folder_name
				folder += '/'
				self._velocities = np.load (folder + VELOCITIES_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._velocities
	
	@velocities.setter
	def velocities(self, new_velocities):
		if self.is_persistency_enabled:
			folder = self.folder_name
			folder += '/'
			np.save (folder + VELOCITIES_FILE, new_velocities)
		self._velocities = new_velocities
	
	@property
	def eigenvectors(self):
		return self._eigenvectors
	
	@eigenvectors.getter
	def eigenvectors(self):
		if self._eigenvectors is None and self.is_persistency_enabled:
			try:
				folder = self.folder_name
				folder += '/'
				self._eigenvectors = np.load (folder + EIGENVECTORS_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._eigenvectors
	
	@eigenvectors.setter
	def eigenvectors(self, new_eigenvectors):
		if self.is_persistency_enabled:
			folder = self.folder_name
			folder += '/'
			np.save (folder + EIGENVECTORS_FILE, new_eigenvectors)
		self._eigenvectors = new_eigenvectors
	
	@property
	def eigenvalues(self):
		return self._eigenvalues
	
	@eigenvalues.getter
	def eigenvalues(self):
		if self._eigenvalues is None and self.is_persistency_enabled:
			try:
				folder = self.folder_name
				folder += '/'
				self._eigenvalues = np.load (folder + EIGENVALUES_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._eigenvalues
	
	@eigenvalues.setter
	def eigenvalues(self, new_eigenvalues):
		if self.is_persistency_enabled:
			folder = self.folder_name
			folder += '/'
			np.save (folder + EIGENVALUES_FILE, new_eigenvalues)
		self._eigenvalues = new_eigenvalues
	
	@property
	def gamma(self):
		return self._gamma
	
	@gamma.getter
	def gamma(self):
		#TODO separate gamma classic and quantum
		if self._gamma is None and self.is_persistency_enabled:
			try:
				folder = self.folder_name
				folder += '/' + str(self.temperature) + '/'
				if self.is_classic:
					folder += 'classic/'
				else:
					folder += 'quantum/'
				self._gamma = np.load (folder + GAMMA_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._gamma
	
	@gamma.setter
	def gamma(self, new_gamma):
		#TODO separate gamma classic and quantum
		if self.is_persistency_enabled:
			folder = self.folder_name
			folder += '/' + str(self.temperature) + '/'
			if self.sigma_in is not None:
				folder += 'sigma_in_' + str(self.sigma_in).replace('.','_') + '/'
			if self.is_classic:
				folder += 'classic/'
			else:
				folder += 'quantum/'
			np.save (folder + GAMMA_FILE, new_gamma)
		self._gamma = new_gamma
	
	@property
	def dos(self):
		return self._dos
	
	@dos.getter
	def dos(self):
		if self._dos is None and self.is_persistency_enabled:
			try:
				folder = self.folder_name
				folder += '/'
				self._dos = np.load (folder + DOS_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._dos
	
	@dos.setter
	def dos(self, new_dos):
		if self.is_persistency_enabled:
			folder = self.folder_name
			folder += '/'
			np.save (folder + DOS_FILE, new_dos)
		self._dos = new_dos
	
	@property
	def occupations(self):
		return self._occupations
	
	@occupations.getter
	def occupations(self):
		if self._occupations is None and self.is_persistency_enabled:
			try:
				folder = self.folder_name
				folder += '/' + str(self.temperature) + '/'
				if self.is_classic:
					folder += 'classic/'
				else:
					folder += 'quantum/'
				self._occupations = np.load (folder + OCCUPATIONS_FILE)
			except FileNotFoundError as e:
				print(e)
		if self._occupations is None:
			frequencies = self.frequencies
			temp = self.temperature
			density = np.zeros_like (frequencies)
			if self.is_classic == False:
				density[frequencies != 0] = 1. / (
						np.exp (constants.hbar * 2 * np.pi * frequencies[frequencies != 0] / constants.k_b / temp) - 1.)
			else:
				density[frequencies != 0] = temp / (
						2 * np.pi * frequencies[frequencies != 0]) / constants.hbar * constants.k_b
			self.occupations = density
		return self._occupations
	
	@occupations.setter
	def occupations(self, new_occupations):
		if self.is_persistency_enabled:
			folder = self.folder_name
			folder += '/' + str(self.temperature) + '/'
			if self.is_classic:
				folder += 'classic/'
			else:
				folder += 'quantum/'
			np.save (folder + OCCUPATIONS_FILE, new_occupations)
		self._occupations = new_occupations
		
	@property
	def k_points(self):
		return self._k_points
	
	@k_points.getter
	def k_points(self):
		if self._k_points is None and self.is_persistency_enabled:
			try:
				folder = self.folder_name
				folder += '/'
				self._k_points = np.load (folder + K_POINTS_FILE)
			except FileNotFoundError as e:
				print(e)
		if self._k_points is None:
			k_size = self.kpts
			n_k_points = np.prod (k_size)
			k_points = np.zeros ((n_k_points, 3))
			for index_k in range (n_k_points):
				k_points[index_k] = np.unravel_index (index_k, k_size, order='F') / k_size
			self.k_points = k_points
		return self._k_points
	
	@k_points.setter
	def k_points(self, new_k_points):
		if self.is_persistency_enabled:
			folder = self.folder_name
			folder += '/'
			np.save (folder + K_POINTS_FILE, new_k_points)
		self._k_points = new_k_points
	
	@property
	def c_v(self):
		return self._c_v
	
	@c_v.getter
	def c_v(self):
		if self._c_v is None and self.is_persistency_enabled:
			try:
				folder = self.folder_name
				folder += '/' + str(self.temperature) + '/'
				if self.is_classic:
					folder += 'classic/'
				else:
					folder += 'quantum/'
				self._c_v = np.load (folder + C_V_FILE)
			except FileNotFoundError as e:
				print(e)
		if self._c_v is None:
			frequencies = self.frequencies
			c_v = np.zeros_like (frequencies)
			if (self.is_classic):
				c_v[:] = constants.k_b * 1e21
			else:
				f_be = self.occupations
				c_v[frequencies != 0] = constants.hbar ** 2 * f_be[frequencies != 0] * (f_be[frequencies != 0] + 1) * (
						2 * np.pi * frequencies[frequencies != 0]) ** 2 / (
						                        constants.k_b * self.temperature ** 2)
			self.c_v = c_v * 1e21
		return self._c_v
	
	@c_v.setter
	def c_v(self, new_c_v):
		if self.is_persistency_enabled:
			folder = self.folder_name
			folder += '/' + str(self.temperature) + '/'
			if self.is_classic:
				folder += 'classic/'
			else:
				folder += 'quantum/'
			np.save (folder + C_V_FILE, new_c_v)
		self._c_v = new_c_v
	
	def save_csv_data(self):
		frequencies = self.frequencies
		lifetime = 1. / self.gamma
		n_modes = frequencies.shape[1]
		if self.is_classic:
			filename = "data_classic"
		else:
			filename = "data_quantum"
		filename = filename + '_' + str (self.temperature)
		filename = filename + ".csv"
		
		filename = self.folder_name + filename
		Logger ().info ('saving ' + filename)
		with open (filename, "w") as csv:
			str_to_write = 'k_x,k_y,k_z,'
			for i in range (n_modes):
				str_to_write += 'frequencies_' + str (i) + ' (THz),'
			for i in range (n_modes):
				str_to_write += 'tau_' + str (i) + ' (ps),'
			for alpha in range (3):
				coord = 'x'
				if alpha == 1:
					coord = 'y'
				if alpha == 2:
					coord = 'z'
				
				for i in range (n_modes):
					str_to_write += 'v^' + coord + '_' + str (i) + ' (km/s),'
			str_to_write += '\n'
			csv.write (str_to_write)
			for k in range (self.q_points ().shape[0]):
				str_to_write = str (self.q_points ()[k, 0]) + ',' + str (self.q_points ()[k, 1]) + ',' + str (
					self.q_points ()[k, 2]) + ','
				for i in range (n_modes):
					str_to_write += str (self.energies[k, i]) + ','
				for i in range (n_modes):
					str_to_write += str (lifetime[k, i]) + ','
				
				for alpha in range (3):
					for i in range (n_modes):
						str_to_write += str (self.velocities[k, i, alpha]) + ','
				str_to_write += '\n'
				csv.write (str_to_write)
import numpy as np
import ballistico.atoms_helper as atom_helper

FREQUENCIES_FILE = 'frequencies.npy'
EIGENVALUES_FILE = 'eigenvalues.npy'
EIGENVECTORS_FILE = 'eigenvectors.npy'
VELOCITIES_FILE = 'velocities.npy'
GAMMA_FILE = 'gamma.npy'
DOS_FILE = 'dos.npy'
OCCUPATIONS_FILE = 'occupations.npy'

IS_PERSISTENCY_ENABLED = True

class Phonons (object):
	def __init__(self, atoms, supercell = (1, 1, 1), kpts = (1, 1, 1), is_classic = False, temperature = 300):
		self.atoms = atoms
		self.supercell = np.array (supercell)
		self.k_size = np.array (kpts)
		self.is_classic = is_classic
		[self.replicated_atoms, self.list_of_replicas] = \
			atom_helper.replicate_atoms (self.atoms, self.supercell)
		
		self.n_k_points = np.prod (self.k_size)
		self.n_modes = self.atoms.get_masses ().shape[0] * 3
		self.n_phonons = self.n_k_points * self.n_modes
		self.temperature = temperature

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
	
	@property
	def frequencies(self):
		return self._frequencies
	
	@frequencies.getter
	def frequencies(self):
		if self._frequencies is None and IS_PERSISTENCY_ENABLED:
			try:
				folder = type(self).__name__
				folder += '/'
				self._frequencies = np.load (folder + FREQUENCIES_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._frequencies
	

	@frequencies.setter
	def frequencies(self, new_frequencies):
		if IS_PERSISTENCY_ENABLED:
			folder = type (self).__name__
			folder += '/'
			np.save (folder + FREQUENCIES_FILE, new_frequencies)
		self._frequencies = new_frequencies
	
	@property
	def velocities(self):
		return self._velocities
	
	@velocities.getter
	def velocities(self):
		if self._velocities is None and IS_PERSISTENCY_ENABLED:
			try:
				folder = type(self).__name__
				folder += '/'
				self._velocities = np.load (folder + VELOCITIES_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._velocities
	
	@velocities.setter
	def velocities(self, new_velocities):
		if IS_PERSISTENCY_ENABLED:
			folder = type (self).__name__
			folder += '/'
			np.save (folder + VELOCITIES_FILE, new_velocities)
		self._velocities = new_velocities
	
	@property
	def eigenvectors(self):
		return self._eigenvectors
	
	@eigenvectors.getter
	def eigenvectors(self):
		if self._eigenvectors is None and IS_PERSISTENCY_ENABLED:
			try:
				folder = type(self).__name__
				folder += '/'
				self._eigenvectors = np.load (folder + EIGENVECTORS_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._eigenvectors
	
	@eigenvectors.setter
	def eigenvectors(self, new_eigenvectors):
		if IS_PERSISTENCY_ENABLED:
			folder = type (self).__name__
			folder += '/'
			np.save (folder + EIGENVECTORS_FILE, new_eigenvectors)
		self._eigenvectors = new_eigenvectors
	
	@property
	def eigenvalues(self):
		return self._eigenvalues
	
	@eigenvalues.getter
	def eigenvalues(self):
		if self._eigenvalues is None and IS_PERSISTENCY_ENABLED:
			try:
				folder = type(self).__name__
				folder += '/'
				self._eigenvalues = np.load (folder + EIGENVALUES_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._eigenvalues
	
	@eigenvalues.setter
	def eigenvalues(self, new_eigenvalues):
		if IS_PERSISTENCY_ENABLED:
			folder = type (self).__name__
			folder += '/'
			np.save (folder + EIGENVALUES_FILE, new_eigenvalues)
		self._eigenvalues = new_eigenvalues
	
	@property
	def gamma(self):
		return self._gamma
	
	@gamma.getter
	def gamma(self):
		#TODO separate gamma classic and quantum
		if self._gamma is None and IS_PERSISTENCY_ENABLED:
			try:
				folder = type(self).__name__
				folder += '/'
				self._gamma = np.load (folder + GAMMA_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._gamma
	
	@gamma.setter
	def gamma(self, new_gamma):
		#TODO separate gamma classic and quantum
		if IS_PERSISTENCY_ENABLED:
			folder = type (self).__name__
			folder += '/'
			np.save (folder + GAMMA_FILE, new_gamma)
		self._gamma = new_gamma
	
	@property
	def dos(self):
		return self._dos
	
	@dos.getter
	def dos(self):
		if self._dos is None and IS_PERSISTENCY_ENABLED:
			try:
				folder = type(self).__name__
				folder += '/'
				self._dos = np.load (DOS_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._dos
	
	@dos.setter
	def dos(self, new_dos):
		if IS_PERSISTENCY_ENABLED:
			folder = type (self).__name__
			folder += '/'
			np.save (folder + DOS_FILE, new_dos)
		self._dos = new_dos
	
	@property
	def occupations(self):
		return self._occupations
	
	@occupations.getter
	def occupations(self):
		#TODO:add a temperature subfolder here
		if self._occupations is None and IS_PERSISTENCY_ENABLED:
			try:
				folder = type(self).__name__
				folder += '/'
				self._occupations = np.load (folder + OCCUPATIONS_FILE)
			except FileNotFoundError as e:
				print (e)
		return self._occupations
	
	@occupations.setter
	def occupations(self, new_occupations):
		#TODO:add a temperature subfolder here
		if IS_PERSISTENCY_ENABLED:
			folder = type (self).__name__
			folder += '/'
			np.save (folder + OCCUPATIONS_FILE, new_occupations)
		self._occupations = new_occupations

import numpy as np
import ballistico.constants as constants

class ConductivityController (object):
	def __init__(self, phonons):
		self.phonons = phonons
	
	def calculate_conductivity(self):
		phonons = self.phonons
		k_mesh = phonons.k_size
		gamma = phonons.gamma[0] + phonons.gamma[1]
		system = phonons.system
		temperature = system.temperature
		n_modes = system.configuration.positions.shape[0] * 3
		n_kpoints = np.prod (k_mesh)
		tau_zero = np.empty_like (gamma)
		tau_zero[(gamma) != 0] = 1 / (gamma[gamma != 0])
		f_be = np.empty_like (phonons.frequencies)
		f_be[phonons.frequencies != 0] = 1. / (
				np.exp (constants.hbar * phonons.frequencies[phonons.frequencies != 0] / (constants.k_b * temperature)) - 1.)
		c_v = constants.hbar ** 2 * f_be * (f_be + 1) * phonons.frequencies ** 2 / (constants.k_b * temperature ** 2)
		volume = np.linalg.det (system.configuration.cell) / 1000.
		
		tau_zero[tau_zero == np.inf] = 0
		c_v[np.isnan (c_v)] = 0
		conductivity_per_mode = np.zeros ((3, 3))
		for index_k in range (n_kpoints):
			for alpha in range (3):
				for beta in range (3):
					for mode in range (n_modes):
						conductivity_per_mode[alpha, beta] += c_v[index_k, mode] * phonons.velocities[
							index_k, mode, beta] * tau_zero[index_k, mode] * phonons.velocities[index_k, mode, alpha]
		
		conductivity_per_mode *= 1.E21 / (volume * n_kpoints)
		conductivity = conductivity_per_mode
		return (conductivity)

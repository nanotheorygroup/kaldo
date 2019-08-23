from opt_einsum import contract
import numpy as np
MAX_ITERATIONS_SC = 500



class ConductivityController:
    def __init__(self, phonons):
        self.phonons = phonons
        frequencies = self.phonons.frequencies.reshape((self.phonons.n_k_points * self.phonons.n_modes), order='C')
        physical_modes = (frequencies > self.phonons.frequency_threshold)
        gamma = self.phonons.gamma.reshape((self.phonons.n_phonons), order='C')

        gamma_physical = np.zeros_like(gamma)
        gamma_physical[physical_modes] = gamma[physical_modes]
        self.tau = np.zeros_like(gamma)
        self.tau[physical_modes] = 1 / gamma_physical[physical_modes]

        # TODO: move this minus sign somewhere else
        self.gamma_tensor = - 1 * self.phonons.gamma_tensor
        self.gamma_tensor = contract('a,ab,b->ab', 1 / frequencies, self.gamma_tensor, frequencies)
        scattering_matrix = np.diag(gamma_physical) + self.gamma_tensor

        # Let's remove the unphysical modes from the matrix
        index = np.outer(physical_modes, physical_modes)
        self.scattering_matrix = scattering_matrix[index].reshape((physical_modes.sum(), physical_modes.sum()), order='C')



    def conductivity(self, lambd):

        # TODO: Change units conversion in this method
        volume = np.linalg.det(self.phonons.atoms.cell) / 1000
        frequencies = self.phonons.frequencies.reshape((self.phonons.n_phonons), order='C')
        physical_modes = (frequencies > self.phonons.frequency_threshold)
        c_v = self.phonons.c_v.reshape((self.phonons.n_phonons), order='C') * 1e21
        velocities = self.phonons.velocities.real.reshape((self.phonons.n_phonons, 3), order='C') / 10
        conductivity_per_mode = np.zeros((self.phonons.n_phonons, 3, 3))
        conductivity_per_mode[physical_modes, :, :] = 1 / (volume * self.phonons.n_k_points) * c_v[physical_modes, np.newaxis, np.newaxis] * \
                                                      velocities[physical_modes, :, np.newaxis] * lambd[physical_modes, np.newaxis, :]
        return conductivity_per_mode

    def calculate_conductivity_inverse(self):
        velocities = self.phonons.velocities.real.reshape((self.phonons.n_phonons, 3), order='C') / 10
        frequencies = self.phonons.frequencies.reshape((self.phonons.n_k_points * self.phonons.n_modes), order='C')
        physical_modes = (frequencies > self.phonons.frequency_threshold)
        scattering_inverse = np.linalg.inv(self.scattering_matrix)
        lambd = np.zeros((self.phonons.n_phonons, 3))
        lambd[physical_modes, :] = scattering_inverse.dot(velocities[physical_modes, :])
        conductivity_per_mode = self.conductivity(lambd)

        #TODO: remove this debug info
        evals = np.linalg.eigvalsh(self.scattering_matrix)
        print('negative eigenvals : ', (evals < 0).sum())
        return conductivity_per_mode

    def transmission_caltech(self, gamma, velocity, length):
        kn = abs(velocity / (length * gamma))
        transmission = (1 - kn * (1 - np.exp(- 1. / kn))) * kn
        return length / abs(velocity) * transmission

    def transmission_matthiesen(self, gamma, velocity, length):
        transmission = (gamma * length / abs(velocity) + 1.) ** (-1)
        return length / abs(velocity) * transmission

    def calculate_conductivity_sc(self, max_n_iterations=None):
        if not max_n_iterations:
            max_n_iterations = MAX_ITERATIONS_SC

        frequencies = self.phonons.frequencies.reshape((self.phonons.n_phonons), order='C')
        physical_modes = (frequencies > self.phonons.frequency_threshold)

        velocities = self.phonons.velocities.real.reshape((self.phonons.n_phonons, 3), order='C') / 10


        lambda_0 = self.tau[:, np.newaxis] * velocities[:, :]
        delta_lambda = np.copy(lambda_0)
        lambda_n = np.copy(lambda_0)
        conductivity_value = np.zeros((3, 3, max_n_iterations))

        for n_iteration in range(max_n_iterations):

            delta_lambda[:, :] = -1 * (self.tau[:, np.newaxis] * self.gamma_tensor[:, physical_modes]).dot(
                delta_lambda[physical_modes, :])
            lambda_n += delta_lambda

            conductivity_value[:, :, n_iteration] = self.conductivity(lambda_n).sum(0)

        conductivity_per_mode = self.conductivity(lambda_n)
        if n_iteration == (max_n_iterations - 1):
            print('Max iterations reached')
        return conductivity_per_mode, conductivity_value

    def calculate_conductivity_rta(self):
        volume = np.linalg.det(self.phonons.atoms.cell)
        gamma = self.phonons.gamma.reshape((self.phonons.n_k_points, self.phonons.n_modes)).copy()
        physical_modes = (self.phonons.frequencies > self.phonons.frequency_threshold)
        tau = 1 / gamma
        tau[np.invert(physical_modes)] = 0
        self.phonons.velocities[np.isnan(self.phonons.velocities)] = 0
        conductivity_per_mode = np.zeros((self.phonons.n_k_points, self.phonons.n_modes, 3, 3))
        conductivity_per_mode[:, :, :, :] = contract('kn,kna,kn,knb->knab', self.phonons.c_v[:, :], self.phonons.velocities[:, :, :], tau[:, :], self.phonons.velocities[:, :, :])
        conductivity_per_mode = 1e22 / (volume * self.phonons.n_k_points) * conductivity_per_mode
        conductivity_per_mode = conductivity_per_mode.reshape((self.phonons.n_phonons, 3, 3))
        return conductivity_per_mode

    def calculate_conductivity_AF(self, gamma_in=None):
        volume = np.linalg.det(self.phonons.atoms.cell)
        if gamma_in is not None:
            gamma = gamma_in * np.ones((self.phonons.n_k_points, self.phonons.n_modes))
        else:
            gamma = self.phonons.gamma.reshape((self.phonons.n_k_points, self.phonons.n_modes)).copy()
        omega = 2 * np.pi * self.phonons.frequencies
        physical_modes = (self.phonons.frequencies[:, :, np.newaxis] > self.phonons.frequency_threshold) * (self.phonons.frequencies[:, np.newaxis, :] > self.phonons.frequency_threshold)

        lorentz = (gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2 / (((gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2) ** 2 +
                                                                           (omega[:, :, np.newaxis] - omega[:, np.newaxis, :]) ** 2)
        lorentz[np.invert(physical_modes)] = 0
        conductivity_per_mode = np.zeros((self.phonons.n_k_points, self.phonons.n_modes, self.phonons.n_modes, 3, 3))
        conductivity_per_mode[:, :, :, :, :] = contract('kn,knma,knm,knmb->knmab', self.phonons.c_v[:, :], self.phonons.velocities_AF[:, :, :, :], lorentz[:, :, :], self.phonons.velocities_AF[:, :, :, :])
        conductivity_per_mode = 1e22 / (volume * self.phonons.n_k_points) * conductivity_per_mode

        kappa = contract('knmab->knab', conductivity_per_mode)
        kappa = kappa.reshape((self.phonons.n_phonons, 3, 3))

        return kappa


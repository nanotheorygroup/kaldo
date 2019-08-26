from opt_einsum import contract
import numpy as np
from .anharmonic import Anharmonic
MAX_ITERATIONS_SC = 500



class Phonons(Anharmonic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        frequencies = self.keep_only_physical(self.frequencies.reshape((self.n_phonons), order='C'))
        gamma = self.keep_only_physical(self.gamma.reshape((self.n_phonons), order='C'))
        self.tau = 1 / gamma

        # TODO: move this minus sign somewhere else
        self.scattering_matrix_without_diagonal = - 1 * self.keep_only_physical(self.gamma_tensor)
        self.scattering_matrix_without_diagonal = contract('a,ab,b->ab', 1 / frequencies, self.scattering_matrix_without_diagonal, frequencies)
        self.scattering_matrix = np.diag(gamma) + self.scattering_matrix_without_diagonal


    def keep_only_physical(self, operator):
        physical_modes = self.physical_modes
        if operator.shape == (self.n_phonons, self.n_phonons):
            index = np.outer(physical_modes, physical_modes)
            return operator[index].reshape((physical_modes.sum(), physical_modes.sum()), order='C')
        else:
            return operator[physical_modes, ...]




    def conductivity(self, lambd):
        physical_modes = (self.frequencies.reshape(self.n_phonons) > self.frequency_threshold)

        # TODO: Change units conversion in this method
        volume = np.linalg.det(self.atoms.cell) / 1000
        c_v = self.keep_only_physical(self.c_v.reshape((self.n_phonons), order='C') * 1e21)
        velocities = self.keep_only_physical(self.velocities.real.reshape((self.n_phonons, 3), order='C') / 10)
        conductivity_per_mode = np.zeros((self.n_phonons, 3, 3))
        conductivity_per_mode[physical_modes, :, :] = 1 / (volume * self.n_k_points) * c_v[:, np.newaxis, np.newaxis] * \
                                                      velocities[:, :, np.newaxis] * lambd[:, np.newaxis, :]
        return conductivity_per_mode

    def calculate_conductivity_inverse(self):
        velocities = self.keep_only_physical(self.velocities.real.reshape((self.n_phonons, 3), order='C') / 10)
        scattering_inverse = np.linalg.inv(self.scattering_matrix)
        lambd = scattering_inverse.dot(velocities[:, :])
        conductivity_per_mode = self.conductivity(lambd)

        #TODO: remove this debug info
        neg_diag = (self.scattering_matrix.diagonal() < 0).sum()
        print('negative on diagonal : ', neg_diag)
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

        frequencies = self.frequencies.reshape((self.n_phonons), order='C')
        physical_modes = (frequencies > self.frequency_threshold)

        velocities = self.keep_only_physical(self.velocities.real.reshape((self.n_phonons, 3), order='C') / 10)


        lambda_0 = self.tau[:, np.newaxis] * velocities[:, :]
        delta_lambda = np.copy(lambda_0)
        lambda_n = np.copy(lambda_0)
        conductivity_value = np.zeros((3, 3, max_n_iterations))

        for n_iteration in range(max_n_iterations):

            delta_lambda[:, :] = -1 * (self.tau[:, np.newaxis] * self.scattering_matrix_without_diagonal[:, :]).dot(
                delta_lambda[:, :])
            lambda_n += delta_lambda

            conductivity_value[:, :, n_iteration] = self.conductivity(lambda_n).sum(0)

        conductivity_per_mode = self.conductivity(lambda_n)
        if n_iteration == (max_n_iterations - 1):
            print('Max iterations reached')

        return conductivity_per_mode, conductivity_value

    def calculate_conductivity_rta(self):
        volume = np.linalg.det(self.atoms.cell)
        gamma = self.gamma.reshape((self.n_k_points, self.n_modes)).copy()
        physical_modes = (self.frequencies > self.frequency_threshold)
        tau = 1 / gamma
        tau[np.invert(physical_modes)] = 0
        self.velocities[np.isnan(self.velocities)] = 0
        conductivity_per_mode = np.zeros((self.n_k_points, self.n_modes, 3, 3))
        conductivity_per_mode[:, :, :, :] = contract('kn,kna,kn,knb->knab', self.c_v[:, :], self.velocities[:, :, :], tau[:, :], self.velocities[:, :, :])
        conductivity_per_mode = 1e22 / (volume * self.n_k_points) * conductivity_per_mode
        conductivity_per_mode = conductivity_per_mode.reshape((self.n_phonons, 3, 3))
        return conductivity_per_mode

    def calculate_conductivity_AF(self, gamma_in=None):
        volume = np.linalg.det(self.atoms.cell)
        if gamma_in is not None:
            gamma = gamma_in * np.ones((self.n_k_points, self.n_modes))
        else:
            gamma = self.gamma.reshape((self.n_k_points, self.n_modes)).copy()
        omega = 2 * np.pi * self.frequencies
        physical_modes = (self.frequencies[:, :, np.newaxis] > self.frequency_threshold) * (self.frequencies[:, np.newaxis, :] > self.frequency_threshold)

        lorentz = (gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2 / (((gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2) ** 2 +
                                                                           (omega[:, :, np.newaxis] - omega[:, np.newaxis, :]) ** 2)
        lorentz[np.invert(physical_modes)] = 0
        conductivity_per_mode = np.zeros((self.n_k_points, self.n_modes, self.n_modes, 3, 3))
        conductivity_per_mode[:, :, :, :, :] = contract('kn,knma,knm,knmb->knmab', self.c_v[:, :], self.velocities_AF[:, :, :, :], lorentz[:, :, :], self.velocities_AF[:, :, :, :])
        conductivity_per_mode = 1e22 / (volume * self.n_k_points) * conductivity_per_mode

        kappa = contract('knmab->knab', conductivity_per_mode)
        kappa = kappa.reshape((self.n_phonons, 3, 3))

        return kappa


    def calculate_conductivity(self, method='rta', max_n_iterations=None):
        if max_n_iterations and method != 'sc':
            raise TypeError('Only self consistent method support n_iteration parameter')

        if method == 'rta':
            conductivity = self.calculate_conductivity_rta()
        elif method == 'af':
            conductivity = self.calculate_conductivity_AF()
        elif method == 'inverse':
            conductivity = self.calculate_conductivity_inverse()
        elif method == 'sc':
            conductivity = self.calculate_conductivity_sc(max_n_iterations)
        else:
            raise TypeError('Conductivity method not recognized')
        return conductivity
from opt_einsum import contract
import numpy as np
MAX_ITERATIONS_SC = 500



class ConductivityController:
    def __init__(self, phonons):
        self.phonons = phonons

    def conductivity(self, mfp):
        # TODO: Change units conversion in this method
        volume = np.linalg.det(self.phonons.atoms.cell) / 1000
        frequencies = self.phonons.frequencies.reshape((self.phonons.n_phonons), order='C')
        physical_modes = (frequencies > self.phonons.frequency_threshold)
        c_v = self.phonons.c_v.reshape((self.phonons.n_phonons), order='C') * 1e21
        velocities = self.phonons.velocities.real.reshape((self.phonons.n_phonons, 3), order='C') / 10
        conductivity_per_mode = np.zeros((self.phonons.n_phonons, 3, 3))
        conductivity_per_mode[physical_modes, :, :] = 1 / (volume * self.phonons.n_k_points) * c_v[physical_modes, np.newaxis, np.newaxis] * \
                                                      velocities[physical_modes, :, np.newaxis] * mfp[physical_modes,np.newaxis, :]
        return conductivity_per_mode

    def calculate_conductivity_inverse(self):

        scattering_matrix = self.phonons.gamma_tensor

        velocities = self.phonons.velocities.real.reshape((self.phonons.n_phonons, 3), order='C') / 10
        frequencies = self.phonons.frequencies.reshape((self.phonons.n_k_points * self.phonons.n_modes), order='C')
        physical_modes = (frequencies > self.phonons.frequency_threshold)  # & (velocities > 0)[:, 2]
        gamma = self.phonons.gamma.reshape((self.phonons.n_phonons), order='C')
        a_in = - 1 * scattering_matrix.reshape((self.phonons.n_phonons, self.phonons.n_phonons), order='C')
        a_in = contract('a,ab,b->ab', 1 / frequencies, a_in, frequencies)
        a_out = np.zeros_like(gamma)
        a_out[physical_modes] = gamma[physical_modes]
        a_out_inverse = np.zeros_like(gamma)
        a_out_inverse[physical_modes] = 1 / a_out[physical_modes]
        a = np.diag(a_out) + a_in

        # Let's remove the unphysical modes from the matrix
        index = np.outer(physical_modes, physical_modes)
        a = a[index].reshape((physical_modes.sum(), physical_modes.sum()), order='C')
        a_inverse = np.linalg.inv(a)
        lambd = np.zeros((self.phonons.n_phonons, 3))
        lambd[physical_modes, :] = a_inverse.dot(velocities[physical_modes, :])
        conductivity_per_mode = self.conductivity(lambd)
        evals = np.linalg.eigvalsh(a)
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
        physical_modes = physical_modes  # & (velocities > 0)[:, 2]

        gamma = self.phonons.gamma.reshape((self.phonons.n_phonons), order='C')
        scattering_matrix = self.phonons.gamma_tensor

        a_in = - 1 * scattering_matrix.reshape((self.phonons.n_phonons, self.phonons.n_phonons), order='C')
        a_in = contract('a,ab,b->ab', 1 / frequencies, a_in, frequencies)
        a_out = np.zeros_like(gamma)
        a_out[physical_modes] = gamma[physical_modes]
        a_out_inverse = np.zeros_like(gamma)
        a_out_inverse[physical_modes] = 1 / a_out[physical_modes]
        a = np.diag(a_out) + a_in
        b = a_out_inverse[:, np.newaxis] * velocities[:, :]
        a_out_inverse_a_in_to_n_times_b = np.copy(b)
        f_n = np.copy(b)
        conductivity_value = np.zeros((3, 3, max_n_iterations))

        for n_iteration in range(max_n_iterations):

            a_out_inverse_a_in_to_n_times_b[:, :] = -1 * (a_out_inverse[:, np.newaxis] * a_in[:, physical_modes]).dot(
                a_out_inverse_a_in_to_n_times_b[physical_modes, :])
            f_n += a_out_inverse_a_in_to_n_times_b

            conductivity_value[:, :, n_iteration] = self.conductivity(f_n).sum(0)

        conductivity_per_mode = self.conductivity(f_n)
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


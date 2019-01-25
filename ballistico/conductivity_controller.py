import ballistico.constants as constants
import numpy as np
from scipy.sparse import csc_matrix

LENGTH_THREESHOLD = 1e20
THREESHOLD = 1e-20


class ConductivityController (object):
    def __init__(self, phonons):
        self.phonons = phonons
        # self.import_scattering_matrix()

    def read_conductivity(self, converged=True):
        folder = self.phonons.folder
        if converged:
            conduct_file = 'BTE.KappaTensorVsT_CONV'
        else:
            conduct_file = 'BTE.KappaTensorVsT_RTA'
        
        conductivity_array = np.loadtxt (folder + conduct_file)
        conductivity_array = np.delete (conductivity_array, 0)
        n_steps = 0
        if converged:
            n_steps = int (conductivity_array[-1])
            conductivity_array = np.delete (conductivity_array, -1)
        return conductivity_array.reshape (3, 3)
    
    def calculate_transmission(self, velocities, length):
        
        prefactor = csc_matrix ((1. / velocities, (range(self.phonons.n_phonons), range(self.phonons.n_phonons))), shape=(self.phonons.n_phonons, self.phonons.n_phonons),
                                 dtype=np.float32)
        gamma_unitless = prefactor.dot (self.phonons.scattering_matrix)
        matthiesen_correction = csc_matrix ((np.sign (velocities)/length, (range(self.phonons.n_phonons), range(self.phonons.n_phonons))), shape=(self.phonons.n_phonons, self.phonons.n_phonons),
                                 dtype=np.float32)
        gamma_unitless_tilde = gamma_unitless + matthiesen_correction
        
        transmission = np.linalg.inv (gamma_unitless_tilde.toarray())
        
        # gamma_unitless = np.diag (length / np.abs(velocities)).dot (gamma)
        # kn = np.linalg.inv(gamma_unitless)
        # func = lambda x : (1. - 1. / x * (1. - np.exp(-1. *  x))) * 1. / x
        # transmission = function_of_operator(func, gamma_unitless)
        
        # gamma_unitless = np.diag (length / np.abs(velocities)).dot (gamma)
        # kn = np.linalg.inv(gamma_unitless)
        # one = np.identity(self.phonons.n_phonons())
        # transmission = (one - kn.dot(one - expm(-gamma_unitless))).dot(kn)
        
        return (transmission / length)
    
    def specific_heat(self, is_classical=False):
        f_be = 1. / (np.exp ((constants.thzoverjoule) * self.phonons.frequencies / constants.kelvinoverjoule /
                             self.phonons.temperature) - 1.
                     + THREESHOLD)
        c_v = np.zeros ((self.phonons.n_phonons))
        if (is_classical):
            c_v[:] = constants.kelvinoverjoule
        else:
            c_v[:] = (constants.thzoverjoule) ** 2 * f_be[:] * (f_be[:] + 1) * self.phonons.frequencies[:] ** 2 / \
                     (constants.kelvinoverjoule * self.phonons.temperature ** 2)
            
        # TODO: get rid of this prefactor
        return 1e21 * c_v
    
    def exact_conductivity(self, is_classical=False, l_x=LENGTH_THREESHOLD, l_y=LENGTH_THREESHOLD,
                           l_z=LENGTH_THREESHOLD, alpha=0, beta=0):
        volume = np.linalg.det(self.phonons.atoms.cell) / 1000.

        length = np.array([l_x, l_y, l_z])
        conductivity_per_mode = np.zeros ((self.phonons.n_phonons))
        # gamma_full = np.diag (1. / (self.tau_zero + THREESHOLD)) - np.array (self.phonons.gamma.toarray ()) +
        # THREESHOLD
        # gamma_full = np.array (self.phonons.gamma.toarray ())
        
        transmission = self.calculate_transmission (self.phonons.velocities[:, alpha], length[alpha]) * length[alpha]
        
        conductivity_per_mode[:] = self.specific_heat(is_classical) * (self.phonons.velocities[:, beta].dot(transmission))
        
        conductivity = np.sum (conductivity_per_mode, 0) / self.phonons.n_k_points / volume
        return conductivity
    
    def transmission_matthiesen(self, rate, velocity, length):
        # TODO: this is not exacly a transmission function, change the names to match the right units.
        # trans =  (rates + abs (velocity) / length) ** (-1)
        # return trans
        trans = (rate + abs (velocity) / length) ** (-1)
        return trans
    
    def transmission_infinite(self, rate, velocity, length):
        return 1. / (rate + THREESHOLD)
    
    def transmission_caltech(self, rate, velocity, length):
        kn = abs (velocity / (length * rate))
        trans = (1 - kn * (1 - np.exp (- 1. / kn))) * kn
        return trans * length / abs (velocity)
    
    def calculate_conductivity(self, is_classic, length_thresholds=None):

        hbar = constants.hbar * 1e12
        k_b = constants.kelvinoverjoule
        phonons = self.phonons
        volume = np.linalg.det(phonons.atoms.cell) / 1000

        omega = phonons.frequencies * 2 * np.pi
        velocities = phonons.velocities.real.reshape((phonons.n_k_points, phonons.n_modes, 3))
        velocities[np.isnan(velocities)] = 0

        omega = omega.reshape((phonons.n_phonons))
        velocities = velocities.reshape(phonons.n_phonons, 3)
        f_be = np.zeros((phonons.n_phonons))

        frequencies = self.phonons.frequencies.reshape(self.phonons.n_k_points * self.phonons.n_modes)
        physical_modes = np.abs(frequencies) > self.phonons.energy_threshold

        index = np.outer(physical_modes, physical_modes)

        conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))

        for alpha in range(3):
            # TODO: here we can probably avoid allocating the tensor new everytime
            scattering_matrix = np.zeros((self.phonons.n_phonons, self.phonons.n_phonons))
            scattering_matrix[index] = -1 * self.phonons.scattering_matrix.reshape((self.phonons.n_phonons,
                                                                                    self.phonons.n_phonons))[index]
            scattering_matrix += np.diag(self.phonons.gamma.flatten())
            scattering_matrix = scattering_matrix[index].reshape((physical_modes.sum(), physical_modes.sum()))
            if length_thresholds:
                if length_thresholds[alpha]:
                    scattering_matrix[:, :] += np.diag(np.abs(velocities[physical_modes, alpha]) / length_thresholds[
                        alpha])

            gamma_inv = np.linalg.inv(scattering_matrix)

            gamma_inv = 1 / (omega[physical_modes, np.newaxis]) * (gamma_inv * omega[np.newaxis, physical_modes])

            # import seaborn as sns
            # import matplotlib.pyplot as plt
            # sns.kdeplot(np.triu(gamma_inv).flatten())
            # sns.kdeplot(np.tril(gamma_inv).flatten())
            # sns.kdeplot(np.diag(gamma_inv).flatten())
            # plt.show()
            for beta in range(3):
                f_be[:] = 1. / (np.exp(hbar * omega[:] / k_b / phonons.temperature) - 1.)

                if (is_classic):
                    conductivity_per_mode[physical_modes, alpha, beta] = 1e21 / (volume * phonons.n_k_points) * k_b *\
                                                                         velocities[physical_modes, alpha] * gamma_inv.dot(
                        velocities[physical_modes, beta])

                else:
                    conductivity_per_mode[physical_modes, alpha, beta] = 1e21 * hbar ** 2 / (k_b *
                                                                                             phonons.temperature ** 2 * volume * phonons.n_k_points) * f_be[physical_modes] * (f_be[physical_modes] + 1)\
                                                            * omega[physical_modes] ** 2 * velocities[physical_modes, alpha] * gamma_inv.dot(velocities[physical_modes, beta])

            total_conductivity = np.sum(conductivity_per_mode, 0)
        return total_conductivity

    def calculate_conductivity_sc(self, is_classic, n_iterations=10, length_thresholds=None):
        hbar = constants.hbar * 1e12
        k_b = constants.kelvinoverjoule

        phonons = self.phonons
        volume = np.linalg.det(phonons.atoms.cell) / 1000
        omegas = phonons.frequencies * 2 * np.pi
        velocities = phonons.velocities.real.reshape((phonons.n_k_points, phonons.n_modes, 3))
        velocities[np.isnan(velocities)] = 0
        scattering_matrix = self.phonons.scattering_matrix.reshape((self.phonons.n_phonons,
                                                                            self.phonons.n_phonons))
        F_n_0 = np.zeros((phonons.n_k_points * phonons.n_modes, 3))
        velocities = velocities.reshape(phonons.n_phonons, 3)
        omegas = omegas.reshape((phonons.n_phonons))

        for alpha in range(3):
            gamma = np.zeros_like(omegas)

            for mu in range(phonons.n_phonons):
                if length_thresholds:
                    if length_thresholds[alpha]:
                        gamma[mu] = phonons.gamma.reshape(phonons.n_phonons)[mu] + \
                                    np.abs(velocities[mu, alpha]) / length_thresholds[alpha]
                    else:
                        gamma[mu] = phonons.gamma.reshape(phonons.n_phonons)[mu]

                else:
                    gamma[mu] = phonons.gamma.reshape(phonons.n_phonons)[mu]
            tau_zero = np.zeros_like(gamma)
            tau_zero[gamma != 0] = 1 / gamma[gamma != 0]
            F_n_0[:, alpha] = tau_zero[:] * velocities[:, alpha] * omegas[:]

        F_n = F_n_0.copy()
        f_be = np.zeros((phonons.n_phonons))
        conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))
        for n_iteration in range(n_iterations):
            for alpha in range(3):
                for beta in range(3):
                    f_be[:] = 1. / (np.exp(hbar * omegas[:] / k_b / phonons.temperature) - 1.)

                    if (is_classic):
                        conductivity_per_mode[:, alpha, beta] = 1e21 / (volume * phonons.n_k_points) * k_b / \
                                                                (omegas[:]) * velocities[:, alpha] * F_n[:, beta]
                    else:
                        conductivity_per_mode[:, alpha, beta] = 1e21 * hbar ** 2 / \
                                                                (k_b * phonons.temperature ** 2 * volume *
                                                                 phonons.n_k_points) * f_be[:] * (f_be[:] + 1) * \
                                                                omegas[:] * velocities[:, alpha] * F_n[:, beta]

            tau_zero = tau_zero.reshape((phonons.n_phonons))

            # calculate the shift in mft
            DeltaF = scattering_matrix.dot(F_n)

            for alpha in range(3):
                F_n[:, alpha] = F_n_0[:, alpha] + tau_zero[:] * DeltaF[:, alpha]

        for alpha in range(3):
            for beta in range(3):
                f_be[:] = 1. / (np.exp(hbar * omegas[:] / k_b / phonons.temperature) - 1.)

                if (is_classic):
                    conductivity_per_mode[:, alpha, beta] = 1e21 / (volume * phonons.n_k_points) * \
                                                            k_b / (omegas[:]) * velocities[:,alpha] * F_n[:, beta]
                else:
                    conductivity_per_mode[:, alpha, beta] = 1e21 * hbar ** 2 \
                                                            / (k_b * phonons.temperature ** 2 * volume *
                                                               phonons.n_k_points) * f_be[:] * (f_be[:] + 1) * \
                                                            omegas[:] * velocities[:, alpha] * F_n[:, beta]

        conductivity = np.sum(conductivity_per_mode, 0)
        return conductivity


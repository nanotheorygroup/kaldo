import numpy as np
from ballistico.Phonons import Phonons
from ballistico.constants import *

class PhononsAnharmonic (Phonons):
    
    def potentials_phonons(self, index_k, index_kp, index_kpp, mu, mu_p, mu_pp,
                           is_plus):
        n_kpoints = np.prod(self.k_size)
        i_k = np.array(np.unravel_index (index_k, self.k_size))
        i_kp = np.array(np.unravel_index (index_kp, self.k_size))
        i_kpp = np.array(np.unravel_index (index_kpp, self.k_size))
        

        kp_point = i_kp / self.k_size
    
        kpp_point = i_kpp / self.k_size
    
        list_of_replicas = self.system.list_of_replicas
        geometry = self.system.configuration.positions
        n_particles = geometry.shape[0]
        n_replicas = list_of_replicas.shape[0]
        n_modes = n_particles * 3
    
        n_grid = self.k_size
        # TODO: I don't know why there's a 10 here, copied by sheng bte
        cellinv = np.linalg.inv (self.system.configuration.cell)
        rlattvec = cellinv * 2 * np.pi * 10.
        realqprime = np.matmul (rlattvec, kp_point)
        realqdprime = np.matmul (rlattvec, kpp_point)
        # print(realqprime, realqdprime)
    
        cell = self.system.replicated_configuration.cell
        cellinv = np.linalg.inv (cell)
        unit_cell_inv = np.linalg.inv (self.system.configuration.cell)
        chi_p = np.zeros (n_replicas).astype (complex)
        chi_pp = np.zeros (n_replicas).astype (complex)
    
        rlatticevec = np.linalg.inv (cell) * np.linalg.det (cell)
        V = np.abs (rlatticevec[0].dot (cell[0]))
        rlatticevec = 2 * np.pi / V * rlatticevec
        for l in range (n_replicas):
            rep_position = self.system.replicated_configuration[l]
        
            sxij = list_of_replicas[l]
            # sxij = sxij - np.round (sxij)
            # dxij = cell.dot (sxij)
            if is_plus:
                chi_p[l] = np.exp (1j * sxij.dot (realqprime))
            else:
                chi_p[l] = np.exp (-1j * sxij.dot (realqprime))
            chi_pp[l] = np.exp (-1j * sxij.dot (realqdprime))
        potential = np.tensordot (self.system.third_order, chi_p, (3, 0))
        potential = np.tensordot (potential, chi_pp, (5, 0)).squeeze ()
    
        masses = self.system.configuration.get_masses ()
        potential /= np.sqrt (masses[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        potential /= np.sqrt (masses[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis])
        potential /= np.sqrt (masses[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])
        a_k = self.eigenvectors[index_k, :, :].T[mu]
        a_kp = self.eigenvectors[index_kp, :, :].T[mu_p]
        a_kpp = self.eigenvectors[index_kpp, :, :].T[mu_pp]

        a_k = a_k.reshape (n_particles * 3)
        a_kp = a_kp.reshape (n_particles * 3)
        a_kpp = a_kpp.reshape (n_particles * 3)
        potential = potential.reshape (n_particles * 3, n_particles * 3, n_particles * 3)
        potential = np.tensordot (potential, a_k, (0, 0))
        if is_plus:
            potential = np.tensordot (potential, a_kp, (0, 0))
        else:
            potential = np.tensordot (potential, np.conj (a_kp), (0, 0))
        potential = np.tensordot (potential, np.conj (a_kpp), (0, 0))
        return potential

    def calculate_gamma(self):
        hbarp = 1.05457172647
    
        print ('Lifetime:')
        nptk = np.prod (self.k_size)

        n_particles = self.system.configuration.positions.shape[0]
        n_modes = n_particles * 3
        gamma_plus = np.zeros ((np.prod(self.k_size), n_modes))
        gamma_minus = np.zeros ((np.prod(self.k_size), n_modes))
        ps_plus = np.zeros ((np.prod(self.k_size), n_modes))
        ps_minus = np.zeros ((np.prod(self.k_size), n_modes))

        # TODO: remove acoustic sum rule
        self.frequencies[0, :3] = 0
        self.velocities[0, :3, :] = 0

        omega = 2 * np.pi * self.frequencies
        density = 1. / (np.exp (hbar * omega / k_b / self.system.temperature) - 1.)
        for index_k in range(np.prod(self.k_size)):
            geometry = self.system.configuration.positions
            n_particles = geometry.shape[0]
            n_modes = n_particles * 3
            phase_space = np.zeros ((2, n_modes))
            gamma = np.zeros ((2, n_modes))
            i_k = np.array (np.unravel_index (index_k, self.k_size))

            for mu in range (n_modes):
    
                energy_diff = np.zeros ((2, nptk, n_modes, nptk, n_modes))
                energy_diff[1] = np.abs (
                    omega[index_k, mu] + omega[:, :, np.newaxis, np.newaxis] - omega[np.newaxis, np.newaxis, :, :])
                energy_diff[0] = np.abs (
                    omega[index_k, mu] - omega[:, :, np.newaxis, np.newaxis] - omega[np.newaxis, np.newaxis, :, :])
    
                density_fact = np.zeros ((2, nptk, n_modes, nptk, n_modes))
                density_fact[1] = density[:, :, np.newaxis, np.newaxis] - density[np.newaxis, np.newaxis, :, :]
                density_fact[0] = .5 * (
                            1 + density[:, :, np.newaxis, np.newaxis] + density[np.newaxis, np.newaxis, :, :])

                sigma = self.calculate_broadening (
                    self.velocities[:, :, np.newaxis, np.newaxis, :] - self.velocities[np.newaxis, np.newaxis, :, :, :])
                
                dirac_delta = np.zeros ((2, nptk, n_modes, nptk, n_modes))

                potential = np.zeros ((2, nptk, n_modes, n_modes)).astype (np.complex)
                delta_condition_plus = ((omega[:, :, np.newaxis, np.newaxis] != 0) & (omega[np.newaxis, np.newaxis, :, :] != 0)) & (
                        energy_diff[1, :, :, :, :] <= (
                            2. * sigma[:, :, :, :]))
                delta_condition_minus = ((omega[:, :, np.newaxis, np.newaxis] != 0) & (
                            omega[np.newaxis, np.newaxis, :, :] != 0)) & (
                                               energy_diff[0, :, :, :, :] <= (2. * sigma[:, :, :, :]))
                delta_condition = np.array([delta_condition_minus, delta_condition_plus])
                
                for is_plus in (1, 0):
                    dirac_delta[is_plus] = density_fact[is_plus, :, :, :, :] * np.exp (
                        -(energy_diff[is_plus, :, :, :, :]) ** 2 / (
                                sigma[:, :, :, :] ** 2)) / sigma[:, :, :, :] / np.sqrt (np.pi) / (
                                                   omega[index_k, mu] * omega[:, :, np.newaxis, np.newaxis] * omega[
                                                                                                              np.newaxis,
                                                                                                              np.newaxis,
                                                                                                              :, :])
                    for mu_p in range (n_modes):
                        for index_kp in range (np.prod (self.k_size)):
                            i_kp = np.array(np.unravel_index (index_kp, self.k_size))
                            # TODO: Umklapp processes are when the reminder is != 0, we could probably separate those
                            if is_plus:
                                i_kpp = ((i_k + i_kp)) % self.k_size
                            else:
                                i_kpp = ((i_k - i_kp)) % self.k_size
                                
                            index_kpp= np.ravel_multi_index (i_kpp, self.k_size)
                            
                            for mu_pp in range (n_modes):
                                

                                if delta_condition[is_plus, index_kp, mu_p, index_kpp, mu_pp]:
                                    
                                    phase_space[is_plus, mu] += dirac_delta[is_plus, index_kp, mu_p, index_kpp, mu_pp]
                                    potential[is_plus, index_kp, mu_p, mu_pp] = self.potentials_phonons (index_k, index_kp, index_kpp, mu, mu_p,
                                                                                               mu_pp, is_plus=is_plus)
                                
                                    gamma[is_plus, mu] += hbarp * np.pi / 4. * np.abs (potential[is_plus, index_kp, mu_p, mu_pp]) ** 2 * dirac_delta[is_plus, index_kp, mu_p, index_kpp, mu_pp]
                    
    
            prefactor = 5.60626442 * 10 ** 8 / nptk
            gamma_plus[index_k] = prefactor * gamma[1]
            gamma_minus[index_k] = prefactor * gamma[0]
            ps_plus[index_k] = phase_space[1] / nptk
            ps_minus[index_k] = phase_space[0] / nptk
        return gamma_plus, gamma_minus, ps_plus, ps_minus

    def calculate_broadening(self, velocity):
        cellinv = np.linalg.inv (self.system.configuration.cell)
        # armstrong to nanometers
        rlattvec = cellinv * 2 * np.pi * 10.
        
        # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
        base_sigma = ((np.tensordot (velocity, rlattvec / self.k_size,[-1,1])) ** 2).sum(axis=-1)
        base_sigma = np.sqrt (base_sigma / 6.)
        return base_sigma

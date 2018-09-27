import numpy as np
from ballistico.Phonons import Phonons
from ballistico.constants import *
import scipy
from sparse import tensordot,COO

class PhononsAnharmonic (Phonons):
    
    
    
    

    def gaussian_delta(self, params):
        # alpha is a factor that tells whats the ration between the width of the gaussian and the width of allowed phase space
        alpha = 2
    
        delta_energy = params[0]
        # allowing processes with width sigma and creating a gaussian with width sigma/2 we include 95% (erf(2/sqrt(2)) of the probability of scattering. The erf makes the total area 1
        sigma = params[1] / alpha
        gauss = 1 / np.sqrt (2 * np.pi * sigma ** 2) * np.exp (- delta_energy ** 2 / (2 * sigma ** 2))
        gauss /= np.erf (2 / np.sqrt (2))
        return gauss

    def calculate_gamma(self, unique_points=None):
        hbarp = 1.05457172647
    
        print ('Lifetime:')
        nptk = np.prod (self.k_size)

        n_particles = self.system.configuration.positions.shape[0]
        n_modes = n_particles * 3
        gamma = np.zeros ((2, np.prod(self.k_size), n_modes))
        ps = np.zeros ((2, np.prod(self.k_size), n_modes))

        # TODO: remove acoustic sum rule
        self.frequencies[0, :3] = 0
        self.velocities[0, :3, :] = 0


        # for index_kp in range (np.prod (self.k_mesh)):
        #     for index_kpp in range (np.prod (self.k_mesh)):
        #         if (tensor_k[0, :, index_kp, index_kpp].sum () > 1):
        #             print (tensor_k[0, :, index_kp, index_kpp].sum ())
        
        # for index_kp in range (np.prod (self.k_mesh)):
        #     for index_kpp in range (np.prod (self.k_mesh)):
        #         print (np.argwhere (tensor_k[1, :, index_kp, index_kpp] == True))

        prefactor = 5.60626442 * 10 ** 8 / nptk
        cellinv = self.system.configuration.cell_inv
        masses = self.system.configuration.get_masses ()

        list_of_replicas = self.system.list_of_replicas
        n_modes = n_particles * 3
        k_size = self.k_size
        n_replicas = list_of_replicas.shape[0]

        # TODO: I don't know why there's a 10 here, copied by sheng bte
        rlattvec = cellinv * 2 * np.pi
        chi = np.zeros ((nptk, n_replicas), dtype=np.complex)

        for index_k in range (np.prod (k_size)):
            i_k = np.array (self.unravel_index (index_k))
            k_point = i_k / k_size
            realq = np.matmul (rlattvec, k_point)
            for l in range (n_replicas):
                chi[index_k, l] = np.exp (1j * list_of_replicas[l].dot (realq))

        scaled_potential = self.system.third_order[0] / np.sqrt (masses[ :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        scaled_potential /= np.sqrt (masses[ np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        scaled_potential /= np.sqrt (masses[ np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])
        
        scaled_potential = scaled_potential.reshape(n_modes, n_replicas, n_modes, n_replicas, n_modes)
        print('Projection started')
        #
        second_eigenv = np.zeros((2, nptk, n_modes, n_modes), dtype=np.complex)
        second_chi = np.zeros((2, nptk, n_replicas), dtype=np.complex)
        # transformed_potential = np.zeros((2, n_modes, nptk, n_modes, nptk, n_modes), dtype=np.complex)

        gamma = np.zeros((2, nptk, n_modes))
        n_particles = self.system.configuration.positions.shape[0]
        n_modes = n_particles * 3
        k_size = self.k_size
        nptk = np.prod (k_size)

        # TODO: remove acoustic sum rule
        # self.frequencies[0, :3] = 0
        # self.velocities[0, :3, :] = 0

        omega = 2 * np.pi * self.frequencies

        density = np.empty_like (omega)

        density[omega != 0] = 1. / (np.exp (hbar * omega[omega != 0] / k_b / self.system.temperature) - 1.)

        omega_product = omega[:, :, np.newaxis, np.newaxis] * omega[np.newaxis, np.newaxis, :, :]

        sigma = self.calculate_broadening (
            self.velocities[:, :, np.newaxis, np.newaxis, :] - self.velocities[np.newaxis, np.newaxis, :, :, :])

        DELTA_THRESHOLD = 2
        delta_correction = scipy.special.erf (DELTA_THRESHOLD / np.sqrt (2))
        # delta_correction = 1
        if unique_points is None:
            list_of_k = np.arange(np.prod (k_size))
        else:
            
            list_of_k = np.array(unique_points)

        third_eigenv = self.eigenvectors.conj ()
        third_chi = chi.conj ()

        for is_plus in (1, 0):
    
            if is_plus:
                density_fact = density[:, :, np.newaxis, np.newaxis] - density[np.newaxis, np.newaxis, :, :]
                second_eigenv = self.eigenvectors
                second_chi = chi
            else:
                density_fact = .5 * (1 + density[:, :, np.newaxis, np.newaxis] + density[np.newaxis, np.newaxis, :, :])
                second_eigenv = self.eigenvectors.conj ()
                second_chi = chi.conj ()
    
    
            for index_k in (list_of_k):
                print (is_plus, index_k)
    
                i_k = np.array (self.unravel_index (index_k))
                for mu in range (n_modes):
                    if omega[index_k, mu] != 0:
                        first = self.eigenvectors[index_k, :, mu]
    
                        if is_plus:
                            energy_diff = np.abs (
                                omega[index_k, mu] + omega[:, :, np.newaxis, np.newaxis] - omega[np.newaxis, np.newaxis, :, :])
                        else:
                            energy_diff = np.abs (
                                omega[index_k, mu] - omega[:, :, np.newaxis, np.newaxis] - omega[np.newaxis, np.newaxis, :, :])
                
                        index_kp_vec = np.arange (np.prod (self.k_size))
                        i_kp_vec = np.array (self.unravel_index (index_kp_vec))
                        i_kpp_vec = i_k[:, np.newaxis] + (int (is_plus) * 2 - 1) * i_kp_vec[:, :]
                        index_kpp_vec = self.ravel_multi_index (i_kpp_vec)
                        delta_energy = energy_diff[index_kp_vec, :, index_kpp_vec, :]
                
                        sigma_small = sigma[index_kp_vec, :, index_kpp_vec, :]
                        condition = (delta_energy < DELTA_THRESHOLD * sigma_small) & (
                                omega[index_kp_vec, :, np.newaxis] != 0) & (omega[index_kpp_vec, np.newaxis, :] != 0)

                        interactions = np.array(np.where (condition)).T
                        # interactions = np.array(np.unravel_index (np.flatnonzero (condition), condition.shape)).T
                        if interactions.size != 0:
                            print('interactions: ', index_k, interactions.size)
                            index_kp_vec = interactions[:, 0]
                            index_kpp_vec = index_kpp_vec[index_kp_vec]
                            mup_vec = interactions[:, 1]
                            mupp_vec = interactions[:, 2]
                    
                            dirac_delta = density_fact[index_kp_vec, mup_vec, index_kpp_vec, mupp_vec]
                    
                            dirac_delta /= omega_product[index_kp_vec, mup_vec, index_kpp_vec, mupp_vec]
                    
                            dirac_delta *= np.exp (- energy_diff[index_kp_vec, mup_vec, index_kpp_vec, mupp_vec] ** 2 / sigma[index_kp_vec, mup_vec, index_kpp_vec, mupp_vec] ** 2) / \
                                           sigma[index_kp_vec, mup_vec, index_kpp_vec, mupp_vec] / np.sqrt (np.pi) / delta_correction
                    
                            ps[is_plus, index_k, mu] += np.sum(dirac_delta)
                            # index_i = np.arange(n_modes)[:, index_kp_vec, np.newaxis, index_kpp_vec, np.newaxis]
                            # index_ip = np.arange(n_modes)[np.newaxis, index_kp_vec, :, index_kpp_vec, np.newaxis]
                            # index_ipp = np.arange(n_modes)[np.newaxis, index_kp_vec, np.newaxis, index_kpp_vec, :]

                            third = third_eigenv[index_kpp_vec, :, mupp_vec]
                            second = second_eigenv[index_kp_vec, :, mup_vec]

                            projected_potential = np.einsum ('wlitj,al,at,aj,ai,w->a', scaled_potential, second_chi[index_kp_vec], third_chi[index_kpp_vec], third , second, first, optimize='greedy')
                            
                            gamma[is_plus, index_k, mu] += np.sum(np.abs (projected_potential) ** 2 * dirac_delta)
                        gamma[is_plus, index_k, mu] /= (omega[index_k, mu])
                        ps[is_plus, index_k, mu] /= (omega[index_k, mu])

        return gamma[1] * prefactor *  hbarp * np.pi / 4., gamma[0] * prefactor *  hbarp * np.pi / 4., ps[1] / nptk, ps[0] / nptk

    def calculate_broadening(self, velocity):
        cellinv = self.system.configuration.cell_inv
        rlattvec = cellinv * 2 * np.pi
        
        # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
        # 10 = armstrong to nanometers
        base_sigma = ((np.tensordot (velocity * 10., rlattvec / self.k_size,[-1,1])) ** 2).sum(axis=-1)
        base_sigma = np.sqrt (base_sigma / 6.)
        return base_sigma

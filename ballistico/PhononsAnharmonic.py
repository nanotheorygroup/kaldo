import numpy as np
from ballistico.Phonons import Phonons
from ballistico.constants import *
from sparse import tensordot,COO

class PhononsAnharmonic (Phonons):
    
    def calculate_delta(self, index_k, mu, is_plus):
        i_k = np.array (self.unravel_index (index_k))


        n_particles = self.system.configuration.positions.shape[0]
        n_modes = n_particles * 3
        k_size = self.k_size
        nptk = np.prod(k_size)

        
        # TODO: remove acoustic sum rule
        # self.frequencies[0, :3] = 0
        # self.velocities[0, :3, :] = 0

        omega = 2 * np.pi * self.frequencies
        
        density = np.empty_like(omega)

        density[omega != 0] = 1. / (np.exp (hbar * omega[omega != 0] / k_b / self.system.temperature) - 1.)

        energy_diff = np.zeros ((2, nptk, n_modes, nptk, n_modes))
        energy_diff[1] = np.abs (
            omega[index_k, mu] + omega[:, :, np.newaxis, np.newaxis] - omega[np.newaxis, np.newaxis, :, :])
        energy_diff[0] = np.abs (
            omega[index_k, mu] - omega[:, :, np.newaxis, np.newaxis] - omega[np.newaxis, np.newaxis, :, :])
    
        omega_product = omega[:, :, np.newaxis, np.newaxis] * omega[np.newaxis, np.newaxis, :, :]

        density_fact = np.zeros ((2, nptk, n_modes, nptk, n_modes))
        density_fact[1] = density[:, :, np.newaxis, np.newaxis] - density[np.newaxis, np.newaxis, :, :]
        density_fact[0] = .5 * (1 + density[:, :, np.newaxis, np.newaxis] + density[np.newaxis, np.newaxis, :, :])
    
        sigma = self.calculate_broadening (
            self.velocities[:, :, np.newaxis, np.newaxis, :] - self.velocities[np.newaxis, np.newaxis, :, :, :])

        total_iterations = 0

        for index_kp in range (np.prod (self.k_size)):
            i_kp = np.array (self.unravel_index (index_kp))
            # TODO: Umklapp processes are when the reminder is != 0, we could probably separate those
            if is_plus:
                i_kpp = i_k + i_kp
            else:
                i_kpp = i_k - i_kp
        
            index_kpp = self.ravel_multi_index (i_kpp)
    
            delta_energy = energy_diff[is_plus, index_kp, :, index_kpp, :]

            sigma_small = sigma[index_kp, :, index_kpp, :]
    
            if omega[index_k, mu] != 0:
                interactions = np.argwhere ((delta_energy < 2 * sigma_small) & (
                            omega[index_kp, :, np.newaxis] != 0) & (omega[index_kpp, np.newaxis, :] != 0))

                if interactions.size != 0:
                    mup_vec = interactions[:,0]
                    mupp_vec = interactions[:,1]
                    n_interactions = mup_vec.shape[0]
                    index_kp_vec = index_kp * np.ones(n_interactions, dtype=int)
                    index_kpp_vec = index_kpp * np.ones(n_interactions, dtype=int)
                    reduced_index = [index_kp_vec, mup_vec, index_kpp_vec, mupp_vec]
                    full_index = [np.ones (n_interactions, dtype=int) * is_plus, index_kp_vec, mup_vec, index_kpp_vec, mupp_vec]
                    dirac_delta = density_fact[full_index]
                    dirac_delta /= (omega[index_k, mu])
                    dirac_delta /= omega_product[reduced_index]

                    dirac_delta *= np.exp (- energy_diff[full_index] ** 2 / sigma[reduced_index] ** 2) / \
                                            sigma[reduced_index] / np.sqrt (np.pi)
                    try:
                        dirac_delta_sparse = dirac_delta_sparse + COO (reduced_index, dirac_delta,
                                                                       shape=(nptk, n_modes, nptk, n_modes))
                    except UnboundLocalError as err:
                        dirac_delta_sparse = COO (np.array (reduced_index), dirac_delta,
                                                  shape=(nptk, n_modes, nptk, n_modes))
                    total_iterations += n_interactions

        if not total_iterations:
            return None
        else:
            return dirac_delta_sparse


    def calculate_gamma(self):
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


        # for index_kp in range (np.prod (self.k_size)):
        #     for index_kpp in range (np.prod (self.k_size)):
        #         if (tensor_k[0, :, index_kp, index_kpp].sum () > 1):
        #             print (tensor_k[0, :, index_kp, index_kpp].sum ())
        
        # for index_kp in range (np.prod (self.k_size)):
        #     for index_kpp in range (np.prod (self.k_size)):
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

        rescaled_potential = self.system.third_order[0] / np.sqrt (masses[ :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        rescaled_potential /= np.sqrt (masses[ np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        rescaled_potential /= np.sqrt (masses[ np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])
        
        rescaled_potential = rescaled_potential.reshape(n_modes, n_replicas, n_modes, n_replicas, n_modes)
        print('Projection started')
        #
        # for is_plus in (1, 0):
        #
        #     if is_plus:
        #         second_eigenv = self.eigenvectors
        #         second_chi = chi
        #     else:
        #         second_eigenv = self.eigenvectors.conj ()
        #         second_chi = chi.conj ()
        #
        #     third_eigenv = self.eigenvectors.conj ()
        #     third_chi = chi.conj ()
        #
        #     projected_potential[is_plus] = np.einsum ('wlitj,kl,kin,qt,qjm,pwr->prknqm', rescaled_potential, second_chi, second_eigenv, third_chi, third_eigenv, self.eigenvectors, optimize='greedy')
        #
        #     # TODO: make it sparse here
        #
        #     projected_potential[is_plus] = np.abs (projected_potential[is_plus]) ** 2
        #
        # print('Projection done')

        second_eigenv = np.zeros((2, nptk, n_modes, n_modes), dtype=np.complex)
        second_chi = np.zeros((2, nptk, n_replicas), dtype=np.complex)
        transformed_potential = np.zeros((2, n_modes, nptk, n_modes, nptk, n_modes), dtype=np.complex)

        gamma = np.zeros((2, nptk, n_modes))
        for is_plus in (1, 0):
            if is_plus:
                second_eigenv[is_plus] = self.eigenvectors
                second_chi[is_plus] = chi
            else:
                second_eigenv[is_plus] = self.eigenvectors.conj ()
                second_chi[is_plus] = chi.conj ()
    
            third_eigenv = self.eigenvectors.conj ()
            third_chi = chi.conj ()
            
            transformed_potential[is_plus] = np.einsum ('wlitj,kl,qt->wkiqj', rescaled_potential, second_chi[is_plus], third_chi, optimize='greedy')
            
        
        for index_k in range (np.prod (k_size)):

                            
            for mu in range (n_modes):
                
                for is_plus in (1, 0):
                # print (index_k, mu)
                    dirac_delta = self.calculate_delta (index_k, mu, is_plus)
                    if dirac_delta:
    
                        index_kp = dirac_delta.coords[0]
                        mup = dirac_delta.coords[1]
                        index_kpp = dirac_delta.coords[2]
                        mupp = dirac_delta.coords[3]
                        ps[is_plus, index_k, mu] += np.sum(dirac_delta[index_kp, mup, index_kpp, mupp])
                        projected_potential = np.einsum('awij,aj,ai,w->a', transformed_potential[is_plus, :, index_kp, :, index_kpp, :], third_eigenv[index_kpp, :, mupp], second_eigenv[is_plus, index_kp, :, mup], self.eigenvectors[index_k, :, mu])
                        gamma[is_plus, index_k, mu] += np.sum(np.abs (projected_potential) ** 2 * dirac_delta.todense()[index_kp, mup, index_kpp, mupp])
    
    

        return gamma[1] * prefactor *  hbarp * np.pi / 4., gamma[0] * prefactor *  hbarp * np.pi / 4., ps[1] / nptk, ps[0] / nptk

    def calculate_broadening(self, velocity):
        cellinv = self.system.configuration.cell_inv
        # armstrong to nanometers
        rlattvec = cellinv * 2 * np.pi
        
        # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
        base_sigma = ((np.tensordot (velocity * 10., rlattvec / self.k_size,[-1,1])) ** 2).sum(axis=-1)
        base_sigma = np.sqrt (base_sigma / 6.)
        return base_sigma

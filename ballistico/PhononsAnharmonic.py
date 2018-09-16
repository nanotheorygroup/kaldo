import numpy as np
from ballistico.Phonons import Phonons
from ballistico.constants import *

class PhononsAnharmonic (Phonons):
    
    def calculate_delta(self, index_k, mu):

        n_particles = self.system.configuration.positions.shape[0]
        n_modes = n_particles * 3
        k_size = self.k_size
        nptk = np.prod(k_size)

        
        # TODO: remove acoustic sum rule
        # self.frequencies[0, :3] = 0
        # self.velocities[0, :3, :] = 0

        omega = 2 * np.pi * self.frequencies
        density = 1. / (np.exp (hbar * omega / k_b / self.system.temperature) - 1.)


        energy_diff = np.zeros ((2, nptk, n_modes, nptk, n_modes))
        energy_diff[1] = np.abs (
            omega[index_k, mu] + omega[:, :, np.newaxis, np.newaxis] - omega[np.newaxis, np.newaxis, :, :])
        energy_diff[0] = np.abs (
            omega[index_k, mu] - omega[:, :, np.newaxis, np.newaxis] - omega[np.newaxis, np.newaxis, :, :])
    
        density_fact = np.zeros ((2, nptk, n_modes, nptk, n_modes))
        density_fact[1] = density[:, :, np.newaxis, np.newaxis] - density[np.newaxis, np.newaxis, :, :]
        density_fact[0] = .5 * (1 + density[:, :, np.newaxis, np.newaxis] + density[np.newaxis, np.newaxis, :, :])
    
        sigma = self.calculate_broadening (
            self.velocities[:, :, np.newaxis, np.newaxis, :] - self.velocities[np.newaxis, np.newaxis, :, :, :])
    
        dirac_delta = np.zeros ((2, nptk, n_modes, nptk, n_modes))
    
        delta_condition_plus = ((omega[:, :, np.newaxis, np.newaxis] != 0) & (
                    omega[np.newaxis, np.newaxis, :, :] != 0)) & (
                                       energy_diff[1, :, :, :, :] <= (
                                       2. * sigma[:, :, :, :]))
        delta_condition_minus = ((omega[:, :, np.newaxis, np.newaxis] != 0) & (
                omega[np.newaxis, np.newaxis, :, :] != 0)) & (
                                        energy_diff[0, :, :, :, :] <= (2. * sigma[:, :, :, :]))
    
        omega_product = omega[:, :, np.newaxis, np.newaxis] * omega[np.newaxis, np.newaxis, :, :]
        coords_plus = np.array (np.argwhere (delta_condition_plus), dtype=int)
        coords_minus = np.array (np.argwhere (delta_condition_minus), dtype=int)

        coords_plus_new = []

        i_kpp = np.zeros ((2, nptk, nptk, 3)).astype (int)
        index_kpp_calc = np.zeros ((2, nptk)).astype (int)

        i_k = np.array (np.unravel_index (index_k, self.k_size), order='C')
        for index_kp in range (np.prod (self.k_size)):
            i_kp = np.array (np.unravel_index (index_kp, self.k_size, order='C'))
            for is_plus in (1, 0):
                # TODO: Umklapp processes are when the reminder is != 0, we could probably separate those
                if is_plus:
                    i_kpp[is_plus, index_k, index_kp, :] = ((i_k + i_kp)) % self.k_size
        
                else:
                    i_kpp[is_plus, index_k, index_kp, :] = ((i_k - i_kp)) % self.k_size
                index_kpp_calc[is_plus, index_kp] = np.ravel_multi_index (
                    i_kpp[is_plus, index_k, index_kp], self.k_size, order='C')
                
        for interaction in np.arange (coords_plus.shape[0]):
            if (coords_plus[interaction, 2] == index_kpp_calc[1, coords_plus[interaction, 0]]):
                coords_plus_new.append (coords_plus[interaction, :])
    
        coords_plus = np.array (coords_plus_new)
    
        coords_minus_new = []
        for interaction in np.arange (coords_minus.shape[0]):
            if (coords_minus[interaction, 2] == index_kpp_calc[0, coords_minus[interaction, 0]]):
                coords_minus_new.append (coords_minus[interaction, :])
    
        coords_minus = np.array (coords_minus_new)
    
        coords = np.array ([coords_minus, coords_plus])
        for is_plus in (1, 0):
        
            if (coords[is_plus].size != 0):
                indexes_reduced = (
                coords[is_plus][:, 0], coords[is_plus][:, 1], coords[is_plus][:, 2], coords[is_plus][:, 3])
                indexes = (np.ones (coords[is_plus][:, 0].shape[0]).astype (int) * is_plus, coords[is_plus][:, 0],
                           coords[is_plus][:, 1], coords[is_plus][:, 2], coords[is_plus][:, 3])
            
                dirac_delta[indexes] = density_fact[indexes] * np.exp (
                    -(energy_diff[indexes]) ** 2 / (sigma[indexes_reduced] ** 2)) / sigma[indexes_reduced] / np.sqrt (
                    np.pi) / (omega_product[indexes_reduced])
                dirac_delta[indexes] /= omega[index_k, mu]
        return dirac_delta, coords

    def project_potential(self, potential, eigenv, chi, is_plus):
        """
        Projection of potential on second and third phonon.
        :param potential: Third order derivative of the potential, rank = (n_replicas, n_particles * 3, n_replicas, n_particles * 3, n_replicas,  n_particles * 3)
        :param eigenv: eigenvector of the dynamical matrix, rank (n_kpoints, n_modes, n_modes)
        :param chi:
        :param n_modes:
        :param nptk:
        :param is_plus:
        :return:
        """
        if is_plus:
            second_eigenv = eigenv
            second_chi = chi
        else:
            second_eigenv = eigenv.conj ()
            second_chi = chi.conj ()
            
        third_eigenv = eigenv.conj()
        third_chi = chi.conj()

        potential = np.einsum('ijkl,mi->jklm', potential, second_chi)
        potential = np.einsum('jklm,ik->jlmi', potential, third_chi)
        potential = np.einsum('jlmi,ikl->jmki', potential, third_eigenv)
        potential = np.einsum('jmki,mlj->kilm', potential, second_eigenv)

        return potential

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

        omega = 2 * np.pi * self.frequencies
        density = 1. / (np.exp (hbar * omega / k_b / self.system.temperature) - 1.)


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
        rlattvec = cellinv * 2 * np.pi * 10.
        chi = np.zeros ((nptk, n_replicas)).astype (complex)

        for index_k in range (np.prod (k_size)):
            i_k = np.array (np.unravel_index (index_k, k_size, order='C'))
            k_point = i_k / k_size
            realq = np.matmul (rlattvec, k_point)
            for l in range (n_replicas):
                sxij = list_of_replicas[l]
                chi[index_k, l] = np.exp (1j * sxij.dot (realq))

        rescaled_potential = self.system.third_order[0] / np.sqrt (masses[ :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        rescaled_potential /= np.sqrt (masses[ np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        rescaled_potential /= np.sqrt (masses[ np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])

        eigenv = np.swapaxes (self.eigenvectors, 1, 2)
        rescaled_potential = rescaled_potential.reshape(n_modes, n_replicas, n_modes, n_replicas, n_modes)
        full_dirac_delta = np.zeros((2, nptk, n_modes, nptk, n_modes, nptk, n_modes))
        projected_potential = np.zeros((2, n_modes, n_modes, nptk, n_modes, nptk)).astype(np.complex)
        for is_plus in (1, 0):
            for i in range(n_particles * 3):
                projected_potential[is_plus, i] = self.project_potential (rescaled_potential[i], eigenv, chi, is_plus)

        for index_k in range (np.prod (k_size)):
            for mu in range (n_modes):
                # print (index_k, mu)
        
                dirac_delta, _ = self.calculate_delta (index_k, mu)
                for is_plus in (1,0):
                    full_dirac_delta[is_plus, index_k, mu] = dirac_delta[is_plus]
        print('phase space completed')

        for index_k in range (np.prod (k_size)):
            for mu in range (n_modes):
                # print (index_k, mu)
                for is_plus in (1,0):
                    first_proj_potential_sq = np.abs (np.einsum ('mijkl,m->ijkl', projected_potential[is_plus], eigenv[index_k, mu, :])) ** 2
    
                    gamma[is_plus, index_k, mu] = np.einsum('ijkl,lkji->', first_proj_potential_sq, full_dirac_delta[is_plus, index_k, mu])
                    ps[is_plus, index_k, mu] = np.einsum('lkji->', full_dirac_delta[is_plus, index_k, mu])

        return gamma[1] * prefactor *  hbarp * np.pi / 4., gamma[0] * prefactor *  hbarp * np.pi / 4., ps[1] / nptk, ps[0] / nptk

    def calculate_broadening(self, velocity):
        cellinv = self.system.configuration.cell_inv
        # armstrong to nanometers
        rlattvec = cellinv * 2 * np.pi * 10.
        
        # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
        base_sigma = ((np.tensordot (velocity, rlattvec / self.k_size,[-1,1])) ** 2).sum(axis=-1)
        base_sigma = np.sqrt (base_sigma / 6.)
        return base_sigma

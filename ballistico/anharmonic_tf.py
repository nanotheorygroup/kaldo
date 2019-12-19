"""
Ballistico
Anharmonic Lattice Dynamics
"""
import numpy as np
import ase.units as units
from ballistico.helpers.tools import timeit, allowed_index_qpp
import logging
import tensorflow as tf


DELTA_THRESHOLD = 2
EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
GAMMATOTHZ = 1e11 * units.mol * EVTOTENJOVERMOL ** 2


class Anharmonic(object):
    def __init__(self, **kwargs):
        self.finite_difference = kwargs.pop('finite_difference')
        self.n_modes = self.finite_difference.n_modes
        self.kpts = np.array(kwargs.pop('kpts'))
        self.n_k_points = np.prod(self.kpts)
        self.frequencies = kwargs.pop('frequencies')
        self.n_phonons = self.n_k_points * self.n_modes
        self.rescaled_eigenvectors = kwargs.pop('rescaled_eigenvectors')
        self.is_gamma_tensor_enabled = kwargs.pop('is_gamma_tensor_enabled')
        self.chi_k = kwargs.pop('chi_k')
        self.velocities = kwargs.pop('velocities')
        self.physical_modes = kwargs.pop('physical_modes')
        self.sigma_in = kwargs.pop('sigma_in')
        self.broadening_shape = kwargs.pop('broadening_shape')

        self.omegas = tf.convert_to_tensor(2 * np.pi * self.frequencies.astype(float))
        self.occupations = tf.convert_to_tensor(kwargs.pop('occupations').astype(float))

    @timeit
    def project_amorphous(self):
        n_replicas = self.finite_difference.n_replicas
        rescaled_eigenvectors = self.rescaled_eigenvectors.astype(float)
        # The ps and gamma matrix stores ps, gamma and then the scattering matrix
        ps_and_gamma = np.zeros((self.n_phonons, 2))
        logging.info('Projection started')
        evect_tf = tf.convert_to_tensor(rescaled_eigenvectors[0])
    
        coords = self.finite_difference.third_order.coords
        data = self.finite_difference.third_order.data
        coords = np.vstack([coords[1], coords[2], coords[0]])
        third_tf = tf.SparseTensor(coords.T, data, (
            self.n_modes * n_replicas, self.n_modes * n_replicas, self.n_modes))
    
        third_tf = tf.sparse.reshape(third_tf, ((self.n_modes * n_replicas) ** 2, self.n_modes))
        for nu_single in range(self.n_phonons):
            out = self.calculate_dirac_delta_amorphous(nu_single)
            if not out:
                continue
            third_nu_tf = tf.sparse.sparse_dense_matmul(third_tf,
                                                        tf.reshape(evect_tf[:, nu_single], ((self.n_modes, 1))))
            third_nu_tf = tf.reshape(third_nu_tf,
                                     (self.n_modes * n_replicas, self.n_modes * n_replicas))
    
            dirac_delta_tf, mup_vec, mupp_vec = out
            scaled_potential_tf = tf.einsum('ij,in,jm->nm', third_nu_tf, evect_tf, evect_tf)
            coords = tf.stack((mup_vec, mupp_vec), axis=-1)
            # pot_times_dirac_tf = tf.SparseTensor(coords, tf.abs(tf.gather_nd(scaled_potential_tf, coords)) ** 2 * dirac_delta_tf, (n_phonons, n_phonons))
    
            pot_times_dirac = tf.reduce_sum(tf.abs(tf.gather_nd(scaled_potential_tf,
                                                                coords)) ** 2 * dirac_delta_tf).numpy() * units._hbar / 8. / self.n_k_points * GAMMATOTHZ
            dirac_delta = tf.reduce_sum(dirac_delta_tf).numpy()
    
            ps_and_gamma[nu_single, 0] = dirac_delta
            ps_and_gamma[nu_single, 1] = pot_times_dirac
            ps_and_gamma[nu_single, 1:] /= self.frequencies.flatten()[nu_single]
    
            THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
            logging.info('calculating third', nu_single, np.round(nu_single / self.n_phonons, 2) * 100,
                      '%')
            logging.info(self.frequencies[0, nu_single], ps_and_gamma[nu_single, 1] * THZTOMEV / (2 * np.pi))
    
        return ps_and_gamma
    
    
    @timeit
    def project_crystal(self):
        is_gamma_tensor_enabled = self.is_gamma_tensor_enabled
        n_replicas = self.finite_difference.n_replicas
        coords = self.finite_difference.third_order.coords
        data = self.finite_difference.third_order.data
        coords = tf.stack([coords[1] * self.n_modes * n_replicas + coords[2], coords[0]], -1)
        third_tf = tf.SparseTensor(coords, data, ((self.n_modes * n_replicas) ** 2, self.n_modes))
        third_tf = tf.cast(third_tf, dtype=tf.complex64)
    
        chi_k = tf.convert_to_tensor(self.chi_k)
        chi_k = tf.cast(chi_k, dtype=tf.complex64)
        evect_tf = tf.convert_to_tensor(self.rescaled_eigenvectors)
        evect_tf = tf.cast(evect_tf, dtype=tf.complex64)
        # The ps and gamma matrix stores ps, gamma and then the scattering matrix
        if is_gamma_tensor_enabled:
            ps_and_gamma = np.zeros((self.n_phonons, 2 + self.n_phonons))
        else:
            ps_and_gamma = np.zeros((self.n_phonons, 2))
        logging.info('Projection started')
        second_minus = tf.math.conj(evect_tf)
        second_minus_chi = tf.math.conj(chi_k)
    
        for nu_single in range(self.n_phonons):
            if nu_single % 200 == 0:
                logging.info('calculating third', nu_single, np.round(nu_single / self.n_phonons, 2) * 100,
                      '%')
            index_k, mu = np.unravel_index(nu_single, (self.n_k_points, self.n_modes), order='C')
    
            third_nu_tf = tf.sparse.sparse_dense_matmul(third_tf, evect_tf[index_k, :, mu, tf.newaxis])
    
            third_nu_tf = tf.cast(
                tf.reshape(third_nu_tf, (n_replicas, self.n_modes, n_replicas, self.n_modes)),
                dtype=tf.complex64)
            third_nu_tf = tf.transpose(third_nu_tf, (0, 2, 1, 3))
            third_nu_tf = tf.reshape(third_nu_tf, (n_replicas * n_replicas, self.n_modes, self.n_modes))
    
    
            for is_plus in (0, 1):
                index_kpp_full = allowed_index_qpp(index_k, is_plus, self.kpts)
                index_kpp_full = tf.cast(index_kpp_full, dtype=tf.int32)
                out = self.calculate_dirac_delta_crystal(index_kpp_full, index_k, mu, is_plus)
                if not out:
                    continue
    
                if is_plus:
    
                    second = evect_tf
                    second_chi = chi_k
                else:
    
                    second = second_minus
                    second_chi = second_minus_chi
    
                third = tf.math.conj(tf.gather(evect_tf, index_kpp_full))
                third_chi = tf.math.conj(tf.gather(chi_k, index_kpp_full))
                dirac_delta, index_kp_vec, mup_vec, index_kpp_vec, mupp_vec = out
    
                # The ps and gamma array stores first ps then gamma then the scattering array
                chi_prod = tf.einsum('kt,kl->ktl', second_chi, third_chi)
                chi_prod = tf.reshape(chi_prod, (self.n_k_points, n_replicas ** 2))
                scaled_potential = tf.tensordot(chi_prod, third_nu_tf, (1, 0))
                scaled_potential = tf.einsum('kij,kim->kjm', scaled_potential, second)
                scaled_potential = tf.einsum('kjm,kjn->kmn', scaled_potential, third)
    
                scaled_potential = tf.gather_nd(scaled_potential, tf.stack([index_kp_vec, mup_vec, mupp_vec], axis=-1))
                pot_times_dirac = tf.abs(
                    scaled_potential) ** 2 * dirac_delta
                if is_gamma_tensor_enabled:
                    # We need to use bincount together with fancy indexing here. See:
                    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
    
                    nup_vec = index_kp_vec * self.n_modes + mup_vec
                    nupp_vec = index_kpp_vec * self.n_modes + mupp_vec
    
                    result = tf.math.bincount(nup_vec, pot_times_dirac, self.n_phonons)
                    if is_plus:
                        ps_and_gamma[nu_single, 2:] -= result
                    else:
                        ps_and_gamma[nu_single, 2:] += result
    
                    result = tf.math.bincount(nupp_vec, pot_times_dirac, self.n_phonons)
                    ps_and_gamma[nu_single, 2:] += result
                ps_and_gamma[nu_single, 0] += tf.reduce_sum(dirac_delta)
                ps_and_gamma[nu_single, 1] += tf.reduce_sum(pot_times_dirac)
            ps_and_gamma[nu_single, 1:] /= self.frequencies.flatten()[nu_single]
            ps_and_gamma[nu_single, 1:] *= units._hbar / 8. / self.n_k_points * GAMMATOTHZ
        return ps_and_gamma
    
    
    def calculate_dirac_delta_crystal(self, index_kpp_full, index_k, mu, is_plus):
        physical_modes = self.physical_modes.reshape((self.n_k_points, self.n_modes))
        if not physical_modes[index_k, mu]:
            return None
        if self.sigma_in:
            sigma_tf = tf.constant(self.sigma_in, dtype=tf.float64)
        else:
            sigma_tf = self.calculate_broadening(index_kpp_full)
        second_sign = (int(is_plus) * 2 - 1)
        omegas_difference = tf.abs(self.omegas[index_k, mu] + second_sign * self.omegas[:, :, tf.newaxis] -
                                   tf.gather(self.omegas, index_kpp_full)[:, tf.newaxis, :])
    
        physical_modes = self.physical_modes.reshape((self.n_k_points, self.n_modes))
    
        condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_tf) & \
                    (physical_modes[:, :, np.newaxis]) & \
                    (physical_modes[index_kpp_full, np.newaxis, :])
        interactions = tf.where(condition)
        if interactions.shape[0] > 0:
            index_kp_vec = tf.cast(interactions[:, 0], dtype=tf.int32)
            index_kpp_vec = tf.gather(index_kpp_full, index_kp_vec)
            mup_vec = tf.cast(interactions[:, 1], dtype=tf.int32)
            mupp_vec = tf.cast(interactions[:, 2], dtype=tf.int32)
            coords_1 = tf.stack((index_kp_vec, mup_vec), axis=-1)
            coords_2 = tf.stack((index_kpp_vec, mupp_vec), axis=-1)
            if sigma_tf.shape != []:
                coords_3 = tf.stack((index_kp_vec, mup_vec, mupp_vec), axis=-1)
                sigma_tf = tf.gather_nd(sigma_tf, coords_3)
            if is_plus:
                dirac_delta_tf = tf.gather_nd(self.occupations, coords_1) - tf.gather_nd(self.occupations, coords_2)
    
            else:
                dirac_delta_tf = 0.5 * (1 + tf.gather_nd(self.occupations, coords_1) + tf.gather_nd(self.occupations, coords_2))
            dirac_delta_tf /= (tf.gather_nd(self.omegas, coords_1) * tf.gather_nd(self.omegas, coords_2))
            dirac_delta_tf = dirac_delta_tf * 1 / tf.sqrt(np.pi * (2 * np.pi * sigma_tf) ** 2) * tf.exp(
                - (self.omegas[index_k, mu] + second_sign * tf.gather_nd(self.omegas, coords_1) - tf.gather_nd(self.omegas, coords_2)) ** 2 / ((2 * np.pi * sigma_tf) ** 2))
            index_kp = index_kp_vec
            mup = mup_vec
            index_kpp = index_kpp_vec
            mupp = mupp_vec
            return tf.cast(dirac_delta_tf, dtype=tf.float32), index_kp, mup, index_kpp, mupp
    
    
    def calculate_dirac_delta_amorphous(self, mu):
        physical_modes = self.physical_modes.reshape((self.n_k_points, self.n_modes))
        if not physical_modes[0, mu]:
            return None
        if self.broadening_shape == 'triangle':
            delta_threshold = 1
        else:
            delta_threshold = DELTA_THRESHOLD
        density_tf = self.occupations
        omega_tf = self.omegas
        sigma_tf = tf.constant(self.sigma_in, dtype=tf.float64)
        for is_plus in (1, 0):
            second_sign = (int(is_plus) * 2 - 1)
            omegas_difference = np.abs(self.omegas[0, mu] + second_sign * self.omegas[0, :, np.newaxis] -
                                       self.omegas[0, np.newaxis, :])
            physical_modes = self.physical_modes.reshape((self.n_k_points, self.n_modes))
            condition = (omegas_difference < delta_threshold * 2 * np.pi * sigma_tf) & \
                    (physical_modes[0, :, np.newaxis]) & \
                    (physical_modes[0, np.newaxis, :])
            interactions = tf.where(condition)
            if interactions.shape[0] > 0:
                # Create sparse index
                mup_vec = interactions[:, 0]
                mupp_vec = interactions[:, 1]
                if is_plus:
                    dirac_delta_tf = tf.gather(density_tf[0], mup_vec) - tf.gather(density_tf[0], mupp_vec)
                else:
                    dirac_delta_tf = 0.5 * (1 + tf.gather(density_tf[0], mup_vec) + tf.gather(density_tf[0], mupp_vec))
                dirac_delta_tf = dirac_delta_tf / tf.gather(omega_tf[0], mup_vec) / tf.gather(omega_tf[0], mupp_vec)
                omegas_difference_tf = tf.abs(self.omegas[0, mu] + second_sign * tf.gather(omega_tf[0], mup_vec) - tf.gather(omega_tf[0],
                                                                                                                              mupp_vec))
    
                if self.broadening_shape == 'gauss':
                    dirac_delta_tf = dirac_delta_tf * 1 / tf.sqrt(np.pi * (2 * np.pi * sigma_tf) ** 2) * np.exp(
                        - omegas_difference_tf ** 2 / ((2 * np.pi * sigma_tf) ** 2))
                elif self.broadening_shape == 'triangle':
                    dirac_delta_tf = dirac_delta_tf * 1. / (2 * np.pi * sigma_tf) * (1 - omegas_difference_tf / (2 * np.pi * sigma_tf))
                else:
                    raise TypeError('Broadening function not implemented.')
    
                try:
                    mup = tf.concat([mup, mup_vec], 0)
                    mupp = tf.concat([mupp, mupp_vec], 0)
                    current_delta = tf.concat([current_delta, dirac_delta_tf], 0)
                except NameError:
                    mup = mup_vec
                    mupp = mupp_vec
                    current_delta = dirac_delta_tf
        try:
            return current_delta, mup, mupp
        except:
            return None
    
    
    def calculate_broadening(self, index_kpp_vec):
        cellinv = self.finite_difference.cell_inv
        k_size = self.kpts
        velocities_tf = tf.convert_to_tensor(self.velocities)
        velocity_difference = velocities_tf[:, :, tf.newaxis, :] - tf.gather(velocities_tf, index_kpp_vec)[:,tf.newaxis, :, :]
        # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
        delta_k = cellinv / k_size
        base_sigma = tf.reduce_sum((tf.tensordot(velocity_difference, delta_k, [-1, 1])) ** 2, axis=-1)
        base_sigma = tf.sqrt(base_sigma / 6.)
        return base_sigma
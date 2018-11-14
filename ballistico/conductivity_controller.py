from ballistico.constants import *
import ballistico.constants as constants
from ballistico.logger import Logger
import numpy as np
from scipy.sparse import csc_matrix
import sys

LENGTH_THREESHOLD = 1e20
THREESHOLD = 1e-20


class ConductivityController (object):
    def __init__(self, phonons):
        self.phonons = phonons
        # self.import_scattering_matrix()
        

    def import_scattering_matrix(self):
        temperature = str (int (self.phonons.temperature))
        filename_gamma = self.phonons.folder + 'T' + temperature + 'K/GGG.Gamma_Tensor'
        gamma_value = []
        row = []
        col = []
        with open (filename_gamma, "rw+") as f:
            for line in f:
                items = line.split ()
                n0 = int (items[0]) - 1
                k0 = int (items[1]) - 1
                n1 = int (items[2]) - 1
                k1 = int (items[3]) - 1
                nu0 = k0 * self.phonons.n_modes + n0
                nu1 = k1 * self.phonons.n_modes + n1
                row.append (nu0)
                col.append (nu1)
                # self.phonons.gamma[nu0, nu1] = float(items[4])
                gamma = float (items[4]) * self.phonons.frequencies[nu1] / (self.phonons.frequencies[nu0] + THREESHOLD) + THREESHOLD
                gamma_value.append (gamma)
        
        self.phonons.gamma = csc_matrix ((gamma_value, (row, col)), shape=(self.phonons.n_phonons, self.phonons.n_phonons),
                                 dtype=np.float32)

    
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
        gamma_unitless = prefactor.dot (self.phonons.gamma)
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
        f_be = 1. / (np.exp (hbar * self.phonons.frequencies / k_b / self.phonons.temperature) - 1. + THREESHOLD)
        c_v = np.zeros ((self.phonons.n_phonons))
        if (is_classical):
            c_v[:] = k_b
        else:
            c_v[:] = hbar ** 2 * f_be[:] * (f_be[:] + 1) * self.phonons.frequencies[:] ** 2 / (k_b * self.phonons.temperature ** 2)
            
        # TODO: get rid of this prefactor
        return 1e21 * c_v
    
    def exact_conductivity(self, is_classical=False, l_x=LENGTH_THREESHOLD, l_y=LENGTH_THREESHOLD, l_z=LENGTH_THREESHOLD, alpha=0, beta=0):
        volume = np.linalg.det(self.phonons.atoms.cell) / 1000.

        length = np.array([l_x, l_y, l_z])
        conductivity_per_mode = np.zeros ((self.phonons.n_phonons))
        # gamma_full = np.diag (1. / (self.tau_zero + THREESHOLD)) - np.array (self.phonons.gamma.toarray ()) + THREESHOLD
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
    
    def calculate_conductivity(self, is_classical, post_processing=None, length=None, converged=False):
        volume = np.linalg.det(self.phonons.atoms.cell) / 1000.
        conductivity_per_mode = np.zeros ((self.phonons.n_k_points, self.phonons.n_modes, 3, 3), dtype=np.complex)
        gamma = self.phonons.gamma
        tau_zero = np.empty_like (gamma)
        tau_zero[(gamma) != 0] = 1 / (gamma[gamma != 0]).astype(np.complex)
        velocities = self.phonons.velocities.reshape(self.phonons.n_k_points, self.phonons.n_modes, 3)
        c_v = self.phonons.c_v.astype(np.complex)
        c_v = c_v.reshape(self.phonons.n_k_points, self.phonons.n_modes)
        for alpha in range (3):
            for beta in range (3):
                conductivity_per_mode[:, :, alpha, beta] += c_v[:, :] * velocities[:, :, beta].conj() * tau_zero[:, :] * velocities[:, :, alpha]
        conductivity_per_mode *= 1 / (volume * self.phonons.n_k_points)
        conductivity_per_mode = conductivity_per_mode.sum (axis=0)
        Logger().info ('\nconductivity = \n' + str (conductivity_per_mode.sum (axis=0)))
        return conductivity_per_mode.sum (axis=0)
    

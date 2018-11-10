import os
import numpy as np
import scipy
import scipy.special
import ballistico.atoms_helper as ath
import ballistico.constants as constants
# import tensorflow as tf
import ballistico.calculator
import sys
from memory_profiler import profile
# tf.enable_eager_execution ()
EIGENVALUES_FILE = 'eigenvalues.npy'
EIGENVECTORS_FILE = 'eigenvectors.npy'
FREQUENCY_K_FILE = 'frequencies.npy'
VELOCITY_K_FILE = 'velocities.npy'
GAMMA_FILE = 'gammas.npy'


class Ballistico (object):
    def __init__(self, atoms, supercell=(1, 1, 1), kpts=(1, 1, 1), second_order=None, third_order=None, is_classic=False, temperature=300):
        # TODO: Keep only the relevant default initializations
        self.atoms = atoms
        self.supercell = np.array(supercell)
        self.k_size = np.array(kpts)
        self.is_classic = is_classic
        self.folder = 'phonons/'
        [self.replicated_atoms, self.list_of_replicas] = \
            ath.replicate_atoms (self.atoms, self.supercell)

        self.n_k_points = np.prod (self.k_size)
        self.n_modes = self.atoms.get_masses().shape[0] * 3
        self.n_phonons = self.n_k_points * self.n_modes
        self._frequencies = None
        self._velocities = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._occupations = None
        self._gamma = None
        self._n_k_points = None
        self._n_modes = None
        self._n_phonons = None
        self.second_order = second_order
        self.third_order = third_order
        self.temperature = temperature
        directory = os.path.dirname (self.folder)
        if not os.path.exists (directory):
            os.makedirs (directory)

    @property
    def frequencies(self):
        return self._frequencies

    @frequencies.getter
    def frequencies(self):
        if self._frequencies is None:
            # try:
            #     self._frequencies = np.load (self.folder + FREQUENCY_K_FILE)
            # except FileNotFoundError as e:
            #     print(e)
            self.calculate_second_all_grid ()
                # np.save (self.folder + FREQUENCY_K_FILE, self._frequencies)
                # np.save (self.folder + VELOCITY_K_FILE, self._velocities)
        return self._frequencies

    @frequencies.setter
    def frequencies(self, new_frequencies):
        self._frequencies = new_frequencies

    @property
    def velocities(self):
        return self._velocities

    @velocities.getter
    def velocities(self):
        if self._velocities is None:
            # try:
            #     self._velocities = np.load (self.folder + VELOCITY_K_FILE)
            # except IOError as e:
            self.calculate_second_all_grid ()
                # np.save (self.folder + VELOCITY_K_FILE, self._velocities)
                # np.save (self.folder + FREQUENCY_K_FILE, self._frequencies)
        return self._velocities

    @velocities.setter
    def velocities(self, new_velocities):
        self._velocities = new_velocities

    @property
    def eigenvalues(self):
        return self._eigenvalues

    @eigenvalues.getter
    def eigenvalues(self):
        if self._eigenvalues is None:
            # self.calculate_second_all_grid ()
            # try:
            #     self._eigenvalues = np.load (self.folder + EIGENVALUES_FILE)
            # except IOError as e:
            self.calculate_second_all_grid ()
                # np.save (self.folder + EIGENVALUES_FILE, self._eigenvalues)
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, new_eigenvalues):
        self._eigenvalues = new_eigenvalues

    @property
    def eigenvectors(self):
        return self._eigenvectors

    @eigenvectors.setter
    def eigenvectors(self, new_eigenvectors):
        self._eigenvectors = new_eigenvectors

    @eigenvectors.getter
    def eigenvectors(self):
        if self._eigenvectors is None:
            # try:
            #     self._eigenvectors = np.load (self.folder + EIGENVECTORS_FILE)
            # except IOError as e:
            self.calculate_second_all_grid ()
                # np.save (self.folder + EIGENVECTORS_FILE, self._eigenvectors)
        return self._eigenvectors

    @property
    def occupations(self):
        return self._occupations

    @occupations.getter
    def occupations(self):
        if self._occupations is None:
            self._occupations = self.calculate_occupations ().squeeze()
        return self._occupations

    @occupations.setter
    def occupations(self, new_occupations):
        self._occupations = new_occupations

    @property
    def gamma(self):
        return self._gamma

    @gamma.getter
    def gamma(self):
        if self._gamma is None:
            # try:
            #     self._gamma = np.load (self.folder + GAMMA_FILE)
            # except IOError as e:
            self.calculate_gamma ()
            #     np.save (self.folder + GAMMA_FILE, self._gamma)
        return np.sum(self._gamma, axis=0)

    @gamma.setter
    def gamma(self, new_gamma):
        self._gamma = new_gamma

    def density_of_states(self, delta=1):
        frequencies = self.frequencies
        k_mesh = self.k_size
        n_modes = frequencies.shape[-1]
        frequencies = frequencies.reshape ((k_mesh[0], k_mesh[1], k_mesh[2], n_modes))
        n_k_points = np.prod (k_mesh)
        # increase_factor = 3
        omega_kl = np.zeros(( n_k_points, n_modes))
        for mode in range(n_modes):
            omega_kl[:, mode] = frequencies[...,mode].flatten()
        # Energy axis and dos
        omega_e = np.linspace (0., np.amax (omega_kl) + 5e-3, num=100)
        dos_e = np.zeros_like (omega_e)
        # Sum up contribution from all q-points and branches
        for omega_l in omega_kl:
            diff_el = (omega_e[:, np.newaxis] - omega_l[np.newaxis, :]) ** 2
            dos_el = 1. / (diff_el + (0.5 * delta) ** 2)
            dos_e += dos_el.sum (axis=1)
        dos_e *= 1. / (n_k_points * np.pi) * 0.5 * delta
        return omega_e, dos_e

    # @profile
    def calculate_second_all_grid(self):
        k_size = self.k_size
        atoms = self.atoms
        second_order = self.second_order
        list_of_replicas = self.list_of_replicas
        frequencies, eigenvalues, eigenvectors, velocities = ballistico.calculator.calculate_second_all_grid(k_size, atoms, second_order, list_of_replicas)
        self._frequencies = frequencies
        self._eigenvalues = eigenvalues
        self._velocities = velocities
        self._eigenvectors = eigenvectors
    
    def calculate_occupations(self):
        temp = self.temperature
        omega = 2 * np.pi * self.frequencies
        eigenvalues = omega * constants.hbar / constants.k_b
        density = np.zeros_like(eigenvalues)
        if self.is_classic == False:
            density[omega != 0] = 1. / (
                    np.exp (constants.hbar * omega[omega != 0] / constants.k_b / self.temperature) - 1.)
        else:
            density[omega != 0] = temp / omega[omega != 0] / constants.hbar * constants.k_b
        return density

    def calculate_gamma(self, sigma_in=None):
        self._gamma = ballistico.calculator.calculate_gamma(
            self.atoms,
            self.frequencies,
            self.velocities,
            self.occupations,
            self.k_size,
            self.eigenvectors,
            self.list_of_replicas,
            self.third_order,
            sigma_in)

    

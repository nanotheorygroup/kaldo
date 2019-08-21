import os


import numpy as np
import ase.units as units
from .harmoniccontroller import HarmonicController
from .anharmoniccontroller import AnharmonicController
from .conductivitycontroller import ConductivityController

EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J
THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15

FREQUENCY_THRESHOLD = 0.001


OCCUPATIONS_FILE = 'occupations.npy'
C_V_FILE = 'c_v.npy'
FOLDER_NAME = 'output'
GAMMA_FILE = 'gamma.npy'
PS_FILE = 'phase_space.npy'








class Phonons:
    def __init__(self, finite_difference, is_classic, temperature, folder=FOLDER_NAME, kpts = (1, 1, 1), sigma_in=None, frequency_threshold=FREQUENCY_THRESHOLD, broadening_shape='gauss'):
        self.finite_difference = finite_difference
        self.atoms = finite_difference.atoms
        self.supercell = np.array (finite_difference.supercell)
        self.kpts = np.array (kpts)
        self.is_classic = is_classic
        self.n_k_points = np.prod (self.kpts)
        self.n_modes = self.atoms.get_masses ().shape[0] * 3
        self.n_phonons = self.n_k_points * self.n_modes
        self.temperature = temperature

        self._occupations = None
        self._full_scattering_plus = None
        self._full_scattering_minus = None
        self._k_points = None
        self.folder_name = folder
        self.sigma_in = sigma_in
        self._c_v = None
        self.is_able_to_calculate = True
        self.broadening_shape = broadening_shape

        if self.is_classic:
            classic_string = 'classic'
        else:
            classic_string = 'quantum'
        folder = self.folder_name + '/' + str (self.temperature) + '/' + classic_string + '/'
        if self.sigma_in is not None:
            folder += 'sigma_in_' + str (self.sigma_in).replace ('.', '_') + '/'
        folders = [self.folder_name, folder]
        for folder in folders:
            if not os.path.exists (folder):
                os.makedirs (folder)
        if frequency_threshold is not None:
            self.frequency_threshold = frequency_threshold
        else:
            self.frequency_threshold = FREQUENCY_THRESHOLD
        self.replicated_cell = self.finite_difference.replicated_atoms.cell
        self.list_of_replicas = self.finite_difference.list_of_replicas()
        self._ps = None
        self._gamma = None
        self._gamma_tensor = None
        self._harmonic_controller = HarmonicController(self)
        self._anharmonic_controller = AnharmonicController(self)



    @property
    def k_points(self):
        return self._harmonic_controller.k_points

    @property
    def dynmat(self):
        return self._harmonic_controller.dynmat

    @property
    def frequencies(self):
        return self._harmonic_controller.frequencies

    @property
    def eigenvalues(self):
        return self._harmonic_controller.eigenvalues

    @property
    def eigenvectors(self):
        return self._harmonic_controller.eigenvectors

    @property
    def velocities(self):
        return self._harmonic_controller.velocities

    @property
    def velocities_AF(self):
        return self._harmonic_controller.velocities_AF

    @property
    def dos(self):
        return self._harmonic_controller.dos


    @property
    def occupations(self):
        return self._occupations

    @occupations.getter
    def occupations(self):
        if self._occupations is None:
            try:
                folder = self.folder_name
                folder += '/' + str(self.temperature) + '/'
                if self.is_classic:
                    folder += 'classic/'
                else:
                    folder += 'quantum/'
                self._occupations = np.load (folder + OCCUPATIONS_FILE)
            except FileNotFoundError as e:
                print(e)
        if self._occupations is None:
            frequencies = self.frequencies

            temp = self.temperature * KELVINTOTHZ
            density = np.zeros_like(frequencies)
            physical_modes = frequencies > self.frequency_threshold

            if self.is_classic is False:
                density[physical_modes] = 1. / (np.exp(frequencies[physical_modes] / temp) - 1.)
            else:
                density[physical_modes] = temp / frequencies[physical_modes]
            self.occupations = density
        return self._occupations

    @occupations.setter
    def occupations(self, new_occupations):
        folder = self.folder_name
        folder += '/' + str(self.temperature) + '/'
        if self.is_classic:
            folder += 'classic/'
        else:
            folder += 'quantum/'
        np.save (folder + OCCUPATIONS_FILE, new_occupations)
        self._occupations = new_occupations


    @property
    def c_v(self):
        return self._c_v

    @c_v.getter
    def c_v(self):
        if self._c_v is None:
            try:
                folder = self.folder_name
                folder += '/' + str(self.temperature) + '/'
                if self.is_classic:
                    folder += 'classic/'
                else:
                    folder += 'quantum/'
                self._c_v = np.load (folder + C_V_FILE)
            except FileNotFoundError as e:
                print(e)
        if self._c_v is None:
            frequencies = self.frequencies
            c_v = np.zeros_like (frequencies)
            physical_modes = frequencies > self.frequency_threshold
            temperature = self.temperature * KELVINTOTHZ

            if (self.is_classic):
                c_v[physical_modes] = KELVINTOJOULE
            else:
                f_be = self.occupations
                c_v[physical_modes] = KELVINTOJOULE * f_be[physical_modes] * (f_be[physical_modes] + 1) * self.frequencies[physical_modes] ** 2 / \
                                      (temperature ** 2)
            self.c_v = c_v
        return self._c_v

    @c_v.setter
    def c_v(self, new_c_v):
        folder = self.folder_name
        folder += '/' + str(self.temperature) + '/'
        if self.is_classic:
            folder += 'classic/'
        else:
            folder += 'quantum/'
        np.save (folder + C_V_FILE, new_c_v)
        self._c_v = new_c_v


    @property
    def gamma(self):
        return self._gamma

    @gamma.getter
    def gamma(self):
        if self._gamma is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._gamma = np.load (folder + GAMMA_FILE)
            except FileNotFoundError as e:
                print(e)
                self._anharmonic_controller.calculate_gamma(is_gamma_tensor_enabled=False)
        return self._gamma

    @gamma.setter
    def gamma(self, new_gamma):
        folder = self.folder_name
        folder += '/'
        np.save (folder + GAMMA_FILE, new_gamma)
        self._gamma = new_gamma

    @property
    def ps(self):
        return self._ps

    @ps.getter
    def ps(self):
        if self._ps is None:
            try:
                folder = self.folder_name
                folder += '/'
                self._ps = np.load (folder + PS_FILE)
            except FileNotFoundError as e:
                print(e)
                self._anharmonic_controller.calculate_gamma(is_gamma_tensor_enabled=False)
        return self._ps

    @ps.setter
    def ps(self, new_ps):
        folder = self.folder_name
        folder += '/'
        np.save (folder + GAMMA_FILE, new_ps)
        self._ps = new_ps

    @property
    def gamma_tensor(self):
        if self._gamma_tensor is None:
            self._anharmonic_controller.calculate_gamma(is_gamma_tensor_enabled=True)
        return  self._gamma_tensor


    @staticmethod
    def apply_boundary_with_cell(cell, cellinv, dxij):
        # exploit periodicity to calculate the shortest distance, which may not be the one we have
        sxij = dxij.dot(cellinv)
        sxij = sxij - np.round(sxij)
        dxij = sxij.dot(cell)
        return dxij


    def calculate_conductivity(self, method='rta', max_n_iterations=None):
        if max_n_iterations and method != 'sc':
            raise TypeError('Only self consistent method support n_iteration parameter')

        conductivity_controller = ConductivityController(self)
        if method == 'rta':
            conductivity = conductivity_controller.calculate_conductivity_rta()
        elif method == 'af':
            conductivity = conductivity_controller.calculate_conductivity_AF()
        elif method == 'inverse':
            conductivity = conductivity_controller.calculate_conductivity_inverse()
        elif method == 'sc':
            conductivity = conductivity_controller.calculate_conductivity_sc(max_n_iterations)
        else:
            raise TypeError('Conductivity method not recognized')
        return conductivity
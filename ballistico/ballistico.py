import os
import numpy as np
import scipy
import scipy.special
import ballistico.atoms_helper as ath
import ballistico.constants as constants
# import tensorflow as tf
import ballistico.calculator
from ballistico.phonons import Phonons
import sys
from memory_profiler import profile
# tf.enable_eager_execution ()

class Ballistico (Phonons):
    def __init__(self,  atoms, supercell=(1, 1, 1), kpts=(1, 1, 1), is_classic=False, temperature=300, second_order=None, third_order=None):
        super(self.__class__, self).__init__(atoms=atoms, supercell=supercell, kpts=kpts)
        self.second_order = second_order
        self.third_order = third_order
        directory = os.path.dirname (self.folder)
        if not os.path.exists (directory):
            os.makedirs (directory)

    @property
    def frequencies(self):
        return super().frequencies

    @frequencies.getter
    def frequencies(self):
        if super (self.__class__, self).frequencies is not None:
            return super (self.__class__, self).frequencies
        self.calculate_second_all_grid ()
        return self._frequencies

    @frequencies.setter
    def frequencies(self, new_frequencies):
        Phonons.frequencies.fset(self, new_frequencies)
        
    @property
    def velocities(self):
        return super().velocities
    
    @velocities.getter
    def velocities(self):
        if super (self.__class__, self).velocities is not None:
            return super (self.__class__, self).velocities
        self.calculate_second_all_grid ()
        return self._velocities

    @velocities.setter
    def velocities(self, new_velocities):
        Phonons.velocities.fset(self, new_velocities)

    @property
    def eigenvalues(self):
        return super().eigenvalues
    
    @eigenvalues.getter
    def eigenvalues(self):
        if super (self.__class__, self).eigenvalues is not None:
            return super (self.__class__, self).eigenvalues
        self.calculate_second_all_grid ()
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, new_eigenvalues):
        Phonons.eigenvalues.fset(self, new_eigenvalues)

    @property
    def eigenvectors(self):
        return super().eigenvectors
    
    @eigenvectors.setter
    def eigenvectors(self, new_eigenvectors):
        Phonons.eigenvectors.fset(self, new_eigenvectors)
        
    @eigenvectors.getter
    def eigenvectors(self):
        if super (self.__class__, self).eigenvectors is not None:
            return super (self.__class__, self).eigenvectors
        else:
            self.calculate_second_all_grid ()
        return self._eigenvectors

    @property
    def dos(self):
        return super().dos
    
    @dos.setter
    def dos(self, new_dos):
        Phonons.dos.fset(self, new_dos)

    @dos.getter
    def dos(self):
        if super (self.__class__, self).dos is not None:
            return super (self.__class__, self).dos
        self.calculate_dos ()
        return self._dos

    @property
    def occupations(self):
        return super().occupations
    
    @occupations.setter
    def occupations(self, new_occupations):
        Phonons.occupations.fset(self, new_occupations)

    @occupations.getter
    def occupations(self):
        if super (self.__class__, self).occupations is not None:
            return super (self.__class__, self).occupations
        self.calculate_occupations ()
        return self._occupations

    @property
    def gamma(self):
        return super ().gamma

    @gamma.setter
    def gamma(self, new_gamma):
        Phonons.gamma.fset(self, new_gamma)
        
    @gamma.getter
    def gamma(self):
        if super (self.__class__, self).gamma is not None:
            return super (self.__class__, self).gamma
        self.calculate_gamma ()
        return self._gamma

    def calculate_dos(self, delta=1):
        self.dos = ballistico.calculator.calculate_density_of_states(
            self.frequencies,
            self.k_size,
            delta)

    # @profile
    def calculate_second_all_grid(self):
        frequencies, eigenvalues, eigenvectors, velocities = ballistico.calculator.calculate_second_all_grid(
            self.k_size,
            self.atoms,
            self.second_order,
            self.list_of_replicas)
        self.frequencies = frequencies
        self.eigenvalues = eigenvalues
        self.velocities = velocities
        self.eigenvectors = eigenvectors
    
    def calculate_occupations(self):
        self.occupations = ballistico.calculator.calculate_occupations(self.frequencies, self.temperature, self.is_classic)

    def calculate_gamma(self, sigma_in=None):
        self.gamma = ballistico.calculator.calculate_gamma(
            self.atoms,
            self.frequencies,
            self.velocities,
            self.occupations,
            self.k_size,
            self.eigenvectors,
            self.list_of_replicas,
            self.third_order,
            sigma_in)

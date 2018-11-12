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
        folder = type (self).__name__
        if not os.path.exists (folder):
            os.makedirs (folder)


    @property
    def frequencies(self):
        return super().frequencies

    @frequencies.getter
    def frequencies(self):
        if super (self.__class__, self).frequencies is not None:
            return super (self.__class__, self).frequencies
        frequencies, eigenvalues, eigenvectors, velocities = ballistico.calculator.calculate_second_all_grid(
            self.k_size,
            self.atoms,
            self.second_order,
            self.list_of_replicas)
        self.frequencies = frequencies
        self.eigenvalues = eigenvalues
        self.velocities = velocities
        self.eigenvectors = eigenvectors
        return frequencies

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
        frequencies, eigenvalues, eigenvectors, velocities = ballistico.calculator.calculate_second_all_grid(
            self.k_size,
            self.atoms,
            self.second_order,
            self.list_of_replicas)
        self.frequencies = frequencies
        self.eigenvalues = eigenvalues
        self.velocities = velocities
        self.eigenvectors = eigenvectors
        return velocities

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
        frequencies, eigenvalues, eigenvectors, velocities = ballistico.calculator.calculate_second_all_grid(
            self.k_size,
            self.atoms,
            self.second_order,
            self.list_of_replicas)
        self.frequencies = frequencies
        self.eigenvalues = eigenvalues
        self.velocities = velocities
        self.eigenvectors = eigenvectors
        return eigenvalues

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
        frequencies, eigenvalues, eigenvectors, velocities = ballistico.calculator.calculate_second_all_grid(
            self.k_size,
            self.atoms,
            self.second_order,
            self.list_of_replicas)
        self.frequencies = frequencies
        self.eigenvalues = eigenvalues
        self.velocities = velocities
        self.eigenvectors = eigenvectors
        return eigenvectors

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
        # TODO: delta needs to be set by the instance
        delta = 1
        dos = ballistico.calculator.calculate_density_of_states(
            self.frequencies,
            self.k_size,
            delta)
        self.dos = dos
        return dos

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
        occupations = ballistico.calculator.calculate_occupations(self.frequencies, self.temperature, self.is_classic)
        self.occupations = occupations
        return occupations

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
        #TODO: Find a way to pass sigma in when needed
        gamma = ballistico.calculator.calculate_gamma(
            self.atoms,
            self.frequencies,
            self.velocities,
            self.occupations,
            self.k_size,
            self.eigenvectors,
            self.list_of_replicas,
            self.third_order)
        self.gamma = gamma
        return gamma
    
    
    
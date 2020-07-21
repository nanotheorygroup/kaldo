import numpy as np
from kaldo.observables.observable import Observable


class PhysicalMode(Observable):

    def __init__(self, frequency, min_frequency=0, max_frequency=None, is_nw=False):
        self.frequency = frequency
        self.n_phonons = self.frequency.size
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.is_nw = is_nw


    def calculate(self, is_loading=False):
        physical_mode = np.ones_like(self.frequency.reshape(self.n_phonons), dtype=bool)
        if self.min_frequency != 0:
            physical_mode = physical_mode & (self.frequency.reshape(self.n_phonons) > self.min_frequency)
        if self.max_frequency is not None:
            physical_mode = physical_mode & (self.frequency.reshape(self.n_phonons) < self.max_frequency)
        if self.is_nw:
            physical_mode[:4] = False
        else:
            physical_mode[:3] = False
        return physical_mode
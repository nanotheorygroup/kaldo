
import numpy as np
import ase.units as units

from kaldo.helpers.logger import get_logger
logging = get_logger()

KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J


def calculate_population(phonons):
    frequency = phonons.frequency.reshape((phonons.n_k_points, phonons.n_modes))
    temp = phonons.temperature * KELVINTOTHZ
    density = np.zeros((phonons.n_k_points, phonons.n_modes))
    physical_mode = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    if phonons.is_classic is False:
        density[physical_mode] = 1. / (np.exp(frequency[physical_mode] / temp) - 1.)
    else:
        density[physical_mode] = temp / frequency[physical_mode]
    return density


def calculate_heat_capacity(phonons):
    frequency = phonons.frequency
    c_v = np.zeros_like(frequency)
    physical_mode = phonons.physical_mode
    temperature = phonons.temperature * KELVINTOTHZ
    if (phonons.is_classic):
        c_v[physical_mode] = KELVINTOJOULE
    else:
        f_be = phonons.population
        c_v[physical_mode] = KELVINTOJOULE * f_be[physical_mode] * (f_be[physical_mode] + 1) * phonons.frequency[
            physical_mode] ** 2 / \
                              (temperature ** 2)
    return c_v


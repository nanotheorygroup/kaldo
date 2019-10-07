import numpy as np
import ase.units as units
from .tools import lazy_property, is_calculated
EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12



def calculate_occupations(phonons):
    frequencies = phonons.frequencies
    temp = phonons.temperature * KELVINTOTHZ
    density = np.zeros_like(frequencies)
    physical_modes = frequencies > phonons.frequency_threshold
    if phonons.is_classic is False:
        density[physical_modes] = 1. / (np.exp(frequencies[physical_modes] / temp) - 1.)
    else:
        density[physical_modes] = temp / frequencies[physical_modes]
    return density


def calculate_c_v(phonons):
    frequencies = phonons.frequencies
    c_v = np.zeros_like (frequencies)
    physical_modes = frequencies > phonons.frequency_threshold
    temperature = phonons.temperature * KELVINTOTHZ

    if (phonons.is_classic):
        c_v[physical_modes] = KELVINTOJOULE
    else:
        f_be = phonons.occupations
        c_v[physical_modes] = KELVINTOJOULE * f_be[physical_modes] * (f_be[physical_modes] + 1) * phonons.frequencies[physical_modes] ** 2 / \
                              (temperature ** 2)
    return c_v


import numpy as np
import ase.units as units

# 1 thz = 0.00414 mev
petahertz = units.J * (2 * np.pi * units._hbar) * 1e15


Bohr = units.Bohr
Rydberg = units.Rydberg
mol = units.mol

mass_factor = units._me * 2 * units.mol * 1e+3

thzoverjoule = petahertz / units.J / 1000
kelvinoverjoule = units.kB / units.J


tenjovermol = 10 * units.J / units.mol

gamma_coeff = units._hbar * units.mol ** 3 / units.J ** 2 * 1e9 * np.pi / 4. / 16 / np.pi ** 4


davide_coeff = 1 / (2 * np.pi) * gamma_coeff * petahertz  # result in mev

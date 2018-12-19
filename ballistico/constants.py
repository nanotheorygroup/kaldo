import numpy as np
import ase.units as units

# 1 thz = 0.00414 mev
petahertz = 4.13566553853599


Bohr = units.Bohr
Rydberg = units.Rydberg
mol = units.mol

mass_factor = 1.8218779 * 6.022e-4

thzoverjoule = petahertz / units.J / 1000
kelvinoverjoule = units.kB / units.J


tenjovermol = 10 * units.J / units.mol

gamma_coeff = 1 / (2 * np.pi) * 1 / (4. * np.pi) ** 3 * mol ** 3 * thzoverjoule * 1 / units.J * 1 / units.J / 1000

davide_coeff = 1 / (2 * np.pi) * gamma_coeff * petahertz  # result in mev

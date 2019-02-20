import numpy as np
import ase.units as units

# 1 thz = 0.00414 mev
petahertz = units.J * (2 * np.pi * units._hbar) * 1e15


Bohr = units.Bohr
Rydberg = units.Rydberg
mol = units.mol

hbar = units._hbar


thzoverjoule = petahertz / units.J / 1000
kelvinoverjoule = units.kB / units.J




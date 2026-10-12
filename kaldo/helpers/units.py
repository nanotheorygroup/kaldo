"""Unit conversion factors shared by kALDo's NumPy-only code paths."""

import ase.units as units

# eV to kALDo's internal 10 J/mol energy unit. An eV/Angstrom^2 force
# constant divided by atomic masses in amu and multiplied by this factor is a
# dynamical-matrix element in (rad/ps)^2.
EV_TO_10J_PER_MOL = units.mol / (10 * units.J)

# Boltzmann constant in J/K.
KB_J_PER_K = units.kB / units.J

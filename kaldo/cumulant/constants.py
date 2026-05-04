from __future__ import annotations

import numpy as np


# --- Physical constants (SI) ------------------------------------------------
# These match the cumulant Julia LDT reference (CODATA 2018). They are not
# imported from ``ase.units`` because that module uses an internal unit
# system (eV-Å convention with ``Ang = 1.0``) which is incompatible with
# the SI-meter ``ANG = 1e-10`` convention the cumulant kernels rely on for
# their ``EV/ANG**2``-style unit conversions in ``dynmat_and_eigs``.
HBAR = 1.054_571_817e-34
KB = 1.380_649e-23
EV = 1.602_176_634e-19
AMU = 1.660_539_068_92e-27
ANG = 1e-10

# Default mass for Ne-solid reference runs (matches Ethan's thermo_out_full).
NE_MASS_AMU = 20.1797

# Frequencies below this THz value are treated as acoustic / unphysical modes
# and excluded from Bose–Einstein sums.
FREQ_TOL_THZ = 1e-3

# eV-scale Boltzmann constant, useful when everything else is in eV/atom.
KB_eV_per_K = KB / EV  # ≈ 8.617e-5 eV/K

# Example: aluminum FCC, ASE EMT calculator
# Computes: 2nd and 3rd order force constants using both serial and parallel methods
# Uses: ASE EMT (built-in, no external files required)
#
# Purpose: Demonstrate that kaldo's parallel third-order calculator produces
#          numerically identical force constants to the serial method.
#
# For the parallel third.calculate() call, pass the EMT class itself (not an
# instance) so each worker process can construct its own isolated calculator.
# For serial, pass a constructed EMT() instance as usual.

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from kaldo.forceconstants import ForceConstants


# --- System setup ---

# Al FCC conventional cubic cell: 4 atoms, a=4.05 Angstrom
atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
supercell = (3, 3, 3)

# --- Serial calculation ---
# Standard finite-difference: one process, one EMT instance.
print("### Computing force constants (serial)")

forceconstants_serial = ForceConstants(
    atoms=atoms,
    supercell=supercell,
    folder='fc_al_serial',
)
forceconstants_serial.second.calculate(
    calculator=EMT(),
    delta_shift=1e-4,
)
# Pass a constructed EMT() instance for the serial run (n_threads=1, default).
forceconstants_serial.third.calculate(
    calculator=EMT(),
    delta_shift=1e-4,
    n_threads=1,
)

print("Serial third-order force constants computed and stored in fc_al_serial/\n\n")

# --- Parallel calculation ---
# Each worker process calls EMT() to create its own independent calculator.
# EMT takes no arguments, so we can pass the class directly as a factory.
# For LAMMPS or other file-based calculators, use a lambda that sets a unique
# tmp directory per process, e.g.:
#   calculator=lambda: LAMMPS(tmp_dir=f'/tmp/kaldo_{os.getpid()}')
# For NEP that reads a file, but doesn't use file-based calculations,
# pass in the nep text file path as usual, no tmp_dir required.
#   calculator=lambda: NEP('nep.txt')

print("### Computing force constants (parallel)")

forceconstants_parallel = ForceConstants(
    atoms=atoms,
    supercell=supercell,
    folder='fc_al_parallel',
)

forceconstants_parallel.second.calculate(
    calculator=EMT(),
    delta_shift=1e-4,
)

# Pass EMT (the class) as the factory for parallel workers.
# n_threads=None uses all available CPU cores; set an explicit integer to limit.
forceconstants_parallel.third.calculate(
    calculator=EMT,
    delta_shift=1e-4,
    n_threads=2,
)

print("Parallel third-order force constants computed and stored in fc_al_parallel/\n\n")

# --- Numerical comparison ---
# Both paths should produce identical sparse tensors up to floating-point noise.

print("### Comparing serial and parallel third-order tensors")

serial_dense   = forceconstants_serial.third.value.todense()
parallel_dense = forceconstants_parallel.third.value.todense()

max_abs_diff = np.max(np.abs(serial_dense - parallel_dense))
rel_diff = max_abs_diff / (np.max(np.abs(serial_dense)) + 1e-30)

print(f"Max absolute difference : {max_abs_diff:.2e}  eV/Angstrom^3")
print(f"Max relative difference : {rel_diff:.2e}")

if max_abs_diff < 1e-6:
    print("PASS: Serial and parallel results are numerically identical.")
else:
    print("FAILED: Serial and parallel differ.\n"\
         "Please ensure the tests in test/test_parallel_third.py is passing.")

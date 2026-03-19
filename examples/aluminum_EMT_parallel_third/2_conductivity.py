# Example: aluminum FCC, ASE EMT calculator
# Computes: phonon dispersion and thermal conductivity from serial and parallel IFCs
# Prerequisite: run 1_force_constants.py first to generate fc_al_serial/ and fc_al_parallel/
#
# Note: EMT is a simplified empirical potential not calibrated for anharmonic
# properties, so computed values will differ from experiment. The key result here
# is that serial and parallel methods produce the *same* thermal conductivity.

import numpy as np
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
from kaldo.controllers import plotter
import os


supercell = (3, 3, 3)

# k-point mesh for phonon BZ integration.
k_points = 7


def compute_conductivity(fc_folder, phonon_folder, label):
    print(f"\n{label}\n")

    forceconstants = ForceConstants.from_folder(
        folder=fc_folder,
        supercell=supercell,
        format='numpy',
    )

    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[k_points, k_points, k_points],
        is_classic=False,  # quantum Bose-Einstein statistics
        temperature=300,   # 300 K
        folder=phonon_folder,
        storage='formatted',
    )

    # Plot dispersion and DOS for the first run; skip for the second to save time.
    if not os.path.exists(phonon_folder):
        os.makedirs(phonon_folder)
        plotter.plot_dispersion(phonons, with_velocity=True, is_showing=False)
        plotter.plot_dos(phonons, is_showing=False)

    # Relaxation-time approximation (RTA) -- fastest, good first estimate
    rta = Conductivity(phonons=phonons, method='rta', storage='numpy',).conductivity.sum(axis=0)
    rta_kappa = np.mean(np.diag(rta))
    print(f"  RTA conductivity        : {rta_kappa:.3f} W/m/K")
    print(f"  RTA matrix (W/m/K):\n{rta}")

    # Direct matrix inversion (exact BTE solution within the harmonic IFC space)
    inv = Conductivity(phonons=phonons, method='inverse', storage='numpy',).conductivity.sum(axis=0)
    inv_kappa = np.mean(np.diag(inv))
    print(f"\n  Inverse (exact) kappa   : {inv_kappa:.3f} W/m/K")
    print(f"  Inverse matrix (W/m/K):\n{inv}")

    return {'rta': rta_kappa, 'inverse': inv_kappa}


# Compute from serial force constants
kappa_serial = compute_conductivity(
    fc_folder='fc_al_serial',
    phonon_folder='ALD_al_serial',
    label='Thermal Conductivity from Serial IFCs',
)

# Compute from parallel force constants
kappa_parallel = compute_conductivity(
    fc_folder='fc_al_parallel',
    phonon_folder='ALD_al_parallel',
    label='Thermal Conductivity from Parallel IFCs',
)

# --- Summary ---
print("  Serial vs Parallel thermal conductivity (W/m/K)")
print(f"  {'Method':<12}  {'Serial':>10}  {'Parallel':>10}  {'Abs Diff':>10}  {'Rel Diff (%)':>10}")
print(f"  {'-'*70}")
for method in ('rta', 'inverse'):
    s = kappa_serial[method]
    p = kappa_parallel[method]
    print(f"  {method.upper():<12}  {s:>10.4f}  {p:>10.4f}  {abs(s-p):>10.2e} {100 * abs(s-p)/s:>10.2e}")

print("\n ## Ab initio value for lattice thermal conductivity of Al ##")
print("\t5.6 W/mK (GGA)  5.8 W/mK (LDA) ")
print("\tWang, Y., Lu, Z., & Ruan, X. (2016) JAP 119(22) ")

# Save for the plotting script
np.save('kappa_serial.npy',   np.array([kappa_serial['rta'], kappa_serial['inverse']]))
np.save('kappa_parallel.npy', np.array([kappa_parallel['rta'], kappa_parallel['inverse']]))

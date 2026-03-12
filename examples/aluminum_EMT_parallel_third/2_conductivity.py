# Example: aluminum FCC, ASE EMT calculator
# Computes: phonon dispersion and thermal conductivity from serial and parallel IFCs
# Prerequisite: run 1_force_constants.py first to generate fc_al_serial/ and fc_al_parallel/
#
# Al thermal conductivity literature reference (experiment): ~237 W/m/K at 300 K
# (Touloukian et al., Thermophysical Properties of Matter, Vol. 1, 1970)
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
# 7x7x7 gives good convergence for FCC Al; increase for higher precision.
k_points = 7


def compute_conductivity(fc_folder, phonon_folder, label):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

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
    rta = Conductivity(phonons=phonons, method='rta').conductivity.sum(axis=0)
    rta_kappa = np.mean(np.diag(rta))
    print(f"  RTA conductivity        : {rta_kappa:.3f} W/m/K")
    print(f"  RTA matrix (W/m/K):\n{rta}")

    # Self-consistent solution of BTE
    sc = Conductivity(phonons=phonons, method='sc', n_iterations=20).conductivity.sum(axis=0)
    sc_kappa = np.mean(np.diag(sc))
    print(f"\n  Self-consistent kappa   : {sc_kappa:.3f} W/m/K")
    print(f"  SC matrix (W/m/K):\n{sc}")

    # Direct matrix inversion (exact BTE solution within the harmonic IFC space)
    inv = Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0)
    inv_kappa = np.mean(np.diag(inv))
    print(f"\n  Inverse (exact) kappa   : {inv_kappa:.3f} W/m/K")
    print(f"  Inverse matrix (W/m/K):\n{inv}")

    return {'rta': rta_kappa, 'sc': sc_kappa, 'inverse': inv_kappa}


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
print(f"\n{'=' * 60}")
print("  Summary: Serial vs Parallel thermal conductivity (W/m/K)")
print(f"{'=' * 60}")
header = f"  {'Method':<12}  {'Serial':>10}  {'Parallel':>10}  {'Diff':>10}"
print(header)
print(f"  {'-'*50}")
for method in ('rta', 'sc', 'inverse'):
    s = kappa_serial[method]
    p = kappa_parallel[method]
    print(f"  {method.upper():<12}  {s:>10.4f}  {p:>10.4f}  {abs(s-p):>10.2e}")

print(f"\n  Experimental Al kappa at 300 K: ~237 W/m/K")
print(f"  (EMT is not calibrated for anharmonic properties;")
print(f"   use a more accurate potential for quantitative comparison.)")

# Save for the plotting script
np.save('kappa_serial.npy',   np.array([kappa_serial['rta'],   kappa_serial['sc'],   kappa_serial['inverse']]))
np.save('kappa_parallel.npy', np.array([kappa_parallel['rta'], kappa_parallel['sc'], kappa_parallel['inverse']]))

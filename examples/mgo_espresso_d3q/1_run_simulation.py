#!/usr/bin/env python
"""
MgO kALDo calculation demonstrating NAC correction
Calculates harmonic properties and thermal conductivity.

Adjust k points in "Replica Settings" under the Configuration header, lower
will run quicker but less accurately.
"""
from kaldo.controllers.plotter import plot_dispersion, plot_crystal
from kaldo.forceconstants import ForceConstants
from kaldo.conductivity import Conductivity
from kaldo.phonons import Phonons
import numpy as np

# Model parameters-
k = 9
kpt_array = np.array([k, k, k])
nrep = int(5)
nrep_third = int(5)
supercell = np.array([nrep, nrep, nrep])
third_supercell = np.array([nrep_third, nrep_third, nrep_third])

# Simulation Settings
temperature = 300
classical_stats = False
# True/False means Classical/Quantum heat capacity + phonon populations

# Correction-dependent Settings
# True = using NAC
# False = without NAC
nac_folder_dic = {
	True: 'forces',
	False: 'forces_no_charges'
}

for io_folder in ['forces', 'forces_no_charges']:
    print_string = "WITHOUT" if io_folder.endswith("no_charges") else "WITH"

    print(f"Loading force constants {print_string.lower()} NAC correction...")
    forceconstant = ForceConstants.from_folder(
        folder=io_folder,
        supercell=supercell,
        third_supercell=third_supercell,
        only_second=True, # set to true if you only want harmonic properties
        is_acoustic_sum=True,
        format='shengbte-d3q')

    print("Creating phonons object...")
    phonons = Phonons(
        forceconstants=forceconstant,
        kpts=kpt_array,
        is_classic=False,
        temperature=300,
        folder=io_folder,
        is_unfolding=True,
        storage='numpy')
    print("\tdone!")

    # Calculate Thermal Conductivity
    print("Calculating thermal conductivity")
    print("\tRTA method...")
    rta_matrix = Conductivity(phonons=phonons, method='rta').conductivity.sum(axis=0)

    print("\tInverse method...")
    inv_matrix = Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0)

    print("\n\n" + "="*48)
    print(f"RESULTS {print_string} NAC CORRECTION")
    print("Thermal Conductivity (W/m/K):")
    print(f"\tRTA method:")
    print(f"    {rta_matrix}\n")
    print(f"\tInverse method:")
    print(f"    {inv_matrix}\n")
    print("\n\n" + "="*48)

    # We'll generate the dispersion here and combine the plots later!
    atoms = forceconstant.atoms
    cell = atoms.cell
    lat = cell.get_bravais_lattice()
    path = cell.bandpath('GXUGLW', npoints=150) # generate the k-path
    print(f"Calculating dispersion...")
    print(f"\tPath: {path}")
    # Save path for reference, if desired (e.g. for use with matdyn.x)
    # np.savetxt('data/kpath.txt', path.kpts)

    plot_dispersion(phonons,
        is_showing=False,
        folder=io_folder,
        manually_defined_path=path)
    print(f"Dispersion saved in {io_folder}!")


#!/usr/bin/env python
"""
MgO kALDo calculation demonstrating NAC correction
Calculates harmonic properties and thermal conductivity.

Adjust k points in "Replica Settings" under the Configuration header, lower
will run quicker but less accurately.
"""

# Edit the number of threads here to control the threading. 
nthr = 1
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(nthr)
tf.config.threading.set_inter_op_parallelism_threads(nthr)
# For GPU control, uncomment this code block!
# gpu_index = 0 # which gpu to select.
# These indices match with the output from the nvidia-smi command on your cli
# devices = tf.config.list_physical_devices("GPU")
# tf.config.set_visible_devices(devices[gpu_index])

import numpy as np
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.controllers.plotter import plot_dos
from kaldo.conductivity import Conductivity

# ============================================================
# Configuration
# ============================================================
# Dispersion settings
n_points = 200
path_string = 'GXUGLW'
unfold_bool = True
output_folder = 'plots/'

# Replica settings
k = 15
kpt_array = np.array([k, k, k])
nrep = int(9)
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
	'True': 'forces',
	'False': 'forces_without_
}

for is_nac_corrected in [True, False]:
	io_folder = nac_folder_dic[is_nac_corrected]
	
	# ============================================================
	# Create Phonon objects from Force Constants with/without NAC 
	# ============================================================
	print("Loading force constants with NAC correction...")
	forceconstant = ForceConstants.from_folder(
	    folder=io_folder,
	    supercell=supercell,
	    third_supercell=third_supercell,
	    only_second=False, # set to true if you only want harmonic properties
	    is_acoustic_sum=True,
	    format='shengbte-d3q')

	print("Creating phonons object...")
	phonons = Phonons(
	    forceconstants=forceconstant,
	    kpts=kpt_array,
	    is_classic=False,
	    temperature=300,
	    folder=io_folder,
	    is_unfolding=unfold_bool,
	    storage='numpy')

	# ============================================================
	# Calculate Harmonic Properties
	# ============================================================
	print("Calculating density of states...")
	plot_dos(phonons,
		is_showing=False,
		folder=io_folder)
	print('\tDOS figure saved!')

	# Calculate dispersion
	atoms = forceconstant.atoms
	cell = atoms.cell
	lat = cell.get_bravais_lattice()
	path = cell.bandpath(pathstring, npoints=npoints)
	print(f"Calculating dispersion along path {pathstring}...")
	print(f'\tPath: {path}')

	# Save path for reference, if desired
	# np.savetxt('tools/kpath.txt', path.kpts)

	# Calculate frequencies and velocities along path
	plot_dispersion(phonons,
		is_showing=False,
		folder=io_folder,
		manually_defined_path=path)

	# ============================================================
	# Calculate Thermal Conductivity
	# ============================================================
	print("Calculating thermal conductivity...")
	print("\tRTA method...")
	rta_matrix = Conductivity(phonons=phonons, method='rta').conductivity.sum(axis=0)

	print("\tInverse method...")
	inv_matrix = Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0)


	print_string = "WITH" if is_nac_corrected else "WITHOUT"
	print("\n\n" + "="*48)
	print(f"RESULTS {print_string} NAC CORRECTION")
	print("="*48)
	print("Thermal Conductivity (W/m/K):")
	print(f"\tRTA method:")
	print(f"    {rta_matrix}\n")
	print(f"\tInverse method:")
	print(f"    {inv_matrix}\n")
	print("="*48 + "\n\n")

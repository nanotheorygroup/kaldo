from kaldo.observables.secondorder import SecondOrder
from kaldo.observables.thirdorder import ThirdOrder
from ase.calculators.lammpslib import LAMMPSlib
from kaldo.forceconstants import ForceConstants
from kaldo.conductivity import Conductivity
from kaldo.phonons import Phonons
from ase.io import read
import numpy as np


def phononics(ge_concentration='0'):

	# Set up Forceconstants Object
	folder_string = 'structures/1728_atom/aSiGe_C'+ge_concentration+'/'
	atoms = read(folder_string+'replicated_atoms.xyz')

	forceconstants = ForceConstants(atoms=atoms,
		folder=folder_string+'/ald')

	# Calculate the 2nd + 3rd Forceconstant matrices
	lammps_inputs = {'lmpcmds': ["pair_style tersoff",
				"pair_coeff * * forcefields/SiCGe.tersoff Si(D) Ge"],
				"log_file" : "min.log",
				"keep_alive":True}
	calc = LAMMPSlib(**lammps_inputs)
	second = forceconstants.second
	second.load(folder=folder_string+'/ald')
	print('Third')
#	second.calculate(calculator=calc)
	third = forceconstants.third
	third.calculate(calculator=calc, is_verbose=True)

	# Create Phonon Object
	phonons = Phonons(forceconstants=forceconstants,
		is_classic=False, # quantum stats
		temperature=300, # 300 K
		folder=folder_string,
		third_bandwidth=0.5/4.135, # 0.5 eV smearing
		broadening_shape='gauss') # shape of smearing

	# Phononic Data
	## These save files to help us look at phononic properties
	## with our plotter (4_plotting.py). These properties are "lazy" which
	## means they won't be calculated unless explicitly called, or required
	## by another calculation.

	np.save('frequency', phonons.frequency)
	np.save('bandwidth', phonons.bandwidth)
	np.save('diffusivity', phonons.diffusivity)
	#np.save('participation', phonons.participation_ratio)

	# Conductivity Object
	# This will print out the total conductivity and save the contribution per mode
	conductivity = Conductivity(phonons=phonons, method='qhgk').conductivity
	np.save('conductivity', 1/3*np.einsum('iaa->i', conductivity))
	print("Thermal Conductivity (W/m/K): %.3f" % conductivity.sum(axis=0).diagonal().mean())

desired_concentrations = [0.1]
for c in desired_concentrations:
	phononics(str(int(c*100)))




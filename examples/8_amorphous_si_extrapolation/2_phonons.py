from ballistico.forceconstants import ForceConstants
from ballistico.conductivity import Conductivity
from ballistico.phonons import Phonons
from ase.io import read,write
import numpy as np
random_seed = 7793

def phoninc_information(ge_concentration=0)

	# Read in 1728 atom system
	atoms = read('structures/replicated_atoms.xyz',format='xyz')
	# Swap in Ge atoms
	if ge_concentration != 0:
		sym = atoms.get_chemical_symbols()
		n_ge_atoms = int(np.round(ge_concentration * len(sym), 0))
		rng = np.random.default_rng(seed=random_seed)
		id = rng.choice(len(atoms), size=n_ge_atoms, replace=False)
		symbols[id] = 'Ge'
		atoms.set_chemical_symbols(symbols.tolist())
	ge_concentration = string(ge_concentration)


	forceconstants = ForceConstants.from_files(atoms=atoms,
		folder='structures/1728_atom_aSi/'+ge_concentration+'/ald')
	phonons = Phonons(forceconstants=forceconstants,
		is_classic=False, # quantum stats
		temperature=300, # 300 K
		folder='structures/ald',
		third_bandwidth=0.5/4.135, #0.5 eV
		broadening_shape='gauss')

	# Phononic Data #############################
	# These save files to help us look at phononic properties
	# with our plotter (3_plotting.py)
	np.save('frequency', phonons.frequency)
	np.save('bandwidth', phonons.bandwidth)
	np.save('diffusivity', phonons.diffusivity)
	#np.save('participation', phonons.participation_ratio)

	# Conductivity Properties #########################
	# This will print out the total conductivity and save
	# the contribution per mode
	conductivity = Conductivity(phonons=phonons, method='qhgk').conductivity
	print("Thermal Conductivity (W/m/K): %.3f" % conductivity.sum(axis=0).diagonal().mean())
	np.save('conductivity', 1/3*np.einsum('iaa->i', conductivity))




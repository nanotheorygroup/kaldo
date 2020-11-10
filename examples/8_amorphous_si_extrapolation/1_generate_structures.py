from kaldo.forceconstants import ForceConstants
from ase.calculators.lammpslib import LAMMPSlib
from kaldo.conductivity import Conductivity
from ase.optimize import LBFGSLineSearch
from kaldo.phonons import Phonons
from ase.io import read,write
import numpy as np
random_seed = 7793

def change_concentration(ge_concentration=0):
	# Read in 1728 atom system
	atoms = read('structures/1728_atom/aSi.xyz', format='xyz')

	# Swap in Ge atoms
	if ge_concentration != 0:
		symbols = np.array(atoms.get_chemical_symbols())
		n_ge_atoms = int(np.round(ge_concentration * len(symbols), 0))
		rng = np.random.default_rng(seed=random_seed)
		id = rng.choice(len(atoms), size=n_ge_atoms, replace=False)
		symbols[id] = 'Ge'
		atoms.set_chemical_symbols(symbols.tolist())
	ge_concentration = str(int(ge_concentration*100))

	# Minimize structure
	lammps_inputs = {'lmpcmds': ["pair_style tersoff",
				"pair_coeff * * forcefields/SiCGe.tersoff Si(D) Ge"],
				"log_file" : "lammps.log",
				"keep_alive":True}
	calc = LAMMPSlib(**lammps_inputs)
	atoms.set_calculator(calc)
	atoms.pbc = True
	search = LBFGSLineSearch(atoms)
	search.run(fmax=.001)
	write('structures/1728_atom/aSiGe_C'+ge_concentration+'.xyz',
		 search.atoms, format='xyz')
	


desired_concentrations = [0.1, 0.2]
for c in desired_concentrations:
	change_concentration(c)

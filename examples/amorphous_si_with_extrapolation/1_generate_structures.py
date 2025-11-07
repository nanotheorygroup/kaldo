from ase.calculators.lammpslib import LAMMPSlib
from ase.optimize import LBFGSLineSearch
from ase.io import read,write
import numpy as np
import os

random_seed = 7793
desired_concentrations = [0.1]

def change_concentration(ge_concentration=0):
	# Read in 1728 atom system
	atoms = read('structures/1728_atom/aSi.xyz')

	# Swap in Ge atoms
	if ge_concentration != 0:
		symbols = np.array(atoms.get_chemical_symbols())
		n_ge_atoms = int(np.round(ge_concentration * len(symbols), 0))
		rng = np.random.default_rng(seed=random_seed)
		id = rng.choice(len(atoms), size=n_ge_atoms, replace=False)
		symbols[id] = 'Ge'
		atoms.set_chemical_symbols(symbols.tolist())
	ge_concentration = str(int(ge_concentration*100))
	folder_string = 'structures/1728_atom/aSiGe_C'+str(ge_concentration)
	if not os.path.exists(folder_string):
		os.makedirs(folder_string)

	# Minimize structure - LAMMPS + ASE
	lammps_inputs = {'lmpcmds': ["pair_style tersoff",
				"pair_coeff * * forcefields/SiCGe.tersoff Si(D) Ge"],
				"log_file" : "min.log",
				"keep_alive":True}
	calc = LAMMPSlib(**lammps_inputs)
	atoms.set_calculator(calc)
	atoms.pbc = True
	search = LBFGSLineSearch(atoms)
	search.run(fmax=.001)
	write('structures/1728_atom/aSiGe_C'+str(ge_concentration)+'/replicated_atoms.xyz',
		 search.atoms)

for c in desired_concentrations:
	change_concentration(c)


# Example: silicon bulk, Tersoff potential
# Computes: force constant potential for silicon bulk (2 atoms per cell)
# Uses: hiPhive, ASE, LAMMPS
# External files: Si.tersoff

from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.io import write, read
from hiphive import ClusterSpace
from hiphive import StructureContainer, ForceConstantPotential
from hiphive.fitting import Optimizer
from hiphive.structure_generation import generate_rattled_structures
import numpy as np
import os

### Generate reference structures and perform force calculations ####
# a0: lattice parameter (Angstrom)
# rattle_std: standard deviation of the distribution of displacements
# number_of_structures: number of structures generated with standard rattle (random displacement) 
# and used in force calculations

a0 = 5.432
rattle_std = 0.01
number_of_structures = 100

# Define input information for the ase LAMMPSlib calculator
# Terosff potential for silicon (Si.tersoff) is used for this example.
cmds = ['pair_style tersoff',
        'pair_coeff * * forcefields/Si.tersoff Si']
calc = LAMMPSlib(lmpcmds=cmds, log_file='lammps_si_bulk_hiphive.log',
                 keep_alive=True)

calc = calc = LAMMPSlib(lmpcmds=cmds, log_file='lammps_si_bulk_hiphive.log',
                        keep_alive=True)
if not os.path.exists('structures'):
    os.mkdir('structures/')

# Generate initial structure
atoms_prim = bulk('Si', 'diamond', a=a0)
n_prim = atoms_prim.get_masses().shape[0]
# Replicate the unit cell 'nrep'=3 times
nrep = 3
supercell = np.array([nrep, nrep, nrep])
initial_structure = atoms_prim.copy() * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])
replicated_structure = initial_structure.copy()

######## Set up the standard rattle (random displacement) scheme #######
# seed_int_struct = np.random.randint(1, 100000, size=1, dtype=np.int64)[0]

# Fix random seed is used here so results can be checked against the reference. 
seed_int_struct = 36000

# Generate and save rattled structures
rattled_structures = generate_rattled_structures(
    initial_structure, number_of_structures, rattle_std, seed=seed_int_struct)
print('Random Seed: %d' % seed_int_struct)

for i in range(len(rattled_structures)):
    single_structures_fname = 'structures/rattled_structures_' + str(i) + '.extxyz'
    try:
        structure = rattled_structures[i]
        structure.calc = None
        new_structure = read(single_structures_fname)
        structure.arrays['forces'] = new_structure.arrays['forces']
        structure.arrays['displacements'] = new_structure.arrays['displacements']
    except FileNotFoundError:
        structure = rattled_structures[i]
        displacements = structure.positions - initial_structure.positions
        structure.set_calculator(calc)
        print('calculating force', i)
        structure.new_array('forces', structure.get_forces())
        structure.new_array('displacements', displacements)
        write(single_structures_fname, structure)
    structure.positions = initial_structure.positions
print('Rattle structures generation is completed!')
print('\n')

################ Set up StructureContainer #####################

# Build StructureContainer
# cutoffs: 2nd and 3rd order cutoffs (Angstrom)
cutoffs = [4.0, 4.0]
cs = ClusterSpace(rattled_structures[0], cutoffs)
sc = StructureContainer(cs)
for structure in rattled_structures:
    sc.add_structure(structure)
print('\n')
print('Structure container building ups are completed!')

############# Develop force constant potential model #################

# Fit models for 2nd and 3rd order 
# fit_method: str to specify which fitting method to use
opt = Optimizer(sc.get_fit_data(), fit_method='least-squares', seed=seed_int_struct)
opt.train()
print(opt)
fcp = ForceConstantPotential(cs, opt.parameters)

########### Generate and save force constant of desired orders #############

fcs = fcp.get_force_constants(initial_structure)
# Set up hiPhive fcs folder
hiphive_filename = 'hiPhive_si_bulk/'
primitive_fname = 'hiPhive_si_bulk/atom_prim.xyz'
if not os.path.isdir(hiphive_filename):
    os.mkdir(hiphive_filename)
write(primitive_fname, atoms_prim, format='xyz')
write('hiPhive_si_bulk/replicated_atoms.xyz', replicated_structure, format='xyz')
fcs.write(hiphive_filename + '/' + 'model2.fcs')
fcs.write(hiphive_filename + '/' + 'model3.fcs')
print('\n')
print('IFC computations are completed!')

# Example 3.1: silicon bulk, LDA pseudo potential 
# Computes: force constant potential for silicon bulk (2 atoms per cell)
# Uses: hiPhive, ASE, Quantum ESPRESSO (QE)
# External files: Si.pz-n-kjpaw_psl.0.1.UPF

from ase.build import bulk
from ase.calculators.espresso import Espresso
from ase.io import write, read
from hiphive import ClusterSpace
from hiphive.fitting import Optimizer
from hiphive import StructureContainer, ForceConstantPotential
from hiphive.structure_generation import generate_rattled_structures
import numpy as np
import os

### Generate reference structures and perform force calculations ####
# a0: lattice parameter (Angstrom)
# rattle_std: standard deviation of the distribution of displacements
# number_of_structures: number of structures generated with standard rattle (random displacement) 
# and used in force calculations

a0 = 5.516326
rattle_std = 0.01
number_of_structures =50

# Define input information for the ASE Espresso calculator
# LDA pseudopotential for si bulk (Si.pz-n-kjpaw_psl.0.1.UPF) is used for this example.
calculator_inputs = {'pseudopotentials': {'Si': 'Si.pz-n-kjpaw_psl.0.1.UPF'},
                     'tstress': True,
                     'tprnfor': True,
                     'outdir':'qe_at_gamma',
                     'input_data':
                         {'system': {'ecutwfc': 16.0},
                          'electrons': {'conv_thr': 1e-10, 'mixing_beta': 0.5},
                          'disk_io': 'low',
                          'pseudo_dir': 'potentials'},
                     'kpts': [1, 1, 1],
                     'koffset': [1, 1, 1]
                     }
calc = Espresso(**calculator_inputs)
if not os.path.exists('structures'):
    os.mkdir('structures/')

# Generate initali structure
atoms_prim = bulk('Si', 'diamond', a=a0)
n_prim = atoms_prim.get_masses().shape[0]
# Replicate the unit cell 'nrep'=3 times
nrep = 3
supercell = np.array([nrep, nrep, nrep])
initial_structure = atoms_prim.copy() * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])
replicated_structure = initial_structure.copy()

######## Set up the standard rattle (random displacement) scheme #######
# seed_int_struct = np.random.randint(1, 100000, size=1, dtype=np.int64)[0]

# Fix random seed here is used so results can be checked against the reference. 
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
opt = Optimizer(sc.get_fit_data(), fit_method = 'least-squares',seed=seed_int_struct)
opt.train()
print(opt)
fcp = ForceConstantPotential(cs, opt.parameters)

########### Generate and save force constant of desired orders #############

fcs = fcp.get_force_constants(initial_structure)
# Set up hiphive fcs folder
hiphive_filename = 'hiphive_si_bulk/'
primitive_fname = 'hiphive_si_bulk/atom_prim.xyz'
if not os.path.isdir(hiphive_filename):
    os.mkdir(hiphive_filename)
write(primitive_fname, atoms_prim, format='xyz')
write('hiphive_si_bulk/replicated_atoms.xyz',replicated_structure,format='xyz')
fcs.write(hiphive_filename + '/' + 'model2.fcs')
fcs.write(hiphive_filename + '/' + 'model3.fcs')
print('\n')
print('ICF computations are completed!')

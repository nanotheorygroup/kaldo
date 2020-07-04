"""
Prepare training structures for si-bulk using an LAMMPSlib calculator and a
general rattle approach for generating displacements.
"""
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.io import write, read
from hiphive import ClusterSpace
from hiphive.fitting import Optimizer
from hiphive import StructureContainer, ForceConstantsPotential
from hiphive.structure_generation import generate_rattled_structures
import numpy as np
import os

# Denote parameters

a0 = 5.432
dim = 3
rattle_amplitude = 0.01
number_of_structures = 100
potential_file = '/forcefields/Si.tersoff'
cmds = ['pair_style tersoff',
        'pair_coeff * * forcefields/Si.tersoff Si']
calc = LAMMPSlib(lmpcmds=cmds, log_file='lammps_si_bulk_hiphive.log',
                 keep_alive=True)
os.mkdir('structures/')


# Generate rattled structures
atoms_prim = bulk('Si', 'diamond', a=a0)
n_prim = atoms_prim.get_masses().shape[0]
supercell = np.array([3, 3, 3])
initial_structure = atoms_prim.copy() * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])
seed_int_struct = np.random.randint(1, 100000, size=1, dtype=np.int64)[0]
rattled_structures = generate_rattled_structures(
    initial_structure, number_of_structures, rattle_amplitude, seed=seed_int_struct)
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

"""
Build up StructureContainer.
"""
# Build StructureContainer
cs = ClusterSpace(rattled_structures[0], [4.0, 4.0])
sc = StructureContainer(cs)
for structure in rattled_structures:
    sc.add_structure(structure)
print('\n')
print('Structure container building ups are completed!')

"""
Compute Interatomic Force Constants (IFC) for desired orders.
"""

# Fit models for 3rd order
opt = Optimizer(sc.get_fit_data(), seed=seed_int_struct)
opt.train()
print(opt)
fcp = ForceConstantsPotential(cs, opt.parameters)

# Derive and save force constants from force potential
fcs = fcp.get_force_constants(initial_structure)

# Set up hiphive fcs folder
hiphive_filename = 'hiphive_si_bulk/'
primitive_fname = 'hiphive_si_bulk/atom_prim.xyz'
if not os.path.isdir(hiphive_filename):
    os.mkdir(hiphive_filename)
write(primitive_fname, atoms_prim, format='xyz')
fcs.write(hiphive_filename + '/' + 'model2.fcs')
fcs.write(hiphive_filename + '/' + 'model3.fcs')
print('\n')
print('ICF computations are completed!')

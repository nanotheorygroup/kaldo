"""
Prepare training structures for si-bulk using an LAMMPSlib calculator and a
general rattle approach for generating displacements.

"""
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.io import write
from hiphive.structure_generation import generate_rattled_structures
import numpy as np
import os

# Denote parameters

a0 = 5.432
dim = 3
rattle_amplitude = 0.01
number_of_structures = 50
potential_file = '/forcefields/Si.tersoff'
cmds = ['pair_style tersoff',
        'pair_coeff * * forcefields/Si.tersoff Si']

calc = LAMMPSlib(lmpcmds=cmds, log_file='lammps_si_bulk_hiphive.log',
                 keep_alive=True)
primitive_fname = 'structures/POSCAR'
structures_fname = 'structures/rattled_structures.extxyz'
supercell = np.array([3,3,3])

# Generate rattled structures

atoms_prim = bulk('Si', 'diamond', a=a0)
n_prim = atoms_prim.get_masses().shape[0]
atoms_ideal = atoms_prim.copy()*(supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])
rattled_structures = generate_rattled_structures(
    atoms_ideal, number_of_structures, rattle_amplitude)

structures = []
for structure in rattled_structures:
    structure.set_calculator(calc)
    forces = structure.get_forces()
    displacements = structure.positions - atoms_ideal.get_positions()
    structure.new_array('forces', forces)
    structure.new_array('displacements', displacements)
    structure.calc = None
    structure.positions = atoms_ideal.get_positions()
    structures.append(structure)

# save structures

if not os.path.isdir(os.path.dirname(structures_fname)):
    os.mkdir(os.path.dirname(structures_fname))
write(primitive_fname, atoms_prim)
write(structures_fname, structures)


print('\n')
print('Rattle structures generation is completed!')
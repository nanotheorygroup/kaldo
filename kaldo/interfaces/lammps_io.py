from ase.calculator.lammpslib import convert_cell
import numpy as np


def write_lammps_dump(atoms, step, fname, write_type='a+'):
'''
    Writes a lammps-dump file with a triclinic simulation box.
'''
    # Input information
    n_atoms = len(atoms)
    cell = atoms.cell
    positions = atoms.positions
    symbols = np.array(atoms.get_chemical_symbols())
    unique = np.unique(symbols)
    for i, type in enumerate(unique):
        indices = np.where(symbols==type)
        symbols[indices] = i+1

    # Rotate
    cell, trans = convert_cell(cell)
    positions = np.matmul(trans, positions.T).T
    xx, xy, xz, _, yy, yz, _, _, zz = cell.flatten()

    # Box Bounds (lammps formula: https://docs.lammps.org/Howto_triclinic.html)
    xlob = 0 + np.min([0, xy, xz, xy+xz])
    xhib = xx + np.max([0, xy, xz, xy+xz])
    ylob = 0 + np.min([0, yz])
    yhib = yy + np.max([0, yz])

    # Stringify info
    box_bounds = ' xy xz yz '
    box_bounds += '\n%16.16e %16.16e %16.16e' % (xlob, xhib, xy)
    box_bounds += '\n%16.16e %16.16e %16.16e' % (ylob, yhib, xz)
    box_bounds += '\n%16.16e %16.16e %16.16e' % (0., zz, yz)
    print(box_bounds)
    # Atoms
    ids = (np.arange(n_atoms) + 1)
    atoms_stack = ' id type x y z\n'
    for i in range(n_atoms):
        atoms_stack += ' %i %s %.5f %.5f %.5f\n'\
                    % (ids[i], symbols[i], *positions[i, :])

    items_dic = {
        'TIMESTEP': '\n%i' % step,
        'NUMBER OF ATOMS': '\n%i' % n_atoms,
        'BOX BOUNDS': box_bounds,
        'ATOMS': atoms_stack
        }

    items_list = ['TIMESTEP', 'NUMBER OF ATOMS', 'BOX BOUNDS', 'ATOMS']
    with open(fname, write_type) as f:
        for ITEM in items_list:
            f.write('ITEM: %s %s \n' % (ITEM, items_dic[ITEM]))


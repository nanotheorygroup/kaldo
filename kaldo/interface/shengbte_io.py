"""
kaldo
Anharmonic Lattice Dynamics
"""
import pandas as pd
import numpy as np
from kaldo.phonons import Phonons
from ase.units import Rydberg, Bohr
from ase import Atoms
import os
import re
from kaldo.grid import Grid, wrap_coordinates
from sparse import COO

BUFFER_PLOT = .2
SHENG_FOLDER_NAME = 'sheng_bte'
SHENGBTE_SCRIPT = 'ShengBTE.x'


def divmod(a, b):
    #TODO: Remove this method
    q = a / b
    r = a % b
    return q, r


def split_index(index, nx, ny, nz):
    #TODO: Remove this method
    tmp1, ix = divmod(index - 1, nx, )
    tmp2, iy = divmod(tmp1, ny)
    iatom, iz = divmod(tmp2, nz)
    ix = ix + 1
    iy = iy + 1
    iz = iz + 1
    iatom = iatom + 1
    return int(ix), int(iy), int(iz), int(iatom)


def read_second_order_matrix(folder, supercell):
    second_file = folder + '/' + 'FORCE_CONSTANTS_2ND'
    if not os.path.isfile(second_file):
        second_file = folder + '/' + 'FORCE_CONSTANTS'
    with open(second_file, 'r') as file:
        first_row = file.readline()
        first_row_split = re.findall(r'\d+', first_row)
        n_rows = int(list(map(int, first_row_split))[0])
        n_replicas = np.prod(supercell)
        n_unit_atoms = int(n_rows / n_replicas)
        n_replicas = np.prod(supercell)

        second_order = np.zeros((n_unit_atoms, 3, supercell[0],
                                 supercell[1], supercell[2], n_unit_atoms, 3))

        line = file.readline()
        while line:
            try:
                i, j = np.fromstring(line, dtype=np.int, sep=' ')
            except ValueError as err:
                print(err)
            i_ix, i_iy, i_iz, i_iatom = split_index(i, supercell[0], supercell[1], supercell[2])
            j_ix, j_iy, j_iz, j_iatom = split_index(j, supercell[0], supercell[1], supercell[2])
            for alpha in range(3):
                if (i_ix == 1) and (i_iy == 1) and (i_iz == 1):
                    second_order[i_iatom - 1, alpha, j_iz - 1, j_iy - 1, j_ix - 1, j_iatom - 1, :] = \
                        np.fromstring(file.readline(), dtype=np.float, sep=' ')
                else:
                    file.readline()
            line = file.readline()
    return second_order


def read_second_order_qe_matrix(filename):
    file = open('%s' % filename, 'r')
    ntype, n_atoms, ibrav = [int(x) for x in file.readline().split()[:3]]
    if (ibrav == 0):
        file.readline()
    for i in np.arange(ntype):
        file.readline()
    for i in np.arange(n_atoms):
        file.readline()
    polar = file.readline()
    if ("T" in polar):
        for i in np.arange(3):
            file.readline()
        for i in np.arange(n_atoms):
            file.readline()
            for j in np.arange(3):
                file.readline()
    supercell = [int(x) for x in file.readline().split()]
    second = np.zeros((3, 3, n_atoms, n_atoms, supercell[0], supercell[1], supercell[2]))
    for i in np.arange(3 * 3 * n_atoms * n_atoms):
        alpha, beta, i_at, j_at = [int(x) - 1 for x in file.readline().split()]
        for j in np.arange(supercell[0] * supercell[1] * supercell[2]):
            readline = file.readline().split()
            t1, t2, t3 = [int(x) - 1 for x in readline[:3]]
            second[alpha, beta, i_at, j_at, t1, t2, t3] = float(readline[3]) * (Rydberg / (
                                Bohr ** 2))
    second = second.transpose(2,0,4,5,6,3,1)
    return second, supercell




def read_third_order_matrix(third_file, atoms, supercell, order='C'):
    n_unit_atoms = atoms.positions.shape[0]
    n_replicas = np.prod(supercell)
    third_order = np.zeros((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
    second_cell_list = []
    third_cell_list = []

    current_grid = Grid(supercell, order=order).grid(is_wrapping=True)
    list_of_index = current_grid
    list_of_replicas = list_of_index.dot(atoms.cell)
    with open(third_file, 'r') as file:
        line = file.readline()
        n_third = int(line)
        for i in range(n_third):
            file.readline()
            file.readline()
            second_cell_position = np.fromstring(file.readline(), dtype=np.float, sep=' ')
            second_cell_index = second_cell_position.dot(np.linalg.inv(atoms.cell)).round(0).astype(int)
            second_cell_list.append(second_cell_index)

            # create mask to find the index
            second_cell_id = (list_of_index[:] == second_cell_index).prod(axis=1)
            second_cell_id = np.argwhere(second_cell_id).flatten()

            third_cell_position = np.fromstring(file.readline(), dtype=np.float, sep=' ')
            third_cell_index = third_cell_position.dot(np.linalg.inv(atoms.cell)).round(0).astype(int)
            third_cell_list.append(third_cell_index)

            # create mask to find the index
            third_cell_id = (list_of_index[:] == third_cell_index).prod(axis=1)
            third_cell_id = np.argwhere(third_cell_id).flatten()

            atom_i, atom_j, atom_k = np.fromstring(file.readline(), dtype=np.int, sep=' ') - 1
            for _ in range(27):
                values = np.fromstring(file.readline(), dtype=np.float, sep=' ')
                alpha, beta, gamma = values[:3].round(0).astype(int) - 1
                third_order[atom_i, alpha, second_cell_id, atom_j, beta, third_cell_id, atom_k, gamma] = values[
                        3]

    third_order = third_order.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas *
                                       n_unit_atoms * 3))
    return third_order


def read_third_order_matrix_2(third_file, atoms, third_supercell, order='C'):
    supercell = third_supercell
    n_unit_atoms = atoms.positions.shape[0]
    n_replicas = np.prod(supercell)
    current_grid = Grid(third_supercell, order=order).grid(is_wrapping=True)
    list_of_index = current_grid
    list_of_replicas = list_of_index.dot(atoms.cell)
    replicated_cell = atoms.cell * supercell
    # replicated_cell_inv = np.linalg.inv(replicated_cell)

    coords = []
    data = []
    second_cell_positions = []
    third_cell_positions = []
    atoms_coords = []
    sparse_data = []
    with open(third_file, 'r') as file:
        line = file.readline()
        n_third = int(line)
        for i in range(n_third):
            file.readline()
            file.readline()

            second_cell_position = np.fromstring(file.readline(), dtype=np.float, sep=' ')
            second_cell_positions.append(second_cell_position)
            d_1 = list_of_replicas[:, :] - second_cell_position[np.newaxis, :]
            # d_1 = wrap_coordinates(d_1,  replicated_cell, replicated_cell_inv)

            mask_second = np.linalg.norm(d_1, axis=1) < 1e-5
            second_cell_id = np.argwhere(mask_second).flatten()

            third_cell_position = np.fromstring(file.readline(), dtype=np.float, sep=' ')
            third_cell_positions.append(third_cell_position)
            d_2 = list_of_replicas[:, :] - third_cell_position[np.newaxis, :]
            # d_2 = wrap_coordinates(d_2,  replicated_cell, replicated_cell_inv)
            mask_third = np.linalg.norm(d_2, axis=1) < 1e-5
            third_cell_id = np.argwhere(mask_third).flatten()

            atom_i, atom_j, atom_k = np.fromstring(file.readline(), dtype=np.int, sep=' ') - 1
            atoms_coords.append([atom_i, atom_j, atom_k])
            small_data = []
            for _ in range(27):

                values = np.fromstring(file.readline(), dtype=np.float, sep=' ')
                alpha, beta, gamma = values[:3].round(0).astype(int) - 1
                coords.append([atom_i, alpha, second_cell_id, atom_j, beta, third_cell_id, atom_k, gamma])
                data.append(values[3])
                small_data.append(values[3])
            sparse_data.append(small_data)

    third_order = COO(np.array(coords).T, np.array(data), shape=(n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
    third_order = third_order.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
    return third_order, np.array(sparse_data), np.array(second_cell_positions), np.array(third_cell_positions), np.array(atoms_coords)


def import_control_file(control_file):
    positions = []
    latt_vecs = []
    lfactor = 1
    with open(control_file, "r") as fo:
        lines = fo.readlines()
    for line in lines:
        if 'lattvec' in line:
            value = line.split('=')[1]
            latt_vecs.append(np.fromstring(value, dtype=np.float, sep=' '))
        if 'elements' in line and not ('nelements' in line):
            value = line.split('=')[1]
            # TODO: only one species at the moment
            value = value.replace('"', '\'')
            value = value.replace(" ", '')
            value = value.replace("\n", '')
            value = value.replace(',', '')
            value = value.replace("''", '\t')
            value = value.replace("'", '')
            elements = value.split("\t")

        if 'types' in line:
            value = line.split('=')[1]

            types = np.fromstring(value, dtype=np.int, sep=' ')
        if 'positions' in line:
            value = line.split('=')[1]
            positions.append(np.fromstring(value, dtype=np.float, sep=' '))
        if 'lfactor' in line:
            lfactor = float(line.split('=')[1].split(',')[0])
        if 'scell' in line:
            value = line.split('=')[1]
            supercell = np.fromstring(value, dtype=np.int, sep=' ')
    # l factor is in nanometer
    cell = np.array(latt_vecs) * lfactor * 10
    positions = np.array(positions).dot(cell)
    list_of_elem = []
    for i in range(len(types)):
        list_of_elem.append(elements[types[i] - 1])

    atoms = Atoms(list_of_elem,
                  positions=positions,
                  cell=cell,
                  pbc=[1, 1, 1])

    print('Atoms object created.')
    return atoms, supercell


def save_second_order_matrix(phonons):

    filename = 'FORCE_CONSTANTS_2ND'
    filename = phonons.folder + '/' + filename
    forceconstants = phonons.forceconstants
    second_order = forceconstants.second_order
    n_atoms_unit_cell = forceconstants.atoms.positions.shape[0]
    n_replicas = phonons.forceconstants.n_replicas
    second_order = second_order.reshape((n_atoms_unit_cell, 3, n_replicas, n_atoms_unit_cell, 3))


    #TODO: this is a bit hacky. ShengBTE wants the whole second order matrix, but actually uses only the reduced one. So we fill the rest with zeros
    with open(filename, 'w+') as file:
        file.write(str(n_atoms_unit_cell * n_replicas) + '\n')
        for i0 in range(n_atoms_unit_cell):
            for l0 in range(n_replicas):
                for i1 in range(n_atoms_unit_cell):
                    for l1 in range(n_replicas):
                        file.write(str(l0 + i0 * n_replicas + 1) + '  ' + str(l1 + i1 * n_replicas + 1) + '\n')
                        if l0 == 0:
                            sub_second = second_order[i0, :, l1, i1, :]
                        else:
                            sub_second = np.zeros((3, 3))

                        try:
                            file.write('%.6f %.6f %.6f\n' % (sub_second[0][0], sub_second[0][1], sub_second[0][2]))
                            file.write('%.6f %.6f %.6f\n' % (sub_second[1][0], sub_second[1][1], sub_second[1][2]))
                            file.write('%.6f %.6f %.6f\n' % (sub_second[2][0], sub_second[2][1], sub_second[2][2]))
                        except TypeError as err:
                            print(err)
    print('second order sheng saved')


def save_second_order_qe_matrix(phonons):
    shenbte_folder = phonons.folder + '/'
    n_replicas = phonons.forceconstants.n_replicas
    n_atoms = int(phonons.n_modes / 3)
    second_order = phonons.second_order.reshape((n_atoms, 3, n_replicas, n_atoms, 3))
    filename = 'espresso.ifc2'
    filename = shenbte_folder + filename
    file = open ('%s' % filename, 'w+')

    list_of_index = phonons.list_of_index()

    file.write (header(phonons))
    for alpha in range (3):
        for beta in range (3):
            for i in range (n_atoms):
                for j in range (n_atoms):
                    file.write('%4d %4d %4d %4d\n' % (alpha + 1, beta + 1, i + 1, j + 1))
                    for id_replica in range(list_of_index.shape[0]):

                        l_vec = (phonons.list_of_index()[id_replica] + 1)
                        for delta in range(3):
                            if l_vec[delta] <= 0:
                                l_vec[delta] = phonons.forceconstants.supercell[delta]


                        file.write('%4d %4d %4d' % (int(l_vec[2]), int(l_vec[1]), int(l_vec[2])))

                        matrix_element = second_order[i, alpha, id_replica, j, beta]

                        matrix_element = matrix_element / Rydberg * (
                                Bohr ** 2)
                        file.write ('\t %.11E' % matrix_element)
                        file.write ('\n')
    file.close ()
    print('second order qe sheng saved')


def save_third_order_matrix(phonons):
    filename = 'FORCE_CONSTANTS_3RD'
    filename = phonons.folder + '/' + filename
    file = open ('%s' % filename, 'w+')
    n_in_unit_cell = len (phonons.atoms.numbers)
    n_replicas = phonons.forceconstants.n_replicas
    third_order = phonons.forceconstants.third_order\
        .reshape((n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3))\
        .todense()

    block_counter = 0
    for i_0 in range (n_in_unit_cell):
        for n_1 in range (n_replicas):
            for i_1 in range (n_in_unit_cell):
                for n_2 in range (n_replicas):
                    for i_2 in range (n_in_unit_cell):
                        three_particles_interaction = third_order[0, i_0, :, n_1, i_1, :, n_2, i_2, :]
                        three_particles_interaction = three_particles_interaction

                        if (np.abs (three_particles_interaction) > 1e-9).any ():
                            block_counter += 1
                            file.write ('\n  ' + str (block_counter))
                            rep_position = phonons.forceconstants.second_order.list_of_replicas[n_1]
                            file.write ('\n  ' + str (rep_position[0]) + ' ' + str (rep_position[1]) + ' ' + str (
                                rep_position[2]))
                            rep_position = phonons.forceconstants.second_order.list_of_replicas[n_2]
                            file.write ('\n  ' + str (rep_position[0]) + ' ' + str (rep_position[1]) + ' ' + str (
                                rep_position[2]))
                            file.write ('\n  ' + str (i_0 + 1) + ' ' + str (i_1 + 1) + ' ' + str (i_2 + 1))

                            for alpha_0 in range (3):
                                for alpha_1 in range (3):
                                    for alpha_2 in range (3):
                                        file.write (
                                            '\n  ' + str (alpha_0 + 1) + ' ' + str (alpha_1 + 1) + ' ' + str (
                                                alpha_2 + 1) + "  %.11E" % three_particles_interaction[
                                                alpha_0, alpha_1, alpha_2])
                            file.write ('\n')
    file.close ()
    with open (filename, 'r') as original:
        data = original.read ()
    with open (filename, 'w+') as modified:
        modified.write ('  ' + str (block_counter) + '\n' + data)
    print('third order sheng saved')


def create_control_file_string(phonons, is_espresso=False):
    k_points = phonons.kpts
    elements = phonons.atoms.get_chemical_symbols ()
    unique_elements = np.unique (phonons.atoms.get_chemical_symbols ())
    string = ''
    string += '&allocations\n'
    string += '\tnelements=' + str(len(unique_elements)) + '\n'
    string += '\tnatoms=' + str(len(elements)) + '\n'
    string += '\tngrid(:)=' + str (k_points[0]) + ' ' + str (k_points[1]) + ' ' + str (k_points[2]) + '\n'
    string += '&end\n'
    string += '&crystal\n'
    string += '\tlfactor=0.1,\n'
    for i in range (phonons.atoms.cell.shape[0]):
        vector = phonons.atoms.cell[i]
        string += '\tlattvec(:,' + str (i + 1) + ')= ' + str (vector[0]) + ' ' + str (vector[1]) + ' ' + str (
            vector[2]) + '\n'
    string += '\telements= '
    for element in np.unique(phonons.atoms.get_chemical_symbols()):
        string += '\"' + element + '\",'
    string +='\n'
    string += '\ttypes='
    for element in phonons.atoms.get_chemical_symbols():
        string += str(type_element_id(phonons.atoms, element) + 1) + ' '
    string += ',\n'
    for i in range (phonons.atoms.positions.shape[0]):
        # TODO: double check this for more complicated geometries
        cellinv = np.linalg.inv (phonons.atoms.cell)
        vector = cellinv.dot(phonons.atoms.positions[i])
        string += '\tpositions(:,' + str (i + 1) + ')= ' + str (vector[0]) + ' ' + str (vector[1]) + ' ' + str (
            vector[2]) + '\n'
    string += '\tscell(:)=' + str (phonons.supercell[0]) + ' ' + str (phonons.supercell[1]) + ' ' + str (
        phonons.supercell[2]) + '\n'
    string += '&end\n'
    string += '&parameters\n'
    string += '\tT=' + str (phonons.temperature) + '\n'
    string += '\tscalebroad=1.0\n'
    string += '&end\n'
    string += '&flags\n'
    if is_espresso:
        string += '\tespresso=.true.\n'
    else:
        string += '\tespresso=.false.\n'
    if phonons.is_classic:
        string += '\tclassical=.true.\n'


    string += '\tnonanalytic=.false.\n'
    string += '\tisotopes=.false.\n'
    string += '&end\n'
    return string


def create_control_file(phonons):
    folder = phonons.folder
    filename = folder + '/CONTROL'
    string = create_control_file_string (phonons)

    with open (filename, 'w+') as file:
        file.write (string)


def header(phonons):

    # this convert masses to qm masses

    nat = len (phonons.atoms.get_chemical_symbols ())

    # TODO: The dielectric calculation is not implemented yet
    dielectric_constant = 1.
    born_eff_charge = 0.000000

    ntype = len (np.unique (phonons.atoms.get_chemical_symbols ()))
    # in quantum espresso ibrav = 0, do not use symmetry and use cartesian vectors to specify symmetries
    ibrav = 0
    header_str = ''
    header_str += str (ntype) + ' '
    header_str += str (nat) + ' '
    header_str += str (ibrav) + ' '

    # TODO: I'd like to have ibrav = 1 and put the actual positions here
    header_str += '0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 \n'
    header_str += matrix_to_string (phonons.atoms.cell)
    mass_factor = 1.8218779 * 6.022e-4

    for i in range (ntype):
        mass = np.unique (phonons.forceconstants.atoms.get_masses ())[i] / mass_factor
        label = np.unique (phonons.forceconstants.atoms.get_chemical_symbols ())[i]
        header_str += str (i + 1) + ' \'' + label + '\' ' + str (mass) + '\n'

    # TODO: this needs to be changed, it works only if all the atoms in the unit cell are different species
    for i in range (nat):
        header_str += str (i + 1) + '  ' + str (i + 1) + '  ' + matrix_to_string (phonons.atoms.positions[i])
    header_str += 'T \n'
    header_str += matrix_to_string (np.diag (np.ones (3)) * dielectric_constant)
    for i in range (nat):
        header_str += str (i + 1) + '\n'
        header_str += matrix_to_string (np.diag (np.ones (3)) * born_eff_charge * (-1) ** i)
    header_str += str (phonons.supercell[0]) + '    '
    header_str += str (phonons.supercell[1]) + '    '
    header_str += str (phonons.supercell[2]) + '\n'
    return header_str




def matrix_to_string(matrix):
    string = ''
    if len (matrix.shape) == 1:
        for i in range (matrix.shape[0]):
            string += '%.7f' % matrix[i] + ' '
        string += '\n'
    else:
        for i in range (matrix.shape[0]):
            for j in range (matrix.shape[1]):
                string += '%.7f' % matrix[i, j] + ' '
            string += '\n'
    return string


def type_element_id(atoms, element_name):
    # TODO: remove this method
    unique_elements = np.unique (atoms.get_chemical_symbols ())
    for i in range(len(unique_elements)):
        element = unique_elements[i]
        if element == element_name:
            return i


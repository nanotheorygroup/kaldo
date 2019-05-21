import pandas as pd
import numpy as np
from ase.units import Bohr, Rydberg

def save_second_order_matrix(phonons):
    filename = 'FORCE_CONSTANTS_2ND'
    filename = phonons.folder_name + '/' + filename
    finite_difference = phonons.finite_difference
    n_atoms_unit_cell = finite_difference.atoms.positions.shape[0]
    n_replicas = np.prod(finite_difference.supercell)
    with open(filename, 'w+') as file:
        file.write(str(n_atoms_unit_cell * n_replicas) + '\n')
        for i0 in range(n_atoms_unit_cell):
            for l0 in range(n_replicas):
                for i1 in range(n_atoms_unit_cell):
                    for l1 in range(n_replicas):

                        file.write(str(l0 + i0 * n_atoms_unit_cell + 1) + '  ' + str(l1 + i1 * n_atoms_unit_cell + 1) + '\n')

                        sub_second = finite_difference.second_order[l0, i0, :, l1, i1, :]
                        file.write('%.6f %.6f %.6f\n' % (sub_second[0][0], sub_second[0][1], sub_second[0][2]))
                        file.write('%.6f %.6f %.6f\n' % (sub_second[1][0], sub_second[1][1], sub_second[1][2]))
                        file.write('%.6f %.6f %.6f\n' % (sub_second[2][0], sub_second[2][1], sub_second[2][2]))


def save_second_order_qe_matrix(phonons):
    shenbte_folder = phonons.folder_name + '/'
    n_replicas = phonons.supercell.prod()
    n_particles = int(phonons.n_modes / 3)
    if phonons.finite_difference.is_reduced_second:
        second_order = phonons.finite_difference.second_order.reshape((n_particles, 3, n_replicas, n_particles, 3))
    else:
        second_order = phonons.finite_difference.second_order.reshape (
            (n_replicas, n_particles, 3, n_replicas, n_particles, 3))[0]
    filename = 'espresso.ifc2'
    filename = shenbte_folder + filename
    file = open ('%s' % filename, 'w+')

    list_of_index = phonons.finite_difference.list_of_index()

    file.write (header(phonons))
    for alpha in range (3):
        for beta in range (3):
            for i in range (n_particles):
                for j in range (n_particles):
                    file.write('%4d %4d %4d %4d\n' % (alpha + 1, beta + 1, i + 1, j + 1))
                    for id_replica in range(list_of_index.shape[0]):

                        l_vec = (phonons.finite_difference.list_of_index()[id_replica] + 1)
                        for delta in range(3):
                            if l_vec[delta] <= 0:
                                l_vec[delta] = phonons.finite_difference.supercell[delta]


                        file.write('%4d %4d %4d' % (int(l_vec[2]), int(l_vec[1]), int(l_vec[2])))

                        matrix_element = second_order[i, alpha, id_replica, j, beta]

                        matrix_element = matrix_element / Rydberg * (
                                Bohr ** 2)
                        file.write ('\t %.11E' % matrix_element)
                        file.write ('\n')
    file.close ()
    print('second order saved')


def save_third_order_matrix(phonons):
    filename = 'FORCE_CONSTANTS_3RD'
    filename = phonons.folder_name + '/' + filename
    file = open ('%s' % filename, 'w+')
    n_in_unit_cell = len (phonons.atoms.numbers)
    n_replicas = np.prod (phonons.supercell)
    third_order = phonons.finite_difference.third_order\
        .reshape((n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3))\
        .todense()

    block_counter = 0
    for i_0 in range (n_in_unit_cell):
        for n_1 in range (n_replicas):
            for i_1 in range (n_in_unit_cell):
                for n_2 in range (n_replicas):
                    for i_2 in range (n_in_unit_cell):
                        three_particles_interaction = third_order[0, i_0, :, n_1, i_1, :, n_2, i_2, :]
                        try:
                            three_particles_interaction = three_particles_interaction.todense()
                        except AttributeError:
                            pass

                        if (np.abs (three_particles_interaction) > 1e-9).any ():
                            block_counter += 1
                            file.write ('\n  ' + str (block_counter))
                            rep_position = phonons.finite_difference.list_of_replicas()[n_1]
                            file.write ('\n  ' + str (rep_position[0]) + ' ' + str (rep_position[1]) + ' ' + str (
                                rep_position[2]))
                            rep_position = phonons.finite_difference.list_of_replicas()[n_2]
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
    print('third order saved')




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
    folder = phonons.folder_name
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
        mass = np.unique (phonons.finite_difference.replicated_atoms.get_masses ())[i] / mass_factor
        label = np.unique (phonons.finite_difference.replicated_atoms.get_chemical_symbols ())[i]
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




def apply_boundary_with_cell(cell, cellinv, dxij):
    # exploit periodicity to calculate the shortest distance, which may not be the one we have
    sxij = dxij.dot(cellinv)
    sxij = sxij - np.round (sxij)
    dxij = sxij.dot(cell)
    return dxij


def apply_boundary(atoms, dxij):
    # TODO: remove this method
    cell = atoms.cell
    cellinv = np.linalg.inv (cell)
    dxij = apply_boundary_with_cell(atoms.cell, cellinv, dxij)
    return dxij


def type_element_id(atoms, element_name):
    # TODO: remove this method
    unique_elements = np.unique (atoms.get_chemical_symbols ())
    for i in range(len(unique_elements)):
        element = unique_elements[i]
        if element == element_name:
            return i

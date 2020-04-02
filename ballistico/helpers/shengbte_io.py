"""
Ballistico
Anharmonic Lattice Dynamics
"""
import pandas as pd
import numpy as np
from ballistico.phonons import Phonons
from ase.units import Rydberg, Bohr
from ase import Atoms
import os
import re
from ballistico.helpers.tools import count_rows

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


def list_of_index(finite_difference):
    list_of_index = finite_difference.list_of_replicas.dot(np.linalg.inv(finite_difference.atoms.cell)).round(0).astype(int)
    return list_of_index


def import_second_and_third_from_sheng(finite_difference):
    atoms = finite_difference.atoms
    supercell = finite_difference.supercell
    folder = finite_difference.folder
    second_file = folder + '/' + 'FORCE_CONSTANTS_2ND'
    if not os.path.isfile(second_file):
        second_file = folder + '/' + 'FORCE_CONSTANTS'
    n_replicas = np.prod(supercell)
    with open(second_file, 'r') as file:
        first_row = file.readline()
        first_row_split = re.findall(r'\d+', first_row)
        n_rows = int(list(map(int, first_row_split))[0])
        n_unit_atoms = int(n_rows / n_replicas)

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
    is_reduced_second = True
    third_order = np.zeros((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
    third_file = folder + '/' + 'FORCE_CONSTANTS_3RD'
    second_cell_list = []
    third_cell_list = []
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
            second_cell_id = (list_of_index(finite_difference)[:] == second_cell_index).prod(axis=1)
            second_cell_id = np.argwhere(second_cell_id).flatten()

            third_cell_position = np.fromstring(file.readline(), dtype=np.float, sep=' ')
            third_cell_index = third_cell_position.dot(np.linalg.inv(atoms.cell)).round(0).astype(int)
            third_cell_list.append(third_cell_index)

            # create mask to find the index
            third_cell_id = (list_of_index(finite_difference)[:] == third_cell_index).prod(axis=1)
            third_cell_id = np.argwhere(third_cell_id).flatten()

            atom_i, atom_j, atom_k = np.fromstring(file.readline(), dtype=np.int, sep=' ') - 1
            for _ in range(27):
                values = np.fromstring(file.readline(), dtype=np.float, sep=' ')
                alpha, beta, gamma = values[:3].round(0).astype(int) - 1
                try:
                    third_order[atom_i, alpha, second_cell_id, atom_j, beta, third_cell_id, atom_k, gamma] = values[
                        3]
                except TypeError as err:
                    print(err)
                except IndexError as err:
                    print(err)
    second_order = second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
    third_order = third_order.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas *
                                       n_unit_atoms * 3))
    return second_order, is_reduced_second, third_order


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
    finite_difference = phonons.finite_difference
    second_order = finite_difference.second_order
    n_atoms_unit_cell = finite_difference.atoms.positions.shape[0]
    n_replicas = phonons.finite_difference.n_replicas

    if not phonons.finite_difference.is_reduced_second:
        second_order = second_order.reshape((n_replicas, n_atoms_unit_cell, 3, n_replicas, n_atoms_unit_cell, 3))
    else:
        second_order = second_order.reshape((n_atoms_unit_cell, 3, n_replicas, n_atoms_unit_cell, 3))


    #TODO: this is a bit hacky. ShengBTE wants the whole second order matrix, but actually uses only the reduced one. So we fill the rest with zeros
    with open(filename, 'w+') as file:
        file.write(str(n_atoms_unit_cell * n_replicas) + '\n')
        for i0 in range(n_atoms_unit_cell):
            for l0 in range(n_replicas):
                for i1 in range(n_atoms_unit_cell):
                    for l1 in range(n_replicas):
                        file.write(str(l0 + i0 * n_replicas + 1) + '  ' + str(l1 + i1 * n_replicas + 1) + '\n')
                        if phonons.finite_difference.is_reduced_second:
                            if l0 == 0:
                                sub_second = second_order[i0, :, l1, i1, :]
                            else:
                                sub_second = np.zeros((3, 3))
                        else:
                            sub_second = second_order[l0, i0, :, l1, i1, :]
                        try:
                            file.write('%.6f %.6f %.6f\n' % (sub_second[0][0], sub_second[0][1], sub_second[0][2]))
                            file.write('%.6f %.6f %.6f\n' % (sub_second[1][0], sub_second[1][1], sub_second[1][2]))
                            file.write('%.6f %.6f %.6f\n' % (sub_second[2][0], sub_second[2][1], sub_second[2][2]))
                        except TypeError as err:
                            print(err)
    print('second order sheng saved')


def save_second_order_qe_matrix(phonons):
    shenbte_folder = phonons.folder + '/'
    n_replicas = phonons.finite_difference.n_replicas
    n_atoms = int(phonons.n_modes / 3)
    if phonons.finite_difference.is_reduced_second:
        second_order = phonons.second_order.reshape((n_atoms, 3, n_replicas, n_atoms, 3))
    else:
        second_order = phonons.second_order.reshape (
            (n_replicas, n_atoms, 3, n_replicas, n_atoms, 3))[0]
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
                                l_vec[delta] = phonons.finite_difference.supercell[delta]


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
    n_replicas = phonons.finite_difference.n_replicas
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
                        three_particles_interaction = three_particles_interaction

                        if (np.abs (three_particles_interaction) > 1e-9).any ():
                            block_counter += 1
                            file.write ('\n  ' + str (block_counter))
                            rep_position = phonons.finite_difference.list_of_replicas[n_1]
                            file.write ('\n  ' + str (rep_position[0]) + ' ' + str (rep_position[1]) + ' ' + str (
                                rep_position[2]))
                            rep_position = phonons.finite_difference.list_of_replicas[n_2]
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


def type_element_id(atoms, element_name):
    # TODO: remove this method
    unique_elements = np.unique (atoms.get_chemical_symbols ())
    for i in range(len(unique_elements)):
        element = unique_elements[i]
        if element == element_name:
            return i


def import_from_shengbte(finite_difference, kpts, is_classic, temperature, folder):

    # # Create a phonon object
    phonons = Phonons(finite_difference=finite_difference,
                      kpts=kpts,
                      is_classic=is_classic,
                      temperature=temperature,
                      folder=folder)

    new_shape = [phonons.kpts[0], phonons.kpts[1], phonons.kpts[2], phonons.n_modes]
    energies = read_energy_data(phonons).reshape(new_shape)
    phonons.frequency = energies / (2 * np.pi)
    phonons.velocity = read_velocity_data(phonons) * 10
    phonons.gamma = read_decay_rate_data(phonons).reshape(new_shape)
    # phonons.scattering_matrix = import_scattering_matrix(phonons)
    return phonons

def qpoints_mapper(phonons):
    q_points = pd.read_csv (phonons.folder + '/BTE.qpoints_full', header=None, delim_whitespace=True)
    return q_points.values


def irreducible_indices(phonons):
    return np.unique(qpoints_mapper(phonons)[:,1])


def q_points(phonons):
    return qpoints_mapper(phonons)[:,2:5]


def read_energy_data(phonons):
    # We read in rad/ps
    omega = pd.read_csv (phonons.folder + '/BTE.omega', header=None, delim_whitespace=True)
    n_qpoints = qpoints_mapper(phonons).shape[0]
    n_branches = omega.shape[1]
    energy_data = np.zeros ((n_qpoints, n_branches))
    for index, reduced_index, q_point_x, q_point_y, q_point_z in qpoints_mapper(phonons):
        energy_data[int (index - 1)] = omega.loc[[int (reduced_index - 1)]].values
    return energy_data


def read_ps_data(phonons, type=None):
    if type == 'plus':
        file = 'BTE.WP3_plus'
    elif type == 'minus':
        file = 'BTE.WP3_minus'
    else:
        file = 'BTE.WP3'
    temperature = str (int (phonons.temperature))
    decay = pd.read_csv (phonons.folder + '/T' + temperature + 'K/' + file, header=None,
                         delim_whitespace=True)
    # decay = pd.read_csv (phonons.folder + 'T' + temperature +
    # 'K/BTE.w_anharmonic', header=None, delim_whitespace=True)
    n_branches = int (decay.shape[0] / irreducible_indices(phonons).max ())
    n_qpoints_reduced = int (decay.shape[0] / n_branches)
    n_qpoints = qpoints_mapper(phonons).shape[0]
    decay = np.delete (decay.values, 0, 1)
    decay = decay.reshape ((n_branches, n_qpoints_reduced))
    decay_data = np.zeros ((n_qpoints, n_branches))
    for index, reduced_index, q_point_x, q_point_y, q_point_z in qpoints_mapper(phonons):
        decay_data[int (index - 1)] = decay[:, int (reduced_index - 1)]
    return decay_data


def read_decay_rate_data(phonons, type=None):
    if type == 'plus':
        file = 'BTE.w_anharmonic_plus'
    elif type == 'minus':
        file = 'BTE.w_anharmonic_minus'
    else:
        file = 'BTE.w_anharmonic'
    temperature = str(int(phonons.temperature))
    decay = pd.read_csv (phonons.folder + '/T' + temperature + 'K/' + file, header=None,
                         delim_whitespace=True)
    # decay = pd.read_csv (phonons.folder + 'T' + temperature +
    # 'K/BTE.w_anharmonic', header=None, delim_whitespace=True)
    n_branches = int (decay.shape[0] / irreducible_indices (phonons).max ())
    n_qpoints_reduced = int (decay.shape[0] / n_branches)
    n_qpoints = qpoints_mapper(phonons).shape[0]
    decay = np.delete(decay.values,0,1)
    decay = decay.reshape((n_branches, n_qpoints_reduced))
    decay_data = np.zeros ((n_qpoints, n_branches))
    for index, reduced_index, q_point_x, q_point_y, q_point_z in qpoints_mapper(phonons):
        decay_data[int (index - 1)] = decay[:, int(reduced_index-1)]
    return decay_data


def read_velocity_data(phonons):
    shenbte_folder = phonons.folder
    velocity = pd.read_csv (shenbte_folder + '/BTE.v_full', header=None, delim_whitespace=True)
    n_velocity = velocity.shape[0]
    n_qpoints = qpoints_mapper(phonons).shape[0]
    n_modes = int(n_velocity / n_qpoints)

    velocity_array = velocity.values.reshape (n_modes, n_qpoints, 3)

    velocity = np.zeros((phonons.kpts[0], phonons.kpts[1], phonons.kpts[2], n_modes, 3))

    z = 0
    for k in range (phonons.kpts[2]):
        for j in range(phonons.kpts[1]):
            for i in range (phonons.kpts[0]):
                velocity[i, j, k, :, :] = velocity_array[:, z, :]
                z += 1
    return velocity


def read_conductivity(phonons, converged=True):
    folder = phonons.folder
    if converged:
        conduct_file = '/BTE.KappaTensorVsT_CONV'
    else:
        conduct_file = '/BTE.KappaTensorVsT_RTA'

    conductivity_array = np.loadtxt (folder + conduct_file)
    conductivity_array = np.delete (conductivity_array, 0)
    n_steps = 0
    if converged:
        n_steps = int (conductivity_array[-1])
        conductivity_array = np.delete (conductivity_array, -1)

    conductivity = conductivity_array.reshape (3, 3)
    return conductivity, n_steps


def import_scattering_matrix(phonons):
    temperature = str(int(phonons.temperature))
    filename_gamma = phonons.folder + '/T' + temperature + 'K/GGG.Gamma_Tensor'
    filename_tau_zero = phonons.folder + '/T' + temperature + 'K/GGG.tau_zero'
    phonons.tau_zero = np.zeros((phonons.n_modes, phonons.n_k_points))
    with open(filename_tau_zero, "r+") as f:
        for line in f:
            items = line.split()
            phonons.tau_zero[int(items[0]) - 1, int(items[1]) - 1] = float(items[2])

    n0 = []
    n1 = []
    k0 = []
    k1 = []
    gamma_value = []

    with open(filename_gamma, "r+") as f:
        for line in f:
            items = line.split()

            n0.append(int(items[0]) - 1)
            k0.append(int(items[1]) - 1)
            n1.append(int(items[2]) - 1)
            k1.append(int(items[3]) - 1)

            gamma_value.append(float(items[4]))
    gamma_tensor = np.zeros((phonons.n_k_points, phonons.n_modes, phonons.n_k_points,phonons.n_modes))
    gamma_tensor[k0, n0, k1, n1] = gamma_value
    return gamma_tensor


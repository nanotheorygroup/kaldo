from ballistico.phonons import Phonons

import pandas as pd
import numpy as np
import os
import subprocess
from ase.units import Bohr, Rydberg


BUFFER_PLOT = .2
SHENG_FOLDER_NAME = 'sheng_bte'
SHENGBTE_SCRIPT = 'ShengBTE.x'


def import_from_shengbte(finite_difference, kpts, is_classic, temperature, is_persistency_enabled, convergence, is_calculating, script=SHENGBTE_SCRIPT):
    # # Create a phonon object
    phonons = Phonons(finite_difference=finite_difference,
                      kpts=kpts,
                      is_classic=is_classic,
                      temperature=temperature,
                      is_persistency_enabled=is_persistency_enabled)
    
    phonons.convergence = convergence

    if is_calculating:
        run(phonons, script)
    try:
        new_shape = [phonons.kpts[0], phonons.kpts[1], phonons.kpts[2], phonons.n_modes]
        phonons.energies = read_energy_data().reshape(new_shape)
        phonons.frequencies = phonons.energies / (2 * np.pi)
        phonons.velocities = read_velocity_data(phonons)
        phonons.gamma = read_decay_rate_data(phonons).reshape(new_shape)
        phonons.scattering_matrix = import_scattering_matrix(phonons)
    except FileNotFoundError:
        # TODO: clean up this replicated logic
        run(phonons, script)
        new_shape = [phonons.kpts[0], phonons.kpts[1], phonons.kpts[2], phonons.n_modes]
        phonons.energies = read_energy_data().reshape(new_shape)
        phonons.frequencies = phonons.energies / (2 * np.pi)
        phonons.velocities = read_velocity_data(phonons)
        phonons.gamma = read_decay_rate_data(phonons).reshape(new_shape)
        phonons.scattering_matrix = import_scattering_matrix(phonons)
    return phonons
    

def save_second_order_matrix(phonons):
    shenbte_folder = SHENG_FOLDER_NAME + '/'
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
    cell_inv = np.linalg.inv(phonons.atoms.cell)

    list_of_index = phonons.finite_difference.list_of_index.dot(cell_inv)
    list_of_index = np.flip(list_of_index, 1)
    list_of_index = np.round(list_of_index)

    file.write (header(phonons))
    for alpha in range (3):
        for beta in range (3):
            for i in range (n_particles):
                for j in range (n_particles):
                    file.write ('\t' + str (alpha + 1) + '\t' + str (beta + 1) + '\t' + str (i + 1)
                                + '\t' + str (j + 1) + '\n')
                    for id_replica in range(list_of_index.shape[0]):
                        index = list_of_index[id_replica]
                        l_vec = np.array(index % phonons.supercell + 1).astype(np.int)
                        file.write ('\t' + str (int(l_vec[0])) + '\t' + str (int(l_vec[1])) + '\t' + str (int(l_vec[
                                                                                                              2])))

                        matrix_element = second_order[j, beta, id_replica, i, alpha]

                        matrix_element = matrix_element / Rydberg * (
                                Bohr ** 2)
                        file.write ('\t %.11E' % matrix_element)
                        file.write ('\n')
    file.close ()
    print('second order saved')


def save_third_order_matrix(phonons):
    filename = 'FORCE_CONSTANTS_3RD'
    filename = SHENG_FOLDER_NAME + '/' + filename
    file = open ('%s' % filename, 'w+')
    n_in_unit_cell = len (phonons.atoms.numbers)
    n_replicas = np.prod (phonons.supercell)
    third_order = phonons.finite_difference.third_order\
        .reshape((n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3))\
        .todense()
    list_of_index = phonons.finite_difference.list_of_index.astype(int)
    list_of_index = np.flip(list_of_index, 1)
    block_counter = 0
    for i_0 in range (n_in_unit_cell):
        for n_1 in range (n_replicas):
            for i_1 in range (n_in_unit_cell):
                for n_2 in range (n_replicas):
                    for i_2 in range (n_in_unit_cell):
                        three_particles_interaction = third_order[0, i_0, :, n_1, i_1, :, n_2, i_2, :]
                        try:
                            three_particles_interaction = three_particles_interaction.todense()
                        except AttributeError as err:
                            pass

                        if (np.abs (three_particles_interaction) > 1e-9).any ():
                            block_counter += 1
                            replica = list_of_index
                            file.write ('\n  ' + str (block_counter))
                            rep_position = apply_boundary (phonons.finite_difference.replicated_atoms,replica[n_1])
                            file.write ('\n  ' + str (rep_position[0]) + ' ' + str (rep_position[1]) + ' ' + str (
                                rep_position[2]))
                            rep_position = apply_boundary (phonons.finite_difference.replicated_atoms,replica[n_2])
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


def run(phonons, script):
    folder = SHENG_FOLDER_NAME
    if not os.path.exists(folder):
        os.makedirs (folder)
    create_control_file(phonons)
    save_second_order_matrix(phonons)
    save_third_order_matrix(phonons)
    return run_script (script, SHENG_FOLDER_NAME)


def create_control_file_string(phonons):
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
    string += '\tespresso=.true.\n'

    if phonons.is_classic:
        string += '\tclassical=.true.\n'

    if phonons.convergence:
        string += '\tconvergence=.true.\n'
    else:
        string += '\tconvergence=.false.\n'

    string += '\tnonanalytic=.false.\n'
    string += '\tisotopes=.false.\n'
    string += '&end\n'
    return string


def create_control_file(phonons):
    folder = SHENG_FOLDER_NAME
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
    header_str += str (phonons.supercell[0]) + ' '
    header_str += str (phonons.supercell[1]) + ' '
    header_str += str (phonons.supercell[2]) + '\n'
    return header_str


def qpoints_mapper():
    q_points = pd.read_csv (SHENG_FOLDER_NAME + '/BTE.qpoints_full', header=None, delim_whitespace=True)
    return q_points.values


def irreducible_indices():
    return np.unique(qpoints_mapper()[:,1])


def q_points():
    return qpoints_mapper()[:,2:5]


def read_energy_data():
    # We read in rad/ps
    omega = pd.read_csv (SHENG_FOLDER_NAME + '/BTE.omega', header=None, delim_whitespace=True)
    n_qpoints = qpoints_mapper().shape[0]
    n_branches = omega.shape[1]
    energy_data = np.zeros ((n_qpoints, n_branches))
    for index, reduced_index, q_point_x, q_point_y, q_point_z in qpoints_mapper():
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
    decay = pd.read_csv (SHENG_FOLDER_NAME + '/T' + temperature + 'K/' + file, header=None,
                         delim_whitespace=True)
    # decay = pd.read_csv (SHENG_FOLDER_NAME + 'T' + temperature +
    # 'K/BTE.w_anharmonic', header=None, delim_whitespace=True)
    n_branches = int (decay.shape[0] / irreducible_indices().max ())
    n_qpoints_reduced = int (decay.shape[0] / n_branches)
    n_qpoints = qpoints_mapper().shape[0]
    decay = np.delete (decay.values, 0, 1)
    decay = decay.reshape ((n_branches, n_qpoints_reduced))
    decay_data = np.zeros ((n_qpoints, n_branches))
    for index, reduced_index, q_point_x, q_point_y, q_point_z in qpoints_mapper():
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
    decay = pd.read_csv (SHENG_FOLDER_NAME + '/T' + temperature + 'K/' + file, header=None,
                         delim_whitespace=True)
    # decay = pd.read_csv (SHENG_FOLDER_NAME + 'T' + temperature +
    # 'K/BTE.w_anharmonic', header=None, delim_whitespace=True)
    n_branches = int (decay.shape[0] / irreducible_indices ().max ())
    n_qpoints_reduced = int (decay.shape[0] / n_branches)
    n_qpoints = qpoints_mapper().shape[0]
    decay = np.delete(decay.values,0,1)
    decay = decay.reshape((n_branches, n_qpoints_reduced))
    decay_data = np.zeros ((n_qpoints, n_branches))
    for index, reduced_index, q_point_x, q_point_y, q_point_z in qpoints_mapper():
        decay_data[int (index - 1)] = decay[:, int(reduced_index-1)]
    return decay_data


def read_velocity_data(phonons):
    shenbte_folder = SHENG_FOLDER_NAME
    velocities = pd.read_csv (shenbte_folder + '/BTE.v_full', header=None, delim_whitespace=True)
    n_velocities = velocities.shape[0]
    n_qpoints = qpoints_mapper().shape[0]
    n_modes = int(n_velocities / n_qpoints)

    velocity_array = velocities.values.reshape (n_modes, n_qpoints, 3)

    velocities = np.zeros((phonons.kpts[0], phonons.kpts[1], phonons.kpts[2], n_modes, 3))

    z = 0
    for k in range (phonons.kpts[2]):
        for j in range(phonons.kpts[1]):
            for i in range (phonons.kpts[0]):
                velocities[i, j, k, :, :] = velocity_array[:, z, :]
                z += 1
    return velocities


def read_conductivity(converged=True):
    folder = SHENG_FOLDER_NAME
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
    return conductivity


def import_scattering_matrix(phonons):
    temperature = str(int(phonons.temperature))
    filename_gamma = SHENG_FOLDER_NAME + '/T' + temperature + 'K/GGG.Gamma_Tensor'
    filename_tau_zero = SHENG_FOLDER_NAME + '/T' + temperature + 'K/GGG.tau_zero'
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


def run_script(cmd, folder=None):
    # print 'Executing: ' + cmd
    if folder:
        p = subprocess.Popen (cmd, cwd=folder, shell=True, stdout=subprocess.PIPE)
    else:
        p = subprocess.Popen (cmd, shell=True, stdout=subprocess.PIPE)
    (output, err) = p.communicate ()
    p.wait ()
    return output.decode('ascii')


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

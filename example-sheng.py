import numpy as np
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.units import Rydberg, Bohr

from ballistico.finite_difference import FiniteDifference


def _matrix_to_string(matrix):
    string = ''
    if len(matrix.shape) == 1:
        for i in range(matrix.shape[0]):
            string += '%.7f' % matrix[i] + ' '
        string += '\n'
    else:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                string += '%.7f' % matrix[i, j] + ' '
            string += '\n'
    return string


def _apply_boundary_with_cell(cell, cellinv, dxij):
    # exploit periodicity to calculate the shortest distance, which may not be the one we have
    sxij = dxij.dot(cellinv)
    sxij = sxij - np.round(sxij)
    dxij = sxij.dot(cell)
    return dxij


def _apply_boundary(atoms, dxij):
    cell = atoms.cell
    cellinv = np.linalg.inv(cell)
    dxij = _apply_boundary_with_cell(atoms.cell, cellinv, dxij)
    return dxij


def _header(finite_difference):
    # this convert masses to qm masses
    atoms = finite_difference.atoms
    supercell = finite_difference.supercell
    nat = len(atoms.get_chemical_symbols())

    # TODO: The dielectric calculation is not implemented yet
    dielectric_constant = 1.
    born_eff_charge = 0.000000

    ntype = len(np.unique(atoms.get_chemical_symbols()))
    # in quantum espresso ibrav = 0, do not use symmetry and use cartesian vectors to specify symmetries
    ibrav = 0
    header_str = ''
    header_str += str(ntype) + ' '
    header_str += str(nat) + ' '
    header_str += str(ibrav) + ' '

    # TODO: I'd like to have ibrav = 1 and put the actual positions here
    header_str += '0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 \n'
    header_str += _matrix_to_string(atoms.cell)
    mass_factor = 1.8218779 * 6.022e-4

    for i in range(ntype):
        mass = np.unique(finite_difference.replicated_atoms.get_masses())[i] / mass_factor
        label = np.unique(finite_difference.replicated_atoms.get_chemical_symbols())[i]
        header_str += str(i + 1) + ' \'' + label + '\' ' + str(mass) + '\n'

    for i in range(nat):
        header_str += str(i + 1) + '  ' + str(i + 1) + '  ' + _matrix_to_string(atoms.positions[i])
    header_str += 'T \n'
    header_str += _matrix_to_string(np.diag(np.ones(3)) * dielectric_constant)
    for i in range(nat):
        header_str += str(i + 1) + '\n'
        header_str += _matrix_to_string(np.diag(np.ones(3)) * born_eff_charge * (-1) ** i)
    header_str += str(supercell[0]) + ' '
    header_str += str(supercell[1]) + ' '
    header_str += str(supercell[2]) + '\n'
    return header_str


def _type_element_id(atoms, element_name):
    # TODO: remove this method
    unique_elements = np.unique(atoms.get_chemical_symbols())
    for i in range(len(unique_elements)):
        element = unique_elements[i]
        if element == element_name:
            return i


def save_second_order_matrix(finite_difference):
    second_order = finite_difference.second_order
    atoms = finite_difference.atoms
    supercell = finite_difference.supercell
    n_particles = len(atoms.get_chemical_symbols())

    n_replicas = supercell.prod()
    n_modes = n_particles * 3
    second_order = second_order.reshape((n_replicas, n_particles, 3, n_replicas, n_particles, 3))
    filename = 'espresso.ifc2'
    n_particles = second_order.shape[1]
    file = open('%s' % filename, 'a+')
    cell_inv = np.linalg.inv(atoms.cell)
    list_of_indices = np.zeros_like(finite_difference.list_of_index, dtype=np.int)
    for replica_id in range(finite_difference.list_of_index.shape[0]):
        list_of_indices[replica_id] = cell_inv.dot(finite_difference.list_of_index[replica_id])
    file.write(_header(finite_difference))

    for alpha in range(3):
        for beta in range(3):
            for i in range(n_particles):
                for j in range(n_particles):
                    file.write('\t' + str(alpha + 1) + '\t' + str(beta + 1) + '\t' + str(i + 1)
                               + '\t' + str(j + 1) + '\n')
                    for id_replica in range(list_of_indices.shape[0]):
                        index = list_of_indices[id_replica]
                        l_vec = np.array(index % supercell + 1).astype(np.int)
                        file.write('\t' + str(l_vec[0]) + '\t' + str(l_vec[1]) + '\t' + str(l_vec[2]))

                        # TODO: WHy are they flipped?
                        matrix_element = second_order[0, j, beta, id_replica, i, alpha]

                        matrix_element = matrix_element / Rydberg * (
                                Bohr ** 2)
                        file.write('\t %.11E' % matrix_element)
                        file.write('\n')
    file.close()
    print('second order saved')


def save_third_order_matrix(finite_difference):
    supercell = finite_difference.supercell
    atoms = finite_difference.atoms
    third_order = finite_difference.third_order
    filename = 'FORCE_CONSTANTS_3RD'
    file = open('%s' % filename, 'w')
    n_in_unit_cell = len(atoms.numbers)
    n_replicas = np.prod(supercell)
    third_order = third_order.reshape(
        (n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3))
    block_counter = 0
    for i_0 in range(n_in_unit_cell):
        for n_1 in range(n_replicas):
            for i_1 in range(n_in_unit_cell):
                for n_2 in range(n_replicas):
                    for i_2 in range(n_in_unit_cell):

                        three_particles_interaction = third_order[0, i_0, :, n_1, i_1, :, n_2, i_2, :]

                        if (np.abs(three_particles_interaction) > 1e-9).any():
                            block_counter += 1
                            replica = finite_difference.list_of_index
                            file.write('\n  ' + str(block_counter))
                            rep_position = _apply_boundary(finite_difference.replicated_atoms, replica[n_1])
                            file.write('\n  ' + str(rep_position[0]) + ' ' + str(rep_position[1]) + ' ' + str(
                                rep_position[2]))
                            rep_position = _apply_boundary(finite_difference.replicated_atoms, replica[n_2])
                            file.write('\n  ' + str(rep_position[0]) + ' ' + str(rep_position[1]) + ' ' + str(
                                rep_position[2]))
                            file.write('\n  ' + str(i_0 + 1) + ' ' + str(i_1 + 1) + ' ' + str(i_2 + 1))

                            for alpha_0 in range(3):
                                for alpha_1 in range(3):
                                    for alpha_2 in range(3):
                                        file.write(
                                            '\n  ' + str(alpha_0 + 1) + ' ' + str(alpha_1 + 1) + ' ' + str(
                                                alpha_2 + 1) + "  %.11E" % three_particles_interaction[
                                                alpha_0, alpha_1, alpha_2])
                            file.write('\n')
    file.close()
    with open(filename, 'r') as original:
        data = original.read()
    with open(filename, 'w') as modified:
        modified.write('  ' + str(block_counter) + '\n' + data)
    print('third order saved')


def create_control_file_string(finite_difference, kpts, temperature, is_classic, convergence):
    k_points = kpts
    atoms = finite_difference.atoms
    elements = finite_difference.atoms.get_chemical_symbols()
    unique_elements = np.unique(atoms.get_chemical_symbols())
    supercell = finite_difference.supercell
    string = ''
    string += '&allocations\n'
    string += '\tnelements=' + str(len(unique_elements)) + '\n'
    string += '\tnatoms=' + str(len(elements)) + '\n'
    string += '\tngrid(:)=' + str(k_points[0]) + ' ' + str(k_points[1]) + ' ' + str(k_points[2]) + '\n'
    string += '&end\n'
    string += '&crystal\n'
    string += '\tlfactor=0.1,\n'
    for i in range(atoms.cell.shape[0]):
        vector = atoms.cell[i]
        string += '\tlattvec(:,' + str(i + 1) + ')= ' + str(vector[0]) + ' ' + str(vector[1]) + ' ' + str(
            vector[2]) + '\n'
    string += '\telements= '
    for element in np.unique(atoms.get_chemical_symbols()):
        string += '\"' + element + '\",'
    string += '\n'
    string += '\ttypes='
    for element in atoms.get_chemical_symbols():
        string += str(_type_element_id(atoms, element) + 1) + ' '
    string += ',\n'
    for i in range(atoms.positions.shape[0]):
        # TODO: double check this for more complicated geometries
        cellinv = np.linalg.inv(atoms.cell)
        vector = cellinv.dot(atoms.positions[i])
        string += '\tpositions(:,' + str(i + 1) + ')= ' + str(vector[0]) + ' ' + str(vector[1]) + ' ' + str(
            vector[2]) + '\n'
    string += '\tscell(:)=' + str(supercell[0]) + ' ' + str(supercell[1]) + ' ' + str(
        supercell[2]) + '\n'
    # if (self.length).any():
    # 	string += '\tlength(:)=' + str(self.length[0]) + ' ' + str(self.length[1]) + ' ' + str(self.length[2]) + '\n'
    string += '&end\n'
    string += '&parameters\n'
    string += '\tT=' + str(temperature) + '\n'
    string += '\tscalebroad=1.0\n'
    string += '&end\n'
    string += '&flags\n'
    string += '\tespresso=.true.\n'

    if is_classic:
        string += '\tclassical=.true.\n'

    if convergence:
        string += '\tconvergence=.true.\n'
    else:
        string += '\tconvergence=.false.\n'

    string += '\tnonanalytic=.false.\n'
    string += '\tisotopes=.false.\n'
    string += '&end\n'
    return string


def create_control_file(finite_difference, kpts, temperature, is_classic, convergence):
    filename = 'CONTROL'
    string = create_control_file_string(finite_difference, kpts, temperature, is_classic, convergence)

    with open(filename, 'w') as file:
        file.write(string)


if __name__ == "__main__":

    atoms = bulk('Si', 'diamond', a=5.432)

    supercell = np.array([3, 3, 3])
    n_replicas = np.prod(supercell)

    calculator = LAMMPSlib
    calculator_inputs = {'lmpcmds': ["pair_style tersoff", "pair_coeff * * forcefields/Si.tersoff Si"],
                         'log_file': 'log_lammps.out'}

    finite_difference = FiniteDifference(atoms=atoms,
                                         supercell=supercell,
                                         calculator=calculator,
                                         calculator_inputs=calculator_inputs,
                                         is_persistency_enabled=False)

    create_control_file(finite_difference, kpts=(3, 3, 3), temperature=300, is_classic=False, convergence=True)
    save_second_order_matrix(finite_difference)
    save_third_order_matrix(finite_difference)

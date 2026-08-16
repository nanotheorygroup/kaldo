"""
kaldo
Anharmonic Lattice Dynamics
"""
import numpy as np
from ase import Atoms
from kaldo.grid import SupercellGrid
from sparse import COO
from kaldo.helpers.logger import get_logger
logging = get_logger()


def _resolve_cell_id(cell_position, cell_inv, current_grid, source):
    """Resolve a Cartesian cell offset to a replica id, independent of the sign
    convention the writer used for half-box (even-supercell) offsets.

    The integer quotient maps either sign convention for an even-cell
    half-box offset to the same periodic class without a bounded search.
    """
    frac = cell_position.dot(cell_inv)
    if np.max(np.abs(frac - np.round(frac))) > 1e-3:
        raise ValueError(f"{source}: cell offset {cell_position} is not a lattice vector of the supercell.")
    return current_grid.class_id(frac.round().astype(int))


def read_third_order_matrix(third_file: str,
                            atoms: Atoms,
                            supercell: tuple[int, int, int],
                            order: str = 'C'):
    """Read third order force constants from a file in VASP format.
    """
    n_unit_atoms = atoms.positions.shape[0]
    n_replicas = np.prod(supercell)
    third_order = np.zeros((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
    current_grid = SupercellGrid(np.diag(supercell), order=order)
    cell_inv = np.linalg.inv(np.array(atoms.cell))
    occupied_blocks = set()

    with open(third_file, 'r') as file:
        first_line = file.readline()
        n_third = int(first_line.strip())
        for i in range(n_third):
            # skip two lines
            file.readline()
            file.readline()

            # next two lines are the positions of the second and third cell
            second_cell_position = np.array([float(x) for x in file.readline().split()])
            second_cell_id = _resolve_cell_id(second_cell_position, cell_inv, current_grid, third_file)

            third_cell_position = np.array([float(x) for x in file.readline().split()])
            third_cell_id = _resolve_cell_id(third_cell_position, cell_inv, current_grid, third_file)

            # index to atom
            atom_i, atom_j, atom_k = np.array([int(x) for x in file.readline().split()]) - 1
            target = (atom_i, second_cell_id, atom_j, third_cell_id, atom_k)
            if target in occupied_blocks:
                raise ValueError(
                    f"{third_file}: duplicate FC3 block resolves to canonical target {target}"
                )
            occupied_blocks.add(target)

            # for x,y,z directions with 3 atoms
            # FC3 assigns each quartet's block directly into its slot (ShengBTE-format writers emit
            # unique (atom, cell) slots here).
            for _ in range(27):
                values = np.array([float(x) for x in file.readline().split()])
                alpha, beta, gamma = values[:3].round(0).astype(int) - 1
                third_order[atom_i, alpha, second_cell_id, atom_j, beta, third_cell_id, atom_k, gamma] = values[
                        3]

    third_order = third_order.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas *
                                       n_unit_atoms * 3))
    return third_order


def import_control_file(control_file):
    positions = []
    latt_vecs = []
    eps_vecs = []
    bec_vecs = []
    lfactor = 1
    masses = None
    with open(control_file, "r") as fo:
        lines = fo.readlines()
    for line in lines:
        if 'lattvec' in line:
            value = line.split('=')[1]
            latt_vecs.append(np.array([float(x.rstrip(',')) for x in value.split()]))
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
            types = np.array([int(x.rstrip(',')) for x in value.split()])
        if 'positions' in line:
            value = line.split('=')[1]
            positions.append(np.array([float(x.rstrip(',')) for x in value.split()]))
        if 'lfactor' in line:
            lfactor = float(line.split('=')[1].split(',')[0])
        # TODO: only one species/mass at the moment
        if 'masses' in line:
            value = line.split('=')[1]
            masses=np.array([float(x.rstrip(',')) for x in value.split()])
            #masses = float(line.split('=')[1].split(',')[0])
        if 'scell' in line:
            value = line.split('=')[1]
            supercell = np.array([int(x.rstrip(',')) for x in value.split()])
        if 'epsilon' in line:
            value = line.split('=')[1]
            eps_vecs.append(np.array([float(x.rstrip(',')) for x in value.split()]))
        if 'born' in line:
            value = line.split('=')[1]
            bec_vecs.append(np.array([float(x.rstrip(',')) for x in value.split()]))
    # l factor is in nanometer
    cell = np.array(latt_vecs) * lfactor * 10
    positions = np.array(positions).dot(cell)
    list_of_elem = []
    if masses is None:
        for i in range(len(types)):
            list_of_elem.append(elements[types[i] - 1])

        atoms = Atoms(list_of_elem,
                      positions=positions,
                      cell=cell,
                      pbc=[1, 1, 1])
    else:
        list_of_masses = []
        for i in range(len(types)):
            list_of_elem.append(elements[types[i] - 1])
            list_of_masses.append(masses[types[i] - 1])
        atoms = Atoms(list_of_elem,
                      positions=positions,
                      cell=cell,
                      masses=list_of_masses,
                      pbc=[1, 1, 1])

    logging.info('Atoms object created.')
    if len(eps_vecs) == 0:
        charges = None
    else:
        charges = np.zeros((len(atoms)+1, 3, 3))
        charges[0, ...] = np.array(eps_vecs)
        charges[1:, ...] = np.array(bec_vecs).reshape((len(atoms), 3, 3))
        logging.info('Charge data found in CONTROL file.')
    return atoms, supercell, charges


def save_second_order_matrix(phonons):

    filename = 'FORCE_CONSTANTS_2ND'
    filename = phonons.folder + '/' + filename
    forceconstants = phonons.forceconstants
    second_order = forceconstants.second
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

                        file.write('%.6f %.6f %.6f\n' % (sub_second[0][0], sub_second[0][1], sub_second[0][2]))
                        file.write('%.6f %.6f %.6f\n' % (sub_second[1][0], sub_second[1][1], sub_second[1][2]))
                        file.write('%.6f %.6f %.6f\n' % (sub_second[2][0], sub_second[2][1], sub_second[2][2]))


    logging.info('second order sheng saved')


def save_third_order_matrix(phonons):
    filename = 'FORCE_CONSTANTS_3RD'
    filename = phonons.folder + '/' + filename
    file = open ('%s' % filename, 'w+')
    n_in_unit_cell = len(phonons.atoms.numbers)
    n_replicas = phonons.forceconstants.n_replicas
    third_order = phonons.forceconstants.third.value\
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
                            rep_position = phonons.forceconstants.second.list_of_replicas[n_1]
                            file.write ('\n  ' + str (rep_position[0]) + ' ' + str (rep_position[1]) + ' ' + str (
                                rep_position[2]))
                            rep_position = phonons.forceconstants.second.list_of_replicas[n_2]
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
    logging.info('third order sheng saved')


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


def type_element_id(atoms, element_name):
    # TODO: remove this method
    unique_elements = np.unique (atoms.get_chemical_symbols ())
    for i in range(len(unique_elements)):
        element = unique_elements[i]
        if element == element_name:
            return i

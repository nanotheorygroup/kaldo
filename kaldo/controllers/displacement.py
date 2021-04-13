
import numpy as np
from sparse import COO
import ase.io
from os import mkdir
from os.path import isdir
from kaldo.helpers.logger import get_logger
logging = get_logger()


def list_of_replicas(atoms, replicated_atoms):
    n_atoms = atoms.positions.shape[0]
    n_replicas = int(replicated_atoms.positions.shape[0] / n_atoms)
    list_of_cells = (replicated_atoms.positions - atoms.positions).reshape((n_replicas, ))
    return list_of_cells


def calculate_gradient(x, input_atoms, alpha, move, save_progress):
    """
    Construct the calculate_gradient based on the given structure and atom object
    Set a copy for the atom object so that
    the progress of the optimization is traceable
    Force is the negative of the calculate_gradient
    """

    atoms = input_atoms.copy()
    input_atoms.positions = np.reshape(x, (int(x.size / 3.), 3))
    gr = -1. * input_atoms.get_forces()
    if np.sum(gr[0:10,:]) == 0: # Check for force convergence, yell if failed.
        logging.warning('WARNING: Force calculation failed.')
        logging.info('Outputting failed geometry calculation to failed.xyz')
        ase.io.write('failed.xyz', images=input_atoms, format='xyz')
        exit(1)
    if save_progress:
        ase.io.write('test')
    grad = np.reshape(gr, gr.size)
    input_atoms.positions = atoms.positions
    return grad


def calculate_single_second(replicated_atoms, atom_id, second_order_delta):
    """
    Compute the numerator of the approximated second matrices
    (approximated force from forward difference -
    approximated force from backward difference )
    """
    n_replicated_atoms = len(replicated_atoms .numbers)
    second_per_atom = np.zeros((3, n_replicated_atoms * 3))
    for alpha in range(3):
        for move in (-1, 1):
            shift = np.zeros((n_replicated_atoms, 3))
            shift[atom_id, alpha] += move * second_order_delta
            second_per_atom[alpha, :] += move * calculate_gradient(replicated_atoms.positions + shift,
                                                                   replicated_atoms,
                                                                   alpha,
                                                                   move)
    return second_per_atom


def calculate_second(atoms, replicated_atoms, second_order_delta, progress=None, is_verbose=False):
    # TODO: remove supercell
    """
    Core method to compute second order force constant matrices
    Approximate the second order force constant matrices
    using central difference formula
    """
    logging.info('Calculating second order potential derivatives, ' +
                 'finite difference displacement: %.3e angstrom'%second_order_delta)
    n_unit_cell_atoms = len(atoms.numbers)
    n_replicated_atoms = len(replicated_atoms.numbers)
    n_atoms = n_unit_cell_atoms
    n_replicas = int(n_replicated_atoms / n_unit_cell_atoms)
    logging.info('Requires force evaluations on %i frames'%(int(3*2*n_atoms)))
    second = np.zeros((n_atoms, 3, n_replicated_atoms * 3))
    if progress != None:
        folder = progress[0]
        progress_indices = progress[1]
        atom_index = progress_indices[0]
        for i in range(atom_index):
            if is_verbose:
                logging.info('loading forces on atom ' + str(i))
            try:
                second[i] = load_single_second(folder, i, n_atoms)
            except FileNotFoundError:
                logging.info('Attempt to load force on atom ' + str(i) + ' failed')
                second[i] = calculate_single_second(replicated_atoms, i, second_order_delta)
        for i in range(atom_index, n_atoms):
            if is_verbose:
                logging.info('calculating forces on atom ' + str(i))
            second[i] = calculate_single_second(replicated_atoms, i, second_order_delta)
    else:
        for i in range(n_atoms):
            if is_verbose:
                logging.info('calculating forces on atom ' + str(i))
            second[i] = calculate_single_second(replicated_atoms, i, second_order_delta)
    second = second.reshape((1, n_unit_cell_atoms, 3, n_replicas, n_unit_cell_atoms, 3))
    second = second / (2. * second_order_delta)
    return second


def calculate_third(atoms, replicated_atoms, third_order_delta, distance_threshold=None, is_verbose=False):
    """
    Compute third order force constant matrices by using the central
    difference formula for the approximation
    """
    logging.info('Calculating third order potential derivatives, ' + 'finite difference displacement: %.3e angstrom'%third_order_delta)
    n_atoms = len(atoms.numbers)
    replicated_atoms = replicated_atoms
    n_replicas = int(replicated_atoms.positions.shape[0] / n_atoms)
    i_at_sparse = []
    i_coord_sparse = []
    jat_sparse = []
    j_coord_sparse = []
    k_sparse = []
    value_sparse = []
    n_forces_to_calculate = n_replicas * (n_atoms * 3) ** 2
    n_forces_done = 0
    n_forces_skipped = 0
    for iat in range(n_atoms):
        for jat in range(n_replicas * n_atoms):
            is_computing = True
            m, j_small = np.unravel_index(jat, (n_replicas, n_atoms))
            if (distance_threshold is not None):
                dxij = atoms.positions[iat] - (list_of_replicas[m] + atoms.positions[j_small])
                if (np.linalg.norm(dxij) > distance_threshold):
                    is_computing = False
                    n_forces_skipped += 9
            if is_computing:
                if is_verbose:
                    logging.info('calculating forces on atoms: ' + str(iat) + ',' + str(jat))
                for icoord in range(3):
                    for jcoord in range(3):
                        value = calculate_single_third(atoms, replicated_atoms, iat, icoord, jat, jcoord,
                                                       third_order_delta)
                        for id in range(value.shape[0]):
                            i_at_sparse.append(iat)
                            i_coord_sparse.append(icoord)
                            jat_sparse.append(jat)
                            j_coord_sparse.append(jcoord)
                            k_sparse.append(id)
                            value_sparse.append(value[id])
                n_forces_done += 9
            if (n_forces_done + n_forces_skipped % 300) == 0:
                logging.info('Calculate third derivatives ' + str
                    (int((n_forces_done + n_forces_skipped) / n_forces_to_calculate * 100)) + '%')

    logging.info('total forces to calculate third : ' + str(n_forces_to_calculate))
    logging.info('forces calculated : ' + str(n_forces_done))
    logging.info('forces skipped (outside distance threshold) : ' + str(n_forces_skipped))
    coords = np.array([i_at_sparse, i_coord_sparse, jat_sparse, j_coord_sparse, k_sparse])
    shape = (n_atoms, 3, n_replicas * n_atoms, 3, n_replicas * n_atoms * 3)
    phifull = COO(coords, np.array(value_sparse), shape)
    phifull = phifull.reshape \
        ((n_atoms * 3, n_replicas * n_atoms * 3, n_replicas * n_atoms * 3))
    return phifull


def calculate_single_third(atoms, replicated_atoms, iat, icoord, jat, jcoord, third_order_delta):
    n_in_unit_cell = len(atoms.numbers)
    n_replicated_atoms = len(replicated_atoms.numbers)
    n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
    phi_partial = np.zeros((n_supercell * n_in_unit_cell * 3))
    for isign in (1, -1):
        for jsign in (1, -1):
            shift = np.zeros((n_replicated_atoms, 3))
            shift[iat, icoord] += isign * third_order_delta
            shift[jat, jcoord] += jsign * third_order_delta
            phi_partial[:] += isign * jsign * calculate_single_third_with_shift(atoms, replicated_atoms, shift)
    return phi_partial / (4. * third_order_delta * third_order_delta)


def calculate_single_third_with_shift(atoms, replicated_atoms, shift):
    n_in_unit_cell = len(atoms.numbers)
    n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
    phi_partial = np.zeros((n_supercell * n_in_unit_cell * 3))
    phi_partial[:] = (-1. * calculate_gradient(replicated_atoms.positions + shift, replicated_atoms))
    return phi_partial

def load_single_second(folder, index, n_atoms):
    second_per_atom = np.zeros((3, n_atoms * 3))
    digits = len(str(n_atoms))
    alpha_names = ['x', 'y', 'z']
    move_names = [None, 'p', 'm']
    for alpha in range(3):
        for move in (-1, 1):
            atoms = ase.io.read(folder+'/'+index.zfill(digits)+'_'+alpha_names[alpha]+move_names[move]+'.xyz', format='extxyz')
            forces = atoms.get_forces()
            second_per_atom[alpha, :] += move * np.reshape(forces, forces.size)
    return second_per_atom


"""
kaldo
Anharmonic Lattice Dynamics
"""
from ase.io import read
from hiphive import ForceConstants
import numpy as np

from kaldo.helpers.logger import get_logger, log_size

logging = get_logger()


def import_second_from_hiphive(folder, n_replicas, n_atoms):
    second_hiphive_file = str(folder) + '/model2.fcs'
    fcs2 = ForceConstants.read(second_hiphive_file)
    second_order = fcs2.get_fc_array(2).transpose(0, 2, 1, 3)
    second_order = second_order.reshape((n_replicas, n_atoms, 3,
                                         n_replicas, n_atoms, 3))
    second_order = second_order[0, np.newaxis]
    return second_order


def import_third_from_hiphive(atoms, supercell, folder):
    third_hiphive_file = str(folder) + '/model3.fcs'
    supercell = np.array(supercell)
    replicated_atom_prime_file = str(folder) + '/replicated_atoms.xyz'
    try:
        replicated_atoms = read(replicated_atom_prime_file)
    except FileNotFoundError:
        logging.warning(
            'Replicated atoms file not found. Please check if the file exists. Using the unit cell atoms instead.')
        replicated_atoms = atoms * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])
    # Derive constants used for third-order reshape
    n_prim = atoms.positions.shape[0]
    n_sc = np.prod(supercell)
    pbc_condition = replicated_atoms.get_pbc()
    dim = len(pbc_condition[pbc_condition == True])
    fcs3 = ForceConstants.read(third_hiphive_file)
    third_order = fcs3.get_fc_array(3).transpose(0, 3, 1, 4, 2, 5).reshape(n_sc, n_prim, dim,
                                                                           n_sc, n_prim, dim, n_sc, n_prim, dim)
    return third_order

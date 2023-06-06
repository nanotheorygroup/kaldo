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
    try:
        second_order = second_order.reshape((1, n_atoms, 3,
                                             1, n_atoms, 3))
        logging.info('Reduced second order matrix loaded.')
    except ValueError:
        second_order = second_order.reshape((n_replicas, n_atoms, 3,
                                             n_replicas, n_atoms, 3))
        logging.info('Full second order matrix loaded.')

    second_order = second_order[0, np.newaxis]
    return second_order


def import_third_from_hiphive(atoms, supercell, folder):
    third_hiphive_file = str(folder) + '/model3.fcs'
    supercell = np.array(supercell)
    n_prim = atoms.positions.shape[0]
    n_sc = np.prod(supercell)
    fcs3 = ForceConstants.read(third_hiphive_file)
    third_order = fcs3.get_fc_array(3).transpose(0, 3, 1, 4, 2, 5)
    try:
        third_order = third_order.reshape((n_prim * 3, n_prim * 3, n_prim * 3))
    except ValueError:
        third_order = third_order.reshape((n_sc * n_prim * 3, n_sc * n_prim * 3, n_sc * n_prim * 3))
    return third_order

"""
Ballistico
Anharmonic Lattice Dynamics
"""
import numpy as np
import time
from itertools import takewhile, repeat
from ballistico.helpers.logger import get_logger
logging = get_logger()



def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            logging.info('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed



def count_rows(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
    return sum(buf.count(b'\n') for buf in bufgen if buf)


def convert_to_poscar(atoms, supercell=None):
    list_of_types = []
    for symbol in atoms.get_chemical_symbols():
        for i in range(np.unique(atoms.get_chemical_symbols()).shape[0]):
            if np.unique(atoms.get_chemical_symbols())[i] == symbol:
                list_of_types.append(str(i))

    poscar = {'lattvec': atoms.cell / 10,
              'positions': (atoms.positions.dot(np.linalg.inv(atoms.cell))).T,
              'elements': atoms.get_chemical_symbols(),
              'types': list_of_types}
    if supercell is not None:
        poscar['na'] = supercell[0]
        poscar['nb'] = supercell[1]
        poscar['nc'] = supercell[2]
    return poscar

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


def wrap_coordinates(dxij, cell=None, cell_inv=None):
    # exploit periodicity to calculate the shortest distance, which may not be the one we have
    if cell is not None and cell_inv is None:
        cell_inv = np.linalg.inv(cell)
    if cell is not None:
        dxij = dxij.dot(cell_inv)
    dxij = dxij - np.round(dxij)
    if cell is not None:
        dxij = dxij.dot(cell)
    return dxij


def unwrap_positions_with_cell(dxij, cell=None, cell_inv=None):
    # exploit periodicity to calculate the shortest distance, which may not be the one we have
    if cell is not None and cell_inv is None:
        cell_inv = np.linalg.inv(cell)
    if cell is not None:
        dxij = dxij.dot(cell_inv)
    dxij = dxij + np.round(dxij)
    dxij = (dxij + (dxij < 0).astype(np.int))
    if cell is not None:
        dxij = dxij.dot(cell)
    return dxij


def convert_to_spg_structure(atoms):
    cell = atoms.cell
    scaled_positions = atoms.get_positions().dot(np.linalg.inv(atoms.cell))
    spg_struct = (cell, scaled_positions, atoms.get_atomic_numbers())
    return spg_struct


def q_index_from_q_vec(q_vec, kpts):
    # the input q_vec is in the unit sphere
    rescaled_qpp = np.round((q_vec * kpts).T, 0).astype(np.int)
    q_index = np.ravel_multi_index(rescaled_qpp, kpts, mode='wrap')
    return q_index


def q_vec_from_q_index(q_index, kpts):
    # the output q_vec is in the unit sphere
    q_vec = np.array(np.unravel_index(q_index, (kpts))).T / kpts
    wrap_coordinates(q_vec)
    return q_vec


def allowed_index_qpp(index_q, is_plus, kpts):
    n_k_points = np.prod(kpts)
    index_qp_full = np.arange(n_k_points)
    q_vec = q_vec_from_q_index(index_q, kpts)
    qp_vec = q_vec_from_q_index(index_qp_full, kpts)
    qpp_vec = q_vec[np.newaxis, :] + (int(is_plus) * 2 - 1) * qp_vec[:, :]
    index_qpp_full = q_index_from_q_vec(qpp_vec, kpts)
    return index_qpp_full


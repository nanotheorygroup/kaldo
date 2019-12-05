"""
Ballistico
Anharmonic Lattice Dynamics
"""
import numpy as np
import time
import os
from itertools import takewhile, repeat

FOLDER_NAME = 'output'
LAZY_PREFIX = '_lazy__'

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def create_folder(phonons, is_reduced_path):
    if phonons.folder:
        folder = phonons.folder
    else:
        folder = FOLDER_NAME
    if phonons.n_k_points > 1:
        kpts = phonons.kpts
        folder += '/' + str(kpts[0]) + '_' + str(kpts[1]) + '_' + str(kpts[2])
    if not is_reduced_path:
        folder += '/' + str(phonons.temperature)
        if phonons.is_classic:
            folder += '/classic'
        else:
            folder += '/quantum'
        if phonons.sigma_in is not None:
            try:
                phonons.sigma_in.size
            except AttributeError:
                folder += '/' + str(phonons.sigma_in)
            else:
                folder += '/vec_' + str(phonons.sigma_in[-2])
                print(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def lazy_property(is_storing, is_reduced_path):
    def _lazy_property(fn):
        attr = LAZY_PREFIX + fn.__name__
        @property
        def __lazy_property(self):
            if not hasattr(self, attr):
                if is_storing:
                    folder = create_folder(self, is_reduced_path)
                    filename = folder + '/' + fn.__name__ + '.npy'
                    try:
                        loaded_attr = np.load (filename)
                    except FileNotFoundError:
                        print(filename, 'not found, calculating', fn.__name__)
                        loaded_attr = fn(self)
                        np.save (filename, loaded_attr)
                    else:
                        print('loading', filename)
                else:
                    loaded_attr = fn(self)
                setattr(self, attr, loaded_attr)
            return getattr(self, attr)

        __lazy_property.__doc__ = fn.__doc__
        return __lazy_property
    return _lazy_property

def is_calculated(property, self, is_reduced_path=False):
    attr = LAZY_PREFIX + property
    try:
        getattr(self, attr)
    except AttributeError:
        try:
            folder = create_folder(self, is_reduced_path)
            filename = folder + '/' + property + '.npy'
            loaded_attr = np.load(filename)
            setattr(self, attr, loaded_attr)
            return True
        except FileNotFoundError:
            return False
    else:
        return True

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


def apply_boundary_with_cell(dxij, replicated_cell=None, replicated_cell_inv=None):
    # exploit periodicity to calculate the shortest distance, which may not be the one we have
    if replicated_cell is not None:
        dxij = dxij.dot(replicated_cell_inv)
    dxij = dxij - np.round(dxij)
    if replicated_cell is not None:
        dxij = dxij.dot(replicated_cell)
    return dxij


def convert_to_spg_structure(atoms):
    cell = atoms.cell
    scaled_positions = atoms.get_positions().dot(np.linalg.inv(atoms.cell))
    spg_struct = (cell, scaled_positions, atoms.get_atomic_numbers())
    return spg_struct
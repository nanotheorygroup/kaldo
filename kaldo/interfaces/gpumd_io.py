"""
kaldo
Anharmonic Lattice Dynamics

I/O for force constants exported from the GPUMD ecosystem (NEP potentials via
calorine + phono3py), serialized as a single compact ``gpumd_fc.npz``. See the
``api_forceconstants`` documentation for the file-format contract.
"""
import os

import numpy as np
import sparse
from ase import Atoms

from kaldo.helpers.logger import get_logger

logging = get_logger()

GPUMD_FC_FILE = 'gpumd_fc.npz'
SUPPORTED_VERSION = 1
EXPECTED_UNITS_FC2 = 'eV/angstrom^2'
EXPECTED_UNITS_FC3 = 'eV/angstrom^3'


def _as_str(value):
    # np.savez stores python str as a 0-d array of dtype '<U..'
    return str(np.asarray(value).item())


def read_gpumd_fc(folder):
    """Read a ``gpumd_fc.npz`` archive and return geometry + force-constant arrays.

    Returns a dict with keys: ``atoms`` (ase.Atoms unit cell), ``supercell``,
    ``third_supercell`` (int 3-tuples), ``fc2`` (float64, shape
    ``(1, n_uc, 3, n_rep2, n_uc, 3)``, eV/A^2), ``fc3`` (sparse.COO, shape
    ``(n_uc*3, n_rep3*n_uc*3, n_rep3*n_uc*3)``, eV/A^3), and
    ``acoustic_sum_applied`` (bool).
    """
    path = os.path.join(str(folder), GPUMD_FC_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found.')
    with np.load(path, allow_pickle=False) as npz:
        version = int(npz['format_version'])
        if version != SUPPORTED_VERSION:
            raise ValueError(f'Unsupported gpumd_fc format_version {version} '
                             f'(supported: {SUPPORTED_VERSION}).')
        units_fc2 = _as_str(npz['units_fc2'])
        units_fc3 = _as_str(npz['units_fc3'])
        if units_fc2 != EXPECTED_UNITS_FC2 or units_fc3 != EXPECTED_UNITS_FC3:
            raise ValueError(f'Unexpected units in gpumd_fc.npz: fc2={units_fc2!r}, '
                             f'fc3={units_fc3!r}; expected {EXPECTED_UNITS_FC2!r}/'
                             f'{EXPECTED_UNITS_FC3!r} (bare IFCs, not mass-weighted).')
        if _as_str(npz['grid_order']) != 'C':
            raise ValueError("gpumd_fc.npz must use grid_order='C'.")

        atoms = Atoms(numbers=npz['atomic_numbers'].astype(int),
                      positions=npz['positions'].astype(np.float64),
                      cell=npz['cell'].astype(np.float64), pbc=True)
        supercell = tuple(int(x) for x in npz['supercell'])
        third_supercell = tuple(int(x) for x in npz['third_supercell'])
        fc2 = np.ascontiguousarray(npz['fc2'], dtype=np.float64)
        fc3 = sparse.COO(npz['fc3_coords'].astype(np.int64),
                         npz['fc3_data'].astype(np.float64),
                         shape=tuple(int(x) for x in npz['fc3_shape']))
        acoustic_sum_applied = bool(npz['acoustic_sum_applied'])

    return {'atoms': atoms, 'supercell': supercell, 'third_supercell': third_supercell,
            'fc2': fc2, 'fc3': fc3, 'acoustic_sum_applied': acoustic_sum_applied}

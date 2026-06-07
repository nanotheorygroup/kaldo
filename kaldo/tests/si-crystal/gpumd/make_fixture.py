"""Generate the committed gpumd_fc.npz fixture from the existing hiphive Si fixture.

Run from the kaldo repo root:
    python kaldo/tests/si-crystal/gpumd/make_fixture.py
This keeps the gpumd fixture numerically identical to the trusted hiphive route,
so the reader test is a true differential oracle.
"""
import os

import numpy as np

from kaldo.forceconstants import ForceConstants

HERE = os.path.dirname(__file__)
HIPHIVE = os.path.join(HERE, '..', 'hiphive')
SUPERCELL = (3, 3, 3)  # match the si-crystal hiphive fixture's supercell


def main():
    fc = ForceConstants.from_folder(folder=HIPHIVE, supercell=SUPERCELL, format='hiphive')
    atoms = fc.atoms
    n_uc = atoms.positions.shape[0]
    n_rep = int(np.prod(SUPERCELL))

    fc2 = np.ascontiguousarray(fc.second.value, dtype=np.float64)
    assert fc2.shape == (1, n_uc, 3, n_rep, n_uc, 3), fc2.shape

    third_raw = np.asarray(fc.third.value, dtype=np.float64)  # dense or sparse
    import sparse as _sparse
    if isinstance(third_raw, _sparse.COO):
        third = third_raw
    else:
        third = _sparse.COO.from_numpy(third_raw)

    np.savez_compressed(
        os.path.join(HERE, 'gpumd_fc.npz'),
        format_version=np.int64(1),
        atomic_numbers=atoms.get_atomic_numbers().astype(np.int64),
        positions=atoms.get_positions().astype(np.float64),
        cell=np.array(atoms.cell).astype(np.float64),
        supercell=np.array(SUPERCELL, dtype=np.int64),
        third_supercell=np.array(SUPERCELL, dtype=np.int64),
        fc2=fc2,
        fc3_coords=third.coords.astype(np.int32),
        fc3_data=third.data.astype(np.float64),
        fc3_shape=np.array(third.shape, dtype=np.int64),
        units_fc2='eV/angstrom^2', units_fc3='eV/angstrom^3',
        grid_order='C', acoustic_sum_applied=np.bool_(False),
        nep_potential='derived-from-hiphive-fixture', generator='make_fixture.py')
    print('wrote', os.path.join(HERE, 'gpumd_fc.npz'))


if __name__ == '__main__':
    main()

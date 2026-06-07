import numpy as np
import sparse
from ase import Atoms

from kaldo.interfaces import gpumd_io


def _write_minimal_npz(path, n_uc=2, supercell=(1, 1, 1)):
    n_rep = int(np.prod(supercell))
    a = 5.43
    atoms = Atoms('Si2', scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
                  cell=np.eye(3) * a, pbc=True)
    fc2 = np.zeros((1, n_uc, 3, n_rep, n_uc, 3), dtype=np.float64)
    for i in range(n_uc):
        for alpha in range(3):
            fc2[0, i, alpha, 0, i, alpha] = 1.0  # diagonal self term
    shape3 = (n_uc * 3, n_rep * n_uc * 3, n_rep * n_uc * 3)
    coords = np.array([[0], [0], [0]], dtype=np.int32)
    data = np.array([0.5], dtype=np.float64)
    np.savez_compressed(
        path, format_version=np.int64(1),
        atomic_numbers=atoms.get_atomic_numbers().astype(np.int64),
        positions=atoms.get_positions().astype(np.float64),
        cell=np.array(atoms.cell).astype(np.float64),
        supercell=np.array(supercell, dtype=np.int64),
        third_supercell=np.array(supercell, dtype=np.int64),
        fc2=fc2, fc3_coords=coords, fc3_data=data,
        fc3_shape=np.array(shape3, dtype=np.int64),
        units_fc2='eV/angstrom^2', units_fc3='eV/angstrom^3',
        grid_order='C', acoustic_sum_applied=np.bool_(False),
        nep_potential='test', generator='test')


def test_read_gpumd_fc_returns_geometry_and_arrays(tmp_path):
    folder = tmp_path
    _write_minimal_npz(folder / 'gpumd_fc.npz', supercell=(1, 1, 1))
    meta = gpumd_io.read_gpumd_fc(str(folder))
    assert meta['atoms'].get_chemical_symbols() == ['Si', 'Si']
    assert tuple(meta['supercell']) == (1, 1, 1)
    assert meta['fc2'].shape == (1, 2, 3, 1, 2, 3)
    assert meta['fc2'].dtype == np.float64
    assert isinstance(meta['fc3'], sparse.COO)
    assert meta['fc3'].shape == (6, 6, 6)


def test_read_gpumd_fc_rejects_wrong_units(tmp_path):
    import numpy as _np
    p = tmp_path / 'gpumd_fc.npz'
    _write_minimal_npz(p)
    data = dict(_np.load(p, allow_pickle=True))
    data['units_fc2'] = _np.array('Ry/bohr^2')
    _np.savez_compressed(p, **data)
    import pytest
    with pytest.raises(ValueError, match='units'):
        gpumd_io.read_gpumd_fc(str(tmp_path))


# Task A.2 — SecondOrder.load
from kaldo.observables.secondorder import SecondOrder  # noqa: E402


def test_secondorder_load_gpumd(tmp_path):
    _write_minimal_npz(tmp_path / 'gpumd_fc.npz', supercell=(1, 1, 1))
    so = SecondOrder.load(folder=str(tmp_path), format='gpumd')
    assert tuple(so.supercell) == (1, 1, 1)
    assert so.value.shape == (1, 2, 3, 1, 2, 3)
    np.testing.assert_allclose(so.value[0, 0, 0, 0, 0, 0], 1.0)


# Task A.3 — ThirdOrder.load
from kaldo.observables.thirdorder import ThirdOrder  # noqa: E402


def test_thirdorder_load_gpumd(tmp_path):
    _write_minimal_npz(tmp_path / 'gpumd_fc.npz', supercell=(1, 1, 1))
    to = ThirdOrder.load(folder=str(tmp_path), format='gpumd')
    assert to.value.shape == (6, 6, 6)
    np.testing.assert_allclose(to.value[0, 0, 0], 0.5)


def test_thirdorder_load_gpumd_threshold(tmp_path):
    _write_minimal_npz(tmp_path / 'gpumd_fc.npz', supercell=(1, 1, 1))
    to = ThirdOrder.load(folder=str(tmp_path), format='gpumd', third_energy_threshold=1.0)
    assert to.value.nnz == 0  # 0.5 < 1.0 threshold drops the only entry


# Task A.4 — ForceConstants.from_folder
from kaldo.forceconstants import ForceConstants  # noqa: E402


def test_from_folder_gpumd(tmp_path):
    _write_minimal_npz(tmp_path / 'gpumd_fc.npz', supercell=(1, 1, 1))
    fc = ForceConstants.from_folder(folder=str(tmp_path), format='gpumd')
    assert fc.second.value.shape == (1, 2, 3, 1, 2, 3)
    assert fc.third.value.shape == (6, 6, 6)
    assert tuple(fc.supercell) == (1, 1, 1)

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


# Task A.5 — differential round-trip oracle from the existing si-crystal hiphive fixture
import os  # noqa: E402

_SI = os.path.join(os.path.dirname(__file__), 'si-crystal')


def test_gpumd_matches_hiphive_oracle():
    sc = (3, 3, 3)
    fc_hip = ForceConstants.from_folder(folder=os.path.join(_SI, 'hiphive'), supercell=sc, format='hiphive')
    fc_gpu = ForceConstants.from_folder(folder=os.path.join(_SI, 'gpumd'), format='gpumd')
    np.testing.assert_allclose(fc_gpu.second.value, fc_hip.second.value, rtol=1e-12, atol=1e-12)
    a = fc_hip.third.value.todense() if hasattr(fc_hip.third.value, 'todense') else fc_hip.third.value
    b = fc_gpu.third.value.todense() if hasattr(fc_gpu.third.value, 'todense') else fc_gpu.third.value
    np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-12)


# Task C.4 — transport-level regression locking format='gpumd' to format='hiphive' oracle
from kaldo.conductivity import Conductivity  # noqa: E402
from kaldo.phonons import Phonons  # noqa: E402


def _kappa_inverse(fc, folder):
    """Compute isotropic inverse-BTE kappa (W/mK) for the given ForceConstants."""
    phonons = Phonons(
        forceconstants=fc,
        kpts=[3, 3, 3],
        is_classic=False,
        temperature=300,
        folder=folder,
        storage='memory',
    )
    cond = Conductivity(phonons=phonons, method='inverse', storage='memory').conductivity.sum(axis=0)
    return float(np.abs(np.mean(cond.diagonal())))


def test_gpumd_transport_matches_hiphive(tmp_path):
    """Lock the format='gpumd' BTE conductivity to the format='hiphive' oracle value.

    The two routes load identical force-constant arrays (fc2 and fc3 are bit-for-bit equal,
    confirmed by test_gpumd_matches_hiphive_oracle) AND the fixture records the same replica
    grid order the hiphive route uses, so the gpumd reader reconstructs an identical
    replica->R-vector mapping.  The two BTE conductivities are therefore numerically equal.
    Both also agree with kaldo's published Si reference of ~154 W/mK to 2 significant figures.
    """
    sc = (3, 3, 3)
    fc_gpumd = ForceConstants.from_folder(folder=os.path.join(_SI, 'gpumd'), format='gpumd')
    fc_hiphive = ForceConstants.from_folder(folder=os.path.join(_SI, 'hiphive'), supercell=sc, format='hiphive')

    kappa_gpumd = _kappa_inverse(fc_gpumd, str(tmp_path / 'gpumd'))
    kappa_hiphive = _kappa_inverse(fc_hiphive, str(tmp_path / 'hiphive'))

    # Same force constants, same replica ordering: kappa must agree to numerical precision.
    np.testing.assert_allclose(kappa_gpumd, kappa_hiphive, rtol=1e-6,
                               err_msg=f'gpumd kappa {kappa_gpumd:.4f} vs hiphive {kappa_hiphive:.4f}')

    # Lock the gpumd path to the known Si reference value (~154 W/mK, 2 sig figs).
    np.testing.assert_approx_equal(kappa_gpumd, 154, significant=2)

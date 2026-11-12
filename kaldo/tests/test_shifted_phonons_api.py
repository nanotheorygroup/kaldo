import logging

import pytest

from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons

SHIFT = (0.5, 0.5, 0.5)


@pytest.fixture(scope="session")
def amorphous_fc():
    return ForceConstants.from_folder(folder="kaldo/tests/si-amorphous", format="eskm")


def _shifted(fc, tmp_path, **kwargs):
    options = dict(kpts=(2, 2, 2), kpts_shift=SHIFT, third_bandwidth=0.5, temperature=300, folder=str(tmp_path))
    options.update(kwargs)
    return Phonons(forceconstants=fc, **options)


def test_shift_warns_rta_only(amorphous_fc, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _shifted(amorphous_fc, tmp_path)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert sum("only Conductivity(method='rta')" in m for m in messages) == 1


def test_unshifted_storage_path_unchanged(amorphous_fc, tmp_path):
    ph = Phonons(forceconstants=amorphous_fc, kpts=(2, 2, 2), temperature=300, folder=str(tmp_path))
    folder = ph.get_folder_from_label('')
    assert "/2_2_2/ifc_" in folder
    assert "_shift_" not in folder


def test_shifted_storage_path_distinct(amorphous_fc, tmp_path):
    ph = _shifted(amorphous_fc, tmp_path)
    assert "2_2_2_shift_0.5_0.5_0.5" in ph.get_folder_from_label('')
    assert ph.kpts_shift.dtype.name == "float64"
    assert ph._is_shifted_grid


@pytest.mark.parametrize("method", ["inverse", "sc", "full", "qhgk"])
def test_shifted_rejects_non_rta(amorphous_fc, tmp_path, method):
    cond = Conductivity(_shifted(amorphous_fc, tmp_path), method=method, storage="numpy")
    with pytest.raises(ValueError, match="shifted"):
        cond.conductivity


@pytest.mark.parametrize("kwargs", [dict(use_q_symmetry=True), dict(broadening_shape="lorentz"),
                                    dict(third_bandwidth=None), dict(is_balanced=True),
                                    dict(include_isotopes=True), dict(kpts=(3, 3, 3))])
def test_shifted_rejects_unsupported_anharmonic(amorphous_fc, tmp_path, kwargs):
    ph = _shifted(amorphous_fc, tmp_path, **kwargs)
    with pytest.raises(NotImplementedError):
        ph._check_shifted_anharmonic_support()


@pytest.mark.parametrize("kwargs", [dict(is_unfolding=True), dict(is_symmetrizing_frequency=True),
                                    dict(is_antisymmetrizing_velocity=True)])
def test_shifted_rejects_unsupported_harmonic_options(amorphous_fc, tmp_path, kwargs):
    with pytest.raises(NotImplementedError):
        _shifted(amorphous_fc, tmp_path, **kwargs)


def test_shifted_rejects_crystal_supercell(tmp_path):
    fc = ForceConstants.from_folder(folder="kaldo/tests/si-crystal/vasp", supercell=[5, 5, 5],
                                    third_supercell=[5, 5, 5], format="vasp-sheng")
    ph = _shifted(fc, tmp_path)
    with pytest.raises(NotImplementedError, match="supercell"):
        ph._check_shifted_anharmonic_support()

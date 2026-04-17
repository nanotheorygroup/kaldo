import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.phonons import Phonons


@pytest.fixture(scope="module")
def nac_second_order():
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    return forceconstants.second


def test_harmonic_with_q_accepts_nac_options(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
        q_index=7,
    )
    assert phonon.nac_method == "gonze"
    assert phonon.nac_debug is True
    assert phonon.nac_debug_folder == "debug"
    assert phonon.q_index == 7


def test_unknown_nac_method_raises_value_error(nac_second_order):
    with pytest.raises(ValueError, match="Unknown nac_method"):
        HarmonicWithQ(
            q_point=np.array([0.1, 0.0, 0.1]),
            second=nac_second_order,
            storage="memory",
            nac_method="bad-method",
        )


def test_gonze_velocity_raises_not_implemented(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
    )
    with pytest.raises(NotImplementedError, match="Gonze-Lee velocity"):
        _ = phonon.velocity


def test_phonons_stores_nac_options():
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[1, 1, 1],
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
    )
    assert phonons.nac_method == "gonze"
    assert phonons.nac_debug is True
    assert phonons.nac_debug_folder == "debug"


def test_gonze_debug_folder_for_index(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
        q_index=3,
    )
    assert phonon._gonze_debug_q_folder().as_posix() == "debug/q-00003"


def test_gonze_debug_folder_for_single_q(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
    )
    assert phonon._gonze_debug_q_folder().as_posix() == "debug/q_0p1_0p0_0p1"

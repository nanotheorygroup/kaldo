import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.interfaces import shengbte_io
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.phonons import Phonons


def attach_reference_nac(second_order, nac_file="examples/nacl_phonopy/espresso.ifc2"):
    _, _, charges = shengbte_io.read_second_order_qe_matrix(nac_file)
    if charges is None:
        raise ValueError(f"No NAC data found in {nac_file}")
    second_order.atoms.info["dielectric"] = charges[0, :, :]
    second_order.atoms.set_array("charges", charges[1:, :, :], shape=(3, 3))
    return second_order


@pytest.fixture(scope="module")
def nac_second_order(tmp_path_factory):
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    forceconstants.second.folder = str(tmp_path_factory.mktemp("wang_runtime_cache"))
    return attach_reference_nac(forceconstants.second)


def test_harmonic_with_q_accepts_wang_nac_options(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="wang",
        nac_debug=True,
        nac_debug_folder="debug",
        q_index=7,
    )
    assert phonon.nac_method == "wang"
    assert phonon.nac_debug is True
    assert phonon.nac_debug_folder == "debug"
    assert phonon.q_index == 7


def test_harmonic_with_q_rejects_wang_without_nac_metadata():
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    with pytest.raises(ValueError, match=r"nac_method='wang' requires"):
        HarmonicWithQ(
            q_point=np.array([0.1, 0.0, 0.1]),
            second=forceconstants.second,
            storage="memory",
            nac_method="wang",
        )


def test_phonons_accepts_wang_nac_options():
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    attach_reference_nac(forceconstants.second)
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[1, 1, 1],
        storage="memory",
        nac_method="wang",
        nac_debug=True,
        nac_debug_folder="debug",
    )
    assert phonons.nac_method == "wang"
    assert phonons.nac_debug is True
    assert phonons.nac_debug_folder == "debug"
    frequencies = phonons.frequency
    assert frequencies.shape == (1, phonons.n_modes)


def test_phonons_rejects_wang_without_nac_metadata():
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    with pytest.raises(ValueError, match=r"nac_method='wang' requires"):
        phonons = Phonons(
            forceconstants=forceconstants,
            kpts=[1, 1, 1],
            storage="memory",
            nac_method="wang",
        )
        _ = phonons.frequency

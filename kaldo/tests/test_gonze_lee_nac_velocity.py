from pathlib import Path

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.interfaces import shengbte_io
from kaldo.observables.harmonic_with_q import HarmonicWithQ


def attach_reference_nac(second_order, nac_file="examples/nacl_phonopy_v2/espresso.ifc2"):
    _, _, charges = shengbte_io.read_second_order_qe_matrix(nac_file)
    if charges is None:
        static_dir = Path(nac_file).parent / "debug" / "static"
        dielectric = np.load(static_dir / "dielectric.npy")
        born = np.load(static_dir / "born.npy")
    else:
        dielectric = charges[0, :, :]
        born = charges[1:, :, :]
    second_order.atoms.info["dielectric"] = dielectric
    second_order.atoms.set_array("charges", born, shape=(3, 3))
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
    forceconstants.second.folder = str(tmp_path_factory.mktemp("gonze_velocity_cache"))
    return attach_reference_nac(forceconstants.second)


def test_gonze_velocity_debug_data_contains_top_level_fields(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        is_unfolding=True,
        nac_method="gonze",
    )
    data = phonon._calculate_gonze_velocity_debug_data()
    assert data["dm_q"].shape == (6, 6)
    assert data["eigenvectors"].shape == (6, 6)
    assert sorted(data["directions"]) == ["d0", "d1", "d2", "d3"]

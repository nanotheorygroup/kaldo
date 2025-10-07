from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import numpy as np
import pytest


@pytest.fixture(scope="session")
def phonons():
    fcs = ForceConstants.from_folder(folder="kaldo/tests/si-tdep", supercell=(5, 5, 5), format="tdep")
    k = 6
    kpts = [k, k, k]
    temperature = 300
    is_classic = False
    phonons = Phonons(
        forceconstants=fcs,
        kpts=kpts,
        is_classic=is_classic,
        temperature=temperature,
        folder=".",
        storage="memory",
        is_unfolding=False,
    )
    return phonons


def test_tdep_conductivity_300(phonons):
    phonons.temperature = 300
    cond = Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0).diagonal().mean()
    expected_cond = 72.0
    np.testing.assert_approx_equal(cond, expected_cond, significant=2)

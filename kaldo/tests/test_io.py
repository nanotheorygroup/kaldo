"""
Unit and regression test for the kaldo package.
"""

import pytest
from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import numpy as np
import shutil
import os


@pytest.fixture(scope="session")
def phonons():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder("kaldo/tests/si-crystal/hiphive", supercell=[3, 3, 3], format="hiphive")
    k_points = 3
    phonons_config = {
        "kpts": [k_points, k_points, k_points],
        "is_classic": False,
        "temperature": 300,  # 'temperature'=300K
        "folder": "ALD_si_bulk",
        "storage": "formatted",
        "third_bandwidth": 0.5,  # fixed: gauge-invariant regression config (#290)
    }

    phonons = Phonons(forceconstants=forceconstants, **phonons_config)
    yield phonons
    
    # Cleanup: remove the ALD_si_bulk folder after tests complete
    if os.path.exists("ALD_si_bulk"):
        shutil.rmtree("ALD_si_bulk")
        print("Cleaned up ALD_si_bulk folder.")


def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory",
                        diffusivity_bandwidth=1.0).conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_allclose(cond, 2.251450, rtol=5e-3, atol=0.0)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_allclose(cond, 300.205922, rtol=5e-3, atol=0.0)

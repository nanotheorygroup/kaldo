"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import pytest


@pytest.fixture(scope="session")
def phonons():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-crystal", supercell=[3, 3, 3], format="eskm")
    phonons_config = {
        "kpts": [5, 5, 5],
        "is_classic": 0,
        "temperature": 300,
        "folder": "ald",
        "storage": "memory",
        "third_bandwidth": 0.1,
        "broadening_shape": "gauss",
        "is_balanced": True,
        "grid_type": "C",
    }

    phonons = Phonons(forceconstants=forceconstants, **phonons_config)
    return phonons


def test_detailed_balance(phonons):
    cond = Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0)
    cond_ref = np.array(
        [
            [531.17210271, 2.04926525, -2.42778198],
            [2.04926665, 530.92668135, -3.18903786],
            [-2.4277473, -3.18902935, 537.86321364],
        ]
    )
    np.testing.assert_array_almost_equal(cond, cond_ref, decimal=3)

"""
Unit and regression test for the kaldo package.
"""

import pytest
from kaldo.conductivity import Conductivity
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import numpy as np


@pytest.fixture(scope="session")
def phonons():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder('kaldo/tests/si-crystal/hiphive', supercell=[3, 3, 3], format='hiphive')
    k_points = 3
    phonons_config = {'kpts': [k_points, k_points, k_points],
                      'is_classic': False,
                      'temperature': 300,  # 'temperature'=300K
                      'folder': 'ALD_si_bulk',
                      'storage': 'formatted'}

    phonons = Phonons(forceconstants=forceconstants, **phonons_config)
    return phonons


def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method='qhgk', storage='memory').conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_approx_equal(cond, 141, significant=3)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method='inverse', storage='memory').conductivity.sum(axis=0).diagonal()))
    np.testing.assert_approx_equal(cond, 154, significant=3)

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
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/ge-crystal/d3q",
        supercell=[10, 10, 10],
        third_supercell=[3, 3, 3],
        format="qe-d3q")
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[3, 3, 3],
        # kpts=[14, 14, 14], # real-world case, test againest as example, and compare value with literature
        is_classic=False,
        temperature=300,
        storage="memory",
    )
    return phonons

def test_lagacy_format():
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/ge-crystal/d3q",
        supercell=[10, 10, 10],
        third_supercell=[3, 3, 3],
        format="qe-d3q")
    
    forceconstants2 = ForceConstants.from_folder(
        folder="kaldo/tests/ge-crystal/d3q",
        supercell=[10, 10, 10],
        third_supercell=[3, 3, 3],
        format="shengbte-d3q")
    
    np.testing.assert_equal(forceconstants.second.value, forceconstants2.second.value)
    np.testing.assert_equal(forceconstants.third.value, forceconstants2.third.value)


def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory").conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_approx_equal(cond, 21, significant=2)


def test_rta_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_approx_equal(cond, 20, significant=2)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_approx_equal(cond, 22, significant=2)

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
        folder="kaldo/tests/si-crystal/vasp",
        supercell=[5, 5, 5],
        third_supercell=[5, 5, 5],
        format="vasp-sheng")
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[3, 3, 3],
        is_classic=False,
        temperature=300,
        third_bandwidth=0.5,  # fixed: gauge-invariant regression config (#290)
        storage="memory",
    )
    return phonons

def test_lagacy_format():
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/si-crystal/vasp",
        supercell=[5, 5, 5],
        third_supercell=[5, 5, 5],
        format="vasp-sheng")
    
    forceconstants2 = ForceConstants.from_folder(
        folder="kaldo/tests/si-crystal/vasp",
        supercell=[5, 5, 5],
        third_supercell=[5, 5, 5],
        format="shengbte")
    
    np.testing.assert_equal(forceconstants.second.value, forceconstants2.second.value)
    np.testing.assert_equal(forceconstants.third.value, forceconstants2.third.value)


def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory",
                        diffusivity_bandwidth=1.0).conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_allclose(cond, 1.693686, rtol=5e-3, atol=0.0)


def test_rta_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_allclose(cond, 16.406325, rtol=5e-3, atol=0.0)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_allclose(cond, 16.576349, rtol=5e-3, atol=0.0)

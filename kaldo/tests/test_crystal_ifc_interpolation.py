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
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[3, 3, 3],
        is_classic=False,
        temperature=300,
        is_unfolding=True,
        third_bandwidth=0.5,  # fixed: gauge-invariant regression config (#290)
        storage="memory",
    )
    return phonons


def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory",
                        diffusivity_bandwidth=1.0).conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    # Pair-specific Wigner--Seitz ties replace the former single-image
    # representative in both the harmonic velocity and FC3 projection.
    np.testing.assert_allclose(cond, 2.252623, rtol=5e-3, atol=0.0)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_allclose(cond, 307.106420, rtol=5e-3, atol=0.0)

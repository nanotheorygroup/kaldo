"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import pytest
import os


@pytest.fixture(scope="session")
def phonons():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-crystal", supercell=[3, 3, 3], format="eskm")

    # Create a phonon object
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[5, 5, 5],
        is_classic=False,
        temperature=300,
        third_bandwidth=0.5,  # fixed: gauge-invariant regression config (#290)
        storage="memory",
    )

    return phonons


def test_sc_finite_size_conductivity_ms(phonons):
    cond_ms = np.abs(
        Conductivity(
            phonons=phonons,
            method="sc",
            max_n_iterations=71,
            storage="memory",
            length=(1e4, 0, 0),
            finite_length_method="ms",
        ).conductivity.sum(axis=0)[0, 0]
    )
    np.testing.assert_allclose(cond_ms, 210.577436, rtol=5e-3, atol=0.0)


def test_rta_finite_size_conductivity_ms(phonons):
    cond_ms = np.abs(
        Conductivity(
            phonons=phonons, method="rta", storage="memory", length=(1e4, 0, 0), finite_length_method="ms"
        ).conductivity.sum(axis=0)[0, 0]
    )
    np.testing.assert_allclose(cond_ms, 167.271969, rtol=5e-3, atol=0.0)


def test_inverse_finite_size_conductivity_ms(phonons):
    cond_ms = np.abs(
        Conductivity(
            phonons=phonons, method="inverse", storage="memory", length=(1e4, 0, 0), finite_length_method="ms"
        ).conductivity.sum(axis=0)[0, 0]
    )
    np.testing.assert_allclose(cond_ms, 198.764287, rtol=5e-3, atol=0.0)

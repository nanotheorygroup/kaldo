"""
Unit and regression test for the kaldo package.

This file is the designated smoke coverage for the DEFAULT (ShengBTE)
adaptive-broadening kernel (see issue #290): that kernel builds per-pair
sigma from velocity differences, which are not invariant under the
eigenvector gauge freedom in degenerate subspaces, so its conductivities
are deterministic per machine but drift by up to a few percent across
BLAS/LAPACK backends. Tolerances here are therefore deliberately loose
sanity bands. Tight regression pinning lives in the per-format test files,
which use gauge-invariant configurations (broadening_kernel='tdep' or
fixed bandwidths).
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
        folder='data',
        forceconstants=forceconstants,
        kpts=[5, 5, 5],
        is_classic=False,
        temperature=300,
        storage="memory",
    )
    phonons.frequency
    return phonons

def test_phase_space(phonons):
    phase_space = phonons.phase_space.sum()
    np.testing.assert_allclose(phase_space, 113.0, rtol=1e-2, atol=0.0)



def test_sc_conductivity(phonons):
    cond = np.abs(
        np.mean(
            Conductivity(phonons=phonons, method="sc", max_n_iterations=71, storage="memory")
            .conductivity.sum(axis=0)
            .diagonal()
        )
    )
    np.testing.assert_allclose(cond, 255.0, rtol=1e-2, atol=0.0)


def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory").conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_allclose(cond, 230.0, rtol=1e-2, atol=0.0)


def test_rta_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_allclose(cond, 226.0, rtol=1e-2, atol=0.0)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_allclose(cond, 256.0, rtol=1e-2, atol=0.0)

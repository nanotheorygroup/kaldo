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
        folder="kaldo/tests/ge-crystal/vasp-d3q",
        supercell=[5, 5, 5],
        third_supercell=[3, 3, 3],
        format="vasp-d3q")
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[3, 3, 3],
        is_classic=False,
        temperature=300,
        storage="memory",
    )
    return phonons

# Reference values re-pinned when the vasp-* loaders were fixed to declare
# the C replica order their readers actually produce (previously declared F
# over C-ordered data). On this cubic fixture the conductivity shift from
# the relabeling is comparable to the cross-backend spread from the
# eigenvector-basis freedom in degenerate subspaces (see #270), so these
# tests are regression sanity bands, not enumeration detectors: references
# sit at the midpoint of the values observed on x86-64/OpenBLAS
# (0.813 / 0.676 / 0.725) and arm64/Accelerate (0.837 / 0.700 / 0.750),
# with rtol covering that spread. The enumeration itself is pinned by
# test_elastic_vasp (cubic tensor form holds at machine precision only
# under C pairing) and by the relabeling-invariance test in
# test_elastic.py.


def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory").conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_allclose(cond, 0.825, rtol=3e-2, atol=0.0)


def test_rta_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_allclose(cond, 0.688, rtol=3e-2, atol=0.0)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    np.testing.assert_allclose(cond, 0.738, rtol=3e-2, atol=0.0)

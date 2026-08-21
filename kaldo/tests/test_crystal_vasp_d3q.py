"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import pytest

# Gauge-invariant regression configuration (issue #290): a fixed
# third_bandwidth (and, for QHGK, a fixed diffusivity_bandwidth) removes the
# eigenvector-gauge sensitivity of the default adaptive broadening, so these
# goldens are machine-independent and pinned tightly. Default-kernel smoke
# coverage lives in test_crystal.py (and one Ge smoke below, where present).


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
        third_bandwidth=0.5,
        storage="memory",
    )
    return phonons

# References pinned under pair-specific Wigner-Seitz interpolation for the
# VASP IFC2 (d3q IFC3 keeps its native periodic convention).


def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory",
                        diffusivity_bandwidth=1.0).conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_allclose(cond, 0.681305, rtol=5e-3, atol=0.0)


def test_rta_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    # Wider band: residual gauge spread across backends.
    np.testing.assert_allclose(cond, 0.412536, rtol=3e-2, atol=0.0)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    # See the residual-gauge note on test_rta_conductivity.
    np.testing.assert_allclose(cond, 0.437976, rtol=3e-2, atol=0.0)


def test_qhgk_conductivity_default_kernel_smoke():
    """Smoke coverage of the DEFAULT adaptive-broadening path on Ge (#290).

    The default ShengBTE kernel is eigenvector-gauge sensitive in degenerate
    subspaces, so its value is deterministic per machine but spreads ~3%
    across BLAS backends. The band is deliberately wide: this test exercises
    the default path, it does not pin physics. Tight pinning lives in the
    fixed-bandwidth tests above.
    """
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
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory").conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_allclose(cond, 0.548183, rtol=3e-2, atol=0.0)

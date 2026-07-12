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
    cond = Conductivity(phonons=phonons, method="qhgk", storage="memory",
                        diffusivity_bandwidth=1.0).conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_allclose(cond, 0.768016, rtol=5e-3, atol=0.0)


def test_rta_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    # Residual gauge sensitivity (issue #290): even with fixed sigma, per-mode
    # |Phi3|^2 and velocities remain gauge-covariant within degenerate clusters,
    # and on this strongly degenerate, low-kappa fixture the mode-summed BTE
    # value still spreads ~2% across backends (arm64 0.6685, x86-64 0.6547).
    # Band centered between the measured backends; only deterministic gauge
    # canonicalization can tighten this further.
    np.testing.assert_allclose(cond, 0.6616, rtol=3e-2, atol=0.0)


def test_inverse_conductivity(phonons):
    cond = np.abs(
        np.mean(Conductivity(phonons=phonons, method="inverse", storage="memory").conductivity.sum(axis=0).diagonal())
    )
    # See the residual-gauge note on test_rta_conductivity (#290):
    # arm64 0.7371, x86-64 0.7194, band centered between them.
    np.testing.assert_allclose(cond, 0.7283, rtol=3e-2, atol=0.0)


def test_qhgk_conductivity_default_kernel_smoke():
    """Smoke coverage of the DEFAULT adaptive-broadening path on Ge (#290).

    The default ShengBTE kernel is eigenvector-gauge sensitive in degenerate
    subspaces, so its value is deterministic per machine but spreads ~3%
    across BLAS backends (measured arm64/Accelerate 0.837 vs x86-64/OpenBLAS
    0.813). The band is centered between the observed backends and is
    deliberately wide: this test exercises the default path, it does not pin
    physics. Tight pinning lives in the fixed-bandwidth tests above.
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
    np.testing.assert_allclose(cond, 0.825, rtol=3e-2, atol=0.0)

"""Behavioral contract between NAC activation and ``is_unfolding``.

The flag selects the ordinary harmonic interpolation only.  Without NAC, an
incommensurate q point may distinguish the legacy periodic-replica Fourier sum
from Wigner--Seitz interpolation, while the two must coincide on the defining
commensurate mesh.  With generic Gonze NAC active, the NAC controller owns the
Wigner--Seitz short-range reconstruction, just as the QE controller does, so
the legacy flag must become observationally inert.

These tests use public frequencies and velocities rather than private mapping
arrays.  This makes the contract sensitive both to eigensystem routing and to
the separate dynamical-matrix derivative routing used by transport.
"""

from itertools import product

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ


NONPOLAR_SUPERCELL = (2, 2, 2)
INCOMMENSURATE_Q = np.array([0.10, 0.20, 0.30], dtype=np.float64)
GONZE_BVK_MATRIX = np.diag([8, 8, 8]).astype(int)
GONZE_Q = np.array([0.073, 0.041, 0.029], dtype=np.float64)


@pytest.fixture(scope="module")
def nonpolar_second(tmp_path_factory):
    """Build deterministic nonpolar IFCs on a small conventional Cu cell."""
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    atoms.wrap()
    forceconstants = ForceConstants(
        atoms=atoms,
        supercell=NONPOLAR_SUPERCELL,
        folder=str(tmp_path_factory.mktemp("nonpolar_unfolding_contract")),
    )
    forceconstants.second.calculate(
        calculator=EMT(),
        delta_shift=1.0e-3,
        is_storing=False,
        symmetrize=False,
        use_symmetry=False,
        n_workers=1,
    )
    assert "dielectric" not in forceconstants.second.atoms.info
    assert "charges" not in forceconstants.second.atoms.arrays
    return forceconstants.second


@pytest.fixture(scope="module")
def gonze_second(tmp_path_factory):
    """Load total polar IFCs through the generic VASP/Phonopy convention."""
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    second = forceconstants.second
    # This fixture supplies total IFCs and separate polar tensors. Remove q2r
    # provenance so the controller must select generic Gonze preparation.
    second.atoms.info.pop("dipole_subtracted_fc", None)
    second._qe_q2r_header = None
    second.folder = str(tmp_path_factory.mktemp("gonze_unfolding_contract"))
    assert "dielectric" in second.atoms.info
    assert np.max(np.abs(second.atoms.get_array("charges"))) > 0.0
    return second


def _harmonic(second, q_point, *, is_unfolding, is_nac=None, bvk_matrix=None):
    return HarmonicWithQ(
        q_point=np.asarray(q_point, dtype=np.float64),
        second=second,
        storage="memory",
        is_unfolding=is_unfolding,
        is_nac=is_nac,
        nac_bvk_supercell_matrix=bvk_matrix,
    )


def test_nonpolar_incommensurate_q_uses_the_requested_interpolation(nonpolar_second):
    """At incommensurate q, the flag must select observably different models."""
    periodic = _harmonic(
        nonpolar_second, INCOMMENSURATE_Q, is_unfolding=False
    )
    unfolded = _harmonic(
        nonpolar_second, INCOMMENSURATE_Q, is_unfolding=True
    )

    assert periodic.is_nac is False
    assert unfolded.is_nac is False
    assert np.max(np.abs(periodic.frequency - unfolded.frequency)) > 1.0e-4
    assert np.max(np.abs(periodic.velocity - unfolded.velocity)) > 1.0e-4


def test_nonpolar_periodic_and_unfolded_agree_on_commensurate_mesh(
    nonpolar_second,
):
    """WS weights cannot change a Fourier sum on its defining q mesh."""
    worst_frequency_difference = 0.0
    for indices in product(range(2), repeat=3):
        q_point = np.asarray(indices, dtype=np.float64) / 2.0
        periodic = _harmonic(nonpolar_second, q_point, is_unfolding=False)
        unfolded = _harmonic(nonpolar_second, q_point, is_unfolding=True)
        difference = np.max(
            np.abs(
                np.sort(periodic.frequency.reshape(-1))
                - np.sort(unfolded.frequency.reshape(-1))
            )
        )
        worst_frequency_difference = max(
            worst_frequency_difference, float(difference)
        )

    assert worst_frequency_difference < 1.0e-6


def test_generic_gonze_nac_owns_interpolation_for_both_flag_values(gonze_second):
    """Active Gonze NAC must route spectra and gradients through one WS kernel."""
    periodic_requested = _harmonic(
        gonze_second,
        GONZE_Q,
        is_unfolding=False,
        bvk_matrix=GONZE_BVK_MATRIX,
    )
    unfolded_requested = _harmonic(
        gonze_second,
        GONZE_Q,
        is_unfolding=True,
        bvk_matrix=GONZE_BVK_MATRIX,
    )

    assert periodic_requested.is_nac is True
    assert unfolded_requested.is_nac is True
    assert (
        periodic_requested._build_nac_static_data_runtime()["convention"]
        == "gonze_total"
    )
    assert (
        unfolded_requested._build_nac_static_data_runtime()["convention"]
        == "gonze_total"
    )
    np.testing.assert_allclose(
        periodic_requested.frequency,
        unfolded_requested.frequency,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        periodic_requested.velocity,
        unfolded_requested.velocity,
        rtol=0.0,
        atol=0.0,
    )

"""Behavioral contract between NAC and IFC interpolation.

Without NAC, an
incommensurate q point may distinguish the legacy periodic-replica Fourier sum
from Wigner--Seitz interpolation, while the two must coincide on the defining
commensurate mesh. With generic Gonze NAC active, the NAC controller owns the
Wigner--Seitz short-range reconstruction. ``auto`` and ``wigner-seitz`` must
therefore agree, while an explicit ``periodic`` request is rejected rather
than silently ignored.

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


def _harmonic(second, q_point, *, interpolation, is_nac=None, bvk_matrix=None):
    return HarmonicWithQ(
        q_point=np.asarray(q_point, dtype=np.float64),
        second=second,
        storage="memory",
        ifc_interpolation=interpolation,
        is_nac=is_nac,
        nac_bvk_supercell_matrix=bvk_matrix,
    )


def test_nonpolar_incommensurate_q_uses_the_requested_interpolation(nonpolar_second):
    """At incommensurate q, the flag must select observably different models."""
    periodic = _harmonic(
        nonpolar_second, INCOMMENSURATE_Q, interpolation="periodic"
    )
    unfolded = _harmonic(
        nonpolar_second, INCOMMENSURATE_Q, interpolation="wigner-seitz"
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
        periodic = _harmonic(nonpolar_second, q_point, interpolation="periodic")
        unfolded = _harmonic(nonpolar_second, q_point, interpolation="wigner-seitz")
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


def test_generic_gonze_nac_auto_and_ws_use_one_kernel(gonze_second):
    """Active Gonze NAC routes spectra and gradients through one WS kernel."""
    automatic = _harmonic(
        gonze_second,
        GONZE_Q,
        interpolation="auto",
        bvk_matrix=GONZE_BVK_MATRIX,
    )
    wigner_seitz = _harmonic(
        gonze_second,
        GONZE_Q,
        interpolation="wigner-seitz",
        bvk_matrix=GONZE_BVK_MATRIX,
    )

    assert automatic.is_nac is True
    assert wigner_seitz.is_nac is True
    assert (
        automatic._build_nac_static_data_runtime()["convention"]
        == "gonze_total"
    )
    assert (
        wigner_seitz._build_nac_static_data_runtime()["convention"]
        == "gonze_total"
    )
    np.testing.assert_allclose(
        automatic.frequency,
        wigner_seitz.frequency,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        automatic.velocity,
        wigner_seitz.velocity,
        rtol=0.0,
        atol=0.0,
    )


def test_generic_gonze_rejects_periodic_override(gonze_second):
    with pytest.raises(ValueError, match="incompatible with active NAC"):
        _harmonic(
            gonze_second,
            GONZE_Q,
            interpolation="periodic",
            bvk_matrix=GONZE_BVK_MATRIX,
        )

"""
Cross-chemistry validation on Stillinger-Weber Si at 100 K.

Uses the SW Si 100K_3UC TDEP fixture vendored under
``kaldo/tests/cumulant_fixtures/SW/`` (originally from
LatticeDynamicsToolkit.jl). Same Si diamond primitive as the production
``reference_si/T300_0`` fixture, but a **different potential (Stillinger-
Weber)** and **temperature (100 K vs 300 K)**. The non-diagonal ssposcar
tiling is identical (det M = 108), so the SNF path + F1/F2 kernels get
exercised on physics that differs from the primary regression set.

Pinned reference values: F1/F2 at 2³, 3³, 5³ produced by an independent
Julia/LatticeDynamicsToolkit.jl run on the same TDEP IFC files. Python
port agrees to 5+ significant digits, confirming the multi-atom F1/F2
kernels match an independent implementation.

Note on physics: F1 here is positive (~+1.10e-4 eV/atom), opposite sign
to DFT-Si-300K where F1 ≈ -1.24e-5. The SW potential plus colder
temperature produces quartic stiffening that raises the free energy
rather than lowering it; both signs are physically valid given the
different potentials. F2 converges monotonically to ~-2.5e-5.

Tests are entirely self-contained (no external paths, no Julia runtime
required).
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest


# SW Si TDEP fixture vendored from LatticeDynamicsToolkit.jl test data.
# Self-contained — no external paths or Julia runtime required.
LDT_SW_BASE = Path(__file__).parent / "cumulant_fixtures" / "SW"
LDT_SW_IFC = LDT_SW_BASE / "100K_3UC"


def _have_sw_fixture():
    return (LDT_SW_BASE / "infile.ucposcar").exists() and \
           (LDT_SW_IFC / "infile.forceconstant").exists()


SI_MASS_AMU = 28.0855
SI_M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)


@pytest.fixture(scope="module")
def sw_si_folder(tmp_path_factory):
    """Stage a TDEP-style folder for SW Si 100K (IFCs + uc/ss posars colocated)."""
    if not _have_sw_fixture():
        pytest.skip("LDT SW Si fixture unavailable")
    d = tmp_path_factory.mktemp("sw_si_100K")
    for fn in ("infile.ucposcar", "infile.ssposcar"):
        shutil.copy(str(LDT_SW_BASE / fn), str(d / fn))
    for fn in (
        "infile.forceconstant",
        "infile.forceconstant_thirdorder",
        "infile.forceconstant_fourthorder",
    ):
        shutil.copy(str(LDT_SW_IFC / fn), str(d / fn))
    return d


@pytest.mark.skipif(not _have_sw_fixture(),
                    reason="LDT SW Si fixture unavailable")
@pytest.mark.parametrize("mesh,ldt_F1", [
    (2, 1.0716049836198086e-4),
    (3, 1.0949229998332265e-4),
    (5, 1.1012932216338007e-4),
])
def test_sw_si_100K_F1_matches_julia_ldt(sw_si_folder, mesh, ldt_F1):
    """SW Si 100 K F1 converges monotonically to +1.10e-4 eV/atom."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant import F1_from_fc

    fc = ForceConstants.from_folder(
        folder=str(sw_si_folder), supercell_matrix=SI_M, format="tdep",
        include_fourth=True,
    )
    r = F1_from_fc(
        fc, masses_amu=np.full(2, SI_MASS_AMU),
        kmesh=(mesh, mesh, mesh), T_K=100.0, use_q_symmetry=True,
    )
    # rtol=5e-5 because Python matches Julia to 5+ significant digits
    # on this fixture. Floor is set by LAPACK eigh / LDT's own Hermitian
    # diagonalization order; tighter than 1e-6 is not feasible.
    np.testing.assert_allclose(
        r["F1"], ldt_F1, rtol=5e-5,
        err_msg=f"SW Si {mesh}^3 F1 drifted from Julia LDT reference",
    )


@pytest.mark.skipif(not _have_sw_fixture(),
                    reason="LDT SW Si fixture unavailable")
@pytest.mark.parametrize("mesh,ldt_F2", [
    (2, -2.2973337994597353e-5),
    (3, -2.432226409659287e-5),
    (5, -2.4806056654284995e-5),
])
def test_sw_si_100K_F2_matches_julia_ldt(sw_si_folder, mesh, ldt_F2):
    """SW Si 100 K F2 converges monotonically to ~-2.5e-5 eV/atom."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant import F2_from_fc

    fc = ForceConstants.from_folder(
        folder=str(sw_si_folder), supercell_matrix=SI_M, format="tdep",
    )
    r = F2_from_fc(
        fc, masses_amu=np.full(2, SI_MASS_AMU),
        kmesh=(mesh, mesh, mesh), T_K=100.0, sigma_THz=None,
        use_q_symmetry=True,
    )
    # rtol=5e-5: Python matches Julia to 5 significant digits. Float-order
    # in the adaptive-sigma clamp + eigenvector gauge fix the floor.
    np.testing.assert_allclose(
        r["F2"], ldt_F2, rtol=5e-5,
        err_msg=f"SW Si {mesh}^3 F2 drifted from Julia LDT reference",
    )


@pytest.mark.skipif(not _have_sw_fixture(),
                    reason="LDT SW Si fixture unavailable")
def test_sw_si_100K_differs_from_dft_si_300K(sw_si_folder):
    """SW-100K and DFT-300K use the SAME primitive but DIFFERENT potentials.

    This is the core cross-chemistry check: F1 should flip sign between the
    two (SW positive, DFT negative at 300K), confirming the cumulant pipeline
    responds correctly to different underlying physics.
    """
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant import F1_from_fc

    fc_sw = ForceConstants.from_folder(
        folder=str(sw_si_folder), supercell_matrix=SI_M, format="tdep",
        include_fourth=True,
    )
    r_sw = F1_from_fc(
        fc_sw, masses_amu=np.full(2, SI_MASS_AMU),
        kmesh=(2, 2, 2), T_K=100.0, use_q_symmetry=True,
    )
    # SW at 100K: positive F1 (~+1.07e-4)
    # DFT at 300K (reference_si regression): negative F1 (-1.17e-5)
    # Confirm they differ in sign and are not close in magnitude
    assert r_sw["F1"] > 0, "SW Si 100 K F1 must be positive"
    assert abs(r_sw["F1"] - (-1.16592e-5)) > 1e-5, (
        "SW Si 100 K F1 must differ meaningfully from DFT Si 300 K"
    )

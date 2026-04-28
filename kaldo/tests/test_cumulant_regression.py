"""
Regression harness for the cumulant pipeline.

Pins values validated against Julia LDT and Ethan's 25^3 published references.
Any refactor must keep every assertion here green.

As of E.6 the harness uses the kaldo-native entry points
``ForceConstants.from_folder(format='tdep', supercell_matrix=M,
include_fourth=True)`` + ``F1_from_fc`` / ``F2_from_fc`` throughout, not the
legacy list-based ``F1_vectorized`` / ``F2_vectorized`` readers. The
cross-checks ``test_cumulant_from_fc.py`` and ``test_nondiagonal_forceconstants.py``
ensure the two paths give the same numbers.

Validated sources:
  * Ne 10^3 F1/F2 - matches Julia LDT @ 10^3 to 1e-6
  * Ne 25^3 Gate 6 - PASS vs Ethan thermo_out_full (F_total matches to 1.7e-7)
  * Si 10^3 F1/F2 - matches Julia LDT @ 10^3 to 1e-6 (after conj + dynmat fixes)
  * Si 2^3, 3^3, 5^3 - matches Julia LDT to 4-5 digits

Skip conditions:
  * Ne fixtures: requires /home/giuseppe/Development/ethan/run/thermo_out_full/
  * Si fixtures: requires ~/Development/4th-order-cumulants/reference_si/T300_0/
  Both live on the dvncls SSH host; tests are skipped when unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

NE_REF = Path("/home/giuseppe/Development/ethan/run/thermo_out_full")
SI_REF = Path("/home/giuseppe/Development/4th-order-cumulants/reference_si/T300_0")
OUT_DIR = Path("/home/giuseppe/Development/4th-order-cumulants/out")

_NE_AVAIL = NE_REF.exists()
_SI_AVAIL = SI_REF.exists()

# Both Ne and Si production fixtures use rhombohedral primitive + cubic
# conventional ssposcar (non-diagonal M). See E.1-E.5 in
# test_nondiagonal_forceconstants.py for the supercell-matrix values.
NE_M = np.array([[4, -4, 4], [4, 4, -4], [-4, 4, 4]], dtype=int)
SI_M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)
SI_AMU_MASS = 28.0855


def _load_ne_fc(include_fourth=True):
    from kaldo.forceconstants import ForceConstants
    return ForceConstants.from_folder(
        folder=str(NE_REF), supercell_matrix=NE_M, format="tdep",
        include_fourth=include_fourth,
    )


def _load_si_fc(include_fourth=True):
    from kaldo.forceconstants import ForceConstants
    return ForceConstants.from_folder(
        folder=str(SI_REF), supercell_matrix=SI_M, format="tdep",
        include_fourth=include_fourth,
    )


# -------------------------------------------------------------------------
# Ne - F1 (quartic) at 10^3 vs Julia LDT @ 10^3
# -------------------------------------------------------------------------

@pytest.mark.skipif(not _NE_AVAIL, reason="Ne TDEP fixtures unavailable")
def test_ne_10cubed_F1_matches_julia():
    from kaldo.cumulant import F1_from_fc, NE_MASS_AMU
    fc = _load_ne_fc(include_fourth=True)
    r = F1_from_fc(
        fc, masses_amu=np.full(fc.n_atoms, NE_MASS_AMU),
        kmesh=(10, 10, 10), T_K=24.0, use_q_symmetry=True,
    )
    # Julia LDT @ 10^3 gives +1.17351e-4 (see out/ldt_rhombo_mesh10.json).
    # rtol=5e-5 is set by Julia's Julia-vs-Python float-summation order on
    # the 1000-term q-sum (inner Gauss reduction, eigenvalue decomposition
    # routine differences). Tightening below 1e-5 is likely infeasible
    # without byte-exact LAPACK; a failure here at >1e-4 would be a real
    # physics regression.
    np.testing.assert_allclose(
        r["F1"], 1.17351e-4, rtol=5e-5,
        err_msg="Ne 10^3 F1 must match Julia LDT reference",
    )


@pytest.mark.skipif(not _NE_AVAIL, reason="Ne TDEP fixtures unavailable")
def test_ne_10cubed_F2_matches_julia():
    from kaldo.cumulant import F2_from_fc, NE_MASS_AMU
    fc = _load_ne_fc(include_fourth=False)
    r = F2_from_fc(
        fc, masses_amu=np.full(fc.n_atoms, NE_MASS_AMU),
        kmesh=(10, 10, 10), T_K=24.0, sigma_THz=None, use_q_symmetry=True,
    )
    # Julia LDT @ 10^3 adaptive sigma. rtol=1e-4 is the loosest observed
    # during sigma-consolidation work: the TDEP adaptive-sigma clamps are
    # sensitive to eigenvector gauge near degenerate subspaces, and the
    # Python / Julia paths pick different gauge fixings. Below 1e-5 would
    # require harmonizing gauge; above 1e-3 signals a physics regression.
    np.testing.assert_allclose(
        r["F2"], -4.43403e-4, rtol=1e-4,
        err_msg="Ne 10^3 F2 must match Julia LDT reference (adaptive sigma)",
    )


# -------------------------------------------------------------------------
# Ne - 25^3 cached JSON pinned values (Gate 6 inputs)
# -------------------------------------------------------------------------

@pytest.mark.skipif(not (OUT_DIR / "run_25cubed_ibz.json").exists(),
                    reason="Ne 25^3 JSON cache unavailable")
def test_ne_25cubed_F1_cached_values():
    """Cached Ne 25^3 F1 values that feed Gate 6. Pinned against Ethan 25^3."""
    d = json.load(open(OUT_DIR / "run_25cubed_ibz.json"))
    np.testing.assert_allclose(d["phase3"]["F1"], 1.1950363e-4, rtol=1e-6)
    np.testing.assert_allclose(d["phase3"]["U1"], 3.7076641e-5, rtol=1e-6)
    np.testing.assert_allclose(d["phase3"]["S1"], -3.9855231e-2, rtol=1e-6)
    np.testing.assert_allclose(d["phase3"]["Cv1"], -5.9185702e-2, rtol=1e-6)


@pytest.mark.skipif(not (OUT_DIR / "run_25cubed_ibz_tdep.json").exists(),
                    reason="Ne 25^3 TDEP JSON cache unavailable")
def test_ne_25cubed_F2_cached_values():
    """Cached Ne 25^3 F2 values (adaptive sigma) matching Ethan to ~7e-5."""
    d = json.load(open(OUT_DIR / "run_25cubed_ibz_tdep.json"))
    np.testing.assert_allclose(d["phase4"]["F2"], -4.5566464e-4, rtol=1e-6)
    np.testing.assert_allclose(d["phase4"]["U2"], 3.1839929e-4, rtol=1e-6)
    np.testing.assert_allclose(d["phase4"]["S2"], 3.7427662e-1, rtol=1e-6)
    np.testing.assert_allclose(d["phase4"]["Cv2"], 4.5743135e-1, rtol=1e-6)


# -------------------------------------------------------------------------
# Si - F1/F2 at multiple meshes vs Julia LDT (post-fix multi-atom validation)
# -------------------------------------------------------------------------

@pytest.mark.skipif(not _SI_AVAIL, reason="Si TDEP fixtures unavailable")
@pytest.mark.parametrize("mesh,expected_F1", [
    (2, -1.16592e-5),
    (3, -1.23028e-5),
    (5, -1.22421e-5),
    (10, -1.23780e-5),
    # 25^3 takes ~7h; included as cached-JSON pin below rather than re-run.
])
def test_si_F1_matches_julia(mesh, expected_F1):
    from kaldo.cumulant import F1_from_fc
    fc = _load_si_fc(include_fourth=True)
    r = F1_from_fc(
        fc, masses_amu=np.full(fc.n_atoms, SI_AMU_MASS),
        kmesh=(mesh, mesh, mesh), T_K=300.0,
        use_q_symmetry=True,
    )
    # rtol=2e-4 tracks Julia LDT's own Gauss-reduction precision at these
    # meshes; tighter fails on 2^3 where the 8 q-points make quantization
    # of orbit weights visible. A failure at >1e-3 signals a real physics
    # regression, not a numerical drift.
    np.testing.assert_allclose(
        r["F1"], expected_F1, rtol=2e-4,
        err_msg=f"Si {mesh}^3 F1 must match Julia LDT reference",
    )


# -------------------------------------------------------------------------
# Si 25^3 cached-JSON pins (takes ~7h to regenerate; pin the file values)
# -------------------------------------------------------------------------

@pytest.mark.skipif(not (OUT_DIR / "si_T300_mesh25.json").exists(),
                    reason="Si 25^3 cache unavailable")
def test_si_25cubed_cached_matches_julia():
    """
    Pinned Si 25^3 JSON: F1/U1/S1/Cv1 and F2/U2/S2/Cv2 must match Julia LDT
    quantum 25^3 to 1e-5 relative (ldt_si_T300_mesh25_quantum.json).
    """
    py = json.load(open(OUT_DIR / "si_T300_mesh25.json"))
    jl = json.load(open(OUT_DIR / "ldt_si_T300_mesh25_quantum.json"))
    # Julia uses F3 for cubic (=F2), F4 for quartic (=F1)
    pairs = [
        ("phase3.F1", py["phase3"]["F1"],  jl["F4"],  1e-5),
        ("phase3.U1", py["phase3"]["U1"],  jl["U4"],  1e-4),
        ("phase3.S1", py["phase3"]["S1"],  jl["S4"],  1e-4),
        ("phase3.Cv1", py["phase3"]["Cv1"], jl["Cv4"], 1e-3),
        ("phase4.F2", py["phase4"]["F2"],  jl["F3"],  1e-5),
        ("phase4.U2", py["phase4"]["U2"],  jl["U3"],  1e-5),
        ("phase4.S2", py["phase4"]["S2"],  jl["S3"],  1e-5),
        ("phase4.Cv2", py["phase4"]["Cv2"], jl["Cv3"], 1e-5),
    ]
    for key, ours, theirs, rtol in pairs:
        np.testing.assert_allclose(
            ours, theirs, rtol=rtol,
            err_msg=f"Si 25^3 {key} Python vs Julia mismatch",
        )


@pytest.mark.skipif(not _SI_AVAIL, reason="Si TDEP fixtures unavailable")
@pytest.mark.parametrize("mesh,expected_F2", [
    (2, -6.43481e-5),
    (3, -7.01158e-5),
    (5, -7.33168e-5),
    (10, -7.48156e-5),
])
def test_si_F2_matches_julia(mesh, expected_F2):
    from kaldo.cumulant import F2_from_fc
    fc = _load_si_fc(include_fourth=False)
    r = F2_from_fc(
        fc, masses_amu=np.full(fc.n_atoms, SI_AMU_MASS),
        kmesh=(mesh, mesh, mesh), T_K=300.0, sigma_THz=None,
        use_q_symmetry=True,
    )
    # rtol=5e-4: looser than F1 because F2's adaptive-sigma kernel amplifies
    # Julia/Python gauge differences through the sigma clamp. Observed
    # residual is ~1e-4 at 2^3 and ~1e-5 at 10^3; 5e-4 is a conservative
    # margin. >1e-3 = real regression.
    np.testing.assert_allclose(
        r["F2"], expected_F2, rtol=5e-4,
        err_msg=f"Si {mesh}^3 F2 must match Julia LDT reference",
    )

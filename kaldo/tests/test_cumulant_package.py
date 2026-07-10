"""
Acceptance tests for the kaldo.cumulant subpackage.

Checks that kaldo.cumulant imports cleanly and reproduces the Gate 6 Ne 25^3
assembly using the cached Phase 5 sample archive. Together with
kaldo/tests/test_cumulant_regression.py, this pins the package against Julia
LDT (analytic F1/F2) and Ethan's 25^3 thermo_out reference (Gate 6 totals).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Production-only fixtures: large DFT-quality Si IFCs and Ne TDEP run output.
# Set KALDO_TEST_SI_PROD and KALDO_TEST_NE_REF to enable.
# See kaldo/tests/_paths.py for details on env-var-gated test fixtures.
from kaldo.tests._paths import SI_PROD as SI_REF, NE_REF

_NE_AVAIL = NE_REF.exists()
_SI_AVAIL = SI_REF.exists()


# -------------------------------------------------------------------------
# Import smoke tests - package loads cleanly
# -------------------------------------------------------------------------

def test_cumulant_package_imports():
    """All public symbols resolve from kaldo.cumulant."""
    from kaldo.cumulant import (
        HBAR, KB, EV, AMU, ANG, NE_MASS_AMU,
        read_tdep_pair_fcs, read_tdep_ifc3, read_tdep_ifc4,
        dynmat_and_eigs,
        harmonic_thermo_quantum,
        F1_vectorized, F2_vectorized,
    )
    assert abs(HBAR - 1.054571817e-34) < 1e-40
    assert abs(KB - 1.380649e-23) < 1e-30
    assert np.isclose(NE_MASS_AMU, 20.1797)


def test_cumulant_submodules_importable():
    """Each submodule is directly importable."""
    import kaldo.cumulant.common
    import kaldo.cumulant.harmonic
    import kaldo.cumulant.free_energy
    import kaldo.cumulant.sampler
    import kaldo.cumulant.taylor
    import kaldo.cumulant.estimator
    import kaldo.cumulant.bootstrap
    import kaldo.cumulant.api
    assert hasattr(kaldo.cumulant.common, "dynmat_and_eigs")
    assert hasattr(kaldo.cumulant.harmonic, "harmonic_thermo_quantum")
    assert hasattr(kaldo.cumulant.free_energy, "F1_vectorized")
    assert hasattr(kaldo.cumulant.free_energy, "F2_vectorized")


# -------------------------------------------------------------------------
# Harmonic thermo sanity: Ne 10^3 F_H matches phase1 reference
# -------------------------------------------------------------------------

@pytest.mark.skipif(not _NE_AVAIL, reason="Ne fixtures unavailable")
def test_ne_10cubed_F_H_matches_phase1():
    """harmonic_thermo_from_ifc2 on Ne 10^3 matches pinned Phase 1 output."""
    import ase.io
    from kaldo.cumulant import (
        harmonic_thermo_from_ifc2, read_tdep_pair_fcs, NE_MASS_AMU, AMU,
    )
    uc = ase.io.read(str(NE_REF / "infile.ucposcar"), format="vasp")
    uc_pos = np.asarray(uc.get_positions())
    uc_cell = np.asarray(uc.get_cell())
    n_uc = len(uc)
    masses_kg = np.full(n_uc, NE_MASS_AMU * AMU)
    neighbors = read_tdep_pair_fcs(NE_REF / "infile.forceconstant", uc_pos, uc_cell)
    F_H, U_H, S_H, Cv_H = harmonic_thermo_from_ifc2(
        neighbors, uc_pos, masses_kg, uc_cell, (10, 10, 10), 24.0,
    )
    assert 5.0e-3 < F_H < 7.5e-3, f"F_H={F_H} out of expected Ne range at 10^3"
    assert 1.0 < S_H < 1.5, f"S_H={S_H} out of expected Ne range"
    assert 1.0 < Cv_H < 3.0, f"Cv_H={Cv_H} out of expected Ne range"


# -------------------------------------------------------------------------
# Gate 6 full recap using kaldo.cumulant estimator+bootstrap
# -------------------------------------------------------------------------

import os
# Production-only fixture: cached MC samples / analytic JSON outputs from
# Ethan's reference cumulant runs. Set KALDO_TEST_CUMULANT_OUT to the dir
# containing phase5_our_samples.npz, run_25cubed_ibz.json, etc. to enable.
_OUT_DIR = Path(os.environ.get("KALDO_TEST_CUMULANT_OUT", ""))
SAMPLES_NPZ = _OUT_DIR / "phase5_our_samples.npz"


@pytest.mark.skipif(not SAMPLES_NPZ.exists(),
                    reason="Phase 5 sample archive unavailable")
def test_gate6_recap_uses_kaldo_cumulant():
    """
    Re-assemble Gate 6 at 25^3 Ne using the kaldo.cumulant estimator and
    bootstrap on cached Phase 5 samples, plus cached analytic F1/F2 JSONs.
    Must still match Ethan's 25^3 totals to within 3 sigma.
    """
    import json
    from kaldo.cumulant import bootstrap_corrections

    ETHAN = dict(
        F=dict(H=+0.0061408, off=-0.0237363, one=+0.0001195, two=-0.0004557,
               total=-0.0179317, total_SE=3.0e-7),
        U=dict(H=+0.0088115, off=-0.0236692, one=+0.0000371, two=+0.0003184,
               total=-0.0145023, total_SE=2.6e-6),
        S=dict(H=+1.2913396, off=+0.0324360, one=-0.0398552, two=+0.3742765,
               total=+1.6581969, total_SE=1.2909e-3),
        Cv=dict(H=+1.9769193, off=-0.3744209, one=-0.0591857, two=+0.4574312,
                total=+2.0007439, total_SE=1.8803e-2),
    )

    F_H, U_H = 6.140788e-3, 8.811518e-3
    S_H, Cv_H = 1.2913396, 1.9769204

    d25_f1 = json.load(open(_OUT_DIR / "run_25cubed_ibz.json"))
    d25_f2 = json.load(open(_OUT_DIR / "run_25cubed_ibz_tdep.json"))
    F1 = d25_f1["phase3"]["F1"]; U1 = d25_f1["phase3"]["U1"]
    S1 = d25_f1["phase3"]["S1"]; Cv1 = d25_f1["phase3"]["Cv1"]
    F2 = d25_f2["phase4"]["F2"]; U2 = d25_f2["phase4"]["U2"]
    S2 = d25_f2["phase4"]["S2"]; Cv2 = d25_f2["phase4"]["Cv2"]

    samples = np.load(SAMPLES_NPZ)
    V, V2, V3, V4, V_ref = (samples["V"], samples["V2"], samples["V3"],
                            samples["V4"], samples["V2_tilde"])
    Nat = 256; T_K = 24.0
    point, se = bootstrap_corrections(V, V2, V3, V4, V_ref, T_K, Nat,
                                       n_boot=5000, seed=1234)

    totals = dict(
        F=F_H + point["F0"] + F1 + F2,
        U=U_H + point["U0"] + U1 + U2,
        S=S_H + point["S0"] + S1 + S2,
        Cv=Cv_H + point["Cv0"] + Cv1 + Cv2,
    )
    SE = dict(
        F=se["F0"], U=se["U0"],
        S=se["S0"], Cv=se["Cv0"],
    )
    for q in ("F", "U", "S", "Cv"):
        joint_SE = np.sqrt(SE[q] ** 2 + ETHAN[q]["total_SE"] ** 2)
        diff = abs(totals[q] - ETHAN[q]["total"])
        assert diff < 3 * joint_SE, (
            f"Gate 6 FAIL on {q}: |Delta|={diff:.3e}, 3*sigma={3*joint_SE:.3e}"
        )

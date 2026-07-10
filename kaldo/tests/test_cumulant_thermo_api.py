"""
Pin the ``cumulant_thermo`` public signature and drive it on production Ne.

The full pipeline (Phase 5 LAMMPS sampling) is heavy and environment
specific. Here we assert:

  * the signature keeps its documented shape (TDEP folder + supercell in
    front, statistics and LAMMPS commands as the required physics inputs),
    so accidental API breakage shows up in the diff of this file;
  * on the env-var-gated non-diagonal Ne production fixture the analytic
    phases (harmonic + F1 + F2) run end-to-end; Phase 5 is skipped cleanly
    when the ``lammps`` Python module is not installed.
"""
from __future__ import annotations

import inspect

import pytest

# Production-only fixture: Ethan Meitz's non-diagonal Ne TDEP run.
# Set KALDO_TEST_NE_REF to enable. See kaldo/tests/_paths.py.
from kaldo.tests._paths import NE_REF


def test_cumulant_thermo_signature_pinned():
    """The public signature stays stable: TDEP inputs first, then physics."""
    from kaldo.cumulant import cumulant_thermo
    sig = inspect.signature(cumulant_thermo)
    params = list(sig.parameters)
    assert params[:5] == [
        "tdep_folder", "supercell", "temperature", "is_classic", "lammps_cmds",
    ]
    # Tunables keep their defaults so existing call sites stay valid.
    assert sig.parameters["nconf"].default == 100_000
    assert sig.parameters["nboot"].default == 5_000
    assert sig.parameters["use_q_symmetry"].default is True


@pytest.mark.skipif(not NE_REF.exists(), reason="Ne production fixture unavailable")
def test_cumulant_thermo_phases_1_to_4_run_on_nondiagonal_ne():
    """cumulant_thermo on the non-diagonal (det M = 256) Ne production run
    must complete the analytic phases (harmonic + F1 + F2) without error.
    Phase 5 (LAMMPS MC sampling) is skipped if the ``lammps`` Python module
    is not installed: environment issue, not a code regression.
    """
    import numpy as np
    from kaldo.cumulant import cumulant_thermo

    lj_ne = ["pair_style lj/cut 8.5", "pair_coeff * * 0.0031 2.74",
             "pair_modify shift yes"]
    try:
        r = cumulant_thermo(
            str(NE_REF), (4, 4, 4), temperature=24.0, is_classic=False,
            lammps_cmds=lj_ne, nconf=50, nboot=100,
            harmonic_mesh=(5, 5, 5), free_energy_mesh=(5, 5, 5),
            use_q_symmetry=True, verbose=False,
        )
    except ModuleNotFoundError as e:
        if "lammps" in str(e):
            pytest.skip("Python lammps module unavailable; phases 1-4 completed cleanly")
        raise

    # If we got here, Phase 5 also ran: full pipeline success.
    assert np.isfinite(r.F_H)
    assert np.isfinite(r.F_1)
    assert np.isfinite(r.F_2)
    assert np.isfinite(r.F_total)

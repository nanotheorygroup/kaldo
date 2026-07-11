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
from pathlib import Path

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
    # The potential comes from exactly one of lammps_cmds / calculator.
    assert sig.parameters["lammps_cmds"].default is None
    assert sig.parameters["calculator"].default is None


def test_cumulant_thermo_requires_exactly_one_potential_source(tmp_path):
    """lammps_cmds and calculator are mutually exclusive and one is required."""
    from kaldo.cumulant import cumulant_thermo
    with pytest.raises(ValueError, match="exactly one"):
        cumulant_thermo(str(tmp_path), (1, 1, 1), 100.0, True)
    with pytest.raises(ValueError, match="exactly one"):
        cumulant_thermo(str(tmp_path), (1, 1, 1), 100.0, True,
                        lammps_cmds=["pair_style lj/cut 8.5"], calculator=object())


SW_FIX = Path(__file__).parent / "cumulant_fixtures" / "SW"
LJ_FIX = Path(__file__).parent / "cumulant_fixtures" / "LJ"


@pytest.mark.skipif(not (LJ_FIX / "infile.ucposcar").exists(), reason="LJ cumulant fixture missing")
def test_cumulant_thermo_end_to_end_with_ase_calculator(tmp_path):
    """Full pipeline (harmonic + F1 + F2 + Phase-5 sampling + bootstrap) on
    the vendored LJ Ar fixture, with ase's LennardJones standing in for
    LAMMPS via the calculator= path. First CI-runnable end-to-end drive of
    cumulant_thermo: every phase executes with real energies.

    The ASE LJ parameters approximate the potential behind the TDEP fit, so
    F_0 (which measures potential-vs-Taylor anharmonicity) is asserted
    finite and small rather than pinned.
    """
    import shutil

    import numpy as np
    from ase.calculators.lj import LennardJones

    from kaldo.cumulant import cumulant_thermo

    folder = tmp_path / "lj_ar"
    folder.mkdir()
    for fn in ("infile.ucposcar", "infile.ssposcar"):
        shutil.copy(str(LJ_FIX / fn), str(folder / fn))
    for fn in ("infile.forceconstant", "infile.forceconstant_thirdorder",
               "infile.forceconstant_fourthorder"):
        shutil.copy(str(LJ_FIX / "80K_4UC" / fn), str(folder / fn))

    r = cumulant_thermo(
        str(folder), (4, 4, 4), temperature=80.0, is_classic=True,
        calculator=LennardJones(epsilon=0.0104, sigma=3.4, rc=8.5),
        nconf=10, nboot=20,
        harmonic_mesh=(4, 4, 4), free_energy_mesh=(2, 2, 2),
        use_q_symmetry=True, verbose=False,
    )

    for name in ("F_H", "F_0", "F_1", "F_2", "F_total",
                 "U_total", "S_total", "Cv_total", "F_total_SE"):
        assert np.isfinite(getattr(r, name)), f"{name} is not finite"
    assert r.N_conf == 10
    assert np.all(np.isfinite(r.V)) and np.any(r.V != 0.0)
    # The potentials differ slightly, but the constant correction must be a
    # sane per-atom energy, not a blow-up (the historical failure mode was
    # ~1e12 from frame/cell mismatches).
    assert abs(r.F_0) < 1.0


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

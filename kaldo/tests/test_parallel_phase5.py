"""Phase-5 process-pool sampling: one calculator per worker.

Serial ``n_workers=1`` is unchanged (existing thermo API tests). These
tests cover ``n_workers>1``: fixed-displacement parity, jumped-stream
sampling, unique LAMMPS logs, and ``n_workers=0`` rejection.

Heavy TDEP IFC4 tables are not pickled into workers; unit tests use tiny
synthetic contractors so spawn stays cheap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones

from kaldo.cumulant.contractors import SCContractors
from kaldo.cumulant.sampler import SCSampler
from kaldo.parallel import get_executor

AR_IFC = Path(__file__).parent / "cumulant_fixtures" / "LJ" / "Argon_80K_4UC"


class _SpringCalc(Calculator):
    """Analytic ASE calculator: E = 0.5 k * sum |r - r_ref|^2 (isotropic)."""

    implemented_properties = ["energy", "free_energy"]

    def __init__(self, r_ref, k=3.0):
        super().__init__()
        self._r_ref = np.array(r_ref, dtype=float)
        self._k = float(k)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        du = atoms.get_positions() - self._r_ref
        e = 0.5 * self._k * float(np.sum(du * du))
        self.results = {"energy": e, "free_energy": e}


def _have_ar():
    return (AR_IFC / "infile.forceconstant").exists()


def _lammps_importable():
    try:
        from ase.calculators.lammpslib import LAMMPSlib  # noqa: F401
        import lammps  # noqa: F401
        return True
    except Exception:
        return False


def _tiny_contractors(n_atoms=4):
    """A few synthetic pairs/triplets/quartets — small enough to pickle."""
    n_pair = 3
    a1_2 = np.array([0, 1, 2], dtype=np.int64) % n_atoms
    a2_2 = np.array([1, 2, 3], dtype=np.int64) % n_atoms
    phi2 = np.repeat(0.2 * np.eye(3)[None, :, :], n_pair, axis=0)
    n_tri = 2
    a1_3 = np.array([0, 1], dtype=np.int64) % n_atoms
    a2_3 = np.array([1, 2], dtype=np.int64) % n_atoms
    a3_3 = np.array([2, 3], dtype=np.int64) % n_atoms
    phi3 = np.zeros((n_tri, 3, 3, 3))
    phi3[:, 0, 0, 0] = 0.05
    n_q = 2
    a1_4 = np.array([0, 1], dtype=np.int64) % n_atoms
    a2_4 = np.array([1, 2], dtype=np.int64) % n_atoms
    a3_4 = np.array([2, 3], dtype=np.int64) % n_atoms
    a4_4 = np.array([3, 0], dtype=np.int64) % n_atoms
    phi4 = np.zeros((n_q, 3, 3, 3, 3))
    phi4[:, 0, 0, 0, 0] = 0.01
    return SCContractors(_from_arrays=dict(
        n_atoms_sc=n_atoms,
        a1_2=a1_2, a2_2=a2_2, phi2=phi2,
        a1_3=a1_3, a2_3=a2_3, a3_3=a3_3, phi3=phi3,
        a1_4=a1_4, a2_4=a2_4, a3_4=a3_4, a4_4=a4_4, phi4=phi4,
    ))


def _einstein_sampler(n_atoms=4, seed=1):
    k = 0.5
    ifc2 = np.zeros((3 * n_atoms, 3 * n_atoms))
    for i in range(n_atoms):
        ifc2[3 * i:3 * i + 3, 3 * i:3 * i + 3] = k * np.eye(3)
    masses = np.full(n_atoms, 40.0)
    return SCSampler(ifc2, masses, T_K=80.0, is_classic=True, seed=seed)


def _phase5_atoms(n_atoms=4):
    rng = np.random.default_rng(0)
    eq = rng.uniform(1.0, 4.0, size=(n_atoms, 3))
    cell = np.eye(3) * 10.0
    atoms = Atoms("Ar" * n_atoms, positions=eq, cell=cell, pbc=True)
    return atoms, eq, cell


def test_eval_u_chunk_parallel_matches_serial():
    """Two process workers on a fixed ``u`` list match one-process evaluation."""
    from kaldo.cumulant._phase5_worker import eval_u_chunk

    n = 4
    contractors = _tiny_contractors(n)
    atoms, eq, cell = _phase5_atoms(n)
    rng = np.random.default_rng(0)
    us = 0.01 * rng.standard_normal((8, n, 3))
    calc = _SpringCalc(eq, k=2.5)
    base = dict(
        contractors=contractors,
        species=list(atoms.get_chemical_symbols()),
        eq=eq,
        cell=cell,
        lammps_cmds=None,
        calculator=calc,
    )
    serial = eval_u_chunk(dict(base, us=us))
    mid = 4
    specs = [dict(base, us=us[:mid]), dict(base, us=us[mid:])]
    with get_executor(backend="process", n_workers=2) as executor:
        futs = [executor.submit(eval_u_chunk, s) for s in specs]
        parts = [f.result() for f in futs]
    for i, key in enumerate(("V", "V2", "V3", "V4")):
        got = np.concatenate([p[i] for p in parts])
        np.testing.assert_allclose(
            got, serial[i], rtol=1e-7, atol=1e-12,
            err_msg=f"{key} parallel vs serial mismatch",
        )


def test_run_phase5_chunk_n_workers_2_finite():
    """Jumped-stream shards return finite V/V2/V3/V4 (not bit-identical to serial)."""
    from kaldo.cumulant._phase5_worker import run_phase5_chunk, run_phase5_loop
    from kaldo.cumulant._energy_evaluator import BatchEnergyEvaluator

    n = 4
    nconf = 8
    contractors = _tiny_contractors(n)
    sampler = _einstein_sampler(n, seed=11)
    atoms, eq, cell = _phase5_atoms(n)
    atoms.calc = _SpringCalc(eq, k=2.5)
    energy_eval = BatchEnergyEvaluator(atoms, eq)
    serial = run_phase5_loop(nconf, sampler, contractors, energy_eval)

    base = dict(
        contractors=_tiny_contractors(n),
        sampler=_einstein_sampler(n, seed=11),
        species=list(atoms.get_chemical_symbols()),
        eq=eq,
        cell=cell,
        lammps_cmds=None,
        calculator=_SpringCalc(eq, k=2.5),
        seed=11,
    )
    specs = [
        dict(base, sampler=_einstein_sampler(n, seed=11), n_local=4, worker_id=0),
        dict(base, sampler=_einstein_sampler(n, seed=11), n_local=4, worker_id=1),
    ]
    with get_executor(backend="process", n_workers=2) as executor:
        futs = [executor.submit(run_phase5_chunk, s) for s in specs]
        parts = [f.result() for f in futs]
    V = np.concatenate([p[0] for p in parts])
    assert V.shape == (nconf,)
    assert np.all(np.isfinite(V))
    assert np.all(np.isfinite(np.concatenate([p[1] for p in parts])))
    assert np.all(np.isfinite(serial[0]))
    # Jumped streams are independent of serial default_rng; not bit-identical.


def test_cumulant_thermo_n_workers_zero_raises(tmp_path):
    from kaldo.cumulant import cumulant_thermo
    with pytest.raises(ValueError, match="n_workers"):
        cumulant_thermo(
            str(tmp_path), (1, 1, 1), 80.0, True,
            calculator=object(), n_workers=0,
        )


@pytest.mark.skipif(not _have_ar(), reason="LJ Ar cumulant fixture missing")
def test_cumulant_thermo_n_workers_2_ase_calculator():
    """Phase 5 with two processes and ASE LJ: finite F0, no crash."""
    from kaldo.cumulant import cumulant_thermo

    r = cumulant_thermo(
        str(AR_IFC), (4, 4, 4), temperature=80.0, is_classic=True,
        calculator=LennardJones(epsilon=0.0104, sigma=3.4, rc=8.5),
        nconf=8, nboot=16,
        harmonic_mesh=(2, 2, 2), free_energy_mesh=(2, 2, 2),
        use_q_symmetry=True, verbose=False, n_workers=2,
    )
    assert r.N_conf == 8
    assert np.isfinite(r.F_0) and abs(r.F_0) < 1.0
    assert np.all(np.isfinite(r.V)) and r.V.shape == (8,)
    assert np.all(np.isfinite(r.V2)) and np.all(np.isfinite(r.V4))


@pytest.mark.skipif(not _have_ar(), reason="LJ Ar cumulant fixture missing")
@pytest.mark.skipif(not _lammps_importable(), reason="Python lammps module unavailable")
def test_phase5_n_workers_2_lammps_smoke(tmp_path, monkeypatch):
    """Two LAMMPS processes: unique logs, finite energies on a tiny sample."""
    from kaldo.cumulant import cumulant_thermo

    monkeypatch.setenv("KALDO_CUMULANT_LAMMPS_LOG_DIR", str(tmp_path))
    r = cumulant_thermo(
        str(AR_IFC), (4, 4, 4), temperature=80.0, is_classic=True,
        lammps_cmds=[
            "pair_style lj/cut 8.5",
            "pair_coeff * * 0.0104 3.4",
            "pair_modify shift yes",
        ],
        nconf=4, nboot=8,
        harmonic_mesh=(2, 2, 2), free_energy_mesh=(2, 2, 2),
        use_q_symmetry=True, verbose=False, n_workers=2,
    )
    assert np.isfinite(r.F_0)
    assert r.V.shape == (4,)
    assert np.all(np.isfinite(r.V))
    logs = list(tmp_path.glob("cumulant_thermo_lammps_*.log"))
    assert len(logs) >= 2
    assert len({p.name for p in logs}) == len(logs)

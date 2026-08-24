"""Phase-5 config worker (one LAMMPS per process).

This module is imported by spawn workers. It must not import
``kaldo.cumulant.thermodynamics`` or ``kaldo.forceconstants`` itself; those
pull TensorFlow. Parent-package ``__init__`` may still run on spawn — that
is a separate issue. Keep *this* file's imports limited to sampler,
contractors, the energy evaluator, and ASE.

Each worker owns:
  * its own ``LAMMPSlib`` / ASE calculator + ``BatchEnergyEvaluator``
  * a jumped RNG stream on a pickled ``SCSampler``
  * in-process V2/V3/V4 (no inner thread pool)
"""
from __future__ import annotations

import os

import numpy as np

from kaldo.cumulant._energy_evaluator import BatchEnergyEvaluator


def _lammps_log_path(spec=None):
    """Per-process LAMMPS log so n_workers>1 does not collide on one file."""
    log_dir = None
    if spec:
        log_dir = spec.get("log_dir")
    if not log_dir:
        log_dir = os.environ.get("KALDO_CUMULANT_LAMMPS_LOG_DIR", "/tmp")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"cumulant_thermo_lammps_{os.getpid()}.log")


def _load_contractors(spec):
    """Unpickle a tiny contractor table, or rebuild from ``tdep_folder``.

    Production Phase-5 shards must not pickle ``phi4`` into every worker
    (hundreds of MB per process). Pass ``tdep_folder`` and rebuild locally.
    Tests may still ship a small ``contractors`` object.
    """
    contractors = spec.get("contractors")
    if contractors is None:
        folder = spec.get("tdep_folder")
        if not folder:
            raise ValueError("Phase-5 worker needs contractors= or tdep_folder=")
        from .contractors import SCContractors
        contractors = SCContractors.from_tdep_folder(
            folder, include_fourth=spec.get("include_fourth", True),
        )
    return contractors


def _instantiate_calculator(calculator):
    """Return a calculator instance from a class, factory, or live instance."""
    if calculator is None:
        return None
    if isinstance(calculator, type):
        return calculator()
    # ASE calculators have ``calculate``; factories are bare callables.
    if callable(calculator) and not hasattr(calculator, "calculate"):
        return calculator()
    return calculator


def build_energy_eval(spec):
    """Build a worker-local ``BatchEnergyEvaluator`` from picklable spec fields."""
    from ase import Atoms

    atoms = Atoms(
        spec["species"],
        positions=spec["eq"],
        cell=spec["cell"],
        pbc=True,
    )
    if spec.get("lammps_cmds") is not None:
        from ase.calculators.lammpslib import LAMMPSlib

        kwargs = dict(spec.get("lammps_kwargs") or {})
        atoms.calc = LAMMPSlib(
            lmpcmds=list(spec["lammps_cmds"]),
            keep_alive=True,
            log_file=_lammps_log_path(spec),
            **kwargs,
        )
    else:
        atoms.calc = _instantiate_calculator(spec["calculator"])
        if atoms.calc is None:
            raise ValueError("Phase-5 worker needs lammps_cmds= or calculator=")
    return BatchEnergyEvaluator(atoms, spec["eq"])


def run_phase5_loop(nconf, sampler, contractors, energy_eval, verbose=False, logger=None):
    """Serial Phase-5 loop over ``nconf`` draws. Used by n_workers=1 and by each worker."""
    V = np.zeros(nconf)
    V2 = np.zeros(nconf)
    V3 = np.zeros(nconf)
    V4 = np.zeros(nconf)
    V2_tilde = np.zeros(nconf)
    dV2_tilde_dT = np.zeros(nconf)
    import time
    t0 = time.time()
    for n in range(nconf):
        u, z = sampler.draw_with_z()
        V[n] = energy_eval.energy(u)
        V2[n] = contractors.V2(u)
        V3[n] = contractors.V3(u)
        V4[n] = contractors.V4(u)
        V2_tilde[n], dV2_tilde_dT[n] = sampler.V2_tilde_and_dT_from_z(z)
        if verbose and logger is not None and (n + 1) % max(1, nconf // 10) == 0:
            logger.info(f"  n={n+1}/{nconf}  ({time.time()-t0:.1f}s)")
    return V, V2, V3, V4, V2_tilde, dV2_tilde_dT


def eval_u_chunk(spec):
    """Evaluate V/V2/V3/V4 on a fixed list of displacements (no RNG).

    Used by tests so parallel vs serial can match bit-for-bit on contractors
    and energy, independent of jumped sample streams.
    """
    contractors = _load_contractors(spec)
    energy_eval = build_energy_eval(spec)
    us = spec["us"]
    n = len(us)
    V = np.empty(n)
    V2 = np.empty(n)
    V3 = np.empty(n)
    V4 = np.empty(n)
    for i, u in enumerate(us):
        V[i] = energy_eval.energy(u)
        V2[i] = contractors.V2(u)
        V3[i] = contractors.V3(u)
        V4[i] = contractors.V4(u)
    return V, V2, V3, V4


def run_phase5_chunk(spec):
    """Process-pool entry: jumped RNG, own LAMMPS, serial contractors."""
    contractors = _load_contractors(spec)
    sampler = spec["sampler"]
    sampler.rng = np.random.Generator(
        np.random.PCG64(spec["seed"]).jumped(int(spec["worker_id"]))
    )
    energy_eval = build_energy_eval(spec)
    return run_phase5_loop(int(spec["n_local"]), sampler, contractors, energy_eval)

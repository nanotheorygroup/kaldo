"""
Unified ``cumulant_thermo`` API.

Mirrors Ethan Meitz's ``CumulantAnalysis.jl`` ``crystal_thermodynamic_properties``
top-level runner. Given a TDEP folder (IFC2/IFC3/IFC4 plus structures),
compute the full cumulant thermodynamics

    F_total = F_H + F_0 + F_1 + F_2

and analogously U_total / S_total / Cv_total, with bootstrap SEs on the
constant correction (F_0 etc.) and zero SE on the analytic contributions.

Validated against Ethan's 25^3 Ne published values (Gate 6 PASS: F_total
matches to 1.7e-7 eV/atom < Ethan's own SE).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from kaldo.helpers.logger import get_logger
from .harmonic import harmonic_thermo
from .sampler import SCSampler
from .contractors import SCContractors
from .bootstrap import bootstrap_corrections
from kaldo.forceconstants import ForceConstants

logging = get_logger()


@dataclass
class CumulantResult:
    T_K: float
    Nat: int
    N_conf: int
    N_boot: int
    # Per-atom totals and component parts
    # F: eV / atom, U: eV / atom, S: kB / atom, Cv: kB / atom
    F_H: float
    U_H: float
    S_H: float
    Cv_H: float
    F_0: float
    U_0: float
    S_0: float
    Cv_0: float
    F_1: float
    U_1: float
    S_1: float
    Cv_1: float
    F_2: float
    U_2: float
    S_2: float
    Cv_2: float
    F_total: float
    U_total: float
    S_total: float
    Cv_total: float
    # Standard Errors, only from the 0th order correction
    F_total_SE: float
    U_total_SE: float
    S_total_SE: float
    Cv_total_SE: float
    # Raw energy samples (for size studies / re-bootstrapping)
    V: np.ndarray
    V2: np.ndarray
    V3: np.ndarray
    V4: np.ndarray
    V2_tilde: np.ndarray


def _parallel_phonon_frequencies(ph, n_workers):
    """``HarmonicWithQ`` frequencies on ``ph``'s mesh, sharded over processes."""
    from concurrent.futures import as_completed

    from kaldo.parallel import get_executor

    from ._harmonic_worker import freq_q_chunk, init_freq_worker

    q_points = np.asarray(ph._reciprocal_grid.unitary_grid(is_wrapping=False))
    n_q = int(q_points.shape[0])
    n_cpu = n_workers if n_workers is not None else (os.cpu_count() or 1)
    n_cpu = max(1, min(int(n_cpu), n_q))
    chunks = [c for c in np.array_split(q_points, n_cpu) if c.size]
    second = ph.forceconstants.second
    # Materialize IFC2 as numpy so spawn pickles arrays, not TF tensors.
    second._dynmat = np.asarray(second.dynmat)
    hwq_kwargs = dict(
        distance_threshold=ph.forceconstants.distance_threshold,
        folder=ph.folder,
        storage="memory",
        is_nw=ph.is_nw,
        is_unfolding=ph.is_unfolding,
        is_amorphous=ph._is_amorphous,
    )
    parts = [None] * len(chunks)
    with get_executor(
        backend="process",
        n_workers=len(chunks),
        initializer=init_freq_worker,
        initargs=(1, second, hwq_kwargs),
    ) as executor:
        futures = {executor.submit(freq_q_chunk, chunk): i
                   for i, chunk in enumerate(chunks)}
        for fut in as_completed(futures):
            parts[futures[fut]] = fut.result()
    return np.concatenate(parts, axis=0)


def _harmonic_thermo(forceconstants, kmesh, T_K, is_classic=False, n_workers=1):
    """Phase-1 F_H/U_H/S_H/Cv_H from ``Phonons`` frequencies.

    Uses kaldo's ``HarmonicWithQ`` dynamical matrix (including the
    per-pair Fourier phases on non-diagonal TDEP supercells from PR #301),
    then the cumulant classical/quantum closed-form sums.
    ``n_workers=1`` is the in-process ``Phonons.frequency`` loop; ``>1``
    shards that q-loop across processes (same dynmat, same ``n_workers``
    flag as F1/F2 and Phase 5).
    """
    from kaldo.parallel import is_parallel
    from kaldo.phonons import Phonons

    ph = Phonons(
        forceconstants=forceconstants,
        kpts=list(kmesh),
        temperature=T_K,
        is_classic=is_classic,
        storage="memory",
    )
    if is_parallel(n_workers):
        freqs = _parallel_phonon_frequencies(ph, n_workers)
    else:
        freqs = np.asarray(ph.frequency)  # (n_q, n_modes) THz
    n_q, n_modes = freqs.shape
    n_uc = n_modes // 3
    F, U, S, Cv = harmonic_thermo(
        freqs, T_K, n_q * n_uc, is_classic=is_classic,
    )
    return dict(F_H=F, U_H=U, S_H=S, Cv_H=Cv)


def cumulant_thermo(
    tdep_folder: str,
    supercell: tuple[int, int, int],
    temperature: float,
    is_classic: bool,
    lammps_cmds: Sequence[str] | str | None = None,
    nconf: int = 100_000,
    nboot: int = 5_000,
    harmonic_mesh: Sequence[int] = (30, 30, 30),
    free_energy_mesh: Sequence[int] = (25, 25, 25),
    seed: int = 987654,
    use_q_symmetry: bool = True,
    lammps_kwargs: dict | None = None,
    calculator=None,
    verbose: bool = True,
    n_workers: int | None = 1,
) -> CumulantResult:
    """
    Full cumulant thermodynamics on TDEP IFC inputs.

    Parameters
    ----------
    tdep_folder : str
        TDEP folder with infile.ucposcar / infile.ssposcar and the
        infile.forceconstant* files (IFC2, IFC3 and IFC4 are all required).
    supercell : tuple[int, int, int]
        Diagonal supercell tiling. Ignored (inferred from the structure
        files) when the ssposcar tiling is non-diagonal.
    temperature : float
        Temperature in Kelvin.
    is_classic : bool
        True for classical statistics, False for quantum. Required.
        Controls Phase-1 harmonic free energy (classical
        ``kT ln(hbar omega / kT)`` vs quantum Bose), analytic F1/F2
        occupations (classical high-T / LDT closed form vs Bose), and
        the Phase-5 sampler ensemble.
    lammps_cmds : str or sequence of str, optional
        ASE ``LAMMPSlib`` commands (pair_style, pair_coeff, ...) defining
        the potential; the sampler hits this calculator in a tight loop and
        reuses one live LAMMPS instance with Julia-style
        ``run 1 pre no`` (neighbor list built once; large skin).
        For example:
        ["pair_style lj/cut 6.955", "pair_coeff * * 0.0032135 2.782", "pair_modify shift yes"]
        Exactly one of ``lammps_cmds`` and ``calculator`` must be given.
    nconf : int
        Number of configurations to sample for 0th order correction. Default is 100_000.
    nboot : int
        Number of bootstrap samples to estimate error of 0th order correction. Default is 5_000.
    harmonic_mesh : tuple[int, int, int]
        q-mesh for harmonic thermo. Frequencies come from ``kaldo.Phonons``
        (SNF-correct dynmat). Default is (30, 30, 30).
    free_energy_mesh : tuple[int, int, int]
        q-mesh for analytic F1/F2. For optimal runtime/accuracy trade-off you should
        run a convergence study. Default is (25, 25, 25), but smaller meshes often work well.
    seed : int
        Random seed for the sampler.
    use_q_symmetry : bool
        Restricts the F1/F2 outer q1 loop to the spglib IBZ; disabling is
        meant for debugging. Default is True.
    lammps_kwargs : dict
        Keyword arguments for the LAMMPSlib calculator. LAMMPS commands should be provided via
        the `lammps_cmds` argument. `atom_types` and `atom_type_masses` will be populated automatically
        based on the unit-cell species and masses, but can be overridden if needed. See the
        ASE LAMMPSlib documentation for more details. Only valid with ``lammps_cmds``.
    calculator : ase Calculator, optional
        Any ASE calculator (an ML potential, EMT, ase's LennardJones, ...)
        to evaluate the sampled configurations instead of LAMMPS. The
        Phase-5 energies then go through the plain
        ``atoms.get_potential_energy()`` path (correct, without the LAMMPS
        neighbor-list reuse speedup). Exactly one of ``lammps_cmds`` and
        ``calculator`` must be given.
    verbose : bool
        If True, print progress messages. Default is True.
    n_workers : int or None
        Process-pool size for Phase-1 frequencies (full harmonic q-mesh),
        analytic F1/F2 (outer IBZ q1 loop), and Phase-5 sampling (one
        LAMMPS / calculator per process). ``1`` (default) is serial; ``>1``
        that many processes; ``None`` uses all CPUs. This is the only
        parallelism control for ``cumulant_thermo``; V2/V3/V4 contractions
        stay in-process. See :func:`kaldo.cumulant.free_energy.F1_vectorized`.
        When ``n_workers > 1`` and ``calculator=`` is used, pass a picklable
        instance or a no-arg factory (same rule as finite-difference
        ``n_workers``).

    Returns
    -------
    CumulantResult
    """

    from ase import Atoms
    from kaldo.cumulant.free_energy import _check_n_workers
    from kaldo.parallel import is_parallel, get_executor, validate_parallel_calculator

    _check_n_workers(n_workers)

    if (lammps_cmds is None) == (calculator is None):
        raise ValueError(
            "cumulant_thermo requires exactly one of lammps_cmds= (an ASE "
            "LAMMPSlib potential definition) or calculator= (any ASE calculator)."
        )
    if calculator is not None and lammps_kwargs:
        raise ValueError("lammps_kwargs is only valid together with lammps_cmds.")

    if lammps_cmds is not None:
        # Normalize lammps commands to a list of strings
        if isinstance(lammps_cmds, str):
            lammps_cmds = [lammps_cmds]
        else:
            lammps_cmds = list(lammps_cmds)

        # Copy so neither a caller-supplied dict nor a shared default is mutated
        # by the atom_types / atom_type_masses auto-population below.
        lammps_kwargs = dict(lammps_kwargs) if lammps_kwargs else {}

        if "lmpcmds" in lammps_kwargs:
            raise ValueError(
                "lmpcmds should be provided directly to `cumulant_thermo` "
                "via the lammps_cmds argument, not through lammps_kwargs."
            )

    forceconstants = ForceConstants.from_folder(
            tdep_folder,
            supercell=supercell,
            format="tdep",
            include_fourth=True
        )

    uc = forceconstants.atoms
    uc_positions = np.asarray(uc.get_positions())
    uc_cell = np.asarray(uc.get_cell())

    # Since we trust ASE to parse these, print them out
    # so the user can at least debug if something goes wrong 
    species_uc = uc.get_chemical_symbols()
    masses_amu_uc = uc.get_masses()
    if verbose:
        logging.info(f"Parsed unit-cell species with ASE as: {species_uc}")
        logging.info(f"Parsed unit-cell masses with ASE as: {masses_amu_uc}")

    # ---- Phase 1: harmonic (closed form) ----
    if verbose:
        logging.info(f"Phase 1: harmonic {harmonic_mesh} (Phonons dynmat) ...")
    t0 = time.time()
    harm = _harmonic_thermo(
        forceconstants, tuple(harmonic_mesh), temperature, is_classic=is_classic,
        n_workers=n_workers,
    )
    if verbose:
        logging.info(f"  F_H={harm['F_H']:+.4e}  U_H={harm['U_H']:+.4e}  "
                     f"S_H={harm['S_H']:+.4e}  Cv_H={harm['Cv_H']:+.4e}  "
              f"({time.time()-t0:.1f}s)")

    # ---- Phase 3: F1 analytic ----
    if verbose:
        logging.info(f"Phase 3: F1/S1/U1/Cv1 at mesh {free_energy_mesh} ...")
    t0 = time.time()
    from .free_energy import F1_from_fc
    res1 = F1_from_fc(
        forceconstants, masses_amu=masses_amu_uc,
        kmesh=tuple(free_energy_mesh), T_K=temperature,
        use_q_symmetry=use_q_symmetry, is_classic=is_classic,
        n_workers=n_workers,
    )
    if verbose:
        logging.info(f"  F1={res1['F1']:+.4e}  U1={res1['U1']:+.4e}  "
                     f"S1={res1['S1']:+.4e}  Cv1={res1['Cv1']:+.4e}  "
              f"({time.time()-t0:.1f}s)")

    # ---- Phase 4: F2 analytic ----
    if verbose:
        logging.info(f"Phase 4: F2/S2/U2/Cv2 at mesh {free_energy_mesh} ...")
    t0 = time.time()
    from .free_energy import F2_from_fc
    res2 = F2_from_fc(
        forceconstants, masses_amu=masses_amu_uc,
        kmesh=tuple(free_energy_mesh), T_K=temperature, sigma_THz=None,
        use_q_symmetry=use_q_symmetry, is_classic=is_classic,
        n_workers=n_workers,
    )
    if verbose:
        logging.info(f"  F2={res2['F2']:+.4e}  U2={res2['U2']:+.4e}  "
                     f"S2={res2['S2']:+.4e}  Cv2={res2['Cv2']:+.4e}  "
              f"({time.time()-t0:.1f}s)")

    # ---- Phase 5: MC constant correction (F0) ----
    if verbose:
        logging.info(f"Phase 5: sampling N={nconf} configs ...")

    # The whole of Phase 5 works in ONE atom frame: infile.ssposcar order.
    #  * The LAMMPS Atoms is read straight from ssposcar, so its cell,
    #    species and equilibrium positions are exactly TDEP's supercell
    #    (this also sidesteps ForceConstant.replicated_atoms, whose cell is
    #    only meaningful for diagonal tilings).
    #  * SCContractors index displacements in ssposcar order.
    #  * The sampler IFC2 (from irred_to_full, natively replica-major:
    #    replica_id * n_uc + atom_uc) is permuted into ssposcar order once,
    #    so the drawn displacement u is already in the contractor / LAMMPS
    #    frame and needs no per-configuration reindexing.
    import ase.io
    from kaldo.interfaces.tdep_io import build_supercell_replica_mapping

    sc = ase.io.read(str(Path(tdep_folder) / "infile.ssposcar"), format="vasp")
    species_sc = sc.get_chemical_symbols()
    masses_amu_sc = sc.get_masses()
    n_sc = len(sc)

    mapping = build_supercell_replica_mapping(uc, sc)
    n_uc = len(uc)
    # replica-major index of each ssposcar atom: file atom f sits at
    # (atom_of_sc[f], replica_id_of_sc[f]) -> replica_id * n_uc + atom_uc.
    replica_major_of_file = mapping["replica_id_of_sc"] * n_uc + mapping["atom_of_sc"]

    ifc2_replica_major = forceconstants.irred_to_full(order=2).reshape((3 * n_sc, 3 * n_sc))
    # Reindex rows/cols from replica-major to ssposcar order (3x3 blocks).
    dof_perm = (replica_major_of_file[:, None] * 3 + np.arange(3)).reshape(-1)
    ifc2_sc = ifc2_replica_major[np.ix_(dof_perm, dof_perm)]

    if verbose:
        logging.info(f"Phase 5: Remapped IFC2 to ssposcar frame, shape {ifc2_sc.shape} ...")

    sampler = SCSampler(
        ifc2_sc,
        masses_amu_sc,
        T_K=temperature,
        is_classic=is_classic,
        seed=seed
    )
    if verbose:
        logging.info(f"Phase 5: Built sampler ...")

    # Serial keeps contractors in-process. Parallel workers rebuild from
    # the TDEP folder so we do not pickle phi4.
    contractors = None
    if not is_parallel(n_workers):
        contractors = SCContractors.from_tdep_folder(tdep_folder, include_fourth=True)
        if verbose:
            logging.info("Phase 5: Build contractors ...")

    sc_cell_A = np.asarray(sc.get_cell())
    sc_pos_eq_A = np.asarray(sc.get_positions())

    if calculator is not None and is_parallel(n_workers):
        validate_parallel_calculator(calculator, "cumulant_thermo")

    from ._phase5_worker import run_phase5_loop, run_phase5_chunk, _lammps_log_path

    if calculator is not None:
        if verbose:
            logging.info(f"Phase 5: Using ASE calculator {type(calculator).__name__} ...")
    else:
        # Build mappings for the LAMMPS calculator
        _, unique_species_idx = np.unique(species_uc, return_index=True)
        if "atom_types" not in lammps_kwargs:
            lammps_kwargs["atom_types"] = {species_uc[idx]: i + 1 for i, idx in enumerate(unique_species_idx)}
        if "atom_type_masses" not in lammps_kwargs:
            lammps_kwargs["atom_type_masses"] = {species_uc[idx]: masses_amu_uc[idx]
                                                 for i, idx in enumerate(unique_species_idx)}
        if verbose:
            logging.info("Phase 5: Building LAMMPS calculator ...")
            logging.info(f"  LAMMPS atom_types: {lammps_kwargs['atom_types']}")
            logging.info(f"  LAMMPS atom_type_masses: {lammps_kwargs['atom_type_masses']}")

    t0 = time.time()
    if not is_parallel(n_workers):
        at = Atoms(species_sc, positions=sc_pos_eq_A, cell=sc_cell_A, pbc=True)
        if calculator is not None:
            at.calc = calculator
        else:
            from ase.calculators.lammpslib import LAMMPSlib
            at.calc = LAMMPSlib(
                lmpcmds=list(lammps_cmds),
                keep_alive=True,
                log_file=_lammps_log_path(),
                **lammps_kwargs
            )
        # Every sampled configuration shares topology (same atoms, same box; only
        # positions move), so keep one LAMMPS instance alive and, per config,
        # scatter coordinates + `run 1 pre no` (LDT pattern: recompute pe without
        # rebuilding neighbors; large skin keeps the list valid). The first
        # energy() call does the full ASE setup. See BatchEnergyEvaluator.
        from ._energy_evaluator import BatchEnergyEvaluator
        energy_eval = BatchEnergyEvaluator(at, sc_pos_eq_A)
        V, V2, V3, V4, V2_tilde, dV2_tilde_dT = run_phase5_loop(
            nconf, sampler, contractors, energy_eval,
            verbose=verbose, logger=logging,
        )
    else:
        n_cpu = n_workers if n_workers is not None else (os.cpu_count() or 1)
        n_cpu = max(1, min(int(n_cpu), nconf))
        counts = [len(c) for c in np.array_split(np.arange(nconf), n_cpu)]
        counts = [c for c in counts if c > 0]
        if verbose:
            logging.info(f"Phase 5: starting {len(counts)} processes "
                         f"({nconf} configs, one LAMMPS each) ...")
        from multiprocessing import shared_memory
        progress_shape = (len(counts), 8)  # one cache line per worker
        progress_shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(progress_shape)) * np.dtype(np.int64).itemsize,
        )
        progress = np.ndarray(progress_shape, dtype=np.int64, buffer=progress_shm.buf)
        progress.fill(0)
        base = dict(
            sampler=sampler,
            tdep_folder=str(Path(tdep_folder)),
            include_fourth=True,
            species=list(species_sc),
            eq=np.asarray(sc_pos_eq_A),
            cell=np.asarray(sc_cell_A),
            lammps_cmds=lammps_cmds,
            lammps_kwargs=lammps_kwargs,
            calculator=calculator,
            seed=seed,
            progress_shm_name=progress_shm.name,
            progress_shape=progress_shape,
        )
        specs = []
        for i, n_local in enumerate(counts):
            spec = dict(base)
            spec["n_local"] = int(n_local)
            spec["worker_id"] = i
            specs.append(spec)
        from concurrent.futures import FIRST_COMPLETED, wait
        parts = [None] * len(specs)
        try:
            with get_executor(backend="process", n_workers=len(specs)) as executor:
                futures = {executor.submit(run_phase5_chunk, spec): i
                           for i, spec in enumerate(specs)}
                pending = set(futures)
                report_every = max(1, (nconf + 49) // 50)
                next_report = report_every
                while pending:
                    done, pending = wait(
                        pending, timeout=1.0, return_when=FIRST_COMPLETED,
                    )
                    for fut in done:
                        i = futures[fut]
                        parts[i] = fut.result()
                    elapsed = time.time() - t0
                    completed = int(progress[:, 0].sum())
                    if verbose and (completed >= next_report or not pending):
                        logging.info(
                            f"  Phase 5: {completed}/{nconf} "
                            f"({100.0*completed/nconf:.0f}%)  "
                            f"elapsed={elapsed:.1f}s"
                        )
                        while next_report <= completed:
                            next_report += report_every
        finally:
            progress_shm.close()
            progress_shm.unlink()
        V = np.concatenate([p[0] for p in parts])
        V2 = np.concatenate([p[1] for p in parts])
        V3 = np.concatenate([p[2] for p in parts])
        V4 = np.concatenate([p[3] for p in parts])
        V2_tilde = np.concatenate([p[4] for p in parts])
        dV2_tilde_dT = np.concatenate([p[5] for p in parts])

    # Classical: V_ref = V2, dV_ref_dT ≡ 0.
    # Quantum: V_ref = V2_tilde (Bose-reweighted), with explicit ∂Ṽ₂/∂T.
    if is_classic:
        V_ref = V2
        dV_ref_dT = None
    else:
        V_ref = V2_tilde
        dV_ref_dT = dV2_tilde_dT
    point, se = bootstrap_corrections(
        V, V2, V3, V4, V_ref, temperature, n_sc,
        n_boot=nboot, seed=seed + 1, verbose=False,
        dV_ref_dT=dV_ref_dT,
    )
    if verbose:
        logging.info(f"F0={point['F0']:+.4e} +- {se['F0']:.2e}  "
                     f"U0={point['U0']:+.4e} +- {se['U0']:.2e}  "
              f"S0={point['S0']:+.4e} +- {se['S0']:.2e}  "
              f"Cv0={point['Cv0']:+.4e} +- {se['Cv0']:.2e}")

    # ---- Assemble totals (Julia bootstrap.jl convention) ----
    F_total = harm["F_H"] + point["F0"] + res1["F1"] + res2["F2"]
    U_total = harm["U_H"] + point["U0"] + res1["U1"] + res2["U2"]
    S_total = harm["S_H"] + point["S0"] + res1["S1"] + res2["S2"]
    Cv_total = harm["Cv_H"] + point["Cv0"] + res1["Cv1"] + res2["Cv2"]

    result = CumulantResult(
        T_K=temperature, Nat=n_sc, N_conf=nconf, N_boot=nboot,
        F_H=harm["F_H"], U_H=harm["U_H"], S_H=harm["S_H"], Cv_H=harm["Cv_H"],
        F_0=point["F0"], U_0=point["U0"],
        S_0=point["S0"], Cv_0=point["Cv0"],
        F_1=res1["F1"], U_1=res1["U1"], S_1=res1["S1"], Cv_1=res1["Cv1"],
        F_2=res2["F2"], U_2=res2["U2"], S_2=res2["S2"], Cv_2=res2["Cv2"],
        F_total=F_total, U_total=U_total, S_total=S_total, Cv_total=Cv_total,
        F_total_SE=se["F0"], U_total_SE=se["U0"],
        S_total_SE=se["S0"], Cv_total_SE=se["Cv0"],
        V=V, V2=V2, V3=V3, V4=V4, V2_tilde=V2_tilde,
    )
    return result


def print_thermo_table(result: CumulantResult) -> None:
    """Print the result in Ethan's ``*_mean.txt`` column layout."""
    print(f"# Cumulant thermo, T = {result.T_K} K, Nat = {result.Nat}, "
          f"N_conf = {result.N_conf}")
    # The 0th-order (constant) correction is the only stochastic term, so its
    # standard error IS the total standard error (analytic F1/F2 carry none).
    rows = [
        ("F", "eV/atom", result.F_H, result.F_0, result.F_1, result.F_2,
         result.F_total, result.F_total_SE),
        ("U", "eV/atom", result.U_H, result.U_0, result.U_1, result.U_2,
         result.U_total, result.U_total_SE),
        ("S", "kB/atom", result.S_H, result.S_0, result.S_1, result.S_2,
         result.S_total, result.S_total_SE),
        ("Cv", "kB/atom", result.Cv_H, result.Cv_0, result.Cv_1, result.Cv_2,
         result.Cv_total, result.Cv_total_SE),
    ]
    col = 14
    for name, unit, H, zeroth, a1, a2, tot, se in rows:
        print(f"\n# {name} [{unit}]")
        labels = (f"{name}_H", f"{name}_0", f"{name}_1", f"{name}_2",
                  f"{name}_total")
        print("".join(f"{lab:>{col}}" for lab in labels))
        print("".join(f"{v:+{col}.7f}" for v in (H, zeroth, a1, a2, tot)))
        print("".join(f"{v:+{col}.7f}" for v in (0.0, se, 0.0, 0.0, se)))

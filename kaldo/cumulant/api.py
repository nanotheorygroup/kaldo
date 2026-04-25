"""
Unified ``cumulant_thermo`` API.

Mirrors Ethan Meitz's ``CumulantAnalysis.jl`` ``crystal_thermodynamic_properties``
top-level runner. Given a TDEP IFC directory + a Julia-dumped SC-remapped
IFC HDF5, compute the full cumulant thermodynamics:

    F_total = F_H + F_offset + F_1 + F_2

and analogously U_total / S_total / Cv_total, with bootstrap SEs on the
constant-correction piece (F_offset etc.) and zero SE on the analytic
contributions.

Validated against Ethan's 25^3 Ne published values (Gate 6 PASS: F_total
matches to 1.7e-7 eV/atom < Ethan's own SE).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import ase.io

from .common import NE_MASS_AMU, AMU, read_tdep_pair_fcs, read_tdep_ifc3, read_tdep_ifc4
from .harmonic import compute_all_frequencies_THz, harmonic_thermo_quantum
from .free_energy import F1_vectorized, F2_vectorized
from .sampler import SCSampler
from .taylor import SCContractors
from .bootstrap import bootstrap_corrections


@dataclass
class CumulantResult:
    """Per-atom cumulant thermodynamics plus the raw Phase 5 samples."""

    T_K: float
    Nat: int
    N_conf: int
    # Per-atom totals and component parts
    F_H: float; U_H: float; S_H: float; Cv_H: float
    F_offset: float; U_offset: float; S_offset: float; Cv_offset: float
    F_1: float; U_1: float; S_1: float; Cv_1: float
    F_2: float; U_2: float; S_2: float; Cv_2: float
    F_total: float; U_total: float; S_total: float; Cv_total: float
    # SEs (analytic parts are 0; only offset has nonzero bootstrap SE)
    F_offset_SE: float; U_offset_SE: float; S_offset_SE: float; Cv_offset_SE: float
    F_total_SE: float; U_total_SE: float; S_total_SE: float; Cv_total_SE: float
    # raw Phase 5 samples (for size studies / re-bootstrapping)
    V: np.ndarray; V2: np.ndarray; V3: np.ndarray; V4: np.ndarray; V2_tilde: np.ndarray


def _harmonic_thermo(neighbors_pair, uc_positions, uc_cell, masses_amu, kmesh, T_K):
    masses_kg = masses_amu * AMU
    freqs = compute_all_frequencies_THz(
        neighbors_pair, uc_positions, masses_kg, uc_cell, kmesh
    )
    n_q = freqs.shape[0]
    n_uc = len(uc_positions)
    F, U, S, Cv = harmonic_thermo_quantum(freqs, T_K, n_q * n_uc)
    return dict(F_H=F, U_H=U, S_H=S, Cv_H=Cv)


def cumulant_thermo(
    ifc_dir: str | Path,
    ifcs_sc_h5: str | Path,
    temperature: float = 24.0,
    nconf: int = 1000,
    nboot: int = 5000,
    quantum: bool = True,
    harmonic_mesh: Sequence[int] = (30, 30, 30),
    free_energy_mesh: Sequence[int] = (25, 25, 25),
    lammps_cmds: Sequence[str] = (
        "pair_style lj/cut 6.955",
        "pair_coeff * * 0.0032135 2.782",
        "pair_modify shift yes",
    ),
    species: str = "Ne",
    species_mass_amu: float | None = None,
    seed: int = 987654,
    use_q_symmetry: bool = False,
    verbose: bool = True,
    forceconstants=None,
) -> CumulantResult:
    """
    Full cumulant thermodynamics on TDEP IFC inputs.

    Parameters
    ----------
    ifc_dir : str | Path
        Directory with ``infile.ucposcar``, ``infile.ssposcar``,
        ``infile.forceconstant``, ``infile.forceconstant_thirdorder``,
        ``infile.forceconstant_fourthorder``.
    ifcs_sc_h5 : str | Path
        HDF5 with SC-remapped IFC2/3/4 (one entry per supercell pair/
        triplet/quartet). Produced by Julia's ``dump_remapped_ifcs.jl``.
    temperature, nconf, nboot, quantum : phase-5 MC + bootstrap knobs.
    harmonic_mesh, free_energy_mesh : primitive-cell q-mesh for harmonic
        thermo and analytic F1/F2 respectively.
    lammps_cmds : ASE ``LAMMPSlib`` pair_style + pair_coeff etc. The
        sampler hits this calculator in a tight loop.
    species : chemical symbol used to build ASE atoms.
    species_mass_amu : atomic mass in amu. Defaults to NE_MASS_AMU for
        ``species='Ne'``; must be supplied for anything else.
    use_q_symmetry : if True, restrict F1/F2 outer q1 loop to spglib IBZ
        reps. ~|point-group|x speedup at high-symmetry crystals with
        bit-for-bit identical results.
    forceconstants : kaldo.forceconstants.ForceConstants, optional
        If supplied, skip reading IFC2/IFC3/IFC4 from ``ifc_dir`` and use
        this pre-loaded ForceConstants object (must include fourth).
        ``ifc_dir`` is still used as the ASE Atoms source for primitive
        and supercell geometry.

    Returns
    -------
    CumulantResult
    """
    if species_mass_amu is None:
        if species == "Ne":
            species_mass_amu = NE_MASS_AMU
        else:
            raise ValueError(
                f"species_mass_amu must be supplied when species={species!r}"
            )

    ifc_dir = Path(ifc_dir); ifcs_sc_h5 = Path(ifcs_sc_h5)
    uc = ase.io.read(str(ifc_dir / "infile.ucposcar"), format="vasp")
    ss = ase.io.read(str(ifc_dir / "infile.ssposcar"), format="vasp")
    uc_positions = np.asarray(uc.get_positions())
    uc_cell = np.asarray(uc.get_cell())
    n_uc = len(uc); n_sc = len(ss)
    masses_amu = np.full(n_uc, species_mass_amu)
    masses_amu_sc = np.full(n_sc, species_mass_amu)
    masses_kg = masses_amu * AMU

    if forceconstants is None:
        # Legacy list-based path (preserves backward compat).
        neighbors_pair = read_tdep_pair_fcs(
            ifc_dir / "infile.forceconstant", uc_positions, uc_cell
        )
        quartets = read_tdep_ifc4(ifc_dir / "infile.forceconstant_fourthorder", n_uc)
        triplets = read_tdep_ifc3(ifc_dir / "infile.forceconstant_thirdorder", n_uc)
    else:
        neighbors_pair = None
        quartets = None
        triplets = None

    # ---- Phase 1: harmonic (closed form) ----
    if verbose:
        print(f"Phase 1: harmonic {harmonic_mesh} ...", flush=True)
    t0 = time.time()
    if forceconstants is None:
        harm = _harmonic_thermo(
            neighbors_pair, uc_positions, uc_cell, masses_amu,
            tuple(harmonic_mesh), temperature,
        )
    else:
        from .free_energy import _neighbors_from_fc
        harm_neighbors = _neighbors_from_fc(forceconstants)
        harm = _harmonic_thermo(
            harm_neighbors, uc_positions, uc_cell, masses_amu,
            tuple(harmonic_mesh), temperature,
        )
    if verbose:
        print(f"  F_H={harm['F_H']:+.4e}  U_H={harm['U_H']:+.4e}  "
              f"S_H={harm['S_H']:+.4e}  Cv_H={harm['Cv_H']:+.4e}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    # ---- Phase 3: F_1 analytic (quartic) ----
    if verbose:
        print(f"Phase 3: F1/S1/U1/Cv1 at mesh {free_energy_mesh} ...", flush=True)
    t0 = time.time()
    if forceconstants is None:
        res1 = F1_vectorized(
            neighbors_pair, quartets, masses_kg, uc_positions, uc_cell,
            tuple(free_energy_mesh), temperature,
            use_q_symmetry=use_q_symmetry, atoms=uc if use_q_symmetry else None,
        )
    else:
        from .free_energy import F1_from_fc
        res1 = F1_from_fc(
            forceconstants, masses_amu=masses_amu,
            kmesh=tuple(free_energy_mesh), T_K=temperature,
            use_q_symmetry=use_q_symmetry,
        )
    if verbose:
        print(f"  F1={res1['F1']:+.4e}  U1={res1['U1']:+.4e}  "
              f"S1={res1['S1']:+.4e}  Cv1={res1['Cv1']:+.4e}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    # ---- Phase 4: F_2 analytic (cubic) ----
    if verbose:
        print(f"Phase 4: F2/S2/U2/Cv2 at mesh {free_energy_mesh} ...", flush=True)
    t0 = time.time()
    if forceconstants is None:
        res2 = F2_vectorized(
            neighbors_pair, triplets, masses_kg, uc_positions, uc_cell,
            tuple(free_energy_mesh), temperature, sigma_THz=None,
            use_q_symmetry=use_q_symmetry, atoms=uc if use_q_symmetry else None,
        )
    else:
        from .free_energy import F2_from_fc
        res2 = F2_from_fc(
            forceconstants, masses_amu=masses_amu,
            kmesh=tuple(free_energy_mesh), T_K=temperature, sigma_THz=None,
            use_q_symmetry=use_q_symmetry,
        )
    if verbose:
        print(f"  F2={res2['F2']:+.4e}  U2={res2['U2']:+.4e}  "
              f"S2={res2['S2']:+.4e}  Cv2={res2['Cv2']:+.4e}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    # ---- Phase 5: MC constant correction (F_offset) ----
    if verbose:
        print(f"Phase 5: sampling N={nconf} configs ...", flush=True)
    sampler = SCSampler(
        ifcs_sc_h5, masses_amu_sc, T_K=temperature, quantum=quantum, seed=seed,
    )
    contractors = SCContractors(ifcs_sc_h5)
    sc_cell_A = np.asarray(ss.get_cell())
    sc_pos_eq_A = np.asarray(ss.get_positions())

    from ase import Atoms
    from ase.calculators.lammpslib import LAMMPSlib
    at = Atoms(species * n_sc, positions=sc_pos_eq_A, cell=sc_cell_A, pbc=True)
    at.calc = LAMMPSlib(
        lmpcmds=list(lammps_cmds), atom_types={species: 1},
        keep_alive=True, log_file="/tmp/cumulant_thermo_lammps.log",
    )
    V = np.zeros(nconf); V2 = np.zeros(nconf); V3 = np.zeros(nconf)
    V4 = np.zeros(nconf); V2_tilde = np.zeros(nconf)
    t0 = time.time()
    for n in range(nconf):
        u, z = sampler.draw_with_z()
        at.set_positions(sc_pos_eq_A + u)
        V[n] = at.get_potential_energy()
        V2[n] = contractors.V2(u)
        V3[n] = contractors.V3(u)
        V4[n] = contractors.V4(u)
        V2_tilde[n] = sampler.V2_tilde_from_z(z)
        if verbose and (n + 1) % max(1, nconf // 10) == 0:
            print(f"  n={n+1}/{nconf}  ({time.time()-t0:.1f}s)", flush=True)

    V_ref = V2_tilde if quantum else V2
    point, se = bootstrap_corrections(
        V, V2, V3, V4, V_ref, temperature, n_sc,
        n_boot=nboot, seed=seed + 1, verbose=False,
    )
    if verbose:
        print(f"  F_offset={point['F_offset']:+.4e} +- {se['F_offset']:.2e}  "
              f"U_offset={point['U_offset']:+.4e} +- {se['U_offset']:.2e}  "
              f"S_offset={point['S_offset']:+.4e} +- {se['S_offset']:.2e}  "
              f"Cv_offset={point['Cv_offset']:+.4e} +- {se['Cv_offset']:.2e}",
              flush=True)

    # ---- Assemble totals (Julia bootstrap.jl convention) ----
    F_total = harm["F_H"] + point["F_offset"] + res1["F1"] + res2["F2"]
    U_total = harm["U_H"] + point["U_offset"] + res1["U1"] + res2["U2"]
    S_total = harm["S_H"] + point["S_offset"] + res1["S1"] + res2["S2"]
    Cv_total = harm["Cv_H"] + point["Cv_offset"] + res1["Cv1"] + res2["Cv2"]

    return CumulantResult(
        T_K=temperature, Nat=n_sc, N_conf=nconf,
        F_H=harm["F_H"], U_H=harm["U_H"], S_H=harm["S_H"], Cv_H=harm["Cv_H"],
        F_offset=point["F_offset"], U_offset=point["U_offset"],
        S_offset=point["S_offset"], Cv_offset=point["Cv_offset"],
        F_1=res1["F1"], U_1=res1["U1"], S_1=res1["S1"], Cv_1=res1["Cv1"],
        F_2=res2["F2"], U_2=res2["U2"], S_2=res2["S2"], Cv_2=res2["Cv2"],
        F_total=F_total, U_total=U_total, S_total=S_total, Cv_total=Cv_total,
        F_offset_SE=se["F_offset"], U_offset_SE=se["U_offset"],
        S_offset_SE=se["S_offset"], Cv_offset_SE=se["Cv_offset"],
        F_total_SE=se["F_offset"], U_total_SE=se["U_offset"],
        S_total_SE=se["S_offset"], Cv_total_SE=se["Cv_offset"],
        V=V, V2=V2, V3=V3, V4=V4, V2_tilde=V2_tilde,
    )


def print_ethan_table(result: CumulantResult) -> None:
    """Print the result in Ethan's ``*_mean.txt`` column layout."""
    print(f"# Cumulant thermo, T = {result.T_K} K, Nat = {result.Nat}, "
          f"N_conf = {result.N_conf}")
    rows = [
        ("F", "eV/atom", result.F_H, result.F_offset, result.F_1, result.F_2,
         result.F_total, result.F_offset_SE, result.F_total_SE),
        ("U", "eV/atom", result.U_H, result.U_offset, result.U_1, result.U_2,
         result.U_total, result.U_offset_SE, result.U_total_SE),
        ("S", "kB/atom", result.S_H, result.S_offset, result.S_1, result.S_2,
         result.S_total, result.S_offset_SE, result.S_total_SE),
        ("Cv", "kB/atom", result.Cv_H, result.Cv_offset, result.Cv_1, result.Cv_2,
         result.Cv_total, result.Cv_offset_SE, result.Cv_total_SE),
    ]
    for name, unit, H, off, a1, a2, tot, offse, totse in rows:
        print(f"\n# {name} [{unit}]")
        print(f"       {name}_H          {name}_offset         "
              f"{name}_1            {name}_2            {name}_total")
        print(f"  {H:+.7f}      {off:+.7f}      "
              f"{a1:+.7f}      {a2:+.7f}      {tot:+.7f}")
        print(f"  {'0':>14}      {offse:.7f}      "
              f"{'0':>14}      {'0':>14}      {totse:.7f}")

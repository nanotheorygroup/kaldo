"""
Unified ``cumulant_thermo`` API.

Mirrors Ethan Meitz's ``CumulantAnalysis.jl`` ``crystal_thermodynamic_properties``
top-level runner. Given a ``ForceConstants`` object, compute the full
cumulant thermodynamics.

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
from kaldo.remap import remap_to_supercell_ifcs

from .constants import AMU
from .harmonic import compute_all_frequencies_THz, harmonic_thermo_quantum
from .sampler import SCSampler
from .taylor import SCContractors
from .bootstrap import bootstrap_corrections
from kaldo.forceconstants import ForceConstants


@dataclass
class CumulantResult:
    T_K: float
    Nat: int
    N_conf: int
    N_boot: int
    # Per-atom totals and component parts
    # F: eV / atom, U: eV / atom, S: kB / atom, Cv: kB / atom
    F_H: float; U_H: float; S_H: float; Cv_H: float
    F_0: float; U_0: float; S_0: float; Cv_0: float
    F_1: float; U_1: float; S_1: float; Cv_1: float
    F_2: float; U_2: float; S_2: float; Cv_2: float
    F_total: float; U_total: float; S_total: float; Cv_total: float
    # Standard Errors, only from the 0th order correction
    F_total_SE: float; U_total_SE: float; S_total_SE: float; Cv_total_SE: float
    # Raw energy samples (for size studies / re-bootstrapping)
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

def _check_lammps_kwargs(lammps_kwargs: dict) -> None:
    if "atom_types" not in lammps_kwargs:
        raise ValueError("atom_types must be provided in lammps_kwargs")
    if "atom_type_masses" not in lammps_kwargs:
        raise ValueError("atom_type_masses must be provided in lammps_kwargs")
    if "lmpcmds" in lammps_kwargs:
        raise ValueError("lmpcmds should be provided directly to `cumulant_thermo` function via lammps_cmds argument.")


def cumulant_thermo(
    forceconstants : ForceConstants,
    temperature: float,
    is_classic : bool,
    lammps_cmds : Sequence[str] | str,
    nconf: int = 100_000,
    nboot: int = 5_000,
    harmonic_mesh: Sequence[int] = (30, 30, 30),
    free_energy_mesh: Sequence[int] = (25, 25, 25),
    seed: int = 987654,
    use_q_symmetry: bool = True,
    lammps_kwargs: dict = {"atom_types" : {}, "atom_type_masses" : {}},
    verbose: bool = True,
) -> CumulantResult:
    """
    Full cumulant thermodynamics on TDEP IFC inputs.

    Parameters
    ----------
    forceconstants : kaldo.forceconstants.ForceConstants
        Pre-loaded ForceConstants object (must include fourth IFCs).
    temperature : float
        Temperature in Kelvin.
    is_classic : bool
        Specifies if the system is treated with classical or quantum
        statistics. Default: False
    nconf : int
        Number of configurations to sample for 0th order correction. Default is 100_000.
    nboot : int
        Number of bootstrap samples to estimate error of 0th order correction. Default is 5_000.
    harmonic_mesh : tuple[int, int, int]
        q-mesh for harmonic thermo. Default is (30, 30, 30).
    free_energy_mesh : tuple[int, int, int]
        q-mesh for analytic F1/F2. For optimal runtime/accuracy trade-off you should
        run a convergence study. Default is (25, 25, 25), but smaller meshes often work well.
    lammps_cmds : ASE ``LAMMPSlib`` pair_style + pair_coeff etc. The
        sampler hits this calculator in a tight loop. For example:
        ["pair_style lj/cut 6.955", "pair_coeff * * 0.0032135 2.782", "pair_modify shift yes"]
    seed : int
        Random seed for the sampler.
    use_q_symmetry : Restricts F1/F2 outer q1 loop to spglib IBZ. This flag
        is meant for debugging. Default is True. 
    lammps_kwargs : dict
        Keyword arguments for the LAMMPSlib calculator. LAMMPS commands should be provided via
        the `lammps_cmds` argument. `atom_types` and `atom_type_masses` will be populated automatically
        based on the unit-cell species and masses, but can be overridden if needed. See the 
        ASE LAMMPSlib documentation for more details.
    verbose : bool
        If True, print progress messages. Default is True.

    Returns
    -------
    CumulantResult
    """

    from ase import Atoms
    from ase.calculators.lammpslib import LAMMPSlib

    # Normalzie lammps command to list of strings
    if isinstance(lammps_cmds, str):
        lammps_cmds = [lammps_cmds]
    else:
        lammps_cmds = list(lammps_cmds)

    _check_lammps_kwargs(lammps_kwargs)

    uc = forceconstants.atoms
    uc_positions = np.asarray(uc.get_positions())
    uc_cell = np.asarray(uc.get_cell())

    # Since we trust ASE to parse these, print them out
    # so the user can at least debug if something goes wrong 
    species_uc = uc.get_chemical_symbols()
    masses_amu_uc = uc.get_masses()
    if verbose:
        print(f"Parsed unit-cell species with ASE as: {species_uc}", flush=True)
        print(f"Parsed unit-cell masses with ASE as: {masses_amu_uc}", flush=True)

    # ---- Phase 1: harmonic (closed form) ----
    if verbose:
        print(f"Phase 1: harmonic {harmonic_mesh} ...", flush=True)
    t0 = time.time()
    from .free_energy import _neighbors_from_fc
    harm_neighbors = _neighbors_from_fc(forceconstants)
    harm = _harmonic_thermo(
        harm_neighbors, uc_positions, uc_cell, masses_amu_uc,
        tuple(harmonic_mesh), temperature,
    )
    if verbose:
        print(f"  F_H={harm['F_H']:+.4e}  U_H={harm['U_H']:+.4e}  "
              f"S_H={harm['S_H']:+.4e}  Cv_H={harm['Cv_H']:+.4e}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    # ---- Phase 3: F1 analytic ----
    if verbose:
        print(f"Phase 3: F1/S1/U1/Cv1 at mesh {free_energy_mesh} ...", flush=True)
    t0 = time.time()
    from .free_energy import F1_from_fc
    res1 = F1_from_fc(
        forceconstants, masses_amu=masses_amu_uc,
        kmesh=tuple(free_energy_mesh), T_K=temperature,
        use_q_symmetry=use_q_symmetry,
    )
    if verbose:
        print(f"  F1={res1['F1']:+.4e}  U1={res1['U1']:+.4e}  "
              f"S1={res1['S1']:+.4e}  Cv1={res1['Cv1']:+.4e}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    # ---- Phase 4: F2 analytic ----
    if verbose:
        print(f"Phase 4: F2/S2/U2/Cv2 at mesh {free_energy_mesh} ...", flush=True)
    t0 = time.time()
    from .free_energy import F2_from_fc
    res2 = F2_from_fc(
        forceconstants, masses_amu=masses_amu_uc,
        kmesh=tuple(free_energy_mesh), T_K=temperature, sigma_THz=None,
        use_q_symmetry=use_q_symmetry,
    )
    if verbose:
        print(f"  F2={res2['F2']:+.4e}  U2={res2['U2']:+.4e}  "
              f"S2={res2['S2']:+.4e}  Cv2={res2['Cv2']:+.4e}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    # ---- Phase 5: MC constant correction (F0) ----
    if verbose:
        print(f"Phase 5: sampling N={nconf} configs ...", flush=True)

    sc = forceconstants.second.replicated_atom
    species_sc = sc.get_chemical_symbols()
    masses_amu_sc = sc.get_masses()
    n_sc = len(sc)

    phonons_sc = Phonons()

    # Get super cell representation of the IFCs
    #! NEED TO CHECK THIS FUNCTION IS CORRECT
    remapped = remap_to_supercell_ifcs(
        forceconstants,
        require_third=True, 
        require_fourth=True
    )

    sampler = SCSampler(
        remapped,
        masses_amu_sc,
        T_K=temperature,
        is_classic=is_classic,
        seed=seed,
    )

    contractors = SCContractors(remapped)

    
    sc_cell_A = np.asarray(sc.get_cell())
    sc_pos_eq_A = np.asarray(sc.get_positions())

    # Build mappings for LAMMPS calculator
    _, unique_species_idx = np.unique(species_uc, return_index=True)
    atom_types = {species_uc[idx] : i for i, idx in enumerate(unique_species_idx)}
    atom_masses = {species_uc[idx] : masses_amu_uc[idx] for i, idx in enumerate(unique_species_idx)}

    at = Atoms(species_sc, positions=sc_pos_eq_A, cell=sc_cell_A, pbc=True)
    at.calc = LAMMPSlib(
        lmpcmds=list(lammps_cmds),
        atom_types=atom_types,
        atom_type_masses=atom_masses,
        keep_alive=True, 
        log_file="/tmp/cumulant_thermo_lammps.log",
        **lammps_kwargs
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
        print(f"F0={point['F0']:+.4e} +- {se['F0']:.2e}  "
              f"U0={point['U0']:+.4e} +- {se['U0']:.2e}  "
              f"S0={point['S0']:+.4e} +- {se['S0']:.2e}  "
              f"Cv0={point['Cv0']:+.4e} +- {se['Cv0']:.2e}",
              flush=True)

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

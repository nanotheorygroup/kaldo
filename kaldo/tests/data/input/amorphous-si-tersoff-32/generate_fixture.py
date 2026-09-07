#!/usr/bin/env python3
"""Generate the 32-atom amorphous-Si IFC interpolation fixture.

The script has no external model download.  It uses ASE's implementation of
the Si parameters published by Tersoff, Phys. Rev. B 37, 6991 (1988), melts
and quenches a deterministic random packing, relaxes the resulting periodic
cell, and evaluates IFC2/IFC3 through kALDo's public finite-displacement path.

Run this script from the repository root.  ``--structure-only`` is useful for
auditing the melt--quench before committing the more expensive IFC3 run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import ase
import ase.io
from ase import Atoms, units
from ase.calculators.tersoff import Tersoff, TersoffParameters
from ase.constraints import FixCom
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import thermalize_momenta
from ase.optimize import FIRE
import numpy as np

# Executing a script by path puts this fixture directory, not the repository
# root, first on sys.path.  Bind regeneration to the checkout that contains the
# script instead of an unrelated editable-installed kALDo worktree.
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPOSITORY_ROOT))

N_ATOMS = 32
SEED = 20260817
DENSITY_KG_M3 = 2330.0
MIN_INITIAL_DISTANCE_ANGSTROM = 2.0
TIME_STEP_FS = 0.5
MELT_TEMPERATURE_K = 3500.0
FINAL_TEMPERATURE_K = 300.0
MELT_STEPS = 4000
QUENCH_STEPS = 10000
EQUILIBRATION_STEPS = 2000
RELAX_FMAX_EV_ANGSTROM = 1.0e-5
IFC2_DISPLACEMENT_ANGSTROM = 1.0e-3
IFC3_DISPLACEMENT_ANGSTROM = 1.0e-3
IFC3_DISTANCE_THRESHOLD_ANGSTROM = 3.25
IFC_ARTIFACTS = (
    "replicated_atoms.xyz",
    "replicated_atoms_third.xyz",
    "second.npy",
    "third.npz",
)


def tersoff_parameters():
    """Return the published elemental-Si parameter set used by ASE tests."""
    return {
        ("Si", "Si", "Si"): TersoffParameters(
            A=3264.7,
            B=95.373,
            lambda1=3.2394,
            lambda2=1.3258,
            lambda3=1.3258,
            beta=0.33675,
            gamma=1.0,
            m=3.0,
            n=22.956,
            c=4.8381,
            d=2.0417,
            h=0.0,
            R=3.0,
            D=0.2,
        )
    }


def make_calculator():
    """Construct an independent Tersoff calculator for one worker."""
    return Tersoff(parameters=tersoff_parameters())


def cell_length_angstrom():
    """Return the cubic length corresponding to the target mass density."""
    mass_kg = N_ATOMS * 28.0855 * 1.66053906660e-27
    volume_angstrom3 = mass_kg / DENSITY_KG_M3 / 1.0e-30
    return volume_angstrom3 ** (1.0 / 3.0)


def random_packing(rng):
    """Build a deterministic periodic random packing without close contacts."""
    length = cell_length_angstrom()
    positions = []
    attempts = 0
    while len(positions) < N_ATOMS:
        attempts += 1
        if attempts > 1_000_000:
            raise RuntimeError("could not construct the initial random packing")
        candidate = rng.random(3) * length
        if positions:
            delta = candidate - np.asarray(positions)
            delta -= length * np.rint(delta / length)
            if np.min(np.linalg.norm(delta, axis=1)) < MIN_INITIAL_DISTANCE_ANGSTROM:
                continue
        positions.append(candidate)
    return Atoms(
        "Si" * N_ATOMS,
        positions=np.asarray(positions),
        cell=np.eye(3) * length,
        pbc=True,
    )


def canonicalize_static_structure(atoms):
    """Wrap the retained basis and remove trajectory-only momenta."""
    atoms.wrap()
    if atoms.has("momenta"):
        atoms.set_momenta(None)
    atoms.calc = None
    return atoms


def melt_quench(output, force=False):
    """Create and relax the deterministic amorphous structure."""
    structure_path = output / "amorphous_si_32.extxyz"
    if structure_path.exists() and not force:
        atoms = canonicalize_static_structure(ase.io.read(structure_path))
        ase.io.write(structure_path, atoms, format="extxyz")
        return atoms

    rng = np.random.default_rng(SEED)
    atoms = random_packing(rng)
    atoms.calc = make_calculator()

    # Remove the largest random-packing forces before starting dynamics.
    print("pre-relaxing the random packing", flush=True)
    FIRE(atoms, logfile=None).run(fmax=2.0, steps=500)
    thermalize_momenta(atoms, MELT_TEMPERATURE_K, rng=rng)
    atoms.set_constraint(FixCom())
    dynamics = Langevin(
        atoms,
        TIME_STEP_FS * units.fs,
        temperature_K=MELT_TEMPERATURE_K,
        friction=0.02 / units.fs,
        fixcm=False,
        rng=rng,
        logfile=None,
    )
    print(f"melting at {MELT_TEMPERATURE_K:g} K for {MELT_STEPS} steps", flush=True)
    dynamics.run(MELT_STEPS)
    print(f"quenching to {FINAL_TEMPERATURE_K:g} K", flush=True)
    chunk = 100
    for start in range(0, QUENCH_STEPS, chunk):
        fraction = min((start + chunk) / QUENCH_STEPS, 1.0)
        temperature = MELT_TEMPERATURE_K + fraction * (
            FINAL_TEMPERATURE_K - MELT_TEMPERATURE_K
        )
        dynamics.set_temperature(temperature_K=temperature)
        dynamics.run(min(chunk, QUENCH_STEPS - start))
    dynamics.set_temperature(temperature_K=FINAL_TEMPERATURE_K)
    print(f"equilibrating for {EQUILIBRATION_STEPS} steps", flush=True)
    dynamics.run(EQUILIBRATION_STEPS)

    atoms.set_constraint()
    print("relaxing the retained inherent structure", flush=True)
    converged = FIRE(atoms, logfile=None).run(fmax=RELAX_FMAX_EV_ANGSTROM, steps=5000)
    if not converged:
        raise RuntimeError(
            "final FIRE relaxation did not reach "
            f"{RELAX_FMAX_EV_ANGSTROM:g} eV/angstrom"
        )
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces())
    atoms.info.update(
        {
            "generator": "ASE Tersoff deterministic melt-quench",
            "seed": SEED,
            "potential_reference": "Tersoff, Phys. Rev. B 37, 6991 (1988)",
            "potential_energy_eV": energy,
            "max_force_eV_per_A": float(np.max(np.linalg.norm(forces, axis=1))),
        }
    )
    canonicalize_static_structure(atoms)
    ase.io.write(structure_path, atoms, format="extxyz")
    return atoms


def calculate_ifcs(atoms, output, n_workers):
    """Recompute and store compact IFC2 and sparse IFC3 with kALDo.

    ``is_storing=False`` is intentional: a reproducibility script must never
    accept an older tensor merely because the expected output path exists.
    Both objects are saved explicitly after the fresh calculation.
    """
    import kaldo
    from kaldo.forceconstants import ForceConstants

    imported_root = Path(kaldo.__file__).resolve().parents[1]
    if imported_root != REPOSITORY_ROOT:
        raise RuntimeError(
            f"generator imported kALDo from {imported_root}, expected its "
            f"containing checkout {REPOSITORY_ROOT}"
        )

    forceconstants = ForceConstants(
        atoms=atoms,
        supercell=(1, 1, 1),
        third_supercell=(1, 1, 1),
        folder=str(output),
        is_acoustic_sum=True,
    )
    print("calculating IFC2", flush=True)
    forceconstants.second.calculate(
        calculator=make_calculator,
        delta_shift=IFC2_DISPLACEMENT_ANGSTROM,
        n_workers=n_workers,
        is_storing=False,
        symmetrize=False,
    )
    forceconstants.second.save("second")
    ase.io.write(
        output / "replicated_atoms.xyz",
        forceconstants.second.replicated_atoms,
        format="extxyz",
    )
    print("calculating sparse IFC3", flush=True)
    forceconstants.third.calculate(
        calculator=make_calculator,
        delta_shift=IFC3_DISPLACEMENT_ANGSTROM,
        distance_threshold=IFC3_DISTANCE_THRESHOLD_ANGSTROM,
        n_workers=n_workers,
        is_storing=False,
        symmetrize=False,
    )
    forceconstants.third.save("third")


def sha256(path):
    """Return the hexadecimal SHA-256 digest of one retained artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(atoms, output):
    """Write generation settings and checksums for later finalization."""
    mass_kg = np.sum(atoms.get_masses()) * 1.66053906660e-27
    volume_m3 = atoms.get_volume() * 1.0e-30
    retained = [
        "amorphous_si_32.extxyz",
        "replicated_atoms.xyz",
        "replicated_atoms_third.xyz",
        "second.npy",
        "third.npz",
    ]
    present = [name for name in retained if (output / name).is_file()]
    metadata = {
        "generator_version": 2,
        "ase_version": ase.__version__,
        "numpy_version": np.__version__,
        "seed": SEED,
        "n_atoms": N_ATOMS,
        "density_kg_m3": mass_kg / volume_m3,
        "cell_length_angstrom": float(np.asarray(atoms.cell)[0, 0]),
        "potential_energy_eV": float(atoms.info["potential_energy_eV"]),
        "max_force_eV_per_A": float(atoms.info["max_force_eV_per_A"]),
        "tersoff_reference": "J. Tersoff, Phys. Rev. B 37, 6991 (1988)",
        "protocol": {
            "time_step_fs": TIME_STEP_FS,
            "melt_temperature_K": MELT_TEMPERATURE_K,
            "final_temperature_K": FINAL_TEMPERATURE_K,
            "melt_steps": MELT_STEPS,
            "quench_steps": QUENCH_STEPS,
            "equilibration_steps": EQUILIBRATION_STEPS,
            "relax_fmax_eV_per_A": RELAX_FMAX_EV_ANGSTROM,
            "ifc2_displacement_angstrom": IFC2_DISPLACEMENT_ANGSTROM,
            "ifc3_displacement_angstrom": IFC3_DISPLACEMENT_ANGSTROM,
            "ifc3_distance_threshold_angstrom": IFC3_DISTANCE_THRESHOLD_ANGSTROM,
        },
        "sha256": {name: sha256(output / name) for name in present},
    }
    (output / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--force-structure", action="store_true")
    parser.add_argument("--n-workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.structure_only and args.force_structure:
        stale_ifcs = [name for name in IFC_ARTIFACTS if (args.output / name).exists()]
        if stale_ifcs:
            raise RuntimeError(
                "refusing to replace the structure while leaving existing IFCs "
                f"in place: {', '.join(stale_ifcs)}. Use a fresh --output for "
                "a structure-only audit, or omit --structure-only to recompute IFCs."
            )
    atoms = melt_quench(args.output, force=args.force_structure)
    if not args.structure_only:
        calculate_ifcs(atoms, args.output, args.n_workers)
    write_metadata(atoms, args.output)


if __name__ == "__main__":
    main()

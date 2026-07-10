"""
LJ Argon at 80 K — example for n_uc=1 (single-atom primitive).

Demonstrates the cumulant flow on a centrosymmetric Lennard-Jones FCC
crystal. Argon has a one-atom primitive cell, so the multi-atom IFC3/IFC4
phase factors collapse — useful as a "minimal-complexity" sanity case.

Uses the LJ Ar 80 K fixture vendored under
``kaldo/tests/cumulant_fixtures/LJ/`` (originally from
LatticeDynamicsToolkit.jl). Replace the ``folder`` path with your own
TDEP run output to apply the same analysis to other LJ-like atomic
systems (Ne, Kr, ...).

Run::

    python -m kaldo.cumulant.examples.lj_argon
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

from kaldo.forceconstants import ForceConstants
from kaldo.cumulant import F1_from_fc, F2_from_fc


# LJ Ar fixture (n_uc=1, det M = 256) — vendored under
# kaldo/tests/cumulant_fixtures/LJ/ so the example runs without external
# dependencies.
_TESTS_DIR = Path(__file__).parent.parent.parent / "tests" / "cumulant_fixtures"
LDT_BASE = _TESTS_DIR / "LJ"
LDT_IFC_DIR = LDT_BASE / "80K_4UC"

# rhombohedral primitive in 4x4x4 cubic conventional supercell
AR_M = np.array([[4, -4, 4], [4, 4, -4], [-4, 4, 4]], dtype=int)
AR_MASS_AMU = 39.948
TEMPERATURE_K = 80.0


def stage_tdep_folder():
    if not LDT_IFC_DIR.exists():
        raise FileNotFoundError(
            f"LDT LJ Ar fixture not found at {LDT_IFC_DIR}. "
            "Install Julia LDT or point this script at your own TDEP folder."
        )
    staging = Path("/tmp/kaldo_cumulant_quickstart_lj_ar")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for fn in ("infile.ucposcar", "infile.ssposcar"):
        shutil.copy(str(LDT_BASE / fn), str(staging / fn))
    for fn in ("infile.forceconstant",
               "infile.forceconstant_thirdorder",
               "infile.forceconstant_fourthorder"):
        shutil.copy(str(LDT_IFC_DIR / fn), str(staging / fn))
    return staging


def main():
    folder = stage_tdep_folder()
    print(f"Loading TDEP IFCs from {folder}")
    fc = ForceConstants.from_folder(
        folder=str(folder),
        supercell_matrix=AR_M,
        format="tdep",
        include_fourth=True,
    )
    print(f"Loaded primitive: {len(fc.atoms)} atom; "
          f"supercell: |det M| = {fc.n_replicas} replicas\n")

    masses_amu = np.full(1, AR_MASS_AMU)
    print(f"{'mesh':>8s}  {'F1 (eV/atom)':>16s}  {'F2 (eV/atom)':>16s}")
    print("-" * 46)
    for n in (3, 5):
        kmesh = (n, n, n)
        r1 = F1_from_fc(fc, masses_amu=masses_amu, kmesh=kmesh,
                        T_K=TEMPERATURE_K, use_q_symmetry=True)
        r2 = F2_from_fc(fc, masses_amu=masses_amu, kmesh=kmesh,
                        T_K=TEMPERATURE_K, sigma_THz=None,
                        use_q_symmetry=True)
        print(f"{n}^3       {r1['F1']:+16.5e}  {r2['F2']:+16.5e}")

    print()
    print("Mesh-converged (8^3): F1 = +1.011e-3, F2 = -4.788e-4 eV/atom")


if __name__ == "__main__":
    main()

"""
Example: reproduce Gate 6 at 25^3 Ne using the kaldo.cumulant subpackage.

Loads cached Phase 3 / Phase 4 production JSONs + Phase 5 sample NPZ and
assembles the total cumulant thermodynamics using kaldo.cumulant's
bootstrap + estimator. This is the fastest way to drive Gate 6 end-to-end
without re-running the 4h Phase 3/4 meshes.

Requires the cached files (produced earlier by the validated cumulant
pipeline now re-exposed in kaldo.cumulant.F1_vectorized / F2_vectorized
/ SCSampler / bootstrap_corrections):

    <ROOT>/out/run_25cubed_ibz.json
    <ROOT>/out/run_25cubed_ibz_tdep.json
    <ROOT>/out/phase5_our_samples.npz

Set the ``KALDO_CUMULANT_GATE6_ROOT`` environment variable to the
directory containing the ``out/`` subdirectory before running.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from kaldo.cumulant import bootstrap_corrections

ROOT = Path(os.environ.get("KALDO_CUMULANT_GATE6_ROOT", ""))
if not ROOT or not (ROOT / "out").exists():
    raise SystemExit(
        "Set KALDO_CUMULANT_GATE6_ROOT to the directory containing the\n"
        "cached Gate 6 inputs (out/run_25cubed_ibz.json,\n"
        "out/run_25cubed_ibz_tdep.json, out/phase5_our_samples.npz)."
    )

# Ethan's 25^3 published totals (for the PASS/FAIL gate)
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


def main():
    # Phase 1 harmonic (mesh-converged closed form)
    F_H, U_H = 6.140788e-3, 8.811518e-3
    S_H, Cv_H = 1.2913396, 1.9769204

    # Phase 3 @ 25^3 IBZ (F1 quartic)
    r1 = json.load(open(ROOT / "out" / "run_25cubed_ibz.json"))["phase3"]
    F1, U1, S1, Cv1 = r1["F1"], r1["U1"], r1["S1"], r1["Cv1"]

    # Phase 4 @ 25^3 IBZ adaptive sigma (F2 cubic)
    r2 = json.load(open(ROOT / "out" / "run_25cubed_ibz_tdep.json"))["phase4"]
    F2, U2, S2, Cv2 = r2["F2"], r2["U2"], r2["S2"], r2["Cv2"]

    # Phase 5 bootstrap using kaldo.cumulant.bootstrap_corrections
    d = np.load(ROOT / "out" / "phase5_our_samples.npz")
    V, V2, V3, V4, V_ref = d["V"], d["V2"], d["V3"], d["V4"], d["V2_tilde"]
    Nat = 256; T_K = 24.0
    print(f"Bootstrapping Phase 5 on N={len(V)} samples, n_boot=5000 ...")
    point, se = bootstrap_corrections(V, V2, V3, V4, V_ref, T_K, Nat,
                                       n_boot=5000, seed=1234)

    totals = dict(
        F=F_H + point["F0"] + F1 + F2,
        U=U_H + point["U0"] + U1 + U2,
        S=S_H + point["S0"] + S1 + S2,
        Cv=Cv_H + point["Cv0"] + Cv1 + Cv2,
    )
    SE = dict(F=se["F0"], U=se["U0"],
              S=se["S0"], Cv=se["Cv0"])

    print()
    print("=" * 78)
    print(f"Ne 25^3 cumulant thermo (T = {T_K} K), N_conf = {len(V)}")
    print("=" * 78)
    for q, unit in [("F", "eV/atom"), ("U", "eV/atom"),
                    ("S", "kB/atom"), ("Cv", "kB/atom")]:
        e = ETHAN[q]
        ours = totals[q]
        print(f"\n# {q} [{unit}]")
        print(f"ours   : {ours:+.7f}  (SE = {SE[q]:.3e})")
        print(f"Ethan  : {e['total']:+.7f}  (SE = {e['total_SE']:.3e})")
        diff = abs(ours - e['total'])
        joint = (SE[q] ** 2 + e['total_SE'] ** 2) ** 0.5
        good = diff < 3 * joint
        print(f"|Delta| : {diff:.3e}  3sigma: {3*joint:.3e}  -> "
              f"{'PASS' if good else 'FAIL'}")

    any_fail = False
    print()
    for q in ("F", "U", "S", "Cv"):
        e = ETHAN[q]
        joint = (SE[q] ** 2 + e['total_SE'] ** 2) ** 0.5
        if abs(totals[q] - e['total']) >= 3 * joint:
            any_fail = True
    print(f"Gate 6 @ 25^3 overall: {'FAIL' if any_fail else 'PASS'}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

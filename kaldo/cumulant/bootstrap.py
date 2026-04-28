"""
Bootstrap standard errors for the F_offset / S_offset / U_offset / Cv_offset
Monte-Carlo estimates.

Mirrors Julia ``CumulantAnalysis.jl/src/bootstrap.jl::bootstrap_corrections``.
For n_boot iterations, resample sample indices with replacement, recompute
``calculate_cumulants`` on the resampled arrays, and take the std of each
quantity over the boot distribution.

The analytic corrections (F1, F2, S1, ..., Cv2) have no bootstrap SE -
they come from closed-form mode-basis sums with no sampling.
"""
from __future__ import annotations

import time

import numpy as np

from .common import KB_eV_per_K
from .estimator import calculate_cumulants


def bootstrap_corrections(V, V2, V3, V4, V_ref, T_K, Nat, n_boot=5000, seed=None,
                           verbose=False):
    """
    Parameters
    ----------
    V, V2, V3, V4, V_ref : (N_conf,) arrays in eV per supercell.
    T_K : temperature in K.
    Nat : number of atoms in the supercell (for per-atom normalization).
    n_boot : bootstrap resample count.
    seed : RNG seed.
    verbose : print progress every 10%.

    Returns
    -------
    (point, se) : two dicts with keys F_offset, U_offset (eV/atom) and
        S_offset, Cv_offset (kB/atom).
    """
    rng = np.random.default_rng(seed)
    N = len(V)

    F_c, S_c, U_c, Cv_c = calculate_cumulants(V, V2, V3, V4, V_ref, T_K)
    point = dict(
        F_offset=F_c / Nat,
        U_offset=U_c / Nat,
        S_offset=S_c / (Nat * KB_eV_per_K),
        Cv_offset=Cv_c / (Nat * KB_eV_per_K),
    )

    F_b = np.empty(n_boot); S_b = np.empty(n_boot)
    U_b = np.empty(n_boot); Cv_b = np.empty(n_boot)
    t0 = time.time()
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        f, s, u, cv = calculate_cumulants(
            V[idx], V2[idx], V3[idx], V4[idx], V_ref[idx], T_K
        )
        F_b[i] = f; S_b[i] = s; U_b[i] = u; Cv_b[i] = cv
        if verbose and (i + 1) % max(1, n_boot // 10) == 0:
            print(f"  boot {i+1}/{n_boot}  ({time.time()-t0:.1f}s)", flush=True)

    se = dict(
        F_offset=np.std(F_b, ddof=0) / Nat,
        U_offset=np.std(U_b, ddof=0) / Nat,
        S_offset=np.std(S_b, ddof=0) / (Nat * KB_eV_per_K),
        Cv_offset=np.std(Cv_b, ddof=0) / (Nat * KB_eV_per_K),
    )
    return point, se

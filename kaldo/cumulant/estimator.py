"""
Monte-Carlo estimator for the constant cumulant correction F_offset.

Mirrors Julia LDT's `calculate_cumulants(V, V2, V3, V4, V_ref, T, AnalyticalEstimator)`
followed by `constant_corrections(c0, T)` from `cumulant_corrections.jl`.

Given N_conf harmonic-canonical samples of per-configuration energies
(V total, V_2, V_3, V_4 Taylor expansions, V_2_tilde harmonic reference),
compute:

    X        = V - V2 - V3 - V4
    F_const  = <X>                              (eV per supercell)
    dF/dT    = -cov(X, V_ref) / (kB T^2)
    d2F/dT2  = LDT central-moment formula
    F_offset = F_const / N_atoms_supercell      (eV / atom)
    S_offset = -dF/dT / (N_atoms * kB)          (kB / atom)
    U_offset = F_offset + T * S_offset
    Cv_offset = -T * d2F/dT2 / (N_atoms * kB)
"""
from __future__ import annotations

from .constants import KB_eV_per_K


def dA_dT(A, V, T):
    """Julia ``dA/dT(A, V, T) = cov(A, V) / (kB T^2)``.

    ``A`` and ``V`` must be in the same energy units (eV recommended).
    """
    Am = A - A.mean()
    Vm = V - V.mean()
    return (Am * Vm).mean() / (KB_eV_per_K * T * T)


def d2A_dT2(A, V, T, dA=None):
    """Julia ``d2A/dT2(A, V, T, dA)``. See ``cumulant_corrections.jl:8-13``."""
    if dA is None:
        dA = dA_dT(A, V, T)
    dAV = dA_dT(A * V, V, T)
    dVV = dA_dT(V, V, T)
    d_prod = A.mean() * dVV + V.mean() * dA
    return (-2 * dA / T) + (dAV - d_prod) / (KB_eV_per_K * T * T)


def calculate_cumulants(V, V2, V3, V4, V_ref, T):
    """
    Returns ``(F_const, S_const, U_const, Cv_const)`` per supercell.

    All inputs are arrays of length ``N_conf``, in eV per supercell.
    ``T`` is temperature in Kelvin. Caller converts supercell totals to
    per-atom units (divide by ``N_atoms_supercell``, divide S/Cv by ``kB``).
    """
    X = V - V2 - V3 - V4
    kappa = X.mean()
    dkappa = dA_dT(X, V_ref, T)
    ddkappa = d2A_dT2(X, V_ref, T, dA=dkappa)

    F_const = kappa
    S_const = -dkappa
    U_const = F_const + T * S_const
    Cv_const = -T * ddkappa
    return F_const, S_const, U_const, Cv_const

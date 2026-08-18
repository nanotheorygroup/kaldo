"""
Monte-Carlo estimator for the constant cumulant correction F_0.

Mirrors Julia CumulantAnalysis
``calculate_cumulants`` + ``constant_corrections`` from
``cumulant_corrections.jl``.

Given N_conf harmonic-canonical samples of per-configuration energies
(V total, V_2, V_3, V_4 Taylor expansions, V_ref harmonic reference),
compute:

    X         = V - V2 - V3 - V4
    F_const   = <X>                                 (eV per supercell)
    ∂F/∂T     = cov(X, V_ref) / (kB T^2)            (eV/K per supercell)
    ∂²F/∂T²   = LDT central-moment formula
                + cov(X, dV_ref_dT) / (kB T^2)      (quantum Bose-weight term)
    F_0       = F_const / N_atoms                   (eV / atom)
    S_0       = -(∂F/∂T) / (N_atoms * kB)           (kB / atom)
    U_0       = F_0 + T * S_0 * kB                  (via caller)
    Cv_0      = -T * (∂²F/∂T²) / (N_atoms * kB)     (kB / atom)

``V_ref`` is ``V2`` (classical) or ``V2_tilde`` (quantum). ``dV_ref_dT`` is
the explicit ∂V2_tilde/∂T through the Bose weight (zero classically).
"""
from __future__ import annotations

import numpy as np

from .constants import KB_eV_per_K


def dA_dT(A, V, T):
    """Julia ``∂A/∂T(A, V, T) = cov(A, V) / (kB T^2)`` (population cov)."""
    Am = A - A.mean()
    Vm = V - V.mean()
    return (Am * Vm).mean() / (KB_eV_per_K * T * T)


def d2A_dT2(A, V, T, dA=None):
    """Julia ``∂²A/∂T²`` without the explicit ``dV_ref_dT`` piece."""
    if dA is None:
        dA = dA_dT(A, V, T)
    dAV = dA_dT(A * V, V, T)
    dVV = dA_dT(V, V, T)
    d_prod = A.mean() * dVV + V.mean() * dA
    return (-2 * dA / T) + (dAV - d_prod) / (KB_eV_per_K * T * T)


def calculate_cumulants(V, V2, V3, V4, V_ref, T, dV_ref_dT=None):
    """
    Returns ``(F_const, S_const, U_const, Cv_const)`` per supercell.

    All inputs are arrays of length ``N_conf``, in eV per supercell
    (``dV_ref_dT`` in eV/K if given). ``T`` is temperature in Kelvin.
    Caller converts supercell totals to per-atom units (divide by
    ``N_atoms_supercell``, divide S/Cv by ``kB``).
    """
    X = V - V2 - V3 - V4
    kappa = X.mean()
    dkappa = dA_dT(X, V_ref, T)
    ddkappa = d2A_dT2(X, V_ref, T, dA=dkappa)
    if dV_ref_dT is not None:
        # Explicit T-dependence of Ṽ₂ through the Bose weight g(T).
        # Matches CumulantAnalysis ``cumulant_corrections.jl`` Val{0}.
        dV_ref_dT = np.asarray(dV_ref_dT)
        Xm = X - X.mean()
        dVm = dV_ref_dT - dV_ref_dT.mean()
        ddkappa = ddkappa + (Xm * dVm).mean() / (KB_eV_per_K * T * T)

    F_const = kappa
    S_const = -dkappa
    U_const = F_const + T * S_const
    Cv_const = -T * ddkappa
    return F_const, S_const, U_const, Cv_const

"""
Harmonic thermodynamics: F_H, U_H, S_H, Cv_H from the phonon dispersion.

Closed-form Bose-Einstein sums over a regular q-mesh. Matches Ethan
Meitz's LDT `harmonic_thermo` to machine precision.
"""
from __future__ import annotations

import numpy as np

from .common import HBAR, KB, EV, FREQ_TOL_THZ


def harmonic_thermo_quantum(freqs_THz_all, temperature_k, n_atoms_total):
    """
    Free energy, internal energy, entropy, heat capacity per atom from a
    full mesh of phonon frequencies in THz.

    Parameters
    ----------
    freqs_THz_all : array-like
        Frequencies at every (q, band) of the mesh, in THz. Sign preserved
        for imaginary modes; those below FREQ_TOL_THZ are excluded.
    temperature_k : float
        Temperature in Kelvin.
    n_atoms_total : int
        Total atom count in the sum (n_q * n_atoms_primitive).

    Returns
    -------
    (F, U, S, Cv)
        F, U in eV/atom. S, Cv in kB/atom.
    """
    omega = 2 * np.pi * np.asarray(freqs_THz_all).ravel() * 1e12  # rad/s
    mask = omega > (2 * np.pi * FREQ_TOL_THZ * 1e12)
    w = omega[mask]
    kBT = KB * temperature_k
    x = HBAR * w / kBT
    # numerically stable: use expm1 for large x
    e_x = np.exp(x)
    em = e_x - 1.0
    # Free energy per mode: (hbar*omega/2) + kBT * ln(1 - exp(-x))
    F_J = 0.5 * HBAR * w + kBT * np.log1p(-np.exp(-x))
    # Internal energy per mode: hbar*omega * (0.5 + 1/(e^x - 1))
    U_J = HBAR * w * (0.5 + 1.0 / em)
    # Entropy per mode: kB * (x/(e^x - 1) - ln(1 - e^-x))
    S_JK = KB * (x / em - np.log1p(-np.exp(-x)))
    # Heat capacity per mode: kB * x^2 * e^x / (e^x - 1)^2
    Cv_JK = KB * x ** 2 * e_x / (em ** 2)

    F = F_J.sum() / n_atoms_total / EV
    U = U_J.sum() / n_atoms_total / EV
    S = S_JK.sum() / n_atoms_total / KB
    Cv = Cv_JK.sum() / n_atoms_total / KB
    return F, U, S, Cv


def monkhorst_pack_qcart(kmesh, uc_cell):
    """Gamma-centered MP q-grid in Cartesian reciprocal space (rad/Å)."""
    nx, ny, nz = kmesh
    frac = np.array([[ix / nx, iy / ny, iz / nz]
                     for ix in range(nx) for iy in range(ny) for iz in range(nz)])
    recip = 2 * np.pi * np.linalg.inv(uc_cell).T
    return frac @ recip  # (n_q, 3)


def compute_all_frequencies_THz(neighbors_pair, uc_positions, masses_kg, uc_cell, kmesh):
    """Diagonalize D(q) at every point of the MP mesh; return (n_q, n_b) in THz."""
    from .common import dynmat_and_eigs
    cart = monkhorst_pack_qcart(kmesh, uc_cell)
    n_q = cart.shape[0]
    n_b = 3 * len(uc_positions)
    freqs_rad_s = np.empty((n_q, n_b))
    for iq, q in enumerate(cart):
        om, _ = dynmat_and_eigs(neighbors_pair, uc_positions, masses_kg, q)
        freqs_rad_s[iq] = om
    return freqs_rad_s / (2 * np.pi * 1e12)


def harmonic_thermo_from_ifc2(neighbors_pair, uc_positions, masses_kg, uc_cell,
                               kmesh, temperature_k):
    """Convenience: build frequencies on the mesh, then compute F_H/U_H/S_H/Cv_H."""
    freqs = compute_all_frequencies_THz(
        neighbors_pair, uc_positions, masses_kg, uc_cell, kmesh
    )
    n_atoms_total = freqs.size // (3 * len(uc_positions)) * len(uc_positions)
    # = n_q * n_uc; freqs has shape (n_q, 3 * n_uc)
    n_q = freqs.shape[0]
    n_uc = len(uc_positions)
    return harmonic_thermo_quantum(freqs, temperature_k, n_q * n_uc)

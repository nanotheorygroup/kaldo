"""
Harmonic thermodynamics: F_H, U_H, S_H, Cv_H from the phonon dispersion.

Closed-form Bose-Einstein sums over a regular q-mesh. Matches Ethan
Meitz's LDT `harmonic_thermo` to machine precision.
"""
from __future__ import annotations

import numpy as np

from .constants import HBAR, KB, EV, ANG, FREQ_TOL_THZ


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


def dynmat_and_eigs(neighbors_pair, uc_positions, masses_kg, q_cart):
    """
    Build and diagonalize the mass-weighted dynamical matrix at a single q.

    Uses the **sum convention** (= TDEP / LDT convention):
        D_{a,b}(q) = sum_R Phi_{a,b}(R) exp(i q . R) / sqrt(m_a m_b)
    where R is the lattice vector between primitive cells. This makes the
    eigenvectors compatible with the IFC3 / IFC4 triplet and quartet
    pretransforms in this package - which phase only by lattice vectors.

    For single-atom-per-cell systems (Ne) the convention doesn't matter
    (tau_i = 0). For multi-atom primitives (Si, diamond) the atomic
    convention `exp(iq.(r_j - r_i))` with r_j = tau_j + R produces
    eigenvectors shifted by `exp(iq.(tau_j - tau_i))` relative to the sum
    convention - which breaks the IFC3/IFC4 quartet contraction for
    multi-atom cells.

    Returns ``(omegas, egvs)``:
      * ``omegas`` (n_bands,): frequencies in rad/s, with sign preserved
        for imaginary modes (negative omega**2 -> negative omega).
      * ``egvs`` (n_bands, n_bands): complex eigenvectors of the
        dynamical matrix, column-indexed by band.
    """
    n = len(neighbors_pair)
    nb = 3 * n
    D = np.zeros((nb, nb), dtype=complex)
    for i, il in enumerate(neighbors_pair):
        for (j, rj, _lp, phi) in il:
            # R is the pure lattice vector between cell of atom i and cell of
            # atom j. rj = tau_j + R and uc_positions[j] = tau_j, so
            # R = rj - tau_j = rj - uc_positions[j].
            R = rj - uc_positions[j]
            ph = np.exp(1j * np.dot(q_cart, R))
            D[3*i:3*i+3, 3*j:3*j+3] += phi * ph / np.sqrt(masses_kg[i] * masses_kg[j])
    D = 0.5 * (D + D.conj().T)
    w2, egv = np.linalg.eigh(D * (EV / ANG ** 2))
    return np.sign(w2) * np.sqrt(np.abs(w2)), egv

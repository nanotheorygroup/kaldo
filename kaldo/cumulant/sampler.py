"""
Supercell-Gamma canonical sampler (quantum or classical).

Matches Julia LDT's `canonical_configs!` approach: diagonalize D(Gamma) on
the FULL supercell (3 N_atoms eigenmodes, minus 3 acoustic translations),
draw one real Gaussian per non-zero mode, and map back to physical
displacements:

    u_{a, alpha} = sum_lambda sqrt(hbar(2n+1)/(2 m_a omega_lambda))
                   * phi_{a, alpha, lambda} * z_lambda

Bit-for-bit agreement with Julia on Ne rhombohedral SC (T=24 K quantum and
classical) confirmed during the cumulant port.
"""
from __future__ import annotations
import numpy as np

from .common import HBAR, KB, EV, AMU, ANG, FREQ_TOL_THZ


class SCSampler:
    """
    Sample from the quantum or classical canonical harmonic ensemble on the
    supercell (Gamma-point diagonalization of D_sc).

    The supercell IFC2 is consumed from a generic remap dictionary.

    Parameters
    ----------
    remapped
        ``kaldo.remap.SupercellIFCRemap``.
    masses_amu_sc : (n_sc,) array
        Atomic masses in amu for each supercell atom.
    T_K : float
        Temperature in Kelvin.
    quantum : bool, default True
        Quantum Bose-Einstein sampling; False for classical equipartition.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        remapped,
        masses_amu_sc: np.ndarray,
        T_K: float,
        is_classic: bool = True,
        seed: int | None = None,
    ):
        sc_ifc2 = remapped.ifc2
        n_sc = int(remapped.n_atoms_sc)
        a1 = np.asarray(sc_ifc2.a1, dtype=int)
        a2 = np.asarray(sc_ifc2.a2, dtype=int)
        phi = np.asarray(sc_ifc2.phi)
        if phi.shape[-2:] != (3, 3):
            phi = np.moveaxis(phi, -1, 0)
        self.n_sc = n_sc
        self.masses_amu = np.asarray(masses_amu_sc)
        self.masses_kg = self.masses_amu * AMU
        self.T = float(T_K)
        self.is_classic = bool(is_classic)

        nb = 3 * n_sc
        D = np.zeros((nb, nb))
        inv_sqrt_m = 1.0 / np.sqrt(self.masses_kg)
        for k in range(a1.size):
            i = a1[k]; j = a2[k]
            D[3*i:3*i+3, 3*j:3*j+3] += phi[k] * (inv_sqrt_m[i] * inv_sqrt_m[j])
        D = 0.5 * (D + D.T)
        D_SI = D * (EV / ANG ** 2)
        omega2, phi_vec = np.linalg.eigh(D_SI)
        self.omega = np.sign(omega2) * np.sqrt(np.abs(omega2))
        self.phi_vec = phi_vec
        self.ok = self.omega > 2 * np.pi * FREQ_TOL_THZ * 1e12

        # Bose-Einstein populations and per-mode variance
        x = HBAR * self.omega / (KB * self.T)
        self.n_pop = np.zeros_like(self.omega)
        self.n_pop[self.ok] = 1.0 / np.expm1(x[self.ok])

        if self.is_classic:
            self.var_per_mode = np.where(
                self.ok, KB * self.T / (self.omega ** 2), 0.0
            )
        else:
            self.var_per_mode = np.where(
                self.ok, HBAR * (2 * self.n_pop + 1) / (2 * self.omega), 0.0
            )
        self.amp = np.sqrt(self.var_per_mode)
        self.rng = np.random.default_rng(seed)

    def draw_with_z(self):
        """Return ``(u_Ang, z)`` with ``u`` shape (n_sc, 3) in Angstroms."""
        nb = 3 * self.n_sc
        z = self.rng.standard_normal(nb)
        z[~self.ok] = 0.0
        inv_sqrt_m = 1.0 / np.sqrt(np.repeat(self.masses_kg, 3))
        u_flat_m = inv_sqrt_m * (self.phi_vec @ (self.amp * z))
        u_A = u_flat_m.reshape(self.n_sc, 3) / ANG
        return u_A, z

    def draw(self):
        """Return one harmonic sample as (n_sc, 3) displacement in Angstroms."""
        u, _ = self.draw_with_z()
        return u

    def V2_tilde_from_z(self, z):
        """
        Reference V2 computed from the harmonic canonical weights at the
        drawn z (in eV).

        Quantum : per mode ``w = 4 n (n+1) / (2 n + 1)^2``.
        Classical: ``w = 1`` everywhere.
        """
        ok = self.ok

        if self.is_classic:
            w = np.where(ok, 1.0, 0.0)
        else:
            denom = (2 * self.n_pop + 1) ** 2
            w = np.zeros_like(self.omega)
            w[ok] = 4.0 * self.n_pop[ok] * (self.n_pop[ok] + 1.0) / denom[ok]

        per_mode_J = np.where(
            ok, HBAR * self.omega * (2 * self.n_pop + 1) / 4.0, 0.0
        )
        return (w * per_mode_J * (z ** 2)).sum() / EV

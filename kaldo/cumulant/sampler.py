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

AMU_TO_KG = 1.660539040E-27
EMU_TO_KG = 9.10938356E-31
AMU_TO_EMU = AMU_TO_KG/EMU_TO_KG

HARTREE_TO_EV = 27.21138602
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
BOHR_TO_A = 0.52917721067
A_TO_BOHR = 1.0 / BOHR_TO_A
IFC_2nd_EVA_TO_HARTREEBOHR = EV_TO_HARTREE / (A_TO_BOHR**2)

KB_EV = 8.6173303E-5
KB_HARTREE = KB_EV*EV_TO_HARTREE

class SCSampler:
    """
    Sample from the quantum or classical canonical harmonic ensemble on the
    supercell (Gamma-point diagonalization of D_sc).

    Parameters
    ----------
    ifc2_sc_eVAng : (3 n_sc, 3 n_sc) array
        Supercell IFC2 in eV/Angstrom^2, in the same atom order as
        ``masses_amu_sc`` (and as the displacements this sampler returns).
    masses_amu_sc : (n_sc,) array
        Atomic masses in amu for each supercell atom.
    T_K : float
        Temperature in Kelvin.
    is_classic : bool, default True
        Classical equipartition sampling; False for quantum Bose-Einstein.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        ifc2_sc_eVAng: np.ndarray, # assumes in eV / Ang^2
        masses_amu_sc: np.ndarray, # in amu
        T_K: float,
        is_classic: bool = True,
        seed: int | None = None,
    ):

        self.n_sc = len(masses_amu_sc)
        self.masses_emu = masses_amu_sc * AMU_TO_EMU
        self.T = float(T_K)
        self.is_classic = bool(is_classic)

        self.inv_sqrt_m_emu_expanded = 1.0 / np.sqrt(np.repeat(self.masses_emu, 3))
        
        D = 3
        dynmat = np.zeros((D * self.n_sc, D * self.n_sc))
        for i in range(self.n_sc):
            for j in range(i, self.n_sc):
                block = ifc2_sc_eVAng[D*i:D*i+D, D*j:D*j+D]

                mass_factor = np.sqrt(self.masses_emu[i] * self.masses_emu[j])

                dynmat[D*i:D*i+D, D*j:D*j+D] = (
                    IFC_2nd_EVA_TO_HARTREEBOHR * block / mass_factor
                )

                dynmat[D*j:D*j+D, D*i:D*i+D] = (
                    dynmat[D*i:D*i+D, D*j:D*j+D].T
                )

        omega2, evecs = np.linalg.eigh(dynmat)
        self.omega = np.sign(omega2) * np.sqrt(np.abs(omega2))
        self.evecs = evecs

        # Set rigid translation modes to exactly 0.0
        rtm_idx = np.argsort(self.omega)[:D]
        self.omega[rtm_idx] = 0.0

        if np.any(self.omega < 0.0):
            raise ValueError("Imaginary phonon modes detected.")
        
        self.ok = self.omega > 0.0 # mask for rigid translation modes

        # Bose-Einstein populations and per-mode variance
        x = self.omega / (KB_HARTREE * self.T)
        self.n_pop = np.zeros_like(self.omega)
        self.n_pop[self.ok] = 1.0 / np.expm1(x[self.ok])

        self.var_per_mode = np.zeros_like(self.omega)

        if self.is_classic:
            self.var_per_mode[self.ok] = (
                KB_HARTREE * self.T / np.square(self.omega[self.ok])
            )
        else:
            self.var_per_mode[self.ok] = (
                (2.0 * self.n_pop[self.ok] + 1.0)
                / (2.0 * self.omega[self.ok])
            )
            
        self.amp = np.sqrt(self.var_per_mode)
        self.rng = np.random.default_rng(seed)

        # Precompute mode weights w(T) and explicit ∂w/∂T factor for V2_tilde.
        # Classical: w ≡ 1, ∂w/∂T ≡ 0.
        # Quantum: w = 4n(n+1)/(2n+1)² = sech²(ħω/2kT),
        #          (1/w) ∂w/∂T = (x/T) tanh(x/2) with x = ħω/kT
        #          (LDT ``_mode_weight`` / ``_mode_weight_dT``).
        self._mode_weight = np.zeros_like(self.omega)
        self._mode_weight_dT_over_w = np.zeros_like(self.omega)
        if self.is_classic:
            self._mode_weight[self.ok] = 1.0
        else:
            n = self.n_pop
            denom = (2.0 * n + 1.0) ** 2
            self._mode_weight[self.ok] = (
                4.0 * n[self.ok] * (n[self.ok] + 1.0) / denom[self.ok]
            )
            x = np.zeros_like(self.omega)
            x[self.ok] = self.omega[self.ok] / (KB_HARTREE * self.T)
            # Cap extreme x (same spirit as LDT: avoid overflow, weight → 0)
            large = x > 100.0
            self._mode_weight_dT_over_w[self.ok] = (x[self.ok] / self.T) * np.tanh(
                0.5 * x[self.ok]
            )
            self._mode_weight_dT_over_w[large] = 0.0
            self._mode_weight[large] = 0.0

    def draw_with_z(self):
        """Return ``(u_Ang, z)`` with ``u`` shape (n_sc, 3) in Angstroms."""
        nb = 3 * self.n_sc
        z = self.rng.standard_normal(nb)
        z[~self.ok] = 0.0
        u_flat_m = self.inv_sqrt_m_emu_expanded * (self.evecs @ (self.amp * z))
        u_A = u_flat_m.reshape(self.n_sc, 3) * BOHR_TO_A
        return u_A, z

    def draw(self):
        """Return one harmonic sample as (n_sc, 3) displacement in Angstroms."""
        u, _ = self.draw_with_z()
        return u

    def V2_tilde_from_z(self, z):
        """
        Harmonic reference energy ``V_2_tilde`` for the drawn ``z`` (in eV).

        Per LatticeDynamicsToolkit's ``_v2_tilde_coefficients`` the per-mode
        coefficient is::

            c_lambda = mode_weight_lambda * 0.5 * omega_lambda^2 * sigma_lambda^2

        and ``V_2_tilde = sum_lambda c_lambda * z_lambda^2``, where
        ``sigma_lambda^2`` is the mode variance (``self.var_per_mode``; here in
        mass-weighted normal-mode coordinates, so the atomic mass is already
        folded in) and the mode weight is

            classical : ``w = 1``
            quantum   : ``w = 4 n (n + 1) / (2 n + 1)^2``.

        With ``w = 1`` (classical) this is exactly the harmonic potential
        energy ``0.5 u^T Phi u`` of the drawn configuration; the quantum weight
        reweights each mode for the control-variate reference used by the
        entropy / heat-capacity estimator.
        """
        v2t, _ = self.V2_tilde_and_dT_from_z(z)
        return v2t

    def V2_tilde_and_dT_from_z(self, z):
        """Return ``(V2_tilde, dV2_tilde_dT)`` in eV and eV/K.

        ``dV2_tilde_dT`` is the *explicit* Bose-weight derivative only
        (LDT ``dcoeffs = _mode_weight_dT * c``), needed for the quantum Cv
        estimator. Classical: identically zero.
        """
        ok = self.ok
        z = np.asarray(z)
        base = np.where(ok, 0.5 * self.omega ** 2 * self.var_per_mode, 0.0)
        coeffs = self._mode_weight * base
        # dcoeffs = [(1/w) ∂w/∂T] * c = (∂w/∂T) * base
        dcoeffs = self._mode_weight_dT_over_w * coeffs
        z2 = z ** 2
        v2t = float((coeffs * z2).sum() * HARTREE_TO_EV)
        dv2t = float((dcoeffs * z2).sum() * HARTREE_TO_EV)
        return v2t, dv2t

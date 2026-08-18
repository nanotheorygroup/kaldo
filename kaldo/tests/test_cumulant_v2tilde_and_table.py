"""
Two small cumulant correctness fixes.

1. ``SCSampler.V2_tilde_from_z`` is the harmonic reference energy used as the
   control variate for the entropy / heat-capacity estimator. Per
   LatticeDynamicsToolkit's ``_v2_tilde_coefficients`` the per-mode coefficient
   is ``w * 0.5 * omega^2 * sigma^2`` (sigma^2 = mode variance). With the
   classical weight ``w = 1`` this must equal the exact harmonic potential
   energy ``0.5 u^T Phi u`` of the drawn configuration. The previous
   implementation used ``omega * (2n+1)/4`` for the coefficient, which is only
   correct in the quantum branch (there ``sigma^2 = (2n+1)/(2 omega)``) and is
   wrong classically (where ``sigma^2 = kT/omega^2``), giving ~30 % error.

2. ``print_thermo_table`` referenced ``result.F_offset`` / ``F_offset_SE`` etc.,
   fields that ``CumulantResult`` does not have (it exposes ``F_0`` and
   ``F_total_SE``), so any call raised ``AttributeError``.
"""
from __future__ import annotations

import numpy as np

from kaldo.cumulant.sampler import SCSampler, HARTREE_TO_EV, KB_HARTREE


def _random_spd_ifc2(n_at=6, seed=0):
    """A random symmetric positive-definite supercell IFC2 (eV/A^2)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3 * n_at, 3 * n_at))
    phi = A @ A.T + 3.0 * np.eye(3 * n_at)  # SPD -> real positive frequencies
    return 0.5 * (phi + phi.T)


def test_v2_tilde_classical_equals_exact_harmonic_energy():
    """Classical V2_tilde must equal 0.5 uᵀ Φ u for the drawn configuration."""
    phi = _random_spd_ifc2(n_at=6, seed=1)
    masses = np.full(6, 28.0855)
    s = SCSampler(phi, masses, T_K=300.0, is_classic=True, seed=42)

    max_err = 0.0
    scale = 0.0
    for _ in range(50):
        u, z = s.draw_with_z()
        uf = u.reshape(-1)
        harmonic = 0.5 * uf @ phi @ uf
        v2t = s.V2_tilde_from_z(z)
        max_err = max(max_err, abs(v2t - harmonic))
        scale = max(scale, abs(harmonic))
    assert scale > 0.0
    assert max_err < 1e-9 * scale, (
        f"classical V2_tilde deviates from 0.5 uΦu by {max_err:.3e} eV "
        f"(scale {scale:.3f} eV)"
    )


def test_v2_tilde_quantum_is_reweighted_harmonic_energy():
    """Quantum V2_tilde = sum_lambda w_lambda * (0.5 omega^2 sigma^2) z^2, with
    0 < w < 1, so its mean sits below the classical harmonic energy but stays
    positive and finite."""
    phi = _random_spd_ifc2(n_at=6, seed=2)
    masses = np.full(6, 28.0855)
    sc_q = SCSampler(phi, masses, T_K=300.0, is_classic=False, seed=7)
    sc_c = SCSampler(phi, masses, T_K=300.0, is_classic=True, seed=7)

    N = 2000
    vq = np.empty(N)
    vc = np.empty(N)
    for i in range(N):
        _u, z = sc_q.draw_with_z()
        vq[i] = sc_q.V2_tilde_from_z(z)
    for i in range(N):
        u, z = sc_c.draw_with_z()
        vc[i] = 0.5 * u.reshape(-1) @ phi @ u.reshape(-1)

    assert np.all(np.isfinite(vq))
    assert vq.mean() > 0.0
    # Quantum weight 4n(n+1)/(2n+1)^2 <= 1, so the reweighted reference does
    # not exceed the classical harmonic energy (allow a little sampling slack).
    assert vq.mean() <= vc.mean() * 1.05


def test_print_thermo_table_runs(capsys):
    """print_thermo_table must not raise (regression: it referenced F_offset)."""
    from kaldo.cumulant.thermodynamics import CumulantResult, print_thermo_table

    z3 = np.zeros(3)
    r = CumulantResult(
        T_K=100, Nat=216, N_conf=1000, N_boot=10,
        F_H=0.07, U_H=0.07, S_H=0.37, Cv_H=0.74,
        F_0=-1e-5, U_0=1e-5, S_0=-4e-4, Cv_0=-2e-3,
        F_1=1e-4, U_1=1e-4, S_1=-4e-4, Cv_1=-2e-3,
        F_2=-2e-5, U_2=-2e-5, S_2=7e-4, Cv_2=2e-3,
        F_total=0.0698, U_total=0.0699, S_total=0.37, Cv_total=0.74,
        F_total_SE=2e-6, U_total_SE=2e-6, S_total_SE=3e-5, Cv_total_SE=1e-4,
        V=z3, V2=z3, V3=z3, V4=z3, V2_tilde=z3,
    )
    print_thermo_table(r)
    out = capsys.readouterr().out
    # A block per observable, with the 0th-order column relabeled F_0 (not F_offset).
    assert "F_0" in out and "F_total" in out
    assert "F_offset" not in out
    assert "+0.0698000" in out  # F_total value rendered

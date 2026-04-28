"""
Analytic cumulant free-energy corrections F1 (quartic) and F2 (cubic).

Implements Julia LDT's `free_energy_fourthorder` (F1, <V_4>) and
`free_energy_thirdorder` (F2, <V_3 V_3>) as vectorized Python on a
regular Monkhorst-Pack q-mesh. Validated bit-for-bit against Julia LDT
on Ne (n_uc=1) and Si diamond (n_uc=2) at multiple meshes.

Physics formulas follow Julia's conventions exactly:
  * IFC2 dynamical matrix in sum convention: D(q) = sum_R Phi(R) e^{iq.R}
    (see `common.dynmat_and_eigs`).
  * IFC3/IFC4 quartet phase factors use pure lattice vectors.
  * Psi_3 and Psi_4 are contracted with *conjugated* eigenvectors
    (`conj(e1) conj(e2) conj(e3)` for F2, consistent with LDT's
    `evp2 .= conj.(evp2)` before the inner sum).
"""
from __future__ import annotations

import time

import numpy as np

from .common import (
    HBAR, KB, EV, ANG, FREQ_TOL_THZ,
    dynmat_and_eigs,
)


# ---------------------------------------------------------------------------
# Quartic: F1 = <V_4>_0 / (4! * 2)  (LDT prefactor /32)
# ---------------------------------------------------------------------------

def flatten_quartets(quartets_per_atom, masses_kg, uc_cell):
    """
    One-time flatten: concat all quartets into per-quartet arrays so
    ``build_psi4_realspace_v`` can be a vectorized scatter.

    ``masses_kg`` and the ``EV/ANG**4`` factor are absorbed into the
    stored IFC; ``uc_cell`` converts fractional lattice vectors to
    Cartesian.
    """
    Q_a1, Q_a2, Q_a3, Q_a4 = [], [], [], []
    Q_lv2c, Q_lv3c, Q_lv4c = [], [], []
    Q_ifc = []
    inv_sqrt_m = 1.0 / np.sqrt(masses_kg)
    for a1, quarts in enumerate(quartets_per_atom):
        for (a2, a3, a4, lv2f, lv3f, lv4f, ifcs) in quarts:
            Q_a1.append(a1); Q_a2.append(a2); Q_a3.append(a3); Q_a4.append(a4)
            Q_lv2c.append(lv2f @ uc_cell)
            Q_lv3c.append(lv3f @ uc_cell)
            Q_lv4c.append(lv4f @ uc_cell)
            m = inv_sqrt_m[a1] * inv_sqrt_m[a2] * inv_sqrt_m[a3] * inv_sqrt_m[a4]
            Q_ifc.append(ifcs * (m * (EV / (ANG ** 4))))
    return dict(
        a1=np.array(Q_a1, dtype=np.int32),
        a2=np.array(Q_a2, dtype=np.int32),
        a3=np.array(Q_a3, dtype=np.int32),
        a4=np.array(Q_a4, dtype=np.int32),
        lv2c=np.array(Q_lv2c),
        lv3c=np.array(Q_lv3c),
        lv4c=np.array(Q_lv4c),
        ifc=np.array(Q_ifc),
        nb=3 * len(quartets_per_atom),
    )


def build_psi4_realspace_v(QD, q1_cart, q2_cart):
    """
    Vectorized pretransform: A[a1_alpha, a2_beta, a3_gamma, a4_delta]
    for given (q1, q2). Phase factor per quartet:

        exp(i(-q1.lv2 + q2.lv3 - q2.lv4))

    matching Julia LDT `pretransform_phi4!`.
    """
    nb = QD["nb"]
    iqr = -(QD["lv2c"] @ q1_cart) + (QD["lv3c"] @ q2_cart) - (QD["lv4c"] @ q2_cart)
    phase = np.exp(1j * iqr)
    scaled = QD["ifc"] * phase[:, None, None, None, None]
    a1 = QD["a1"]; a2 = QD["a2"]; a3 = QD["a3"]; a4 = QD["a4"]
    A = np.zeros((nb, nb, nb, nb), dtype=complex)
    I = np.arange(3)
    a1_idx = (a1[:, None] * 3 + I[None, :])[:, :, None, None, None]
    a2_idx = (a2[:, None] * 3 + I[None, :])[:, None, :, None, None]
    a3_idx = (a3[:, None] * 3 + I[None, :])[:, None, None, :, None]
    a4_idx = (a4[:, None] * 3 + I[None, :])[:, None, None, None, :]
    a1_idx, a2_idx, a3_idx, a4_idx = np.broadcast_arrays(a1_idx, a2_idx, a3_idx, a4_idx)
    np.add.at(A, (a1_idx, a2_idx, a3_idx, a4_idx), scaled)
    return A


def F1_vectorized(neighbors_pair, quartets, masses_kg, uc_positions, uc_cell,
                  kmesh, T_K, use_q_symmetry=False, atoms=None):
    """
    F1 / S1 / Cv1 / U1 quartic cumulant evaluator on a regular MP q-mesh.

    When ``use_q_symmetry=True`` (and ``atoms`` is supplied), the outer q1
    loop is restricted to the IBZ reps returned by
    ``kaldo.phonons._get_ir_kgrid_data`` and each contribution is weighted
    by its orbit size. The inner q2 loop still ranges over the full BZ.

    The F1 integrand Psi_4(q1, q2) * (2n+1)(2n+1) / (omega_1 omega_2) is
    invariant under q1 -> Sq1 for any crystal point-group operation S
    (after the full-BZ sum over q2): the quartet lattice sum is
    translationally invariant, so Psi_4(Sq1, q2) summed over q2 equals
    Psi_4(q1, q2) summed over q2 by re-indexing.

    Parameters
    ----------
    neighbors_pair, quartets : TDEP IFC2 / IFC4 lists from common.read_tdep_*.
    masses_kg : (n_uc,) atom masses in kg.
    uc_positions, uc_cell : primitive cell geometry in Angstrom.
    kmesh : 3-tuple of mesh dimensions.
    T_K : temperature in Kelvin.
    use_q_symmetry : opt-in IBZ reduction of the outer q1 loop.
    atoms : ASE Atoms (needed only when use_q_symmetry=True).

    Returns
    -------
    dict with keys ``F1``, ``S1``, ``Cv1``, ``U1`` (units eV/atom for F, U
    and kB/atom for S, Cv).
    """
    nx, ny, nz = kmesh
    n_uc = len(uc_positions); nb = 3 * n_uc
    recip = 2 * np.pi * np.linalg.inv(uc_cell).T
    nq = nx * ny * nz
    frac = np.array([[ix/nx, iy/ny, iz/nz]
                     for ix in range(nx) for iy in range(ny) for iz in range(nz)])
    cart = frac @ recip

    # q-symmetry: restrict outer q1 loop to IBZ reps, weighted by orbit size.
    if use_q_symmetry:
        if atoms is None:
            raise ValueError("use_q_symmetry=True requires atoms= ASE Atoms of primitive cell")
        from kaldo.phonons import _get_ir_kgrid_data
        ir_mapping, _, ibz_indices, _ = _get_ir_kgrid_data(
            atoms, kpts=list(kmesh), grid_type='C')
        orbit_sizes = np.bincount(ir_mapping, minlength=nq)
        q1_indices = list(ibz_indices)
        q1_weights = {int(iq): int(orbit_sizes[iq]) for iq in ibz_indices}
        print(f"use_q_symmetry: reducing q1 from {nq} to {len(q1_indices)} "
              f"IBZ reps (avg orbit size {nq / len(q1_indices):.1f})")
    else:
        q1_indices = list(range(nq))
        q1_weights = {iq: 1 for iq in q1_indices}

    t0 = time.time()
    omegas = np.empty((nq, nb))
    egvs = np.empty((nq, nb, nb), dtype=complex)
    for iq, q in enumerate(cart):
        omegas[iq], egvs[iq] = dynmat_and_eigs(
            neighbors_pair, uc_positions, masses_kg, q
        )
    print(f"eigs {nq} qs in {time.time()-t0:.1f}s")

    # Planck distribution and derivatives.
    x = HBAR * omegas / (KB * T_K)
    ok = omegas > 2 * np.pi * FREQ_TOL_THZ * 1e12
    n_tab = np.zeros_like(omegas)
    dn_tab = np.zeros_like(omegas)
    ddn_tab = np.zeros_like(omegas)
    if np.any(ok):
        ex = np.exp(x[ok])
        em = ex - 1.0
        n_tab[ok] = 1.0 / em
        dn_tab[ok] = (x[ok] / T_K) * ex / (em ** 2)
        y = x[ok] / T_K
        coef = -2.0 - x[ok] + 2.0 * x[ok] * ex / em
        ddn_tab[ok] = (y * ex / (T_K * em ** 2)) * coef
    two_np1 = 2 * n_tab + 1
    inv_w = np.zeros_like(omegas)
    inv_w[ok] = 1.0 / omegas[ok]

    # Per-q eigenvector outer products: M[q, b, i, j] = e(q,b,i) * conj(e(q,b,j))
    t1 = time.time()
    M = np.einsum("qib,qjb->qbij", egvs, np.conj(egvs))
    print(f"building per-q outer products  done in {time.time()-t1:.1f}s, shape {M.shape}")

    QD = flatten_quartets(quartets, masses_kg, uc_cell)
    print(f"flattened {QD['a1'].shape[0]} quartets")

    n_q1 = len(q1_indices)
    print(f"F1/S1/Cv1 double-q loop over {n_q1}x{nq}={n_q1*nq} (q1,q2) pairs"
          + (" [q1 in IBZ]" if use_q_symmetry else ""))
    t2 = time.time()
    F1_acc = 0.0
    S1_acc = 0.0
    Cv1_acc = 0.0
    for _i, iq1 in enumerate(q1_indices):
        q1c = cart[iq1]
        M1 = M[iq1]
        inv_w1 = inv_w[iq1]
        two1 = two_np1[iq1]
        dn1 = dn_tab[iq1]
        ddn1 = ddn_tab[iq1]
        ok1 = ok[iq1]
        w1 = q1_weights[iq1]
        for iq2 in range(nq):
            q2c = cart[iq2]
            A = build_psi4_realspace_v(QD, q1c, q2c)
            T = np.einsum("kab,abcd->kcd", M1, A)
            Psi4 = np.einsum("kcd,lcd->kl", T, M[iq2])
            psi_re = np.real(Psi4)
            mask = ok1[:, None] & ok[iq2][None, :]
            inv_w_prod = inv_w1[:, None] * inv_w[iq2][None, :]

            two2 = two_np1[iq2][None, :]
            dn2 = dn_tab[iq2][None, :]
            ddn2 = ddn_tab[iq2][None, :]
            f_w = two1[:, None] * two2 * inv_w_prod
            s_w = -(2.0 * dn1[:, None] * two2 + two1[:, None] * 2.0 * dn2) * inv_w_prod
            cv_w = -(2.0 * ddn1[:, None] * two2 + 2.0 * ddn2 * two1[:, None]
                     + 8.0 * dn1[:, None] * dn2) * T_K * inv_w_prod

            F1_acc += w1 * (psi_re * f_w * mask).sum()
            S1_acc += w1 * (psi_re * s_w * mask).sum()
            Cv1_acc += w1 * (psi_re * cv_w * mask).sum()
        if (_i + 1) % max(1, n_q1 // 20) == 0:
            print(f"  q1={_i+1}/{n_q1}  elapsed={time.time()-t2:.1f}s", flush=True)
    print(f"F1 loop total {time.time()-t2:.1f}s")

    prefac = HBAR * HBAR / (32.0 * nq * nq * n_uc)
    F1 = prefac * F1_acc / EV
    S1 = prefac * S1_acc / KB
    Cv1 = prefac * Cv1_acc / KB
    U1 = F1 + T_K * (S1 * KB) / EV
    return dict(F1=F1, S1=S1, Cv1=Cv1, U1=U1)


# ---------------------------------------------------------------------------
# Cubic: F2 = <V_3 V_3>_0 / (3! * 2 * 4)  (LDT prefactor /48)
# ---------------------------------------------------------------------------

def flatten_triplets(triplets_per_atom, masses_kg, uc_cell):
    """
    One-time flatten of triplets for ``build_psi3_realspace``. Masses and
    eV/A^3 units are absorbed into the stored IFC.
    """
    T_a1, T_a2, T_a3 = [], [], []
    T_lv2c, T_lv3c = [], []
    T_ifc = []
    inv_sqrt_m = 1.0 / np.sqrt(masses_kg)
    for a1, trips in enumerate(triplets_per_atom):
        for (a2, a3, lv2f, lv3f, ifcs) in trips:
            T_a1.append(a1); T_a2.append(a2); T_a3.append(a3)
            T_lv2c.append(lv2f @ uc_cell)
            T_lv3c.append(lv3f @ uc_cell)
            m = inv_sqrt_m[a1] * inv_sqrt_m[a2] * inv_sqrt_m[a3]
            T_ifc.append(ifcs * (m * (EV / (ANG ** 3))))
    return dict(
        a1=np.array(T_a1, dtype=np.int32),
        a2=np.array(T_a2, dtype=np.int32),
        a3=np.array(T_a3, dtype=np.int32),
        lv2c=np.array(T_lv2c),
        lv3c=np.array(T_lv3c),
        ifc=np.array(T_ifc),
        nb=3 * len(triplets_per_atom),
    )


def build_psi3_realspace(TD, q2_cart, q3_cart):
    """
    A[ia, ib, ic] = sum_triplet ifc[alpha,beta,gamma]
                    * exp(i(-q2.lv2 - q3.lv3))
                    / sqrt(m_a1 m_a2 m_a3)
                    * EV/ANG^3
    scattered onto (nb, nb, nb). Phase matches Julia LDT
    `pretransform_phi3!`.
    """
    nb = TD["nb"]
    iqr = -(TD["lv2c"] @ q2_cart) - (TD["lv3c"] @ q3_cart)
    phase = np.exp(1j * iqr)
    scaled = TD["ifc"] * phase[:, None, None, None]
    a1 = TD["a1"]; a2 = TD["a2"]; a3 = TD["a3"]
    A = np.zeros((nb, nb, nb), dtype=complex)
    I = np.arange(3)
    a1_idx = (a1[:, None] * 3 + I[None, :])[:, :, None, None]
    a2_idx = (a2[:, None] * 3 + I[None, :])[:, None, :, None]
    a3_idx = (a3[:, None] * 3 + I[None, :])[:, None, None, :]
    a1_idx, a2_idx, a3_idx = np.broadcast_arrays(a1_idx, a2_idx, a3_idx)
    np.add.at(A, (a1_idx, a2_idx, a3_idx), scaled)
    return A


def planck_and_derivs(omega, T_K):
    """Planck factor n(omega, T) and first / second temperature derivatives."""
    x = HBAR * omega / (KB * T_K)
    ok = omega > 2 * np.pi * FREQ_TOL_THZ * 1e12
    n = np.zeros_like(omega)
    dn = np.zeros_like(omega)
    ddn = np.zeros_like(omega)
    if np.any(ok):
        ex = np.exp(x[ok])
        em = ex - 1.0
        n[ok] = 1.0 / em
        dn[ok] = (x[ok] / T_K) * ex / (em ** 2)
        y = x[ok] / T_K
        coef = -2.0 - x[ok] + 2.0 * x[ok] * ex / em
        ddn[ok] = (y * ex / (T_K * em ** 2)) * coef
    return n, dn, ddn, ok


# --- Adaptive sigma helpers -------------------------------------------------

def compute_group_velocity(neighbors_pair, uc_positions, masses_kg, q_cart,
                           omega, egv, dq=1e-4):
    """
    Numerical group velocity v[alpha, b] = d omega_b / d q_alpha via central
    finite differences. Returns (3, nb) in (rad/s) per (1/A).

    Note: at high-symmetry q (e.g. Brillouin zone face centers), dq cancels
    to zero by symmetry and the result is noise; this is physically correct
    since dω/dq = 0 there. Adaptive σ then falls to its baseline floor.

    Prefer :func:`compute_group_velocity_analytic` for new code: same result
    with no extra dynmat diagonalizations.
    """
    nb = egv.shape[0]
    v = np.zeros((3, nb))
    for alpha in range(3):
        dq_vec = np.zeros(3); dq_vec[alpha] = dq
        om_p, _ = dynmat_and_eigs(neighbors_pair, uc_positions, masses_kg, q_cart + dq_vec)
        om_m, _ = dynmat_and_eigs(neighbors_pair, uc_positions, masses_kg, q_cart - dq_vec)
        v[alpha] = (om_p - om_m) / (2.0 * dq)
    return v


def compute_group_velocity_analytic(neighbors_pair, uc_positions, masses_kg,
                                    q_cart, omega, egv, deg_tol=1e9):
    """
    Hellmann-Feynman group velocity v[alpha, b] = d omega_b / d q_alpha.

    Matches Julia LDT's ``group_velocities`` (src/harmonic/dispersion.jl):

        d D(q) / d q_alpha = sum_R Phi(R) * i * R_alpha * exp(i q . R)
                             / sqrt(m_i m_j)    * (EV / ANG^2)
        d omega^2 / d q_alpha [b]  =  Re(<e_b | dD/dq_alpha | e_b>)
        d omega / d q_alpha [b]    =  (d omega^2 / d q_alpha) / (2 * omega_b)

    Degenerate subspaces (Δω < ``deg_tol`` rad/s; default 1 GHz) get
    Julia's projected eigendecomposition: within each degenerate block
    S, the velocity is computed from the eigenvalues of the projected
    Hermitian ``U_S† · dD/dq_α · U_S``, and all bands in S are assigned
    the block-average. This makes the output gauge-invariant under
    unitary mixing within S.

    ω below 1e11 rad/s (~16 GHz) is treated as acoustic and v is set to 0.

    Returns (3, nb) in (rad/s) per (1/A), same units as the FD routine.
    """
    from .common import EV as _EV, ANG as _ANG
    n = len(neighbors_pair)
    nb = 3 * n
    dDdq = np.zeros((3, nb, nb), dtype=complex)
    for i, il in enumerate(neighbors_pair):
        for (j, rj, _lp, phi) in il:
            R = rj - uc_positions[j]
            ph = np.exp(1j * np.dot(q_cart, R))
            inv_mij = 1.0 / np.sqrt(masses_kg[i] * masses_kg[j])
            for alpha in range(3):
                dDdq[alpha, 3*i:3*i+3, 3*j:3*j+3] += (
                    phi * (1j * R[alpha]) * ph * inv_mij
                )
    for alpha in range(3):
        dDdq[alpha] = 0.5 * (dDdq[alpha] + dDdq[alpha].conj().T)
    dDdq *= (_EV / _ANG ** 2)

    subspaces = []
    s_start = 0
    for b in range(1, nb):
        if abs(omega[b] - omega[b - 1]) > deg_tol:
            subspaces.append(list(range(s_start, b)))
            s_start = b
    subspaces.append(list(range(s_start, nb)))

    v_dwsq = np.zeros((3, nb))
    for alpha in range(3):
        dDa = dDdq[alpha]
        for S in subspaces:
            if len(S) == 1:
                b = S[0]
                tmp = dDa @ egv[:, b]
                v_dwsq[alpha, b] = np.real(np.vdot(egv[:, b], tmp))
            else:
                US = egv[:, S]
                H_sub = US.conj().T @ dDa @ US
                H_sub = 0.5 * (H_sub + H_sub.conj().T)
                w_sub = np.linalg.eigvalsh(H_sub)
                avg = float(np.mean(w_sub))
                for b in S:
                    v_dwsq[alpha, b] = avg

    v = np.zeros((3, nb))
    for b in range(nb):
        if abs(omega[b]) > 1e11:
            v[:, b] = v_dwsq[:, b] / (2.0 * omega[b])
    return v


def compute_default_smearing(omegas, FREQ_TOL=0.0):
    """
    Per-band default smearing (unit-agnostic): max nearest-neighbor gap in the
    sorted per-band frequency list, floored at max_default/5.

    Shim on top of :func:`kaldo.controllers.anharmonic.calculate_default_smearing_per_band`
    so the formula lives in one canonical place. ``FREQ_TOL`` is accepted for
    backwards compatibility with earlier cumulant callers and ignored.
    """
    from kaldo.controllers.anharmonic import calculate_default_smearing_per_band
    return calculate_default_smearing_per_band(omegas)


def adaptive_sigma(radius_inv_ang, group_vel_alpha_nb, default_sigma_nb, scale=1.0):
    """
    TDEP-style adaptive σ (per-q, per-band).

    Shim on :func:`kaldo.controllers.anharmonic.calculate_adaptive_sigma_tdep`.
    Cumulant's historical signature is per-q with ``velocity`` as ``(3, n_b)``
    (α, band); kaldo expects ``(n_k, n_b, 3)``. We reshape to a single-q call
    and drop the leading axis. Formula and clamping are identical on both
    sides, so the output matches bit-for-bit in the overlapping unit regime.
    """
    from kaldo.controllers.anharmonic import calculate_adaptive_sigma_tdep
    v = np.asarray(group_vel_alpha_nb).T[np.newaxis, ...]  # (1, n_b, 3)
    sig = calculate_adaptive_sigma_tdep(
        radius=radius_inv_ang, velocity=v,
        default_sigma=default_sigma_nb, scale=scale,
    )
    return sig[0]  # strip leading q-axis


def F2_vectorized(neighbors_pair, triplets, masses_kg, uc_positions, uc_cell,
                  kmesh, T_K, sigma_THz=None, use_q_symmetry=False, atoms=None):
    """
    F2 / S2 / Cv2 / U2 cubic cumulant evaluator on a regular MP q-mesh.

    If ``sigma_THz`` is None (default), uses adaptive per-mode σ matching
    Julia LDT's convention. If a float, uses a fixed isotropic σ.

    ``use_q_symmetry=True`` reduces the outer q1 loop to spglib IBZ reps
    weighted by orbit size. See ``F1_vectorized`` for the invariance
    argument (same structure; q3 = -q1-q2 is symmetry-consistent).

    Returns a dict with keys ``F2``, ``S2``, ``Cv2``, ``U2``.
    """
    nx, ny, nz = kmesh
    nq = nx * ny * nz
    n_uc = len(uc_positions); nb = 3 * n_uc
    recip = 2 * np.pi * np.linalg.inv(uc_cell).T
    frac = np.array([[ix/nx, iy/ny, iz/nz]
                     for ix in range(nx) for iy in range(ny) for iz in range(nz)])
    cart = frac @ recip

    if use_q_symmetry:
        if atoms is None:
            raise ValueError("use_q_symmetry=True requires atoms= ASE Atoms of primitive cell")
        from kaldo.phonons import _get_ir_kgrid_data
        ir_mapping, _, ibz_indices, _ = _get_ir_kgrid_data(
            atoms, kpts=list(kmesh), grid_type='C')
        orbit_sizes = np.bincount(ir_mapping, minlength=nq)
        q1_indices = list(ibz_indices)
        q1_weights = {int(iq): int(orbit_sizes[iq]) for iq in ibz_indices}
        print(f"use_q_symmetry: reducing q1 from {nq} to {len(q1_indices)} "
              f"IBZ reps (avg orbit size {nq / len(q1_indices):.1f})")
    else:
        q1_indices = list(range(nq))
        q1_weights = {iq: 1 for iq in q1_indices}

    t0 = time.time()
    omegas = np.empty((nq, nb))
    egvs = np.empty((nq, nb, nb), dtype=complex)
    for iq, q in enumerate(cart):
        omegas[iq], egvs[iq] = dynmat_and_eigs(
            neighbors_pair, uc_positions, masses_kg, q
        )
    print(f"eigs {nq} qs in {time.time()-t0:.1f}s")

    if sigma_THz is None:
        t_sig = time.time()
        prim_vol_A3 = abs(np.linalg.det(uc_cell))
        radius = (3.0 / (prim_vol_A3 * nq * 4.0 * np.pi)) ** (1.0 / 3.0)
        default_sigma_bands = compute_default_smearing(omegas)
        sigma_table = np.empty((nq, nb))
        for iq, q in enumerate(cart):
            v = compute_group_velocity_analytic(
                neighbors_pair, uc_positions, masses_kg, q, omegas[iq], egvs[iq],
            )
            sigma_table[iq] = adaptive_sigma(radius, v, default_sigma_bands)
        print(f"adaptive sigma: {time.time()-t_sig:.1f}s, "
              f"range {sigma_table.min():.2e}..{sigma_table.max():.2e} rad/s "
              f"(~{sigma_table.min()/(2*np.pi*1e12):.3f}..{sigma_table.max()/(2*np.pi*1e12):.3f} THz)")
    else:
        sigma_rad_s = 2 * np.pi * sigma_THz * 1e12
        sigma_table = np.full((nq, nb), sigma_rad_s)

    # Planck table
    n_tab = np.empty((nq, nb)); dn_tab = np.empty_like(n_tab); ddn_tab = np.empty_like(n_tab)
    ok_tab = np.empty((nq, nb), dtype=bool)
    for iq in range(nq):
        n_tab[iq], dn_tab[iq], ddn_tab[iq], ok_tab[iq] = planck_and_derivs(omegas[iq], T_K)

    # q3 lookup table from q1+q2+q3 = 0 mod G
    frac_rounded = np.round(frac * np.array([nx, ny, nz])[None, :]).astype(int) \
                   % np.array([nx, ny, nz])[None, :]
    lookup = np.full((nx, ny, nz), -1, dtype=int)
    for iq, (i, j, k) in enumerate(frac_rounded):
        lookup[i, j, k] = iq

    TD = flatten_triplets(triplets, masses_kg, uc_cell)
    print(f"flattened {TD['a1'].shape[0]} triplets, nb={nb}")

    n_q1 = len(q1_indices)
    print(f"F2/S2/Cv2 double-q loop over {n_q1}x{nq}={n_q1*nq} (q1,q2) pairs"
          + (" [q1 in IBZ]" if use_q_symmetry else ""))
    t2 = time.time()
    F2 = 0.0
    S2 = 0.0
    Cv2 = 0.0

    inv_w = np.zeros_like(omegas)
    inv_w[ok_tab] = 1.0 / omegas[ok_tab]

    for _i, iq1 in enumerate(q1_indices):
        q1c = cart[iq1]
        i1, j1, k1 = frac_rounded[iq1]
        e1 = egvs[iq1]
        w1 = omegas[iq1]; n1 = n_tab[iq1]; dn1 = dn_tab[iq1]; ddn1 = ddn_tab[iq1]
        ok1 = ok_tab[iq1]
        w_q1 = q1_weights[iq1]
        for iq2 in range(nq):
            i2, j2, k2 = frac_rounded[iq2]
            i3 = (-i1 - i2) % nx
            j3 = (-j1 - j2) % ny
            k3 = (-k1 - k2) % nz
            iq3 = lookup[i3, j3, k3]
            q2c = cart[iq2]; q3c = cart[iq3]

            A = build_psi3_realspace(TD, q2c, q3c)

            e2 = egvs[iq2]; e3 = egvs[iq3]
            w2 = omegas[iq2]; w3 = omegas[iq3]
            n2 = n_tab[iq2]; n3 = n_tab[iq3]
            ok2 = ok_tab[iq2]; ok3 = ok_tab[iq3]

            # Psi_3 via conjugated eigenvectors (LDT convention).
            e1c = np.conj(e1); e2c = np.conj(e2); e3c = np.conj(e3)
            T1 = np.einsum("abc,ak->kbc", A, e1c)
            T2 = np.einsum("kbc,bl->klc", T1, e2c)
            Psi3 = np.einsum("klc,cm->klm", T2, e3c)
            psisq = np.abs(Psi3) ** 2

            w1_ = w1[:, None, None]; w2_ = w2[None, :, None]; w3_ = w3[None, None, :]
            mask = ok1[:, None, None] & ok2[None, :, None] & ok3[None, None, :]
            s1 = sigma_table[iq1][:, None, None]
            s2 = sigma_table[iq2][None, :, None]
            s3 = sigma_table[iq3][None, None, :]
            sigma_combo = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
            denom1 = w1_ + w2_ + w3_
            Re1 = denom1 / (denom1 ** 2 + sigma_combo ** 2)
            denom2 = w1_ + w2_ - w3_
            Re2 = denom2 / (denom2 ** 2 + sigma_combo ** 2)

            n1_ = n1[:, None, None]; n2_ = n2[None, :, None]; n3_ = n3[None, None, :]
            dn1_ = dn_tab[iq1][:, None, None]; dn2_ = dn_tab[iq2][None, :, None]; dn3_ = dn_tab[iq3][None, None, :]
            ddn1_ = ddn_tab[iq1][:, None, None]; ddn2_ = ddn_tab[iq2][None, :, None]; ddn3_ = ddn_tab[iq3][None, None, :]

            f1 = (n1_ + 1.0) * (n2_ + n3_ + 1.0) + n2_ * n3_
            f2 = n3_ * (n1_ + n2_ + 1.0) - n1_ * n2_

            df1 = (dn1_ * (n2_ + n3_ + 1.0) + (n1_ + 1.0) * (dn2_ + dn3_)
                   + dn2_ * n3_ + n2_ * dn3_)
            df2 = (dn3_ * (n1_ + n2_ + 1.0) + dn1_ * (n3_ - n2_) + dn2_ * (n3_ - n1_))

            ddf1 = (ddn1_ * (n2_ + n3_ + 1.0) + 2.0 * dn1_ * (dn2_ + dn3_)
                    + (n1_ + 1.0) * (ddn2_ + ddn3_) + ddn2_ * n3_ + n2_ * ddn3_
                    + 2.0 * dn2_ * dn3_)
            ddf2 = (ddn3_ * (n1_ + n2_ + 1.0) + dn3_ * (dn1_ + dn2_)
                    + ddn1_ * (n3_ - n2_) + dn1_ * (dn3_ - dn2_)
                    + ddn2_ * (n3_ - n1_) + dn2_ * (dn3_ - dn1_))

            inv_w_prod = np.zeros_like(psisq)
            inv_w_prod[mask] = 1.0 / (w1_ * w2_ * w3_)[mask]

            common_F = (f1 * Re1 + 3.0 * f2 * Re2)
            common_S = (df1 * Re1 + 3.0 * df2 * Re2)
            common_Cv = (ddf1 * Re1 + 3.0 * ddf2 * Re2) * T_K

            integrand_F = psisq * inv_w_prod * common_F / 48.0
            integrand_S = psisq * inv_w_prod * common_S / 48.0
            integrand_Cv = psisq * inv_w_prod * common_Cv / 48.0
            F2 += w_q1 * integrand_F[mask].sum()
            S2 += w_q1 * integrand_S[mask].sum()
            Cv2 += w_q1 * integrand_Cv[mask].sum()
        if (_i + 1) % max(1, n_q1 // 10) == 0:
            print(f"  q1={_i+1}/{n_q1}  elapsed={time.time()-t2:.1f}s", flush=True)
    print(f"F2 loop total {time.time()-t2:.1f}s")

    prefac = HBAR * HBAR / (nq * nq * n_uc)
    F2_eV = -prefac * F2 / EV
    S2_kB = +prefac * S2 / KB
    Cv2_kB = +prefac * Cv2 / KB
    U2_eV = F2_eV + T_K * (S2_kB * KB) / EV
    return dict(F2=F2_eV, S2=S2_kB, Cv2=Cv2_kB, U2=U2_eV)


# ---------------------------------------------------------------------------
# D.1 / D.2: ForceConstants-based entry points
# ---------------------------------------------------------------------------

def _minimum_image_lv(lv_frac, supercell):
    """Wrap fractional lattice vectors into the Wigner-Seitz cell of the
    supercell: each fractional coordinate is shifted by an integer multiple
    of N (the supercell dim along that axis) so the result lies in
    ``(-N/2, N/2]``.

    This is needed because kaldo's ``replicated_positions`` covers only the
    positive octant ``[0, N·a]``, but the cumulant F2 kernel applies
    ``exp(iq·R)`` with signed R. On q-meshes that don't match the supercell
    size, the positive-octant R would give the wrong phase.
    """
    N = np.asarray(supercell, dtype=float)
    lv = np.asarray(lv_frac)
    shifted = lv - np.rint(lv / N) * N
    return shifted


def _replica_lv_frac_table(fc):
    """Return the (n_rep, 3) fractional lattice-vector table per replica.

    Non-diagonal fc (``_replica_table`` attached): use it directly — already
    minimum-image-wrapped into the supercell Wigner-Seitz cell.
    Diagonal fc: compute from ``replicated_positions`` with
    :func:`_minimum_image_lv` so the wrap is correct on F2 meshes smaller
    than the supercell.
    """
    second = fc.second
    if getattr(second, "_replica_table", None) is not None:
        return np.asarray(second._replica_table, dtype=float)
    # Diagonal path
    n_uc = fc.n_atoms
    n_rep = fc.n_replicas
    rep_pos = np.asarray(second.replicated_positions)
    if rep_pos.ndim == 2:
        rep_pos = rep_pos.reshape(n_rep, n_uc, 3)
    uc_pos = np.asarray(fc.atoms.positions)
    uc_cell = np.asarray(fc.atoms.cell)
    inv_cell = np.linalg.inv(uc_cell)
    # Use atom 0 to extract the lattice vectors (they are atom-independent)
    lv_frac_raw = (rep_pos[:, 0, :] - uc_pos[0]) @ inv_cell
    return np.asarray([
        _minimum_image_lv(lv, fc.supercell) for lv in lv_frac_raw
    ])


def _neighbors_from_fc(fc):
    """Reconstruct the legacy ``neighbors_pair`` list-of-tuples from ``fc.second``.

    Legacy shape (per central atom i):
        [(j, rj_cart, lv_frac, phi_3x3), ...]

    kaldo's IFC2 tensor has shape ``(1, n_uc, 3, n_rep, n_uc, 3)``. On a
    non-diagonal fc (SNF) the supercell has a non-diagonal M matrix and
    ``_replica_table`` is attached to ``fc.second``; we use it directly.
    On diagonal fc we compute the minimum-image fractional lattice
    vectors from ``replicated_positions`` via :func:`_minimum_image_lv`.

    ASR is NOT re-imposed here: ``SecondOrder.load`` already applies it if
    requested. The TDEP file is already ASR-exact to float precision.
    """
    second = np.asarray(fc.second.value)[0]  # (n_uc, 3, n_rep, n_uc, 3)
    n_uc = fc.n_atoms
    n_rep = fc.n_replicas
    uc_pos = np.asarray(fc.atoms.positions)
    uc_cell = np.asarray(fc.atoms.cell)
    lv_frac_tab = _replica_lv_frac_table(fc)  # (n_rep, 3)

    neighbors = []
    for i in range(n_uc):
        il = []
        for r in range(n_rep):
            for j in range(n_uc):
                phi = second[i, :, r, j, :]  # (3, 3)
                if not np.any(phi):
                    continue
                lv_frac = lv_frac_tab[r]
                rj = uc_pos[j] + lv_frac @ uc_cell
                il.append((j, rj, lv_frac, phi))
        neighbors.append(il)
    return neighbors


def _triplets_from_fc(fc):
    """Reconstruct the legacy ``triplets`` list from ``fc.third``.

    Legacy shape (per central atom a1):
        [(a2, a3, lv2_frac, lv3_frac, phi_3x3x3), ...]

    ``fc.third.value`` is sparse-COO of shape
    ``(n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)``. ``fc.third.list_of_replicas``
    gives the Cartesian lattice vector of each replica index (n_rep × 3).
    For each unique ``(a1, a2, a3, r2, r3)`` combination we assemble the
    3×3×3 phi and convert r2/r3 to fractional lattice vectors.
    """
    third = np.asarray(fc.third.value.todense())
    # Shape: (n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)
    if getattr(fc.third, "_replica_table", None) is not None:
        lv_frac_tab = np.asarray(fc.third._replica_table, dtype=float)
    else:
        list_rep = fc.third.list_of_replicas  # (n_rep, 3) in Cartesian
        uc_cell = np.asarray(fc.atoms.cell)
        inv_cell = np.linalg.inv(uc_cell)
        lv_frac_raw = list_rep @ inv_cell  # (n_rep, 3) fractional
        lv_frac_tab = np.asarray([
            _minimum_image_lv(lv, fc.supercell) for lv in lv_frac_raw
        ])

    n_uc = fc.n_atoms
    n_rep = fc.n_replicas

    # Find all unique (a1, a2, a3, r2, r3) with any non-zero phi
    per_atom = []
    for a1 in range(n_uc):
        # slice out (3, n_rep, n_uc, 3, n_rep, n_uc, 3)
        sl = third[a1]
        trips = []
        for r2 in range(n_rep):
            for a2 in range(n_uc):
                for r3 in range(n_rep):
                    for a3 in range(n_uc):
                        phi = sl[:, r2, a2, :, r3, a3, :]  # (3, 3, 3)
                        if not np.any(phi):
                            continue
                        lv2 = lv_frac_tab[r2]
                        lv3 = lv_frac_tab[r3]
                        trips.append((a2, a3, lv2, lv3, phi.copy()))
        per_atom.append(trips)
    return per_atom


def _quartets_from_fc(fc):
    """Reconstruct the legacy ``quartets`` list from ``fc.fourth``.

    Legacy shape (per central atom a1):
        [(a2, a3, a4, lv2_frac, lv3_frac, lv4_frac, phi_3x3x3x3), ...]

    ``fc.fourth.value`` is sparse-COO of shape
    ``(n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)``.
    """
    if fc.fourth is None:
        raise ValueError(
            "F1_from_fc requires fc.fourth; load with include_fourth=True"
        )
    fourth = np.asarray(fc.fourth.value.todense())
    if getattr(fc.fourth, "_replica_table", None) is not None:
        lv_frac_tab = np.asarray(fc.fourth._replica_table, dtype=float)
    else:
        list_rep = fc.fourth.list_of_replicas
        uc_cell = np.asarray(fc.atoms.cell)
        inv_cell = np.linalg.inv(uc_cell)
        lv_frac_raw = list_rep @ inv_cell
        lv_frac_tab = np.asarray([
            _minimum_image_lv(lv, fc.supercell) for lv in lv_frac_raw
        ])

    n_uc = fc.n_atoms
    n_rep = int(np.prod(fc.supercell))

    per_atom = []
    for a1 in range(n_uc):
        sl = fourth[a1]  # (3, n_rep, n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)
        quartets_a1 = []
        for r2 in range(n_rep):
            for a2 in range(n_uc):
                for r3 in range(n_rep):
                    for a3 in range(n_uc):
                        for r4 in range(n_rep):
                            for a4 in range(n_uc):
                                phi = sl[:, r2, a2, :, r3, a3, :, r4, a4, :]  # (3,3,3,3)
                                if not np.any(phi):
                                    continue
                                quartets_a1.append(
                                    (a2, a3, a4,
                                     lv_frac_tab[r2], lv_frac_tab[r3], lv_frac_tab[r4],
                                     phi.copy())
                                )
        per_atom.append(quartets_a1)
    return per_atom


def F2_from_fc(fc, masses_amu, kmesh, T_K, sigma_THz=None,
               use_q_symmetry=False):
    """F2 cubic cumulant on a ``ForceConstants`` object (kaldo-native entry).

    Uses ``fc.atoms``, ``fc.second``, ``fc.third`` as the input data source
    and delegates to :func:`F2_vectorized` after reconstructing the legacy
    neighbour-list / triplet-list format. Output is bit-for-bit identical
    to :func:`F2_vectorized` on the same physical inputs.

    Parameters
    ----------
    fc : kaldo.forceconstants.ForceConstants
        Must have ``second`` and ``third`` loaded (e.g. via
        ``ForceConstants.from_folder(..., format='tdep')``).
    masses_amu : (n_uc,) array
        Atomic masses in amu.
    kmesh, T_K, sigma_THz, use_q_symmetry : see :func:`F2_vectorized`.

    Returns
    -------
    dict with keys ``F2``, ``S2``, ``Cv2``, ``U2``.
    """
    from .common import AMU
    uc_pos = np.asarray(fc.atoms.positions)
    uc_cell = np.asarray(fc.atoms.cell)
    masses_kg = np.asarray(masses_amu) * AMU
    neighbors = _neighbors_from_fc(fc)
    triplets = _triplets_from_fc(fc)
    return F2_vectorized(
        neighbors, triplets, masses_kg, uc_pos, uc_cell,
        tuple(kmesh), T_K, sigma_THz=sigma_THz,
        use_q_symmetry=use_q_symmetry,
        atoms=fc.atoms if use_q_symmetry else None,
    )


def F1_from_fc(fc, masses_amu, kmesh, T_K, use_q_symmetry=False):
    """F1 quartic cumulant on a ``ForceConstants`` object (kaldo-native entry).

    Requires ``fc.fourth`` loaded (``include_fourth=True`` in ``from_folder``).
    Delegates to :func:`F1_vectorized` after reconstructing the legacy
    neighbour-list / quartet-list format. Output is bit-for-bit identical
    to :func:`F1_vectorized` on the same physical inputs.
    """
    from .common import AMU
    uc_pos = np.asarray(fc.atoms.positions)
    uc_cell = np.asarray(fc.atoms.cell)
    masses_kg = np.asarray(masses_amu) * AMU
    neighbors = _neighbors_from_fc(fc)
    quartets = _quartets_from_fc(fc)
    return F1_vectorized(
        neighbors, quartets, masses_kg, uc_pos, uc_cell,
        tuple(kmesh), T_K,
        use_q_symmetry=use_q_symmetry,
        atoms=fc.atoms if use_q_symmetry else None,
    )

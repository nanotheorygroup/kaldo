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

import os
import time
from concurrent.futures import as_completed
from multiprocessing import shared_memory

import numpy as np

from kaldo.helpers.logger import get_logger
from .constants import (
    HBAR, KB, EV, ANG, FREQ_TOL_THZ,
)

logging = get_logger()


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
            Q_a1.append(a1)
            Q_a2.append(a2)
            Q_a3.append(a3)
            Q_a4.append(a4)
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


def _quartet_mode_blocks(Mq, atom_left, atom_right):
    """Gather per-quartet 3x3 blocks from mode outer products."""
    xyz = np.arange(3)
    left = 3 * atom_left[:, None] + xyz[None, :]
    right = 3 * atom_right[:, None] + xyz[None, :]
    blocks = Mq[:, left[:, :, None], right[:, None, :]]
    return np.moveaxis(blocks, 1, 0)


def _quartet_mode_blocks_batch(Mq, atom_left, atom_right):
    """Gather per-quartet 3x3 blocks for a batch of q-points."""
    xyz = np.arange(3)
    left = 3 * atom_left[:, None] + xyz[None, :]
    right = 3 * atom_right[:, None] + xyz[None, :]
    blocks = Mq[:, :, left[:, :, None], right[:, None, :]]
    return np.moveaxis(blocks, 2, 1)


def _prepare_psi4_q1(QD, M1, q1_cart):
    """Contract the q1 mode blocks with IFC4 once for an outer-q1 task."""
    M1_blocks = _quartet_mode_blocks(M1, QD["a1"], QD["a2"])
    left = np.einsum("nkab,nabcd->nkcd", M1_blocks, QD["ifc"], optimize=True)
    phase_q1 = np.exp(-1j * (QD["lv2c"] @ q1_cart))
    return left * phase_q1[:, None, None, None]


def _build_psi4_modes_quartet(QD, left_q1, M2, q2_cart):
    """Build mode-space Psi4 directly in quartet space, without scattering."""
    M2_blocks = _quartet_mode_blocks(M2, QD["a3"], QD["a4"])
    phase_q2 = np.exp(1j * ((QD["lv3c"] - QD["lv4c"]) @ q2_cart))
    per_quartet = np.einsum("nkcd,nlcd->nkl", left_q1, M2_blocks, optimize=True)
    return np.einsum("n,nkl->kl", phase_q2, per_quartet, optimize=True)


def _build_psi4_modes_quartet_batch(QD, left_q1, M2, q2_cart):
    """Build mode-space Psi4 for a batch of q2 points."""
    M2_blocks = _quartet_mode_blocks_batch(M2, QD["a3"], QD["a4"])
    phase_q2 = np.exp(1j * (q2_cart @ (QD["lv3c"] - QD["lv4c"]).T))
    per_quartet = np.einsum("nkcd,bnlcd->bnkl", left_q1, M2_blocks, optimize=True)
    return np.einsum("bn,bnkl->bkl", phase_q2, per_quartet, optimize=True)


# ---------------------------------------------------------------------------
# Process-pool q1 reduction: SharedMemory for flattened IFCs (QD/TD) and
# other large read-only arrays. Nested quartet/triplet lists never leave
# the parent. Each worker maps the same blocks once via initializer.
# ---------------------------------------------------------------------------

_QD_ARRAY_KEYS = ('a1', 'a2', 'a3', 'a4', 'lv2c', 'lv3c', 'lv4c', 'ifc')
_TD_ARRAY_KEYS = ('a1', 'a2', 'a3', 'lv2c', 'lv3c', 'ifc')
_F1_Q2_BATCH_SIZE = 4
_F2_Q2_BATCH_SIZE = 8

# Filled by _init_q1_worker in process-pool children; unused on the serial path.
_WORKER_Q1 = None


def _check_n_workers(n_workers):
    if n_workers is not None and n_workers < 1:
        raise ValueError(f"n_workers must be >= 1 or None, got {n_workers}")


def _prefix_table(prefix, table, keys):
    return {f'{prefix}.{k}': table[k] for k in keys}


def _unprefix_table(prefix, arrays, keys, nb):
    out = {k: arrays[f'{prefix}.{k}'] for k in keys}
    out['nb'] = nb
    return out


def _create_shm_array(arr):
    arr = np.ascontiguousarray(arr)
    nbytes = int(arr.nbytes)
    shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
    if nbytes:
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[:] = arr
    return {
        'name': shm.name,
        'shape': tuple(int(s) for s in arr.shape),
        'dtype': np.dtype(arr.dtype).str,
        'nbytes': nbytes,
    }, shm


def _view_shm_array(meta, hold):
    shm = shared_memory.SharedMemory(name=meta['name'])
    hold.append(shm)
    if meta['nbytes'] == 0:
        return np.empty(meta['shape'], dtype=np.dtype(meta['dtype']))
    arr = np.ndarray(meta['shape'], dtype=np.dtype(meta['dtype']), buffer=shm.buf)
    try:
        arr.setflags(write=False)
    except ValueError:
        pass
    return arr


class _SharedArrayStore:
    """Parent-owned SharedMemory blocks for a dict of ndarrays."""

    def __init__(self, arrays):
        self._shms = []
        self.spec = {}
        try:
            for key, arr in arrays.items():
                meta, shm = _create_shm_array(arr)
                self._shms.append(shm)
                self.spec[key] = meta
        except Exception:
            self.close_and_unlink()
            raise

    def close_and_unlink(self):
        for shm in self._shms:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
        self._shms.clear()


def _init_q1_worker(thread_cap, spec, scalars):
    from kaldo.parallel.executor import _init_worker_thread_caps
    _init_worker_thread_caps(int(thread_cap))
    global _WORKER_Q1
    hold = []
    arrays = {k: _view_shm_array(meta, hold) for k, meta in spec.items()}
    _WORKER_Q1 = {'arrays': arrays, 'scalars': scalars, '_hold': hold}


def _q1_chunk_worker(iq1_list):
    key = _WORKER_Q1['scalars']['_accumulate']
    return _ACCUMULATORS[key](iq1_list, _WORKER_Q1['arrays'], _WORKER_Q1['scalars'])


def _run_parallel_q1(accumulate_key, q1_indices, n_workers, arrays, scalars):
    """Map individual q1 points onto processes; reduce (F, S, Cv) with +."""
    from kaldo.parallel import get_executor, is_parallel

    iq1s = [int(i) for i in q1_indices]
    if not is_parallel(n_workers):
        return _ACCUMULATORS[accumulate_key](iq1s, arrays, scalars)

    n_cpu = n_workers if n_workers is not None else (os.cpu_count() or 1)
    n_processes = max(1, min(int(n_cpu), len(iq1s)))
    payload = dict(scalars)
    payload['_accumulate'] = accumulate_key
    store = _SharedArrayStore(arrays)
    F = S = Cv = 0.0
    n_q1 = len(iq1s)
    report_every = max(1, n_q1 // 20)
    t0 = time.time()
    try:
        with get_executor(
            backend='process',
            n_workers=n_processes,
            initializer=_init_q1_worker,
            initargs=(1, store.spec, payload),
        ) as executor:
            futures = [executor.submit(_q1_chunk_worker, [iq1]) for iq1 in iq1s]
            for i, fut in enumerate(as_completed(futures), 1):
                f, s, cv = fut.result()
                F += f
                S += s
                Cv += cv
                if i == 1 or i == n_q1 or i % report_every == 0:
                    elapsed = time.time() - t0
                    logging.info(
                        f"  q1={i}/{n_q1} ({100.0*i/n_q1:.0f}%)  "
                        f"elapsed={elapsed:.1f}s"
                    )
    finally:
        store.close_and_unlink()
    return F, S, Cv


def _accumulate_f1_q1s(iq1_list, arrays, scalars):
    cart = arrays['cart']
    M = arrays['M']
    inv_w = arrays['inv_w']
    two_np1 = arrays['two_np1']
    dn_tab = arrays['dn_tab']
    ddn_tab = arrays['ddn_tab']
    ok = arrays['ok']
    QD = _unprefix_table('qd', arrays, _QD_ARRAY_KEYS, scalars['qd_nb'])
    q1_weights = scalars['q1_weights']
    nq = scalars['nq']
    T_K = scalars['T_K']
    log_every = scalars.get('log_every')
    n_q1 = len(iq1_list)
    t0 = time.time()
    F1_acc = 0.0
    S1_acc = 0.0
    Cv1_acc = 0.0
    for _i, iq1 in enumerate(iq1_list):
        q1c = cart[iq1]
        M1 = M[iq1]
        psi4_left = _prepare_psi4_q1(QD, M1, q1c)
        inv_w1 = inv_w[iq1]
        two1 = two_np1[iq1]
        dn1 = dn_tab[iq1]
        ddn1 = ddn_tab[iq1]
        ok1 = ok[iq1]
        w1 = q1_weights[iq1]
        for iq2_start in range(0, nq, _F1_Q2_BATCH_SIZE):
            iq2_stop = min(iq2_start + _F1_Q2_BATCH_SIZE, nq)
            q2_slice = slice(iq2_start, iq2_stop)
            Psi4 = _build_psi4_modes_quartet_batch(
                QD, psi4_left, M[q2_slice], cart[q2_slice],
            )
            psi_re = np.real(Psi4)
            mask = ok1[None, :, None] & ok[q2_slice][:, None, :]
            inv_w_prod = inv_w1[None, :, None] * inv_w[q2_slice][:, None, :]

            two2 = two_np1[q2_slice][:, None, :]
            dn2 = dn_tab[q2_slice][:, None, :]
            ddn2 = ddn_tab[q2_slice][:, None, :]
            f_w = two1[None, :, None] * two2 * inv_w_prod
            s_w = -(2.0 * dn1[None, :, None] * two2
                    + two1[None, :, None] * 2.0 * dn2) * inv_w_prod
            cv_w = -(2.0 * ddn1[None, :, None] * two2
                     + 2.0 * ddn2 * two1[None, :, None]
                     + 8.0 * dn1[None, :, None] * dn2) * T_K * inv_w_prod

            F1_acc += w1 * (psi_re * f_w * mask).sum()
            S1_acc += w1 * (psi_re * s_w * mask).sum()
            Cv1_acc += w1 * (psi_re * cv_w * mask).sum()
        if log_every and (_i + 1) % log_every == 0:
            logging.info(f"  q1={_i+1}/{n_q1}  elapsed={time.time()-t0:.1f}s")
    return F1_acc, S1_acc, Cv1_acc


def _accumulate_f2_q1s(iq1_list, arrays, scalars):
    cart = arrays['cart']
    egvs = arrays['egvs']
    omegas = arrays['omegas']
    frac_rounded = arrays['frac_rounded']
    lookup = arrays['lookup']
    ok_tab = arrays['ok_tab']
    TD = _unprefix_table('td', arrays, _TD_ARRAY_KEYS, scalars['td_nb'])
    q1_weights = scalars['q1_weights']
    nq = scalars['nq']
    nx = scalars['nx']
    ny = scalars['ny']
    nz = scalars['nz']
    T_K = scalars['T_K']
    is_classic = scalars['is_classic']
    kT = scalars['kT']
    sigma_table = arrays.get('sigma_table')
    n_tab = arrays.get('n_tab')
    dn_tab = arrays.get('dn_tab')
    ddn_tab = arrays.get('ddn_tab')
    log_every = scalars.get('log_every')
    n_q1 = len(iq1_list)
    t0 = time.time()
    F2 = 0.0
    S2 = 0.0
    Cv2 = 0.0
    for _i, iq1 in enumerate(iq1_list):
        i1, j1, k1 = frac_rounded[iq1]
        e1 = egvs[iq1]
        psi3_left = _prepare_psi3_q1(TD, e1)
        w1 = omegas[iq1]
        ok1 = ok_tab[iq1]
        w_q1 = q1_weights[iq1]
        for iq2_start in range(0, nq, _F2_Q2_BATCH_SIZE):
            iq2_stop = min(iq2_start + _F2_Q2_BATCH_SIZE, nq)
            q2_slice = slice(iq2_start, iq2_stop)
            q2_grid = frac_rounded[q2_slice]
            i3 = (-i1 - q2_grid[:, 0]) % nx
            j3 = (-j1 - q2_grid[:, 1]) % ny
            k3 = (-k1 - q2_grid[:, 2]) % nz
            iq3 = lookup[i3, j3, k3]

            Psi3 = _build_psi3_modes_triplet_batch(
                TD, psi3_left, egvs[q2_slice], egvs[iq3],
                cart[q2_slice], cart[iq3],
            )
            psisq = np.abs(Psi3) ** 2

            w1_ = w1[None, :, None, None]
            w2_ = omegas[q2_slice][:, None, :, None]
            w3_ = omegas[iq3][:, None, None, :]
            mask = (ok1[None, :, None, None]
                    & ok_tab[q2_slice][:, None, :, None]
                    & ok_tab[iq3][:, None, None, :])

            inv_w_prod = np.zeros_like(psisq)
            inv_w_prod[mask] = 1.0 / (w1_ * w2_ * w3_)[mask]

            if is_classic:
                w_prod = w1_ * w2_ * w3_
                common_F = np.zeros_like(psisq)
                common_S = np.zeros_like(psisq)
                common_F[mask] = 4.0 * (kT ** 2) / ((HBAR ** 2) * w_prod[mask])
                common_S[mask] = 8.0 * (KB ** 2) * T_K / ((HBAR ** 2) * w_prod[mask])
                common_Cv = common_S
            else:
                s1 = sigma_table[iq1][None, :, None, None]
                s2 = sigma_table[q2_slice][:, None, :, None]
                s3 = sigma_table[iq3][:, None, None, :]
                sigma_combo = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
                denom1 = w1_ + w2_ + w3_
                Re1 = denom1 / (denom1 ** 2 + sigma_combo ** 2)
                denom2 = w1_ + w2_ - w3_
                Re2 = denom2 / (denom2 ** 2 + sigma_combo ** 2)

                n1_ = n_tab[iq1][None, :, None, None]
                n2_ = n_tab[q2_slice][:, None, :, None]
                n3_ = n_tab[iq3][:, None, None, :]
                dn1_ = dn_tab[iq1][None, :, None, None]
                dn2_ = dn_tab[q2_slice][:, None, :, None]
                dn3_ = dn_tab[iq3][:, None, None, :]
                ddn1_ = ddn_tab[iq1][None, :, None, None]
                ddn2_ = ddn_tab[q2_slice][:, None, :, None]
                ddn3_ = ddn_tab[iq3][:, None, None, :]

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

                common_F = (f1 * Re1 + 3.0 * f2 * Re2)
                common_S = (df1 * Re1 + 3.0 * df2 * Re2)
                common_Cv = (ddf1 * Re1 + 3.0 * ddf2 * Re2) * T_K

            integrand_F = psisq * inv_w_prod * common_F / 48.0
            integrand_S = psisq * inv_w_prod * common_S / 48.0
            integrand_Cv = psisq * inv_w_prod * common_Cv / 48.0
            F2 += w_q1 * integrand_F[mask].sum()
            S2 += w_q1 * integrand_S[mask].sum()
            Cv2 += w_q1 * integrand_Cv[mask].sum()
        if log_every and (_i + 1) % log_every == 0:
            logging.info(f"  q1={_i+1}/{n_q1}  elapsed={time.time()-t0:.1f}s")
    return F2, S2, Cv2


_ACCUMULATORS = {
    'f1': _accumulate_f1_q1s,
    'f2': _accumulate_f2_q1s,
}


def F1_vectorized(neighbors_pair, quartets, masses_kg, uc_positions, uc_cell,
                  kmesh, T_K, use_q_symmetry=False, atoms=None, is_classic=False,
                  n_workers=1):
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
    is_classic : if True, use classical occupations ``2n+1 = 2 kT / (ħ ω)``
        (LDT ``quantum=false`` branch); else Bose–Einstein.
    n_workers : int or None
        Process-pool size for the outer q1 loop. ``1`` (default) is serial;
        ``>1`` that many processes; ``None`` uses all CPUs. Flattened IFC
        tables (QD) and other large arrays are attached via
        ``multiprocessing.shared_memory`` so nested quartet lists are never
        pickled into workers.

    Returns
    -------
    dict with keys ``F1``, ``S1``, ``Cv1``, ``U1`` (units eV/atom for F, U
    and kB/atom for S, Cv).
    """
    _check_n_workers(n_workers)
    nx, ny, nz = kmesh
    n_uc = len(uc_positions)
    nb = 3 * n_uc
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
        logging.info(f"use_q_symmetry: reducing q1 from {nq} to {len(q1_indices)} "
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
    logging.info(f"eigs {nq} qs in {time.time()-t0:.1f}s")

    # Occupations: Bose–Einstein or classical (2n+1 = 2 kT / ħω).
    n_tab, dn_tab, ddn_tab, ok = planck_and_derivs(omegas, T_K, is_classic=is_classic)
    two_np1 = 2 * n_tab + 1
    inv_w = np.zeros_like(omegas)
    inv_w[ok] = 1.0 / omegas[ok]

    # Per-q eigenvector outer products: M[q, b, i, j] = e(q,b,i) * conj(e(q,b,j))
    t1 = time.time()
    M = np.einsum("qib,qjb->qbij", egvs, np.conj(egvs))
    logging.info(f"building per-q outer products  done in {time.time()-t1:.1f}s, shape {M.shape}")

    QD = flatten_quartets(quartets, masses_kg, uc_cell)
    logging.info(f"flattened {QD['a1'].shape[0]} quartets")

    n_q1 = len(q1_indices)
    logging.info(f"F1/S1/Cv1 double-q loop over {n_q1}x{nq}={n_q1*nq} (q1,q2) pairs"
          + (" [q1 in IBZ]" if use_q_symmetry else "")
          + (f" n_workers={n_workers}" if n_workers != 1 else ""))
    t2 = time.time()
    arrays = {
        'cart': cart,
        'M': M,
        'inv_w': inv_w,
        'two_np1': two_np1,
        'dn_tab': dn_tab,
        'ddn_tab': ddn_tab,
        'ok': ok,
        **_prefix_table('qd', QD, _QD_ARRAY_KEYS),
    }
    scalars = {
        'qd_nb': int(QD['nb']),
        'nq': int(nq),
        'T_K': float(T_K),
        'q1_weights': q1_weights,
        'log_every': max(1, n_q1 // 20) if n_workers == 1 else None,
    }
    F1_acc, S1_acc, Cv1_acc = _run_parallel_q1(
        'f1', q1_indices, n_workers, arrays, scalars,
    )
    logging.info(f"F1 loop total {time.time()-t2:.1f}s")

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
            T_a1.append(a1)
            T_a2.append(a2)
            T_a3.append(a3)
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


def _triplet_mode_vectors(egv, atoms):
    """Gather Cartesian mode vectors for each triplet atom."""
    xyz = np.arange(3)
    indices = 3 * atoms[:, None] + xyz[None, :]
    return np.moveaxis(egv[indices, :], 2, 1)


def _triplet_mode_vectors_batch(egv, atoms):
    """Gather Cartesian mode vectors for a batch of q-points."""
    xyz = np.arange(3)
    indices = 3 * atoms[:, None] + xyz[None, :]
    return np.moveaxis(egv[:, indices, :], 3, 2)


def _prepare_psi3_q1(TD, e1):
    """Contract q1 mode vectors with IFC3 once for an outer-q1 task."""
    e1_vectors = _triplet_mode_vectors(np.conj(e1), TD["a1"])
    return np.einsum("nka,nabc->nkbc", e1_vectors, TD["ifc"], optimize=True)


def _build_psi3_modes_triplet_batch(TD, left_q1, e2, e3, q2_cart, q3_cart):
    """Build mode-space Psi3 for a batch without a real-space scatter."""
    e2_vectors = _triplet_mode_vectors_batch(np.conj(e2), TD["a2"])
    e3_vectors = _triplet_mode_vectors_batch(np.conj(e3), TD["a3"])
    phase = np.exp(-1j * (q2_cart @ TD["lv2c"].T + q3_cart @ TD["lv3c"].T))
    middle = np.einsum("nkbc,qnlb->qnklc", left_q1, e2_vectors, optimize=True)
    return np.einsum("qnklc,qnmc,qn->qklm", middle, e3_vectors, phase, optimize=True)


def planck_and_derivs(omega, T_K, is_classic=False):
    """Occupation factor n(ω, T) and first / second temperature derivatives.

    Quantum (default): Bose–Einstein ``n = 1/(e^{ħω/kT} - 1)``.

    Classical (``is_classic=True``): high-T limit with
    ``2n + 1 = 2 kT / (ħ ω)``, i.e. ``n = kT/(ħω) - 1/2``, so
    ``∂n/∂T = k/(ħω)`` and ``∂²n/∂T² = 0``. This matches the LDT
    ``quantum=false`` branch for F1 (and the classical occupation
    limit used there).
    """
    ok = omega > 2 * np.pi * FREQ_TOL_THZ * 1e12
    n = np.zeros_like(omega)
    dn = np.zeros_like(omega)
    ddn = np.zeros_like(omega)
    if not np.any(ok):
        return n, dn, ddn, ok
    if is_classic:
        # n = kT/(ħω) - 1/2  ⇒  2n+1 = 2 kT/(ħω)
        n[ok] = KB * T_K / (HBAR * omega[ok]) - 0.5
        dn[ok] = KB / (HBAR * omega[ok])
        # ddn stays zero
    else:
        x = HBAR * omega / (KB * T_K)
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
        dq_vec = np.zeros(3)
        dq_vec[alpha] = dq
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
    from .constants import EV as _EV, ANG as _ANG
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
    TDEP-style adaptive σ (per-q, per-band) via ``adaptive_sigma``.

    Shim on :func:`kaldo.controllers.anharmonic.calculate_adaptive_sigma_tdep`.
    Cumulant's historical signature is per-q with ``velocity`` as ``(3, n_b)``
    (α, band); kaldo expects ``(n_k, n_b, 3)``. We reshape to a single-q call
    and drop the leading axis.

    F2 uses this helper for the quantum resonant denominators.
    """
    from kaldo.controllers.anharmonic import calculate_adaptive_sigma_tdep
    v = np.asarray(group_vel_alpha_nb).T[np.newaxis, ...]  # (1, n_b, 3)
    sig = calculate_adaptive_sigma_tdep(
        radius=radius_inv_ang, velocity=v,
        default_sigma=default_sigma_nb, scale=scale,
    )
    return sig[0]  # strip leading q-axis


def smearingparameter(scaled_rec_basis, group_vel_alpha_nb, default_sigma_nb, scale=1.0):
    """
    TDEP ``qp%smearingparameter`` from Fortran ``anharmonic_free_energy``.

    Not used by F2. Shim on
    :func:`kaldo.controllers.anharmonic.calculate_smearingparameter_tdep`.
    """
    from kaldo.controllers.anharmonic import calculate_smearingparameter_tdep
    v = np.asarray(group_vel_alpha_nb).T[np.newaxis, ...]  # (1, n_b, 3)
    sig = calculate_smearingparameter_tdep(
        scaled_rec_basis=scaled_rec_basis, velocity=v,
        default_sigma=default_sigma_nb, scale=scale,
    )
    return sig[0]

def F2_vectorized(neighbors_pair, triplets, masses_kg, uc_positions, uc_cell,
                  kmesh, T_K, sigma_THz=None, use_q_symmetry=False, atoms=None,
                  is_classic=False, n_workers=1):
    """
    F2 / S2 / Cv2 / U2 cubic cumulant evaluator on a regular MP q-mesh.

    If ``sigma_THz`` is None (default), uses per-mode adaptive σ from the
    Brillouin-zone cell radius and ``|v|``, clamped to the default-smearing
    window. If a float, uses a fixed isotropic σ.

    ``use_q_symmetry=True`` reduces the outer q1 loop to spglib IBZ reps
    weighted by orbit size. See ``F1_vectorized`` for the invariance
    argument (same structure; q3 = -q1-q2 is symmetry-consistent).

    ``is_classic=True`` uses the LDT classical closed form
    ``(kT)^2 / (ω1 ω2 ω3) / 12`` (no resonant principal-value denominators);
    quantum uses Bose occupations with adaptive/fixed σ smearing.

    ``n_workers`` is as in :func:`F1_vectorized`: process-pool over q1,
    with flattened IFC tables (TD) in shared memory.

    Returns a dict with keys ``F2``, ``S2``, ``Cv2``, ``U2``.
    """
    _check_n_workers(n_workers)
    nx, ny, nz = kmesh
    nq = nx * ny * nz
    n_uc = len(uc_positions)
    nb = 3 * n_uc
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
        logging.info(f"use_q_symmetry: reducing q1 from {nq} to {len(q1_indices)} "
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
    logging.info(f"eigs {nq} qs in {time.time()-t0:.1f}s")

    # Adaptive σ only needed for the quantum (resonant) branch.
    # default_smearing from IBZ frequencies; per-mode σ from BZ-cell radius
    # and group velocity on IBZ q-points, then broadcast via ir_mapping.
    if not is_classic:
        if sigma_THz is None:
            t_sig = time.time()
            if atoms is None:
                raise ValueError(
                    "adaptive sigma (sigma_THz=None) requires atoms= ASE Atoms "
                    "so default_smearing / σ can be built on the IBZ"
                )
            from kaldo.phonons import _get_ir_kgrid_data
            from kaldo.controllers.anharmonic import calculate_bz_cell_radius
            ir_mapping, _, ibz_indices, _ = _get_ir_kgrid_data(
                atoms, kpts=list(kmesh), grid_type='C')
            ibz_indices = np.asarray(ibz_indices, dtype=int)
            default_sigma_bands = compute_default_smearing(omegas[ibz_indices])
            radius = calculate_bz_cell_radius(np.linalg.inv(uc_cell), nq)
            sigma_ibz = np.empty((len(ibz_indices), nb))
            for ii, iq in enumerate(ibz_indices):
                v = compute_group_velocity_analytic(
                    neighbors_pair, uc_positions, masses_kg,
                    cart[iq], omegas[iq], egvs[iq],
                )
                sigma_ibz[ii] = adaptive_sigma(
                    radius, v, default_sigma_bands,
                )
            # Broadcast IBZ σ to every full-grid q via its irreducible parent
            # (ir_mapping[iq] is the full-grid index of the IBZ representative).
            sigma_by_full = np.empty((nq, nb))
            for ii, iq in enumerate(ibz_indices):
                sigma_by_full[iq] = sigma_ibz[ii]
            sigma_table = sigma_by_full[np.asarray(ir_mapping, dtype=int)]
            logging.info(f"adaptive_sigma (IBZ): {time.time()-t_sig:.1f}s, "
                  f"range {sigma_table.min():.2e}..{sigma_table.max():.2e} rad/s "
                  f"(~{sigma_table.min()/(2*np.pi*1e12):.3f}..{sigma_table.max()/(2*np.pi*1e12):.3f} THz)")
        else:
            sigma_rad_s = 2 * np.pi * sigma_THz * 1e12
            sigma_table = np.full((nq, nb), sigma_rad_s)
    else:
        sigma_table = None

    # Occupation table (quantum only; classical uses closed-form weights below).
    ok_tab = omegas > 2 * np.pi * FREQ_TOL_THZ * 1e12
    if is_classic:
        n_tab = dn_tab = ddn_tab = None
    else:
        n_tab, dn_tab, ddn_tab, ok_tab = planck_and_derivs(omegas, T_K, is_classic=False)

    # q3 lookup table from q1+q2+q3 = 0 mod G
    frac_rounded = np.round(frac * np.array([nx, ny, nz])[None, :]).astype(int) \
                   % np.array([nx, ny, nz])[None, :]
    lookup = np.full((nx, ny, nz), -1, dtype=int)
    for iq, (i, j, k) in enumerate(frac_rounded):
        lookup[i, j, k] = iq

    TD = flatten_triplets(triplets, masses_kg, uc_cell)
    logging.info(f"flattened {TD['a1'].shape[0]} triplets, nb={nb}")

    n_q1 = len(q1_indices)
    logging.info(f"F2/S2/Cv2 double-q loop over {n_q1}x{nq}={n_q1*nq} (q1,q2) pairs"
          + (" [q1 in IBZ]" if use_q_symmetry else "")
          + (f" n_workers={n_workers}" if n_workers != 1 else ""))
    t2 = time.time()
    kT = KB * T_K
    arrays = {
        'cart': cart,
        'egvs': egvs,
        'omegas': omegas,
        'frac_rounded': frac_rounded,
        'lookup': lookup,
        'ok_tab': ok_tab,
        **_prefix_table('td', TD, _TD_ARRAY_KEYS),
    }
    if sigma_table is not None:
        arrays['sigma_table'] = sigma_table
    if n_tab is not None:
        arrays['n_tab'] = n_tab
        arrays['dn_tab'] = dn_tab
        arrays['ddn_tab'] = ddn_tab
    scalars = {
        'td_nb': int(TD['nb']),
        'nq': int(nq),
        'nx': int(nx),
        'ny': int(ny),
        'nz': int(nz),
        'T_K': float(T_K),
        'is_classic': bool(is_classic),
        'kT': float(kT),
        'q1_weights': q1_weights,
        'log_every': max(1, n_q1 // 10) if n_workers == 1 else None,
    }
    F2, S2, Cv2 = _run_parallel_q1(
        'f2', q1_indices, n_workers, arrays, scalars,
    )
    logging.info(f"F2 loop total {time.time()-t2:.1f}s")

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

    The table must index ``fc.second.value``'s replica axis. That axis is
    ordered by ``second.translation_support`` (unique per-pair vectors when
    a TDEP file provides them), *not* by
    ``second._replica_table`` metadata, which still holds the det(M)
    congruence-class table stamped by ``attach_snf_metadata``.
    Using the class table against a per-pair-indexed tensor phases every
    off-Gamma dynamical matrix with the wrong R and produces hundreds of
    spurious imaginary modes.

    Diagonal fc: compute from ``replicated_positions`` with
    :func:`_minimum_image_lv` so the wrap is correct on F2 meshes smaller
    than the supercell.
    """
    second = fc.second
    support = getattr(second, "translation_support", None)
    if support is not None:
        # The translation support is what actually indexes the value axis
        # (per-pair literal file vectors or periodic representatives).
        return np.asarray(support.translations, dtype=float)
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
    non-diagonal fc (SNF) the replica index runs over the per-pair
    lattice-vector table that indexes ``fc.second.value`` (see
    :func:`_replica_lv_frac_table`). On diagonal fc we compute the
    minimum-image fractional lattice vectors from ``replicated_positions``
    via :func:`_minimum_image_lv`.

    ASR is NOT re-imposed here: ``SecondOrder.load`` already applies it if
    requested. The TDEP file is already ASR-exact to float precision.
    """
    second = np.asarray(fc.second.value)[0]  # (n_uc, 3, n_rep, n_uc, 3)
    n_uc = fc.n_atoms
    uc_pos = np.asarray(fc.atoms.positions)
    uc_cell = np.asarray(fc.atoms.cell)
    lv_frac_tab = _replica_lv_frac_table(fc)  # (n_rep, 3)
    # The tensor's translation axis can be longer than fc.n_replicas when a
    # TDEP file provides literal per-pair vectors; iterate the axis itself.
    n_rep = second.shape[2]
    if lv_frac_tab.shape[0] != n_rep:
        raise ValueError(
            "replica table does not index the IFC2 translation axis: "
            f"{lv_frac_tab.shape[0]} rows vs axis {n_rep}"
        )

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
    third = fc.third.value
    # Shape: (n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3), sparse COO. Never
    # densify: the dense tensor is n_uc^3 * 27 * n_rep^2 doubles.
    support = getattr(fc.third, "translation_support", None)
    if support is not None:
        lv_frac_tab = np.asarray(support.translations, dtype=float)
    else:
        list_rep = fc.third.list_of_replicas  # (n_rep, 3) in Cartesian
        uc_cell = np.asarray(fc.atoms.cell)
        inv_cell = np.linalg.inv(uc_cell)
        lv_frac_raw = list_rep @ inv_cell  # (n_rep, 3) fractional
        lv_frac_tab = np.asarray([
            _minimum_image_lv(lv, fc.supercell) for lv in lv_frac_raw
        ])

    n_uc = fc.n_atoms

    # Assemble 3x3x3 blocks directly from the COO coordinates: only the
    # ~nnz stored entries are visited, in the same (a1, r2, a2, r3, a3)
    # order the dense loops used.
    coords = np.asarray(third.coords)
    data = np.asarray(third.data)
    a1_i, al, r2_i, a2_i, be, r3_i, a3_i, ga = (coords[k] for k in range(8))
    blocks = {}
    for k in range(coords.shape[1]):
        key = (int(a1_i[k]), int(r2_i[k]), int(a2_i[k]), int(r3_i[k]), int(a3_i[k]))
        b = blocks.get(key)
        if b is None:
            b = blocks[key] = np.zeros((3, 3, 3))
        b[al[k], be[k], ga[k]] = data[k]

    per_atom = [[] for _ in range(n_uc)]
    for (a1, r2, a2, r3, a3) in sorted(blocks):
        per_atom[a1].append(
            (a2, a3, lv_frac_tab[r2], lv_frac_tab[r3], blocks[(a1, r2, a2, r3, a3)])
        )
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
    fourth = fc.fourth.value
    # Sparse COO of shape (n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3, n_rep,
    # n_uc, 3). Never densify: the dense tensor is 81 * n_uc^4 * n_rep^3
    # doubles (11+ GB already at 216 atoms), which OOM-kills CI workers.
    support = getattr(fc.fourth, "translation_support", None)
    if support is not None:
        lv_frac_tab = np.asarray(support.translations, dtype=float)
    else:
        list_rep = fc.fourth.list_of_replicas
        uc_cell = np.asarray(fc.atoms.cell)
        inv_cell = np.linalg.inv(uc_cell)
        lv_frac_raw = list_rep @ inv_cell
        lv_frac_tab = np.asarray([
            _minimum_image_lv(lv, fc.supercell) for lv in lv_frac_raw
        ])

    n_uc = fc.n_atoms

    # Assemble 3x3x3x3 blocks directly from the COO coordinates: only the
    # ~nnz stored entries are visited, in the same (a1, r2, a2, r3, a3,
    # r4, a4) order the dense loops used.
    coords = np.asarray(fourth.coords)
    data = np.asarray(fourth.data)
    a1_i, al, r2_i, a2_i, be, r3_i, a3_i, ga, r4_i, a4_i, de = (
        coords[k] for k in range(11)
    )
    blocks = {}
    for k in range(coords.shape[1]):
        key = (int(a1_i[k]), int(r2_i[k]), int(a2_i[k]), int(r3_i[k]),
               int(a3_i[k]), int(r4_i[k]), int(a4_i[k]))
        b = blocks.get(key)
        if b is None:
            b = blocks[key] = np.zeros((3, 3, 3, 3))
        b[al[k], be[k], ga[k], de[k]] = data[k]

    per_atom = [[] for _ in range(n_uc)]
    for key in sorted(blocks):
        a1, r2, a2, r3, a3, r4, a4 = key
        per_atom[a1].append(
            (a2, a3, a4,
             lv_frac_tab[r2], lv_frac_tab[r3], lv_frac_tab[r4],
             blocks[key])
        )
    return per_atom


def F2_from_fc(fc, masses_amu, kmesh, T_K, sigma_THz=None,
               use_q_symmetry=False, is_classic=False, n_workers=1):
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
    kmesh, T_K, sigma_THz, use_q_symmetry, is_classic, n_workers :
        see :func:`F2_vectorized`.

    Returns
    -------
    dict with keys ``F2``, ``S2``, ``Cv2``, ``U2``.
    """
    from .constants import AMU
    uc_pos = np.asarray(fc.atoms.positions)
    uc_cell = np.asarray(fc.atoms.cell)
    masses_kg = np.asarray(masses_amu) * AMU
    neighbors = _neighbors_from_fc(fc)
    triplets = _triplets_from_fc(fc)
    return F2_vectorized(
        neighbors, triplets, masses_kg, uc_pos, uc_cell,
        tuple(kmesh), T_K, sigma_THz=sigma_THz,
        use_q_symmetry=use_q_symmetry,
        atoms=fc.atoms,
        is_classic=is_classic,
        n_workers=n_workers,
    )


def F1_from_fc(fc, masses_amu, kmesh, T_K, use_q_symmetry=False, is_classic=False,
               n_workers=1):
    """F1 quartic cumulant on a ``ForceConstants`` object (kaldo-native entry).

    Requires ``fc.fourth`` loaded (``include_fourth=True`` in ``from_folder``).
    Delegates to :func:`F1_vectorized` after reconstructing the legacy
    neighbour-list / quartet-list format. Output is bit-for-bit identical
    to :func:`F1_vectorized` on the same physical inputs.
    """
    from .constants import AMU
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
        is_classic=is_classic,
        n_workers=n_workers,
    )

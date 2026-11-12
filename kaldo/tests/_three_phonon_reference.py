"""Dense, transparent reference for kALDo's three-phonon RTA phase space and linewidths.

This helper is a yardstick for fast three-phonon routes. It re-implements kALDo's crystal formulas literally, in
plain NumPy, with no symmetry shortcuts: each channel is evaluated with its own partner q'', its own eigenvector and
Fourier-phase conjugations, and the full sparse IFC3. Grid data enter as callables of a fractional q, so the initial
q, the q' set and the q'' lookup may come from different grids (as on a half-step shifted grid).

Conventions mirrored (file pointers refer to the kALDo source at the time of writing):

- IFC3 layout ``(i,a,Rj,j,b,Rk,k,c)`` from ``third.get_interpolation(mode)``; both translation axes index
  ``interpolation.support`` and take the phases ``support.phases(q) = exp(2 pi i q.R)``
  (``kaldo/observables/thirdorder.py`` ``_ThirdOrderIFCInterpolation``, ``kaldo/grid.py`` ``TranslationSupport``).
- Matrix element (``kaldo/controllers/anharmonic.py`` ``sparse_potential_mu``)::

      V(q mu; q' mu', q'' mu'') = sum_{i,Rj,j,Rk,k} Phi[i,Rj,j,Rk,k] e_i(q mu)
                                  * X(q')_{Rj} E(q')_{j mu'} * conj(chi(q'')_{Rk}) conj(e_k(q'' mu''))

  with ``X, E = chi, e`` for absorption (``is_plus=1``) and ``conj(chi), conj(e)`` for decay (``is_plus=0``). The
  initial eigenvector ``e(q mu)`` is not conjugated.
- Absorption has ``q'' = q + q'`` and energy mismatch ``w + w' - w''``; decay has ``q'' = q - q'`` and
  ``w - w' - w''`` (``calculate_dirac_delta_crystal``). Triplets are kept when ``|dw| < 2 * 2 pi sigma`` and both
  partner modes are physical; the weight is ``gaussian_delta(dw, 2 pi sigma)``
  (``kaldo/controllers/dirac_kernel.py``).
- Potential ``|V|^2 * pi * hbar / 4 * GAMMA_TO_THZ / w / n_k / (w' w'')`` (``sparse_potential_mu``).
- Population factors ``n' - n''`` (absorption) and ``0.5 * (1 + n' + n'')`` (decay)
  (``population_delta``, unbalanced).
- Phase space ``sum delta * pop`` divided by ``n_k``; linewidth ``sum hbar_factor * potential * delta * pop``
  (``calculate_ps_and_gamma`` and ``Phonons._select_algorithm_for_phase_space_and_gamma``).
"""

import ase.units as units
import numpy as np
from sparse import COO

from kaldo.controllers.dirac_kernel import gaussian_delta

# Same definitions as kaldo/controllers/anharmonic.py; kept local so this helper does not import TensorFlow.
GAMMA_TO_THZ = 1e11 * units.mol * (units.mol / (10 * units.J)) ** 2
HBAR = units._hbar
DELTA_THRESHOLD = 2


def _ifc3_entries(interpolation):
    """Return COO entries ``(i, Rj, j, Rk, k, value)`` of the rank-8 IFC3 with flattened mode indices."""
    value = interpolation.value
    n_atoms = value.shape[0]
    n_modes = 3 * n_atoms
    n_translations = interpolation.support.size
    shape = (n_modes, n_translations, n_modes, n_translations, n_modes)
    if isinstance(value, COO):
        coo = value.reshape(shape)
    else:
        coo = COO.from_numpy(np.asarray(value).reshape(shape))
    i, t_j, j, t_k, k = (np.asarray(axis, dtype=np.int64) for axis in coo.coords)
    data = np.asarray(coo.data).astype(np.complex128)
    return n_modes, n_translations, i, t_j, j, t_k, k, data


def _fold_to_dense(weight, pair_phase, pair_index, jk_index, n_modes):
    """Fold sparse entries ``weight * pair_phase[(Rj,Rk)]`` onto a dense complex ``(j, k)`` matrix."""
    amplitude = weight * pair_phase[pair_index]
    real = np.bincount(jk_index, weights=amplitude.real, minlength=n_modes * n_modes)
    imag = np.bincount(jk_index, weights=amplitude.imag, minlength=n_modes * n_modes)
    return (real + 1j * imag).astype(np.complex128).reshape((n_modes, n_modes))


def reference_ps_and_gamma(third, interpolation_mode, q_initial, initial_modes, q_primes, vectors, omega, physical,
                           population, partner_lookup, sigma_thz, hbar_factor, n_k):
    """Phase space and linewidth of selected initial modes by dense, literal evaluation of kALDo's formulas.

    Parameters
    ----------
    third : kaldo.observables.thirdorder.ThirdOrder
        Third-order force constants; ``third.get_interpolation(interpolation_mode)`` supplies the rank-8 IFC3
        ``(i,a,Rj,j,b,Rk,k,c)`` and its translation support.
    interpolation_mode : str
        ``'auto'``, ``'periodic'`` or ``'wigner-seitz'``.
    q_initial : np.ndarray, shape (3,)
        Fractional initial q.
    initial_modes : np.ndarray[int]
        Branch indices of the initial phonons at ``q_initial``.
    q_primes : np.ndarray, shape (m, 3)
        Fractional q' points summed over.
    vectors, omega, physical, population : callable
        Functions of a fractional q returning, respectively, mass-rescaled eigenvectors (complex128,
        ``[component, mode]``), angular frequencies in rad/ps (float64), the physical-mode mask (bool) and the
        occupations (float64).
    partner_lookup : callable
        ``partner_lookup(q, q_prime, is_plus)`` returns the fractional q'' (``q + q'`` for absorption, ``q - q'``
        for decay).
    sigma_thz : float
        Fixed Gaussian bandwidth in THz.
    hbar_factor : float
        ``CLASSICAL_HBAR_SCALE`` for classical statistics, ``1`` for quantum.
    n_k : int
        Number of q points normalizing the Brillouin-zone average.

    Returns
    -------
    np.ndarray, shape (len(initial_modes), 2), float64
        Column 0 is the phase space (already divided by ``n_k``), column 1 the linewidth in kALDo's units.
    """
    q_initial = np.asarray(q_initial, dtype=np.float64)
    initial_modes = np.atleast_1d(np.asarray(initial_modes, dtype=np.int64))
    q_primes = np.atleast_2d(np.asarray(q_primes, dtype=np.float64))
    sigma = 2 * np.pi * np.float64(sigma_thz)

    interpolation = third.get_interpolation(interpolation_mode)
    support = interpolation.support
    n_modes, n_translations, i, t_j, j, t_k, k, data = _ifc3_entries(interpolation)
    pair_index = t_j * n_translations + t_k
    jk_index = j * n_modes + k

    vectors_q = np.asarray(vectors(q_initial), dtype=np.complex128)
    omega_q = np.asarray(omega(q_initial), dtype=np.float64)
    physical_q = np.asarray(physical(q_initial), dtype=bool)

    # Partner data per (q', channel): q'' and its harmonic quantities, shared by all initial modes.
    partners = []
    for q_prime in q_primes:
        vectors_p = np.asarray(vectors(q_prime), dtype=np.complex128)
        chi_p = support.phases(q_prime)[0].astype(np.complex128)
        omega_p = np.asarray(omega(q_prime), dtype=np.float64)
        physical_p = np.asarray(physical(q_prime), dtype=bool)
        population_p = np.asarray(population(q_prime), dtype=np.float64)
        for is_plus in (0, 1):
            q_second = np.asarray(partner_lookup(q_initial, q_prime, is_plus), dtype=np.float64)
            second_vectors, second_chi = (vectors_p, chi_p) if is_plus else (np.conj(vectors_p), np.conj(chi_p))
            third_vectors = np.conj(np.asarray(vectors(q_second), dtype=np.complex128))
            third_chi = np.conj(support.phases(q_second)[0].astype(np.complex128))
            partners.append(dict(
                is_plus=is_plus,
                second_vectors=second_vectors,
                third_vectors=third_vectors,
                pair_phase=np.einsum("s,t->st", second_chi, third_chi).reshape(-1),
                omega_p=omega_p,
                omega_pp=np.asarray(omega(q_second), dtype=np.float64),
                physical_p=physical_p,
                physical_pp=np.asarray(physical(q_second), dtype=bool),
                population_p=population_p,
                population_pp=np.asarray(population(q_second), dtype=np.float64),
            ))

    ps_and_gamma = np.zeros((initial_modes.size, 2), dtype=np.float64)
    for row, mu in enumerate(initial_modes):
        if not physical_q[mu]:
            continue
        omega_0 = omega_q[mu]
        weight = data * vectors_q[i, mu]
        for partner in partners:
            sign = 2 * partner["is_plus"] - 1
            delta_omega = omega_0 + sign * partner["omega_p"][:, np.newaxis] - partner["omega_pp"][np.newaxis, :]
            selected = (
                (np.abs(delta_omega) < DELTA_THRESHOLD * sigma)
                & partner["physical_p"][:, np.newaxis]
                & partner["physical_pp"][np.newaxis, :]
            )
            if not selected.any():
                continue
            mu_p, mu_pp = np.nonzero(selected)

            folded = _fold_to_dense(weight, partner["pair_phase"], pair_index, jk_index, n_modes)
            amplitude = partner["second_vectors"].T @ folded @ partner["third_vectors"]

            delta = gaussian_delta(delta_omega[mu_p, mu_pp], sigma)
            potential = (np.abs(amplitude[mu_p, mu_pp]) ** 2 * np.pi * HBAR / 4 * GAMMA_TO_THZ / omega_0 / n_k
                         / partner["omega_p"][mu_p] / partner["omega_pp"][mu_pp])
            n_p = partner["population_p"][mu_p]
            n_pp = partner["population_pp"][mu_pp]
            population_factor = n_p - n_pp if partner["is_plus"] else 0.5 * (1 + n_p + n_pp)

            ps_and_gamma[row, 0] += np.sum(delta * population_factor)
            ps_and_gamma[row, 1] += np.sum(hbar_factor * potential * delta * population_factor)

    ps_and_gamma[:, 0] /= n_k
    return ps_and_gamma

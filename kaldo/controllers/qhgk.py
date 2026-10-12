"""Mode-pair kernels shared by every QHGK diffusivity and conductivity path.

Angular frequencies ``omega`` and the QHGK ``diffusivity_bandwidth`` are in
rad/ps. Each function takes two mode blocks, so a caller can evaluate a full
``(M, M)`` pair field or a bounded ``(B_i, B_j)`` tile with identical values.
Only modes that are physical and have positive frequency contribute.
"""

import ase.units as units
import numpy as np

from kaldo.helpers.units import KB_J_PER_K

# x = _HBAR_OVER_KB * omega / T is dimensionless for omega in rad/ps and T in K.
_HBAR_OVER_KB = units._hbar * 1e12 / KB_J_PER_K
_DEGENERACY_RTOL = 1e-6


def _active(omega, physical_mode):
    return np.asarray(physical_mode, dtype=bool).reshape(omega.shape) & (omega > 0)


def diffusivity_kernel(omega_i, omega_j, bandwidth_i, bandwidth_j, physical_i, physical_j, curve, *,
                       is_including_antiresonant=False, diffusivity_threshold=None):
    """Return ``pi * delta(omega_i - omega_j) / (4 omega_i omega_j)`` for one mode block.

    The broadening width is ``sigma_ij = 2 (b_i + b_j)``. ``diffusivity_threshold``
    keeps the historical rule: pair ``(i, j)`` is kept when
    ``|omega_i - omega_j| < threshold * 2 pi * b_j``, using the partner
    bandwidth, and it applies only to the resonant term. Pairs with an
    inactive mode or zero width contribute zero.
    """
    wi = np.asarray(omega_i, dtype=np.float64).reshape(-1)
    wj = np.asarray(omega_j, dtype=np.float64).reshape(-1)
    bi = np.broadcast_to(np.asarray(bandwidth_i, dtype=np.float64).reshape(-1), wi.shape)
    bj = np.broadcast_to(np.asarray(bandwidth_j, dtype=np.float64).reshape(-1), wj.shape)
    sigma = 2 * (bi[:, np.newaxis] + bj[np.newaxis, :])
    delta = wi[:, np.newaxis] - wj[np.newaxis, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kernel = np.array(curve(delta, sigma), dtype=np.float64)
        if diffusivity_threshold is not None:
            if not np.isfinite(diffusivity_threshold) or diffusivity_threshold < 0:
                raise ValueError("diffusivity_threshold must be finite and nonnegative")
            kernel[np.abs(delta) >= diffusivity_threshold * 2 * np.pi * bj[np.newaxis, :]] = 0
        if is_including_antiresonant:
            kernel += curve(wi[:, np.newaxis] + wj[np.newaxis, :], sigma)
        kernel *= np.pi / 4
        kernel /= wi[:, np.newaxis]
        kernel /= wj[np.newaxis, :]
    valid = _active(wi, physical_i)[:, np.newaxis] & _active(wj, physical_j)[np.newaxis, :] & (sigma > 0)
    kernel[~(valid & np.isfinite(kernel))] = 0
    return kernel


def _single_mode_heat_capacity(x):
    """Stable dimensionless Bose heat capacity ``x^2 e^x / (e^x - 1)^2``."""
    result = np.zeros_like(x)
    small = x < 1e-4
    result[small] = 1 - x[small] ** 2 / 12 + x[small] ** 4 / 240
    moderate = (x >= 1e-4) & (x < 700)
    z = np.exp(-x[moderate])
    result[moderate] = x[moderate] ** 2 * z / (-np.expm1(-x[moderate])) ** 2
    return result


def generalized_heat_capacity(omega_i, omega_j, physical_i, physical_j, *, temperature, is_classic):
    """Return the QHGK pair heat capacity ``C_ij / k_B`` for one mode block.

    ``C_ij = x_i x_j (n_i - n_j) / (x_j - x_i)`` with ``x = hbar omega / (k_B T)``;
    it equals one in the classical limit. Pairs with
    ``|x_i - x_j| <= 1e-6 max(1, |x_i|, |x_j|)`` use the mean of the two
    single-mode heat capacities, which avoids subtracting nearly equal Bose
    occupations. The result is ``float64``.
    """
    wi = np.asarray(omega_i, dtype=np.float64).reshape(-1)
    wj = np.asarray(omega_j, dtype=np.float64).reshape(-1)
    active_i, active_j = _active(wi, physical_i), _active(wj, physical_j)
    valid = active_i[:, np.newaxis] & active_j[np.newaxis, :]
    heat = np.zeros((wi.size, wj.size), dtype=np.float64)
    if is_classic:
        heat[valid] = 1
        return heat
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("quantum temperature must be finite and positive")
    xi = wi * (_HBAR_OVER_KB / temperature)
    xj = wj * (_HBAR_OVER_KB / temperature)
    ni = np.zeros_like(xi)
    nj = np.zeros_like(xj)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ni[active_i] = 1 / np.expm1(xi[active_i])
        nj[active_j] = 1 / np.expm1(xj[active_j])
    dx = xj[np.newaxis, :] - xi[:, np.newaxis]
    scale = np.maximum(1, np.maximum(np.abs(xi[:, np.newaxis]), np.abs(xj[np.newaxis, :])))
    near = np.abs(dx) <= _DEGENERACY_RTOL * scale
    np.divide(xi[:, np.newaxis] * xj[np.newaxis, :] * (ni[:, np.newaxis] - nj[np.newaxis, :]),
              dx, out=heat, where=valid & ~near)
    degenerate = valid & near
    if np.any(degenerate):
        ci = _single_mode_heat_capacity(np.maximum(xi, np.finfo(float).tiny))
        cj = _single_mode_heat_capacity(np.maximum(xj, np.finfo(float).tiny))
        heat[degenerate] = ((ci[:, np.newaxis] + cj[np.newaxis, :]) / 2)[degenerate]
    return heat

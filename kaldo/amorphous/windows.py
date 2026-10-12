"""Spectrum-window boundaries and conservative triangular halos in eigenvalue units."""

from __future__ import annotations

import numpy as np


def approximate_dos_boundaries(bin_edges, density, n_windows):
    """Choose spectral boundaries from a supplied approximate DOS histogram."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    if (edges.ndim != 1 or density.shape != (len(edges)-1,) or
            len(density) < 1 or np.any(np.diff(edges) <= 0) or
            not np.isfinite(edges).all() or not np.isfinite(density).all() or
            np.any(density < 0) or np.sum(density*np.diff(edges)) <= 0 or
            n_windows < 1):
        raise ValueError("invalid approximate DOS")
    cumulative = np.r_[0., np.cumsum(density*np.diff(edges))]
    fractions = np.linspace(0., cumulative[-1], n_windows+1)
    boundaries = np.interp(fractions, cumulative, edges)
    boundaries[0], boundaries[-1] = edges[0], edges[-1]
    return boundaries


def eigenvalue_boundaries_from_full_spectrum(eigenvalues, n_windows):
    """Optional exact mode-count planning pass; avoid splitting degeneracies."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or
            not np.isfinite(values).all() or np.any(np.diff(values) < 0) or
            not isinstance(n_windows, int) or n_windows < 1):
        raise ValueError("invalid full eigenvalue spectrum or window count")
    cuts = []
    for index in range(1, n_windows):
        target = len(values)*index//n_windows
        while target < len(values) and target > 0 and values[target] == values[target-1]:
            target += 1
        if 0 < target < len(values):
            cuts.append((values[target-1]+values[target])/2)
    scale = max(1., float(np.max(np.abs(values))))
    padding = max(1e-8, 1e-6*scale)
    return np.array([values[0]-padding, *sorted(set(cuts)), values[-1]+padding])


def eigenvalue_window_intervals(boundaries, max_bandwidth):
    """Core and halo intervals in eigenvalue units (rad/ps squared).

    Signed frequencies are monotone in eigenvalue, including unstable modes.
    A 4*maximum-bandwidth frequency halo is conservative for triangular QHGK.
    """
    edges = np.asarray(boundaries, dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 2 or
            np.any(np.diff(edges) <= 0) or not np.isfinite(edges).all() or
            not np.isfinite(max_bandwidth) or max_bandwidth < 0):
        raise ValueError("invalid eigenvalue boundaries or maximum bandwidth")
    frequency = np.sign(edges)*np.sqrt(np.abs(edges))
    reach = 4*max_bandwidth
    lower = frequency[:-1]-reach
    upper = frequency[1:]+reach
    # Interval eigensolvers may treat an endpoint as open. Leave a small
    # outward numerical margin so a mode exactly on a DOS/core endpoint is
    # present in at least one halo, even when the linewidth bound is zero.
    margin = 1e-8*max(1., float(np.max(np.abs(edges))))
    halo_lower = np.sign(lower)*lower**2 - margin
    halo_upper = np.sign(upper)*upper**2 + margin
    return [(float(a), float(b), float(c), float(d))
            for a,b,c,d in zip(edges[:-1], edges[1:], halo_lower, halo_upper)]

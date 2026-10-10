"""Exact triangular core/halo planning from frequencies and optional DOS bins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModeWindow:
    core_start: int
    core_stop: int
    halo_start: int
    halo_stop: int

    @property
    def local_core(self):
        return self.core_start - self.halo_start, self.core_stop - self.halo_start


def plan_mode_windows(omega, bandwidth, n_windows):
    """Partition sorted modes and include every possible triangular partner.

    Adjacent exactly degenerate frequencies stay in the same core. The halo
    bound uses the maximum bandwidth in the core and the global maximum; it
    may include extra modes but cannot omit a nonzero triangular pair.
    """
    w, b = np.asarray(omega, dtype=np.float64), np.asarray(bandwidth, dtype=np.float64)
    if (w.ndim != 1 or b.shape != w.shape or not len(w) or
            np.any(np.diff(w) < 0) or not np.isfinite(w).all() or
            not np.isfinite(b).all() or np.any(b < 0) or
            not isinstance(n_windows, int) or n_windows < 1):
        raise ValueError("invalid sorted frequencies, bandwidths, or window count")
    cuts = [0]
    for index in range(1, n_windows):
        target = len(w)*index//n_windows
        while target < len(w) and target > 0 and w[target] == w[target-1]:
            target += 1
        if cuts[-1] < target < len(w):
            cuts.append(target)
    cuts.append(len(w))
    windows = []
    global_max = float(np.max(b))
    for start, stop in zip(cuts[:-1], cuts[1:]):
        reach = 2*(float(np.max(b[start:stop])) + global_max)
        left = int(np.searchsorted(w, w[start]-reach, side="right"))
        right = int(np.searchsorted(w, w[stop-1]+reach, side="left"))
        # The core itself must remain in the halo even for zero bandwidth.
        windows.append(ModeWindow(start, stop, min(start, left), max(stop, right)))
    return windows


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


def frequency_window_intervals(boundaries, max_bandwidth):
    """Return half-open core intervals with conservative triangular halos."""
    boundaries = np.asarray(boundaries, dtype=np.float64)
    if (boundaries.ndim != 1 or len(boundaries) < 2 or
            np.any(np.diff(boundaries) <= 0) or
            not np.isfinite(boundaries).all() or
            not np.isfinite(max_bandwidth) or max_bandwidth < 0):
        raise ValueError("invalid frequency boundaries or maximum bandwidth")
    reach = 4*max_bandwidth
    return [(float(a), float(b), float(a-reach), float(b+reach))
            for a,b in zip(boundaries[:-1], boundaries[1:])]


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

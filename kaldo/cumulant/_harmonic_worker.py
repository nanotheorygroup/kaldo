"""Phase-1 frequency worker: ``HarmonicWithQ`` over a chunk of q-points.

Spawn workers import this module to unpickle ``freq_q_chunk``. TensorFlow
comes in through ``HarmonicWithQ``; that is the same Phonons dynmat path
used serially, just sharded across ``n_workers``.
"""
from __future__ import annotations

import numpy as np

_STATE = {}


def init_freq_worker(n_threads, second, hwq_kwargs):
    """Process-pool initializer: thread cap + pickled ``SecondOrder``."""
    from kaldo.parallel.executor import _init_worker_thread_caps
    _init_worker_thread_caps(int(n_threads))
    _STATE["second"] = second
    _STATE["kwargs"] = hwq_kwargs


def freq_q_chunk(q_points):
    """Frequencies (n_q_chunk, n_modes) in THz for ``q_points``."""
    from kaldo.observables.harmonic_with_q import HarmonicWithQ

    second = _STATE["second"]
    kw = _STATE["kwargs"]
    q_points = np.asarray(q_points)
    n_modes = int(second.n_modes)
    out = np.empty((q_points.shape[0], n_modes), dtype=np.float64)
    for i, q in enumerate(q_points):
        hwq = HarmonicWithQ(q_point=q, second=second, **kw)
        out[i] = np.asarray(hwq.frequency).reshape(-1)
    return out

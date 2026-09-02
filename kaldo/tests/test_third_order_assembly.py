"""Regression tests for the third-order scratch assembly.

The assembly must produce the same tensor through its dense and COO routes,
and it must never remove scratch files before the tensor exists: chunk files
are hours of force evaluations, and a crash mid-assembly used to destroy
them.
"""
import os

import numpy as np
import pytest

from kaldo.controllers.displacement import _assemble_from_scratch_third

N_ATOMS = 4
N_REPLICAS = 1
N3 = N_REPLICAS * N_ATOMS * 3
SHAPE_DENSE = (N_ATOMS * 3, N3, N3)


def _write_chunks(scratch_dir, seed=0):
    """Synthetic per-atom chunk files.

    Coordinates are globally unique per atom, as in production: the compute
    loop visits each (iat, jat) pair exactly once and chunks partition the
    jat range, so no coordinate ever repeats across an atom's chunks.
    """
    rng = np.random.default_rng(seed)
    dense = np.zeros(SHAPE_DENSE)
    for iat in range(N_ATOMS):
        flat = rng.choice(3 * N3 * N3, size=80, replace=False)
        for chunk_id in range(2):
            sl = flat[chunk_id * 40:(chunk_id + 1) * 40]
            alpha, rest = np.divmod(sl, N3 * N3)
            jflat, k = np.divmod(rest, N3)
            jat, beta = np.divmod(jflat, 3)
            values = rng.normal(size=len(sl))
            dense[iat * 3 + alpha, jat * 3 + beta, k] = values
            coords = np.stack([np.full(len(sl), iat), alpha, jat, beta, k])
            np.savez(os.path.join(scratch_dir, f'iat_{iat:05d}_chunk_{chunk_id:04d}.npz'),
                     coords=coords, values=values)
        open(os.path.join(scratch_dir, f'iat_{iat:05d}.done'), 'w').close()
    return dense


def test_assembly_matches_dense_reference(tmp_path):
    scratch = str(tmp_path / 'third_order')
    os.makedirs(scratch)
    expected = _write_chunks(scratch)
    got = _assemble_from_scratch_third(scratch, N_ATOMS, N_REPLICAS,
                                       keep_scratch=True)
    assert got.shape == SHAPE_DENSE
    np.testing.assert_array_equal(got.todense(), expected)


def test_scratch_survives_failed_assembly(tmp_path):
    scratch = str(tmp_path / 'third_order')
    os.makedirs(scratch)
    _write_chunks(scratch)
    before = sorted(os.listdir(scratch))
    # Corrupt one chunk: assembly must raise without having removed anything.
    victim = os.path.join(scratch, f'iat_{N_ATOMS - 1:05d}_chunk_0001.npz')
    with open(victim, 'wb') as fh:
        fh.write(b'not a real npz')
    with pytest.raises(Exception):
        _assemble_from_scratch_third(scratch, N_ATOMS, N_REPLICAS,
                                     keep_scratch=False)
    assert sorted(os.listdir(scratch)) == before


def test_scratch_removed_after_success(tmp_path):
    scratch = str(tmp_path / 'third_order')
    os.makedirs(scratch)
    _write_chunks(scratch)
    _assemble_from_scratch_third(scratch, N_ATOMS, N_REPLICAS,
                                 keep_scratch=False)
    assert not os.path.exists(scratch)

"""
Tests for ``kaldo.parallel.get_executor``:

1. Worker processes are capped to one OpenMP / MKL / OpenBLAS thread by
   default, so internally-multithreaded calculators (PyNEP, numpy+MKL,
   torch CPU) don't oversubscribe when ``n_workers > 1``.
2. User-set env vars are respected (not clobbered by the cap).
3. ``worker_threads=None`` opts out of the cap.
"""

import os

import pytest

from kaldo.parallel import get_executor


def _report_thread_caps(_=None):
    """Pickle-friendly worker task: return the worker's thread-cap env vars."""
    return {
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
    }


def test_default_worker_caps_threads_to_one():
    with get_executor(backend='process', n_workers=2) as executor:
        result = executor.submit(_report_thread_caps).result()
    assert result['OMP_NUM_THREADS'] == '1'
    assert result['MKL_NUM_THREADS'] == '1'
    assert result['OPENBLAS_NUM_THREADS'] == '1'


def test_worker_threads_override():
    with get_executor(backend='process', n_workers=2, worker_threads=4) as executor:
        result = executor.submit(_report_thread_caps).result()
    assert result['OMP_NUM_THREADS'] == '4'
    assert result['MKL_NUM_THREADS'] == '4'


def test_worker_threads_none_respects_caller_env(monkeypatch):
    """With worker_threads=None and OMP_NUM_THREADS set in the caller,
    the worker should inherit the caller's value."""
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    with get_executor(backend='process', n_workers=2, worker_threads=None) as executor:
        result = executor.submit(_report_thread_caps).result()
    assert result['OMP_NUM_THREADS'] == '8'


def test_worker_threads_none_warns_when_omp_unset(monkeypatch):
    """Opting out of the cap with OMP unset is the footgun path — warn loudly."""
    monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
    with pytest.warns(RuntimeWarning, match='OMP_NUM_THREADS'):
        executor = get_executor(backend='process', n_workers=2, worker_threads=None)
    executor.shutdown(wait=True)


def test_default_cap_does_not_override_caller_env(monkeypatch):
    """User-set OMP_NUM_THREADS must survive the worker initializer."""
    monkeypatch.setenv('OMP_NUM_THREADS', '6')
    with get_executor(backend='process', n_workers=2) as executor:
        result = executor.submit(_report_thread_caps).result()
    assert result['OMP_NUM_THREADS'] == '6'

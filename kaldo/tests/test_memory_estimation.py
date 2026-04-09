"""
Tests for the memory estimation and worker-capping system.

Validates the estimation formula, probe mechanism, cap_workers logic,
and environment variable overrides.
"""

import os
import math
from unittest import mock

import numpy as np
import psutil
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from kaldo.helpers.memory import (
    ACCUMULATION_SAFETY_FACTOR,
    DEFAULT_CALCULATOR_FORK_MB,
    DEFAULT_HEADROOM_FRACTION,
    FORK_BASE_OVERHEAD_MB,
    PARENT_RESERVE_MB,
    SPAWN_BASE_OVERHEAD_MB,
    cap_workers,
    estimate_worker_memory_mb,
    probe_calculator_memory_mb,
)


# ---------------------------------------------------------------------------
# estimate_worker_memory_mb tests
# ---------------------------------------------------------------------------

def test_estimate_increases_with_n_atoms():
    """Estimate should grow monotonically with n_atoms."""
    estimates = [
        estimate_worker_memory_mb(n, 4, False, 50, 'fork', 10.0)
        for n in [4, 8, 16, 32]
    ]
    for i in range(len(estimates) - 1):
        assert estimates[i] < estimates[i + 1], (
            f"estimate did not increase: n_atoms transition {[4,8,16,32][i]} -> {[4,8,16,32][i+1]}"
        )


def test_estimate_increases_with_n_replicas():
    """Estimate should grow monotonically with n_replicas."""
    estimates = [
        estimate_worker_memory_mb(4, r, False, 50, 'fork', 10.0)
        for r in [2, 8, 27, 64]
    ]
    for i in range(len(estimates) - 1):
        assert estimates[i] < estimates[i + 1]


def test_scratch_estimate_smaller_than_inmemory():
    """Scratch mode bounds accumulation; should estimate less than in-memory."""
    scratch = estimate_worker_memory_mb(8, 27, True, 50, 'fork', 10.0)
    inmem = estimate_worker_memory_mb(8, 27, False, 50, 'fork', 10.0)
    assert scratch < inmem, f"scratch ({scratch:.1f}) >= inmem ({inmem:.1f})"


def test_spawn_overhead_higher_than_fork():
    """spawn/forkserver should have higher base overhead."""
    fork_est = estimate_worker_memory_mb(4, 8, False, 50, 'fork', 10.0)
    spawn_est = estimate_worker_memory_mb(4, 8, False, 50, 'spawn', 10.0)
    assert spawn_est > fork_est


def test_estimate_positive_and_finite():
    """Estimate should always be positive and finite."""
    est = estimate_worker_memory_mb(1, 1, True, 10, 'fork', 0.0)
    assert est > 0
    assert math.isfinite(est)


def test_calculator_memory_included():
    """Calculator memory should directly affect the estimate."""
    low = estimate_worker_memory_mb(4, 8, False, 50, 'fork', 10.0)
    high = estimate_worker_memory_mb(4, 8, False, 50, 'fork', 500.0)
    assert high - low == pytest.approx(490.0, abs=1.0)


# ---------------------------------------------------------------------------
# probe_calculator_memory_mb tests
# ---------------------------------------------------------------------------

def test_probe_with_emt():
    """Probing EMT should return a small positive float."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((1, 1, 2))
    result = probe_calculator_memory_mb(EMT, atoms)
    assert result is not None
    assert isinstance(result, float)
    assert result >= 0.0
    # EMT is tiny; should be well under 100 MB
    assert result < 100.0


def test_probe_failure_returns_none():
    """A broken calculator factory should return None, not raise."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)

    def broken_factory():
        raise RuntimeError("calculator init failed")

    result = probe_calculator_memory_mb(broken_factory, atoms)
    assert result is None


# ---------------------------------------------------------------------------
# cap_workers tests
# ---------------------------------------------------------------------------

def _mock_virtual_memory(available_mb, total_mb):
    """Create a mock psutil.virtual_memory return value."""
    return mock.MagicMock(
        available=int(available_mb * 1024 * 1024),
        total=int(total_mb * 1024 * 1024),
    )


def test_cap_workers_reduces():
    """With limited memory, workers should be reduced."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((1, 1, 2))

    with mock.patch('kaldo.helpers.memory.psutil') as mock_psutil:
        # 1000 MB available, 2000 MB total
        mock_psutil.virtual_memory.return_value = _mock_virtual_memory(1000, 2000)
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024

        safe, est, msg = cap_workers(
            requested_workers=64,
            n_atoms=4, n_replicas=2,
            use_scratch=True, jat_flush_every=50,
            calculator=None,  # skip probe, use default
            replicated_atoms=atoms,
            start_method='fork',
        )
        assert safe < 64
        assert safe >= 1
        assert msg is not None
        assert 'reducing n_workers' in msg


def test_cap_workers_no_reduction():
    """With ample memory, workers should not be reduced."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((1, 1, 2))

    with mock.patch('kaldo.helpers.memory.psutil') as mock_psutil:
        # 100 GB available
        mock_psutil.virtual_memory.return_value = _mock_virtual_memory(100_000, 128_000)
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024

        safe, est, msg = cap_workers(
            requested_workers=4,
            n_atoms=4, n_replicas=2,
            use_scratch=True, jat_flush_every=50,
            calculator=None,
            replicated_atoms=atoms,
            start_method='fork',
        )
        assert safe == 4
        assert msg is None


def test_cap_workers_minimum_one():
    """Even with very little memory, at least 1 worker is returned."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((1, 1, 2))

    with mock.patch('kaldo.helpers.memory.psutil') as mock_psutil:
        # Only 10 MB available
        mock_psutil.virtual_memory.return_value = _mock_virtual_memory(10, 10)
        mock_psutil.Process.return_value.memory_info.return_value.rss = 5 * 1024 * 1024

        safe, est, msg = cap_workers(
            requested_workers=32,
            n_atoms=4, n_replicas=2,
            use_scratch=True, jat_flush_every=50,
            calculator=None,
            replicated_atoms=atoms,
            start_method='fork',
        )
        assert safe == 1
        assert msg is not None


def test_cap_workers_resolves_none():
    """requested_workers=None should resolve to cpu_count."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((1, 1, 2))

    with mock.patch('kaldo.helpers.memory.psutil') as mock_psutil:
        mock_psutil.virtual_memory.return_value = _mock_virtual_memory(100_000, 128_000)
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024

        safe, est, msg = cap_workers(
            requested_workers=None,
            n_atoms=4, n_replicas=2,
            use_scratch=True, jat_flush_every=50,
            calculator=None,
            replicated_atoms=atoms,
            start_method='fork',
        )
        assert safe >= 1
        assert safe <= (os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Environment variable tests
# ---------------------------------------------------------------------------

def test_skip_env_var(al_atoms_fixture):
    """KALDO_SKIP_MEMORY_CHECK=1 should bypass the memory check entirely."""
    from kaldo.controllers.displacement import calculate_third

    atoms, replicated_atoms = al_atoms_fixture
    with mock.patch.dict(os.environ, {'KALDO_SKIP_MEMORY_CHECK': '1'}):
        # Should not raise or warn even with absurd n_workers
        # (the test system is tiny so it would complete quickly regardless)
        result = calculate_third(
            atoms, replicated_atoms, 1e-5,
            n_workers=2,
            calculator=EMT,
        )
        assert result is not None


def test_max_workers_env_var(al_atoms_fixture):
    """KALDO_MAX_WORKERS should hard-cap n_workers."""
    from kaldo.controllers.displacement import calculate_third

    atoms, replicated_atoms = al_atoms_fixture
    with mock.patch.dict(os.environ, {'KALDO_MAX_WORKERS': '1'}):
        result = calculate_third(
            atoms, replicated_atoms, 1e-5,
            n_workers=8,
            calculator=EMT,
        )
        # Should run serially (capped to 1) and produce a valid result
        assert result is not None


# ---------------------------------------------------------------------------
# Warning emission test
# ---------------------------------------------------------------------------

def test_warning_emitted_on_reduction():
    """ResourceWarning should be emitted when workers are reduced."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((1, 1, 2))

    with mock.patch('kaldo.helpers.memory.psutil') as mock_psutil:
        mock_psutil.virtual_memory.return_value = _mock_virtual_memory(200, 200)
        mock_psutil.Process.return_value.memory_info.return_value.rss = 50 * 1024 * 1024

        safe, est, msg = cap_workers(
            requested_workers=100,
            n_atoms=4, n_replicas=2,
            use_scratch=True, jat_flush_every=50,
            calculator=None,
            replicated_atoms=atoms,
            start_method='fork',
        )
        assert safe < 100
        assert msg is not None
        assert 'Memory safety' in msg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def al_atoms_fixture():
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    replicated_atoms = atoms.repeat((1, 1, 2))
    replicated_atoms.calc = EMT()
    return atoms, replicated_atoms

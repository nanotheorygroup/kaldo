"""
Tests for the memory estimation and worker-resolution system.

Relevant code:
 kaldo/helpers/memory.py - resolve_n_workers, estimate_worker_memory_mb,
                           probe_calculator_memory_mb
"""

import logging as _logging
import os
from unittest import mock

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from kaldo.helpers.memory import (
    estimate_worker_memory_mb,
    probe_calculator_memory_mb,
    resolve_n_workers,
)


@pytest.fixture(scope="module")
def al_atoms_fixture():
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    replicated_atoms = atoms.repeat((1, 1, 2))
    replicated_atoms.calc = EMT()
    return atoms, replicated_atoms


# ---------------------------------------------------------------------------
# estimate_worker_memory_mb
# ---------------------------------------------------------------------------

def test_scratch_estimate_smaller_than_inmemory():
    """Scratch mode bounds accumulation; should estimate less than in-memory."""
    scratch = estimate_worker_memory_mb(8, 27, True, 10.0, 50)
    inmem = estimate_worker_memory_mb(8, 27, False, 10.0, 50)
    assert scratch < inmem, f"scratch ({scratch:.1f}) >= inmem ({inmem:.1f})"


# ---------------------------------------------------------------------------
# probe_calculator_memory_mb
# ---------------------------------------------------------------------------

def test_probe_with_emt():
    """Probing EMT should return a small non-negative float."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((1, 1, 2))
    result = probe_calculator_memory_mb(EMT, atoms)
    assert isinstance(result, float)
    assert result >= 0.0
    assert result < 100.0  # EMT is tiny


def test_probe_failure_propagates():
    """A broken calculator factory should raise, not silently return None."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)

    def broken_factory():
        raise RuntimeError("calculator init failed")

    with pytest.raises(RuntimeError, match="calculator init failed"):
        probe_calculator_memory_mb(broken_factory, atoms)


# ---------------------------------------------------------------------------
# resolve_n_workers
# ---------------------------------------------------------------------------

def _mock_virtual_memory(available_mb, total_mb):
    return mock.MagicMock(
        available=int(available_mb * 1024 * 1024),
        total=int(total_mb * 1024 * 1024),
    )


def test_resolve_returns_request_when_safe():
    """Explicit request within the safe limit should pass through unchanged."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((1, 1, 2))

    with mock.patch('kaldo.helpers.memory.psutil') as mock_psutil:
        mock_psutil.virtual_memory.return_value = _mock_virtual_memory(100_000, 128_000)
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024

        result = resolve_n_workers(
            requested_workers=4,
            n_atoms=4, n_replicas=2,
            use_scratch=True, jat_flush_every=50,
            calculator=EMT,
            replicated_atoms=atoms,
        )
        assert result == 4


def test_resolve_raises_when_explicit_exceeds_safe():
    """Explicit request above the safe limit must raise MemoryError."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((1, 1, 2))

    with mock.patch('kaldo.helpers.memory.psutil') as mock_psutil:
        mock_psutil.virtual_memory.return_value = _mock_virtual_memory(1000, 2000)
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024

        with pytest.raises(MemoryError) as excinfo:
            resolve_n_workers(
                requested_workers=64,
                n_atoms=4, n_replicas=2,
                use_scratch=True, jat_flush_every=50,
                calculator=EMT,
                replicated_atoms=atoms,
            )

    msg = str(excinfo.value)
    assert 'n_workers=64' in msg
    assert 'n_atoms=4' in msg
    assert 'MB/worker' in msg


def test_resolve_raises_in_tight_memory():
    """Even very low memory with a large request should raise MemoryError."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((1, 1, 2))

    with mock.patch('kaldo.helpers.memory.psutil') as mock_psutil:
        mock_psutil.virtual_memory.return_value = _mock_virtual_memory(10, 10)
        mock_psutil.Process.return_value.memory_info.return_value.rss = 5 * 1024 * 1024

        with pytest.raises(MemoryError):
            resolve_n_workers(
                requested_workers=32,
                n_atoms=4, n_replicas=2,
                use_scratch=True, jat_flush_every=50,
                calculator=EMT,
                replicated_atoms=atoms,
            )


def test_resolve_auto_selects_when_none(caplog):
    """n_workers=None should auto-pick and log the choice with atom count."""
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True).repeat((1, 1, 2))

    with mock.patch('kaldo.helpers.memory.psutil') as mock_psutil:
        mock_psutil.virtual_memory.return_value = _mock_virtual_memory(100_000, 128_000)
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024

        with caplog.at_level(_logging.INFO, logger='kaldo'):
            result = resolve_n_workers(
                requested_workers=None,
                n_atoms=4, n_replicas=2,
                use_scratch=True, jat_flush_every=50,
                calculator=EMT,
                replicated_atoms=atoms,
            )

    assert result >= 1
    assert result <= (os.cpu_count() or 1)
    auto_log = [rec.message for rec in caplog.records if 'Auto-selecting' in rec.message]
    assert auto_log, 'Expected an auto-selection INFO log'
    assert 'n_atoms=4' in auto_log[0]

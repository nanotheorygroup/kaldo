"""Tests for GPU-pinned parallel execution in kaldo.parallel.executor.

All tests are mock-only and require no CUDA hardware. The race-freedom test
in Task 5 actually spawns worker processes and inspects os.environ, but still
needs no GPU.
"""

import os
import sys
import warnings

import pytest

from kaldo.parallel import get_executor


def test_gpu_ids_validates_type():
    with pytest.raises(TypeError, match="gpu_ids must be a list"):
        get_executor(backend='process', gpu_ids='0,1')


def test_gpu_ids_validates_element_type():
    with pytest.raises(TypeError, match="gpu_ids must be a list of int"):
        get_executor(backend='process', gpu_ids=['0', '1'])


def test_gpu_ids_validates_duplicates():
    with pytest.raises(ValueError, match="unique"):
        get_executor(backend='process', gpu_ids=[0, 0])


def test_gpu_ids_validates_negative():
    with pytest.raises(ValueError, match="non-negative"):
        get_executor(backend='process', gpu_ids=[-1])


def test_process_empty_gpu_ids_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        get_executor(backend='process', gpu_ids=[])


def test_mpi_rejects_gpu_ids():
    with pytest.raises(NotImplementedError, match="gpu_ids is not supported"):
        get_executor(backend='mpi', gpu_ids=[0])


def test_mpi_rejects_empty_gpu_ids():
    with pytest.raises(NotImplementedError, match="gpu_ids is not supported"):
        get_executor(backend='mpi', gpu_ids=[])

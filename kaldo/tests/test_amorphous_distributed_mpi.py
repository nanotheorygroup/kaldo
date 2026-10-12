"""Row-distributed amorphous QHGK under mpiexec: rank parity, restarts, pipeline, CLI.

Each test launches ``kaldo.tests.mpi_distributed_qhgk_case``, whose ranks
assert against the serial reducer and exit nonzero on mismatch. Select or skip
these tests with ``pytest -m mpi`` / ``pytest -m "not mpi"``.
"""

from functools import lru_cache
import importlib.util
import shutil
import subprocess
import sys

import numpy as np
import pytest

_MPIEXEC = shutil.which("mpiexec")

pytestmark = [
    pytest.mark.mpi,
    pytest.mark.skipif(_MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
                       reason="requires mpiexec and mpi4py (pip install kaldo[mpi])"),
]


@lru_cache(maxsize=1)
def _is_open_mpi():
    version = subprocess.run([_MPIEXEC, "--version"], capture_output=True, text=True).stdout
    return "Open MPI" in version or "OpenRTE" in version


def _launch(ranks, directory, phase, *, timeout=300):
    command = [_MPIEXEC, "-n", str(ranks)]
    if _is_open_mpi():
        # Rank-count parity is the point of these tests, not core count.
        command.append("--oversubscribe")
    command += [sys.executable, "-m", "kaldo.tests.mpi_distributed_qhgk_case", str(directory), phase]
    subprocess.run(command, check=True, timeout=timeout)


@pytest.fixture(scope="module")
def serial_conductivity(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ranks1")
    _launch(1, directory, "resume")
    return np.load(directory / "distributed_results/conductivity.npy")


@pytest.mark.parametrize("ranks", [2, 4])
def test_interrupted_run_resumes_and_matches_one_rank(tmp_path, serial_conductivity, ranks):
    _launch(ranks, tmp_path, "partial")
    _launch(ranks, tmp_path, "resume")
    np.testing.assert_allclose(np.load(tmp_path / "distributed_results/conductivity.npy"),
                               serial_conductivity, rtol=1e-12, atol=1e-13)


def test_interval_pipeline_resumes_to_full_spectrum(tmp_path):
    _launch(2, tmp_path, "pipeline_partial")
    _launch(2, tmp_path, "pipeline_resume")


def test_cli_shard_and_reduce_match_serial(tmp_path):
    _launch(2, tmp_path, "cli")


def test_broadening_shapes_and_mixed_precision_match_serial(tmp_path):
    _launch(2, tmp_path, "shapes")


def test_corrupt_checkpoint_is_rejected_on_every_rank(tmp_path):
    _launch(2, tmp_path, "corrupt", timeout=60)

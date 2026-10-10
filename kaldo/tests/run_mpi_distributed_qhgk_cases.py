"""Run row-distributed MPI parity and restart cases with 1, 2, and 4 ranks."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    with tempfile.TemporaryDirectory(prefix="kaldo-distributed-qhgk-") as scratch:
        results = {}
        for ranks in (1, 2, 4):
            directory = Path(scratch) / f"ranks{ranks}"
            phases = ("partial", "resume") if ranks > 1 else ("resume",)
            for phase in phases:
                subprocess.run(["mpiexec", "-n", str(ranks), sys.executable,
                                "-m", "kaldo.tests.mpi_distributed_qhgk_case",
                                str(directory), phase], check=True)
            results[ranks] = np.load(directory / "distributed_results/conductivity.npy")
        for ranks in (2, 4):
            np.testing.assert_allclose(results[ranks], results[1], rtol=1e-12,
                                       atol=1e-13)
        subprocess.run(["mpiexec", "-n", "2", sys.executable, "-m",
                        "kaldo.tests.mpi_distributed_qhgk_case",
                        str(Path(scratch) / "windows2"), "window"], check=True)
        pipeline_directory = Path(scratch) / "pipeline2"
        for phase in ("pipeline_partial", "pipeline_resume"):
            subprocess.run(["mpiexec", "-n", "2", sys.executable, "-m",
                            "kaldo.tests.mpi_distributed_qhgk_case",
                            str(pipeline_directory), phase], check=True)
        subprocess.run(["mpiexec", "-n", "2", sys.executable, "-m",
                        "kaldo.tests.mpi_distributed_qhgk_case",
                        str(Path(scratch) / "cli2"), "cli"], check=True)
        subprocess.run(["mpiexec", "-n", "2", sys.executable, "-m",
                        "kaldo.tests.mpi_distributed_qhgk_case",
                        str(Path(scratch) / "shapes2"), "shapes"], check=True)
        subprocess.run(["mpiexec", "-n", "2", sys.executable, "-m",
                        "kaldo.tests.mpi_distributed_qhgk_case",
                        str(Path(scratch) / "corrupt2"), "corrupt"],
                       check=True, timeout=30)
    print("Row-distributed QHGK matches serial for 1, 2, 4 ranks and restarts")


if __name__ == "__main__":
    main()

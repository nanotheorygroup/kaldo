"""Run the opt-in local MPI rank and restart integration checks."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    with tempfile.TemporaryDirectory(prefix="kaldo-mpi-qhgk-") as scratch:
        scratch = Path(scratch)
        results = {}
        for ranks in (1, 2, 4):
            directory = scratch / f"ranks{ranks}"
            phases = ("partial", "resume") if ranks > 1 else ("resume",)
            for phase in phases:
                command = ["mpiexec", "-n", str(ranks), sys.executable, "-m",
                           "kaldo.tests.mpi_amorphous_qhgk_case", str(directory), phase]
                subprocess.run(command, check=True)
            manifest = json.loads((directory / "results/manifest.json").read_text())
            assert manifest["complete"] and manifest["rank_count"] == ranks
            assert len(manifest["completed_core_ranges"]) == 6
            results[ranks] = np.load(directory / "results/conductivity.npy")
            assert manifest["evaluated_pairs"] > 0
            print(f"rank_count={ranks} completed={len(manifest['completed_core_ranges'])} "
                  f"evaluated_pairs={manifest['evaluated_pairs']} "
                  f"peak_rss_kib_per_rank={[item['peak_rss_kib'] for item in manifest['metrics_per_rank']]}",
                  flush=True)
        shard_directory = scratch / "sharded_ranks2"
        subprocess.run(["mpiexec", "-n", "2", sys.executable, "-m",
                        "kaldo.tests.mpi_amorphous_qhgk_case", str(shard_directory),
                        "resume", "sharded"], check=True)
        results["sharded"] = np.load(shard_directory / "results/conductivity.npy")
        for ranks in (2, 4):
            np.testing.assert_allclose(results[ranks], results[1], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(results["sharded"], results[1], rtol=1e-13, atol=1e-13)
        print("MPI ranks 1, 2, 4, sharded panels, and interrupted restarts match serial")


if __name__ == "__main__":
    main()

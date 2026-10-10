"""Opt-in MPI integration case: mpiexec -n 1/2/4 python -m kaldo.tests.mpi_amorphous_qhgk_case DIR [partial|resume]."""

import json
import sys
from pathlib import Path

from mpi4py import MPI
import numpy as np

from kaldo.amorphous import (DirectBandwidth, NpyEigensystemSource,
                             ShardedEigenvectorSource, SparseAmorphousHarmonic,
                             calculate_qhgk_isotropic_blocked)
from kaldo.amorphous.mpi import run_mpi_qhgk


def write_input(root):
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "eigensystem.npy").exists():
        rng = np.random.default_rng(123)
        n = 8
        positions = rng.normal(size=(n, 3))
        atom_blocks = np.zeros((n, n, 3, 3))
        for i in range(n):
            for j in range(i + 1, n):
                a = rng.normal(size=(3, 3)) * 0.03
                a = (a + a.T) / 2
                atom_blocks[i, j] = a
                atom_blocks[j, i] = a.T
        for i in range(n):
            atom_blocks[i, i] = -sum(atom_blocks[i, j] for j in range(n) if j != i)
        dense = atom_blocks.transpose(0, 2, 1, 3).reshape(3*n, 3*n)
        _, e = np.linalg.eigh(dense)
        omega = np.arange(1, 3*n + 1, dtype=np.float64) * 0.1
        np.save(root / "eigensystem.npy", np.vstack([omega**2, e]))
        np.save(root / "omega.npy", omega)
        np.save(root / "physical.npy", np.ones(len(omega), bool))
        columns, blocks, displacement, ptr = [], [], [], [0]
        for i in range(n):
            for j in range(n):
                columns.append(j)
                blocks.append(atom_blocks[i, j])
                displacement.append(positions[i] - positions[j])
            ptr.append(len(columns))
        SparseAmorphousHarmonic(
            np.array(ptr), np.array(columns), np.array(blocks),
            np.array(displacement)).save(root / "sparse")

def main():
    root = Path(sys.argv[1])
    phase = sys.argv[2] if len(sys.argv) > 2 else "resume"
    use_shards = len(sys.argv) > 3 and sys.argv[3] == "sharded"
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        write_input(root)
        if use_shards and not (root / "shards/manifest.json").exists():
            ShardedEigenvectorSource.create(
                NpyEigensystemSource(root / "eigensystem.npy"), root / "shards",
                shard_size=5, dtype=np.float64)
    comm.Barrier()
    omega = np.load(root / "omega.npy")
    source = (ShardedEigenvectorSource(root / "shards") if use_shards else
              NpyEigensystemSource(root / "eigensystem.npy"))
    harmonic = SparseAmorphousHarmonic.load(root / "sparse")
    provider = DirectBandwidth(0.035)
    result = run_mpi_qhgk(
        omega, provider, np.ones(len(omega), bool), 300., False, 200., source,
        harmonic, root / "results", block_size=4, compute_dtype=np.float64,
        comm=comm, max_blocks=1 if phase == "partial" else None)
    if comm.rank == 0:
        manifest = json.loads((root / "results/manifest.json").read_text())
        if phase == "partial":
            assert not manifest["complete"]
            print("partial", len(manifest["completed_core_ranges"]))
        else:
            assert manifest["complete"]
            expected = calculate_qhgk_isotropic_blocked(
                omega, provider.evaluate(omega, 300.), np.ones(len(omega), bool),
                300., False, 200., source, harmonic.operators,
                block_size=4, compute_dtype=np.float64)
            for actual, reference in zip(result, expected):
                np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-13)
            print("complete", comm.size, manifest["evaluated_pairs"],
                  float(np.sum(result[0])))


if __name__ == "__main__":
    main()

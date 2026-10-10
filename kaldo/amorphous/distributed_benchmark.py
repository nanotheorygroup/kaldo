"""Opt-in coordinate-row MPI memory, communication, and pair-work probe.

Example: ``mpiexec -n 2 python -m kaldo.amorphous.distributed_benchmark
--atoms 32 64 --block 32 --output /tmp/qhgk-distributed.json``.
"""

import argparse
import json
from pathlib import Path
import tempfile
from time import perf_counter

from mpi4py import MPI
import numpy as np

from .bandwidth import DirectBandwidth
from .checkpoint import RowShardedEigenvectorSource
from .distributed import DistributedSparseAmorphousHarmonic, run_distributed_qhgk
from .qhgk import NpyEigensystemSource
from .sparse_harmonic import SparseAmorphousHarmonic


def _write_case(directory, n_atoms):
    directory.mkdir(parents=True, exist_ok=True)
    m = 3*n_atoms
    rng = np.random.default_rng(n_atoms)
    eigenvectors = np.linalg.qr(rng.normal(size=(m,m)))[0].astype(np.float32)
    omega = np.linspace(0., 12., m)
    np.save(directory / "eigensystem.npy", np.vstack([omega**2, eigenvectors]))
    np.save(directory / "omega.npy", omega)
    ptr, column, blocks, displacement = [0], [], [], []
    for atom in range(n_atoms):
        for neighbor in range(max(0, atom-1), min(n_atoms, atom+2)):
            column.append(neighbor)
            blocks.append(np.eye(3, dtype=np.float32)*(2 if atom == neighbor else -1))
            displacement.append([float(atom-neighbor), 0., 0.])
        ptr.append(len(column))
    SparseAmorphousHarmonic(np.array(ptr), np.array(column), np.array(blocks),
                            np.array(displacement, dtype=np.float32)).save(directory / "harmonic")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    comm = MPI.COMM_WORLD
    if any(n < comm.size for n in args.atoms) or args.block < 1:
        parser.error("each atom count must be at least the rank count; block must be positive")
    if comm.rank == 0:
        scratch = tempfile.TemporaryDirectory(prefix="kaldo-distributed-probe-")
        root = scratch.name
    else:
        scratch, root = None, None
    root = Path(comm.bcast(root, root=0))
    records = []
    for n_atoms in args.atoms:
        directory = root / str(n_atoms)
        if comm.rank == 0:
            _write_case(directory, n_atoms)
        comm.Barrier()
        source = RowShardedEigenvectorSource.create(
            NpyEigensystemSource(directory / "eigensystem.npy"),
            directory / "row_shards", comm, panel_size=args.block,
            dtype=np.float32)
        harmonic = DistributedSparseAmorphousHarmonic(directory / "harmonic", comm)
        omega = np.load(directory / "omega.npy")
        start = perf_counter()
        result = run_distributed_qhgk(
            omega, DirectBandwidth(0.04), omega > 0, 300., False, 1000.,
            source, harmonic, directory / "results", block_size=args.block)
        wall = perf_counter()-start
        walls = comm.gather(wall, root=0)
        if comm.rank == 0:
            manifest = json.loads((directory / "results/manifest.json").read_text())
            records.append({"atoms": n_atoms, "modes": len(omega),
                            "rank_count": comm.size, "block_size": args.block,
                            "evaluated_pairs": manifest["evaluated_pairs"],
                            "total_ordered_pairs": len(omega)**2,
                            "max_wall_seconds": max(walls),
                            "total_conductivity": float(result[0].sum()),
                            "metrics_per_rank": manifest["metrics_per_rank"]})
        comm.Barrier()
    if comm.rank == 0:
        report = {"kind": "synthetic_coordinate_row_distributed_probe",
                  "measurements": records}
        encoded = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(encoded + "\n")
        print(encoded)
        scratch.cleanup()


if __name__ == "__main__":
    main()

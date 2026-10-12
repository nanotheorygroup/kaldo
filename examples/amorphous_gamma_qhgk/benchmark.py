"""Opt-in synthetic precomputed-eigenvector scaling probe.

Run ``python examples/amorphous_gamma_qhgk/benchmark.py --atoms 32 64 128 --block 32``.
The synthetic modes measure memory and I/O behavior, not physical accuracy.
"""

import argparse
import json
import resource
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from kaldo.amorphous import (NpyEigensystemSource, SparseAmorphousHarmonic,
                             calculate_qhgk_isotropic_blocked)


def probe(n_atoms, block, directory):
    m = 3 * n_atoms
    rng = np.random.default_rng(n_atoms)
    basis = np.linalg.qr(rng.normal(size=(m, m)))[0].astype(np.float32)
    omega = np.linspace(0, 12, m)
    directory.mkdir(parents=True, exist_ok=True)
    eigensystem = directory / "eigensystem.npy"
    np.save(eigensystem, np.vstack([omega ** 2, basis]).astype(np.float32))
    source = NpyEigensystemSource(eigensystem)
    del basis
    ptr, columns, blocks, displacement = [0], [], [], []
    for i in range(n_atoms):
        for j in range(max(0, i-1), min(n_atoms, i+2)):
            columns.append(j)
            blocks.append(np.eye(3, dtype=np.float32) * (2 if i == j else -1))
            displacement.append([float(i-j), 0., 0.])
        ptr.append(len(columns))
    harmonic = SparseAmorphousHarmonic(np.array(ptr), np.array(columns),
                                      np.array(blocks), np.array(displacement,
                                                               dtype=np.float32))
    physical = omega > 0
    stats = {}
    start = perf_counter()
    conductivity, diffusivity = calculate_qhgk_isotropic_blocked(
        omega, 0.04, physical, 300., False, 1000., source,
        harmonic.operators, block_size=block, stats=stats)
    elapsed = perf_counter() - start
    return {"atoms": n_atoms, "modes": m, "block_size": block,
            "dtype": "float32 work / float64 accumulation",
            "sparse_blocks": len(columns),
            "sparse_storage_bytes": harmonic.storage_bytes,
            "eigensystem_bytes": eigensystem.stat().st_size,
            "total_ordered_pairs": m*m, "evaluated_pair_fraction": stats["evaluated_pairs"]/(m*m),
            "wall_seconds": elapsed, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "conductivity_sum": float(conductivity.sum()),
            "diffusivity_sum": float(diffusivity.sum()), **stats}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--output", type=Path, help="write JSON measurements here")
    args = parser.parse_args(argv)
    if any(n < 2 for n in args.atoms) or args.block < 1:
        parser.error("atoms must be at least 2 and block must be positive")
    with tempfile.TemporaryDirectory(prefix="kaldo-qhgk-probe-") as scratch:
        measurements = [probe(n, args.block, Path(scratch) / str(n)) for n in args.atoms]
    result = {"kind": "synthetic_precomputed_eigenvector_probe", "measurements": measurements}
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()

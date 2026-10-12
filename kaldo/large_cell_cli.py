"""MPI entry point for the amorphous sparse QHGK stack.

Examples are in ``docs/large_cell_qhgk.md``. Launch with ``mpiexec -n R``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kaldo.helpers.atomic_io import atomic_save_npy, sha256_file
from kaldo.helpers.logger import get_logger
from kaldo.controllers.bandwidth import DirectBandwidth, TabulatedBandwidth
from kaldo.observables.eigenvector_source import RowShardedEigenvectorSource
from kaldo.parallel.distributed_qhgk import DistributedSparseHarmonic, run_distributed_qhgk
from kaldo.parallel.spectrum_slicing import (approximate_dos_boundaries,
                                             eigenvalue_boundaries_from_full_spectrum, run_interval_pipeline)
from kaldo.observables.eigenvector_source import NpyEigensystemSource
from kaldo.parallel.slepc_eigensolver import SlepcRowEigenvectorSource


_LOG = get_logger()


def _provider(args):
    if args.bandwidth_table is not None:
        with np.load(args.bandwidth_table, allow_pickle=False) as table:
            return TabulatedBandwidth(
                table["frequencies"], table["temperatures"], table["values"],
                frequency_unit=args.table_frequency_unit,
                value_unit=args.bandwidth_unit, bounds=args.table_bounds)
    try:
        values = float(args.bandwidth)
    except ValueError:
        values = np.load(args.bandwidth, allow_pickle=False)
    return DirectBandwidth(values, unit=args.bandwidth_unit)


def _common_arguments(parser):
    parser.add_argument("--harmonic", type=Path, required=True,
                        help="atom-block sparse input directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--volume", type=float, required=True,
                        help="cell volume in angstrom cubed")
    parser.add_argument("--classic", action="store_true")
    provider = parser.add_mutually_exclusive_group(required=True)
    provider.add_argument("--bandwidth", help="scalar value or .npy path")
    provider.add_argument("--bandwidth-table", type=Path,
                          help=".npz with frequencies, temperatures, values")
    parser.add_argument("--bandwidth-unit", choices=("rad/ps", "THz", "lifetime_ps"),
                        default="rad/ps")
    parser.add_argument("--table-frequency-unit", choices=("rad/ps", "THz"),
                        default="rad/ps")
    parser.add_argument("--table-bounds", choices=("error", "clamp"), default="error")
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--compute-dtype", choices=("float32", "float64"),
                        default="float32")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    shard_parser = sub.add_parser("shard", help="stream a saved kALDo eigensystem into coordinate-row shards")
    shard_parser.add_argument("--eigensystem", type=Path, required=True)
    shard_parser.add_argument("--omega", type=Path, required=True)
    shard_parser.add_argument("--physical", type=Path, required=True)
    shard_parser.add_argument("--output", type=Path, required=True)
    shard_parser.add_argument("--panel", type=int, default=256)
    shard_parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    reduce_parser = sub.add_parser("reduce", help="reduce precomputed row-sharded eigenvectors")
    _common_arguments(reduce_parser)
    reduce_parser.add_argument("--eigenvectors", type=Path, required=True)
    reduce_parser.add_argument("--omega", type=Path,
                               help="defaults to eigenvector directory/omega.npy")
    reduce_parser.add_argument("--physical", type=Path,
                               help="defaults to eigenvector directory/physical_mode.npy")
    solve_parser = sub.add_parser("solve", help="solve a distributed SLEPc interval")
    solve_parser.add_argument("--harmonic", type=Path, required=True)
    solve_parser.add_argument("--output", type=Path, required=True)
    solve_parser.add_argument("--interval", type=float, nargs=2, metavar=("LOW", "HIGH"),
                              help="eigenvalue interval in (rad/ps)^2; omit for full spectrum")
    solve_parser.add_argument("--tolerance", type=float, default=1e-9)
    solve_parser.add_argument("--eigenvector-dtype", choices=("float32", "float64"),
                              default="float32")
    window_parser = sub.add_parser("windows", help="solve halo intervals and reduce core modes")
    _common_arguments(window_parser)
    planning = window_parser.add_mutually_exclusive_group(required=True)
    planning.add_argument("--eigenvalue-boundaries", type=Path,
                          help="sorted .npy boundaries covering full spectrum")
    planning.add_argument("--dos", type=Path,
                          help=".npz with bin_edges and density in eigenvalue units")
    planning.add_argument("--eigenvalues", type=Path,
                          help="sorted full eigenvalue .npy for optional exact planning")
    window_parser.add_argument("--windows", type=int,
                               help="number of windows with --dos or --eigenvalues")
    window_parser.add_argument("--max-bandwidth", type=float, required=True,
                               help="global upper bound in rad/ps for exact triangular halos")
    window_parser.add_argument("--solver-tolerance", type=float, default=1e-9)
    window_parser.add_argument("--eigenvector-dtype", choices=("float32", "float64"),
                               default="float32")
    args = parser.parse_args(argv)
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError("the amorphous QHGK CLI requires mpi4py. "
                          "Install with: pip install kaldo[mpi]") from exc
    comm = MPI.COMM_WORLD
    if args.command == "shard":
        source = NpyEigensystemSource(args.eigensystem)
        omega = np.load(args.omega, allow_pickle=False).reshape(-1)
        physical = np.load(args.physical, allow_pickle=False).reshape(-1)
        if omega.shape != (source.n_modes,) or physical.shape != omega.shape:
            raise ValueError("omega and physical mask must match eigensystem modes")
        source_hash = sha256_file(args.eigensystem) if comm.rank == 0 else None
        source_hash = comm.bcast(source_hash, root=0)
        RowShardedEigenvectorSource.create(
            source, args.output, comm, panel_size=args.panel,
            dtype=args.dtype, source_sha256=source_hash)
        if comm.rank == 0:
            atomic_save_npy(args.output / "omega.npy", omega)
            atomic_save_npy(args.output / "physical_mode.npy", physical)
            _LOG.info("Row-sharded %d modes: %s", source.n_modes, args.output)
        return
    harmonic = DistributedSparseHarmonic(args.harmonic, comm)
    if args.command == "solve":
        source, omega, physical = SlepcRowEigenvectorSource.solve_interval(
            harmonic, args.output, interval_lambda=args.interval,
            tolerance=args.tolerance, dtype=args.eigenvector_dtype,
            require_full=args.interval is None)
        if comm.rank == 0:
            _LOG.info("Converged %d modes; output: %s", source.n_modes, args.output)
        return
    provider = _provider(args)
    if args.command == "reduce":
        source = RowShardedEigenvectorSource(args.eigenvectors, comm)
        omega_path = args.omega or args.eigenvectors / "omega.npy"
        physical_path = args.physical or args.eigenvectors / "physical_mode.npy"
        result = run_distributed_qhgk(
            np.load(omega_path, allow_pickle=False), provider,
            np.load(physical_path, allow_pickle=False),
            args.temperature, args.classic, args.volume, source, harmonic,
            args.output, block_size=args.block, compute_dtype=args.compute_dtype)
    else:
        if args.eigenvalue_boundaries is not None:
            boundaries = np.load(args.eigenvalue_boundaries, allow_pickle=False)
        elif args.dos is not None:
            if args.windows is None:
                parser.error("--windows is required with --dos")
            with np.load(args.dos, allow_pickle=False) as dos:
                boundaries = approximate_dos_boundaries(
                    dos["bin_edges"], dos["density"], args.windows)
        else:
            if args.windows is None:
                parser.error("--windows is required with --eigenvalues")
            boundaries = eigenvalue_boundaries_from_full_spectrum(
                np.load(args.eigenvalues, allow_pickle=False), args.windows)
        result = run_interval_pipeline(
            harmonic, boundaries,
            args.max_bandwidth, provider, args.temperature, args.classic,
            args.volume, args.output, block_size=args.block,
            compute_dtype=args.compute_dtype,
            solver_options={"tolerance": args.solver_tolerance,
                            "dtype": args.eigenvector_dtype})
    if comm.rank == 0 and result is not None:
        _LOG.info("Total isotropic conductivity: %.12g W/(m K)", float(np.sum(result[0])))
        _LOG.info("Output: %s", args.output)


if __name__ == "__main__":
    main()

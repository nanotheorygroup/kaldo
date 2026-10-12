"""Command line for the large-cell Gamma-point workflow: sparse IFC2 through distributed QHGK.

Run ``python -m kaldo.large_cell_cli COMMAND``; launch the MPI commands with
``mpiexec -n R``. Examples are in ``docs/large_cell_qhgk.md``.

Colored finite-difference IFC2 (serial, one ``ifc-run`` job per shard):
  ifc-plan       probe the force range and create the immutable coloring plan
  ifc-run        evaluate one checkpointed shard of force tasks
  ifc-finalize   merge and validate; write merged/, dynmat_bsr.npz, harmonic/

Harmonic input:
  convert        stream a nonperiodic dense .npy or one-replica Dyn.form

Distributed QHGK (MPI; needs ``pip install kaldo[mpi]``):
  shard          stream a saved kALDo eigensystem into coordinate-row shards
  reduce         reduce precomputed row-sharded eigenvectors
  solve          solve a distributed SLEPc eigenvalue interval (``kaldo[slepc]``)
  windows        solve spectrum halo intervals and reduce their core modes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kaldo.helpers.logger import get_logger


_LOG = get_logger()


def _comm():
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError("distributed QHGK commands require mpi4py. "
                          "Install with: pip install kaldo[mpi]") from exc
    return MPI.COMM_WORLD


def _provider(args):
    from kaldo.controllers.bandwidth import DirectBandwidth, TabulatedBandwidth

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


def _convert(args):
    from kaldo.observables.sparse_harmonic import convert_legacy_dense, convert_legacy_dynform

    if args.derivative and len(args.derivative) != 3:
        raise ValueError("provide exactly three --derivative paths")
    converter = convert_legacy_dense if args.dynamical.suffix == ".npy" else convert_legacy_dynform
    harmonic = converter(
        args.dynamical, args.positions, args.output, masses_path=args.masses,
        derivative_paths=args.derivative or None, dtype=args.dtype,
        apply_acoustic_sum=args.apply_acoustic_sum)
    _LOG.info("Sparse harmonic written to %s: %s", args.output,
              json.dumps({"n_atoms": harmonic.n_atoms,
                          "n_blocks": len(harmonic.column_atom),
                          "storage_bytes": harmonic.storage_bytes,
                          "validation": harmonic.validation_report()}))


def _shard(args):
    from kaldo.helpers.atomic_io import atomic_save_npy, sha256_file
    from kaldo.observables.eigenvector_source import NpyEigensystemSource, RowShardedEigenvectorSource

    comm = _comm()
    source = NpyEigensystemSource(args.eigensystem)
    omega = np.load(args.omega, allow_pickle=False).reshape(-1)
    physical = np.load(args.physical, allow_pickle=False).reshape(-1)
    if omega.shape != (source.n_modes,) or physical.shape != omega.shape:
        raise ValueError("omega and physical mask must match eigensystem modes")
    source_hash = comm.bcast(sha256_file(args.eigensystem) if comm.rank == 0 else None, root=0)
    RowShardedEigenvectorSource.create(source, args.output, comm, panel_size=args.panel,
                                       dtype=args.dtype, source_sha256=source_hash)
    if comm.rank == 0:
        atomic_save_npy(args.output / "omega.npy", omega)
        atomic_save_npy(args.output / "physical_mode.npy", physical)
        _LOG.info("Row-sharded %d modes: %s", source.n_modes, args.output)


def _harmonic(args):
    from kaldo.parallel.distributed_qhgk import DistributedSparseHarmonic

    return DistributedSparseHarmonic(args.harmonic, _comm())


def _solve(args):
    from kaldo.parallel.slepc_eigensolver import SlepcRowEigenvectorSource

    harmonic = _harmonic(args)
    source, _, _ = SlepcRowEigenvectorSource.solve_interval(
        harmonic, args.output, interval_lambda=args.interval, tolerance=args.tolerance,
        dtype=args.eigenvector_dtype, require_full=args.interval is None)
    if harmonic.rank == 0:
        _LOG.info("Converged %d modes; output: %s", source.n_modes, args.output)


def _report(rank, result, output):
    if rank == 0 and result is not None:
        _LOG.info("Total isotropic conductivity: %.12g W/(m K)", float(np.sum(result[0])))
        _LOG.info("Output: %s", output)


def _reduce(args):
    from kaldo.observables.eigenvector_source import RowShardedEigenvectorSource
    from kaldo.parallel.distributed_qhgk import run_distributed_qhgk

    harmonic = _harmonic(args)
    source = RowShardedEigenvectorSource(args.eigenvectors, harmonic.comm)
    omega_path = args.omega or args.eigenvectors / "omega.npy"
    physical_path = args.physical or args.eigenvectors / "physical_mode.npy"
    result = run_distributed_qhgk(
        np.load(omega_path, allow_pickle=False), _provider(args), np.load(physical_path, allow_pickle=False),
        args.temperature, args.classic, args.volume, source, harmonic, args.output,
        block_size=args.block, compute_dtype=args.compute_dtype)
    _report(harmonic.rank, result, args.output)


def _windows(args):
    from kaldo.parallel.spectrum_slicing import (approximate_dos_boundaries,
                                                 eigenvalue_boundaries_from_full_spectrum, run_interval_pipeline)

    if args.eigenvalue_boundaries is not None:
        boundaries = np.load(args.eigenvalue_boundaries, allow_pickle=False)
    elif args.windows is None:
        raise ValueError("--windows is required with --dos or --eigenvalues")
    elif args.dos is not None:
        with np.load(args.dos, allow_pickle=False) as dos:
            boundaries = approximate_dos_boundaries(dos["bin_edges"], dos["density"], args.windows)
    else:
        boundaries = eigenvalue_boundaries_from_full_spectrum(
            np.load(args.eigenvalues, allow_pickle=False), args.windows)
    harmonic = _harmonic(args)
    result = run_interval_pipeline(
        harmonic, boundaries, args.max_bandwidth, _provider(args), args.temperature, args.classic,
        args.volume, args.output, block_size=args.block, compute_dtype=args.compute_dtype,
        solver_options={"tolerance": args.solver_tolerance, "dtype": args.eigenvector_dtype})
    _report(harmonic.rank, result, args.output)


def _reduction_arguments(parser):
    parser.add_argument("--harmonic", type=Path, required=True, help="atom-block sparse input directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--volume", type=float, required=True, help="cell volume in angstrom cubed")
    parser.add_argument("--classic", action="store_true")
    provider = parser.add_mutually_exclusive_group(required=True)
    provider.add_argument("--bandwidth", help="scalar value or .npy path")
    provider.add_argument("--bandwidth-table", type=Path, help=".npz with frequencies, temperatures, values")
    parser.add_argument("--bandwidth-unit", choices=("rad/ps", "THz", "lifetime_ps"), default="rad/ps")
    parser.add_argument("--table-frequency-unit", choices=("rad/ps", "THz"), default="rad/ps")
    parser.add_argument("--table-bounds", choices=("error", "clamp"), default="error")
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float32")


def _parser():
    from kaldo.controllers.colored_fd import build_dynmat

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    planning = commands.add_parser("ifc-plan", help="probe the force range and create the coloring plan")
    planning.add_argument("--workdir", type=Path, required=True)
    planning.add_argument("--structure", type=Path, required=True, help="ASE-readable physical Gamma cell")
    planning.add_argument("--calculator", required=True, help="importable module:factory")
    planning.add_argument("--calculator-description", help="stable model/version identity for checkpoint safety")
    planning.add_argument("--delta-shift", type=float, required=True, help="central-difference shift in Angstrom")
    planning.add_argument("--shards", type=int, default=1)
    planning.add_argument("--expected-atoms", type=int, help="fail unless the structure has exactly this many atoms")
    planning.add_argument("--n-probe", type=int, default=12)
    planning.add_argument("--tolerance", type=float, default=1e-5)
    planning.add_argument("--seed", type=int, default=0)
    planning.add_argument("--bin-width", type=float, default=0.5)
    planning.add_argument("--cutoff-hint", type=float, help="calculator descriptor cutoff in Angstrom")
    planning.add_argument("--cutoff-multiplier", type=float, default=2.0)
    planning.set_defaults(handler=build_dynmat.plan)
    worker = commands.add_parser("ifc-run", help="evaluate one checkpointed shard")
    worker.add_argument("--workdir", type=Path, required=True)
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--flush-tasks", type=int, default=1)
    worker.set_defaults(handler=build_dynmat.run)
    merging = commands.add_parser("ifc-finalize", help="merge, validate, and write merged/, dynmat, harmonic/")
    merging.add_argument("--workdir", type=Path, required=True)
    merging.add_argument("--response-threshold", type=float,
                         help="maximum relative omitted response; defaults to the probe tolerance")
    merging.add_argument("--residual-threshold", type=float, default=1e-8)
    merging.set_defaults(handler=build_dynmat.finalize)

    convert = commands.add_parser("convert", help="convert a nonperiodic .npy or one-replica Dyn.form")
    convert.add_argument("dynamical", type=Path)
    convert.add_argument("positions", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--masses", type=Path)
    convert.add_argument("--derivative", type=Path, action="append", default=[],
                         help="one each for x, y, z; verifies the displacement convention")
    convert.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    convert.add_argument("--apply-acoustic-sum", action="store_true")
    convert.set_defaults(handler=_convert)

    shard = commands.add_parser("shard", help="stream a saved kALDo eigensystem into coordinate-row shards")
    shard.add_argument("--eigensystem", type=Path, required=True)
    shard.add_argument("--omega", type=Path, required=True)
    shard.add_argument("--physical", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--panel", type=int, default=256)
    shard.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    shard.set_defaults(handler=_shard)
    reduce = commands.add_parser("reduce", help="reduce precomputed row-sharded eigenvectors")
    _reduction_arguments(reduce)
    reduce.add_argument("--eigenvectors", type=Path, required=True)
    reduce.add_argument("--omega", type=Path, help="defaults to eigenvector directory/omega.npy")
    reduce.add_argument("--physical", type=Path, help="defaults to eigenvector directory/physical_mode.npy")
    reduce.set_defaults(handler=_reduce)
    solve = commands.add_parser("solve", help="solve a distributed SLEPc interval")
    solve.add_argument("--harmonic", type=Path, required=True)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--interval", type=float, nargs=2, metavar=("LOW", "HIGH"),
                       help="eigenvalue interval in (rad/ps)^2; omit for full spectrum")
    solve.add_argument("--tolerance", type=float, default=1e-9)
    solve.add_argument("--eigenvector-dtype", choices=("float32", "float64"), default="float32")
    solve.set_defaults(handler=_solve)
    windows = commands.add_parser("windows", help="solve halo intervals and reduce core modes")
    _reduction_arguments(windows)
    planning_source = windows.add_mutually_exclusive_group(required=True)
    planning_source.add_argument("--eigenvalue-boundaries", type=Path,
                                 help="sorted .npy boundaries covering full spectrum")
    planning_source.add_argument("--dos", type=Path, help=".npz with bin_edges and density in eigenvalue units")
    planning_source.add_argument("--eigenvalues", type=Path,
                                 help="sorted full eigenvalue .npy for optional exact planning")
    windows.add_argument("--windows", type=int, help="number of windows with --dos or --eigenvalues")
    windows.add_argument("--max-bandwidth", type=float, required=True,
                         help="global upper bound in rad/ps for exact triangular halos")
    windows.add_argument("--solver-tolerance", type=float, default=1e-9)
    windows.add_argument("--eigenvector-dtype", choices=("float32", "float64"), default="float32")
    windows.set_defaults(handler=_windows)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()

"""Opt-in row-distributed MPI QHGK integration case."""

import json
import sys
from pathlib import Path

from mpi4py import MPI
import numpy as np

from kaldo.amorphous import (DirectBandwidth, NpyEigensystemSource,
                             NumpyEigenvectorSource, SparseAmorphousHarmonic,
                             calculate_qhgk_isotropic_blocked)
from kaldo.amorphous.checkpoint import RowShardedEigenvectorSource
from kaldo.amorphous.distributed import (DistributedSparseAmorphousHarmonic,
                                         run_distributed_qhgk)
from kaldo.amorphous.pipeline import run_interval_pipeline
from kaldo.amorphous.windows import plan_mode_windows
from kaldo.tests.mpi_amorphous_qhgk_case import write_input


class PrecomputedIntervalSource:
    """Use the tracked test eigensystem as a deterministic interval solver."""

    @classmethod
    def solve_interval(cls, harmonic, directory, *, interval_lambda, **kwargs):
        root = harmonic.directory.parent
        omega = np.load(root / "omega.npy")
        eigenvalues = omega**2
        first = int(np.searchsorted(eigenvalues, interval_lambda[0], side="left"))
        stop = int(np.searchsorted(eigenvalues, interval_lambda[1], side="right"))
        full = NpyEigensystemSource(root / "eigensystem.npy")
        local = NumpyEigenvectorSource(full.read_modes(first, stop, dtype=np.float64))
        source = RowShardedEigenvectorSource.create(
            local, directory, harmonic.comm, panel_size=5, dtype=np.float64)
        if harmonic.rank == 0:
            np.save(Path(directory) / "eigenvalues.npy", eigenvalues[first:stop])
            np.save(Path(directory) / "omega.npy", omega[first:stop])
            np.save(Path(directory) / "physical_mode.npy", np.ones(stop-first, bool))
        harmonic.comm.Barrier()
        return source, omega[first:stop], np.ones(stop-first, bool)


def main():
    root = Path(sys.argv[1])
    phase = sys.argv[2] if len(sys.argv) > 2 else "resume"
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        write_input(root)
    comm.Barrier()
    row_directory = root / "row_shards"
    if phase == "cli":
        from kaldo.amorphous.cli import main as cli_main
        cli_main(["shard", "--eigensystem", str(root / "eigensystem.npy"),
                  "--omega", str(root / "omega.npy"),
                  "--physical", str(root / "physical.npy"),
                  "--output", str(row_directory), "--panel", "5",
                  "--dtype", "float64"])
    if not (row_directory / "manifest.json").exists():
        source = NpyEigensystemSource(root / "eigensystem.npy")
        RowShardedEigenvectorSource.create(source, row_directory, comm,
                                           panel_size=5, dtype=np.float64)
    row_source = RowShardedEigenvectorSource(row_directory, comm)
    harmonic = DistributedSparseAmorphousHarmonic(root / "sparse", comm)
    omega = np.load(root / "omega.npy")
    provider = DirectBandwidth(0.035)
    if phase == "corrupt":
        result_dir = root / "corrupt_results"
        run_distributed_qhgk(
            omega, provider, np.ones(len(omega), bool), 300., False, 200.,
            row_source, harmonic, result_dir, block_size=4,
            compute_dtype=np.float64, max_blocks=1)
        if comm.rank == 0:
            next(result_dir.glob("rows_*.npz")).write_bytes(b"damaged checkpoint")
        comm.Barrier()
        try:
            run_distributed_qhgk(
                omega, provider, np.ones(len(omega), bool), 300., False, 200.,
                row_source, harmonic, result_dir, block_size=4,
                compute_dtype=np.float64)
        except ValueError as exc:
            rejected = "invalid distributed checkpoint" in str(exc)
        else:
            rejected = False
        assert all(comm.allgather(rejected))
        if comm.rank == 0:
            print("corrupt restart rejected collectively", comm.size)
        return
    if phase == "shapes":
        adjusted = omega.copy()
        adjusted[0] = 0.
        adjusted[2] = adjusted[1] + 1e-11
        physical = np.ones(len(adjusted), bool)
        physical[[0,3]] = False
        for shape in ("triangle", "gauss", "lorentz"):
            for dtype in (np.float64, np.float32):
                result = run_distributed_qhgk(
                    adjusted, provider, physical, 300., False, 200.,
                    row_source, harmonic,
                    root / f"shape_{shape}_{np.dtype(dtype).name}",
                    block_size=5, broadening_shape=shape,
                    compute_dtype=dtype)
                if comm.rank == 0:
                    reference = calculate_qhgk_isotropic_blocked(
                        adjusted, provider.evaluate(adjusted, 300.), physical,
                        300., False, 200.,
                        NpyEigensystemSource(root / "eigensystem.npy"),
                        SparseAmorphousHarmonic.load(root / "sparse").operators,
                        block_size=5, broadening_shape=shape,
                        compute_dtype=np.float64)
                    if dtype == np.float64:
                        for actual, expected in zip(result, reference):
                            np.testing.assert_allclose(actual, expected,
                                                       rtol=1e-12, atol=1e-13)
                    else:
                        relative = abs(result[0].sum()-reference[0].sum()) / abs(reference[0].sum())
                        assert relative < 1e-6
        if comm.rank == 0:
            print("distributed shapes and mixed precision complete", comm.size)
        return
    if phase == "cli":
        from kaldo.amorphous.cli import main as cli_main
        cli_main(["reduce", "--harmonic", str(root / "sparse"),
                  "--eigenvectors", str(row_directory),
                  "--omega", str(root / "omega.npy"),
                  "--physical", str(root / "physical.npy"),
                  "--output", str(root / "cli_results"),
                  "--temperature", "300", "--volume", "200",
                  "--bandwidth", "0.035", "--block", "4",
                  "--compute-dtype", "float64"])
        if comm.rank == 0:
            expected = calculate_qhgk_isotropic_blocked(
                omega, provider.evaluate(omega, 300.), np.ones(len(omega), bool),
                300., False, 200., NpyEigensystemSource(root / "eigensystem.npy"),
                SparseAmorphousHarmonic.load(root / "sparse").operators,
                block_size=4, compute_dtype=np.float64)[0]
            np.testing.assert_allclose(np.load(root / "cli_results/conductivity.npy"),
                                       expected, rtol=1e-12, atol=1e-13)
            print("distributed CLI complete", comm.size)
        return
    if phase.startswith("pipeline"):
        result = run_interval_pipeline(
            harmonic, [0., 0.7, 2.3, 6.], 0.035, provider,
            300., False, 200., root / "pipeline_results",
            solver=PrecomputedIntervalSource, block_size=4,
            compute_dtype=np.float64,
            max_windows=1 if phase == "pipeline_partial" else None)
        if comm.rank == 0:
            manifest = json.loads((root / "pipeline_results/manifest.json").read_text())
            if phase == "pipeline_partial":
                assert not manifest["complete"] and manifest["core_mode_count"] < len(omega)
                print("interval pipeline partial", comm.size)
            else:
                assert manifest["complete"] and manifest["core_mode_count"] == len(omega)
                reference = calculate_qhgk_isotropic_blocked(
                    omega, provider.evaluate(omega, 300.), np.ones(len(omega), bool),
                    300., False, 200., NpyEigensystemSource(root / "eigensystem.npy"),
                    SparseAmorphousHarmonic.load(root / "sparse").operators,
                    block_size=4, compute_dtype=np.float64)
                for actual, expected in zip(result, reference):
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
                print("interval pipeline complete", comm.size, float(result[0].sum()))
        return
    if phase == "window":
        collected_c = np.zeros(len(omega)) if comm.rank == 0 else None
        collected_d = np.zeros(len(omega)) if comm.rank == 0 else None
        full_source = NpyEigensystemSource(root / "eigensystem.npy")
        for number, window in enumerate(plan_mode_windows(
                omega, provider.evaluate(omega, 300.), 3)):
            subspace = NumpyEigenvectorSource(full_source.read_modes(
                window.halo_start, window.halo_stop, dtype=np.float64))
            source = RowShardedEigenvectorSource.create(
                subspace, root / f"window_{number}/row_shards", comm,
                panel_size=5, dtype=np.float64)
            local_start, local_stop = window.local_core
            result = run_distributed_qhgk(
                omega[window.halo_start:window.halo_stop], provider,
                np.ones(window.halo_stop-window.halo_start, bool),
                300., False, 200., source, harmonic,
                root / f"window_{number}/results", block_size=4,
                compute_dtype=np.float64, core_start=local_start,
                core_stop=local_stop)
            if comm.rank == 0:
                collected_c[window.core_start:window.core_stop] = result[0][local_start:local_stop]
                collected_d[window.core_start:window.core_stop] = result[1][local_start:local_stop]
        if comm.rank == 0:
            reference = calculate_qhgk_isotropic_blocked(
                omega, provider.evaluate(omega, 300.), np.ones(len(omega), bool),
                300., False, 200., full_source,
                SparseAmorphousHarmonic.load(root / "sparse").operators,
                block_size=4, compute_dtype=np.float64)
            for actual, expected in zip((collected_c, collected_d), reference):
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
            print("distributed window complete", comm.size, float(collected_c.sum()))
        return
    result = run_distributed_qhgk(
        omega, provider, np.ones(len(omega), bool), 300., False, 200.,
        row_source, harmonic, root / "distributed_results", block_size=4,
        compute_dtype=np.float64, max_blocks=2 if phase == "partial" else None)
    if comm.rank == 0:
        manifest = json.loads((root / "distributed_results/manifest.json").read_text())
        if phase == "partial":
            assert not manifest["complete"]
            assert len(manifest["completed_core_ranges"]) == 2
            print("distributed partial", comm.size)
        else:
            assert manifest["complete"]
            reference = calculate_qhgk_isotropic_blocked(
                omega, provider.evaluate(omega, 300.), np.ones(len(omega), bool),
                300., False, 200., NpyEigensystemSource(root / "eigensystem.npy"),
                SparseAmorphousHarmonic.load(root / "sparse").operators,
                block_size=4, compute_dtype=np.float64)
            for actual, expected in zip(result, reference):
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
            print("distributed complete", comm.size, float(np.sum(result[0])))


if __name__ == "__main__":
    main()

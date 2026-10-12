"""Row-distributed MPI QHGK integration case, launched by ``test_amorphous_distributed_mpi.py``.

Usage: ``mpiexec -n R python -m kaldo.tests.mpi_distributed_qhgk_case DIR PHASE``.
"""

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


def write_input(root):
    """Write a small dense-coupled sparse harmonic and an orthonormal eigensystem once."""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "eigensystem.npy").exists():
        return
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
    columns = np.tile(np.arange(n), n)
    displacement = (positions[:, None, :] - positions[None, :, :]).reshape(-1, 3)
    SparseAmorphousHarmonic(np.arange(0, n*n + 1, n), columns, atom_blocks.reshape(-1, 3, 3),
                            displacement).save(root / "sparse")


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

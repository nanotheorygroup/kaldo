"""Optional PETSc/SLEPc real-symmetric interval solve to row shards.

PETSc and SLEPc are imported only when solving. Their MPI communicator must
match the harmonic input's communicator. Eigenvectors are written directly
from distributed PETSc Vec ownership into local row-shard files; no rank
gathers a complete eigenvector.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import bsr_matrix

from kaldo.helpers.atomic_io import atomic_save_npy, atomic_write_json, sha256_file
from .checkpoint import RowShardedEigenvectorSource
from .distributed import DistributedSparseAmorphousHarmonic


class SlepcRowEigenvectorSource(RowShardedEigenvectorSource):
    """Row-sharded source produced by a distributed SLEPc interval solve."""

    @classmethod
    def solve_interval(cls, harmonic, directory, *, interval_lambda=None,
                       tolerance=1e-9, max_iterations=None, dtype=np.float32,
                       require_full=False, zero_tolerance=1e-12,
                       factor_solver=None):
        if not isinstance(harmonic, DistributedSparseAmorphousHarmonic):
            raise ValueError("SLEPc solve requires distributed sparse harmonic input")
        try:
            from petsc4py import PETSc
            from slepc4py import SLEPc
        except ImportError as exc:
            raise ImportError("PETSc/SLEPc interval solves require petsc4py and slepc4py. "
                              "Install with: pip install kaldo[slepc]") from exc
        comm = harmonic.comm
        rank, size = comm.Get_rank(), comm.Get_size()
        if PETSc.COMM_WORLD.getSize() != size or PETSc.COMM_WORLD.getRank() != rank:
            raise ValueError("PETSc communicator must match the MPI world communicator")
        if np.dtype(PETSc.ScalarType) != np.dtype("float64"):
            raise ValueError("amorphous interval solves require real float64 PETSc scalars")
        if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("eigenvector checkpoint dtype must be float32 or float64")
        if tolerance <= 0 or not np.isfinite(tolerance) or zero_tolerance < 0:
            raise ValueError("invalid eigensolver tolerance")
        if factor_solver is None:
            factor_solver = "mumps" if size > 1 else "petsc"
        if factor_solver not in ("mumps", "petsc") or (size > 1 and factor_solver == "petsc"):
            raise ValueError("MPI interval slicing requires an inertia-capable parallel factor solver such as MUMPS")
        m = 3*harmonic.n_atoms
        local_m = 3*harmonic.n_local_atoms
        local_bsr = bsr_matrix((np.asarray(harmonic.blocks, dtype=np.float64),
                                harmonic.column_atom, harmonic.row_ptr),
                               shape=(local_m, m))
        local_csr = local_bsr.tocsr()
        pointers = np.asarray(local_csr.indptr, dtype=PETSc.IntType)
        columns = np.asarray(local_csr.indices, dtype=PETSc.IntType)
        values = np.asarray(local_csr.data, dtype=PETSc.ScalarType)
        matrix = PETSc.Mat().createAIJ(
            size=((local_m, m), (local_m, m)), csr=(pointers, columns, values),
            comm=PETSc.COMM_WORLD)
        matrix.assemble()
        if matrix.getOwnershipRange() != (3*harmonic.atom_start, 3*harmonic.atom_stop):
            raise ValueError("PETSc row ownership differs from atom-row shards")
        if interval_lambda is None:
            diagonal = local_csr.diagonal() if local_csr.shape[0] == m else np.array([
                local_csr[row, 3*harmonic.atom_start+row]
                for row in range(local_m)])
            radius = np.asarray(np.abs(local_csr).sum(axis=1)).reshape(-1) - np.abs(diagonal)
            local_lower = float(np.min(diagonal-radius))
            local_upper = float(np.max(diagonal+radius))
            from mpi4py import MPI
            lower = comm.allreduce(local_lower, op=MPI.MIN)
            upper = comm.allreduce(local_upper, op=MPI.MAX)
            padding = max(1e-8, 1e-6*max(1., abs(lower), abs(upper)))
            interval_lambda = (lower-padding, upper+padding)
        lower, upper = map(float, interval_lambda)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("invalid eigenvalue interval")
        eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
        eps.setOperators(matrix)
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.ALL)
        eps.setInterval(lower, upper)
        eps.getST().setType(SLEPc.ST.Type.SINVERT)
        ksp = eps.getST().getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.CHOLESKY)
        if factor_solver == "mumps":
            pc.setFactorSolverType("mumps")
            # SLEPc's inertia check needs MUMPS to return pivot counts.
            PETSc.Options()["st_mat_mumps_icntl_13"] = "1"
        if max_iterations is None:
            eps.setTolerances(tol=tolerance)
        else:
            eps.setTolerances(tol=tolerance, max_it=max_iterations)
        # Target-machine KSP/PC and spectrum-slicing options can be supplied
        # through PETSc's options database without changing this adapter.
        eps.setFromOptions()
        start_time = perf_counter()
        eps.solve()
        elapsed = perf_counter()-start_time
        if int(eps.getConvergedReason()) <= 0:
            raise RuntimeError(f"SLEPc interval solve did not converge: {eps.getConvergedReason()}")
        nconv = eps.getConverged()
        if require_full and nconv != m:
            raise RuntimeError(f"SLEPc converged {nconv} of {m} requested eigenpairs")
        eigenvalues = np.array([float(np.real(eps.getEigenvalue(index)))
                                for index in range(nconv)])
        ordering = np.argsort(eigenvalues, kind="stable")
        eigenvalues = eigenvalues[ordering]
        errors = np.array([eps.getErrorEstimate(int(index)) for index in ordering])
        if (not np.isfinite(errors).all() or
                np.any(errors > max(1e-7, 100*tolerance))):
            raise RuntimeError("SLEPc eigenpair residual estimate exceeds tolerance")
        directory = Path(directory)
        if rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
        filename = (f"rank_{rank:05d}_atoms_{harmonic.atom_start:09d}_"
                    f"{harmonic.atom_stop:09d}.npy")
        target = directory / filename
        mapped = np.lib.format.open_memmap(target, mode="w+", dtype=dtype,
                                           shape=(local_m, nconv))
        vector = matrix.createVecRight()
        for column, original in enumerate(ordering):
            eps.getEigenpair(int(original), vector)
            mapped[:, column] = vector.getArray(readonly=True)
        mapped.flush()
        del mapped
        local = {"rank": rank, "atom_start": harmonic.atom_start,
                 "atom_stop": harmonic.atom_stop, "file": filename,
                 "sha256": sha256_file(target)}
        shards = comm.gather(local, root=0)
        if rank == 0:
            omega = np.sign(eigenvalues)*np.sqrt(np.abs(eigenvalues))
            physical = eigenvalues > zero_tolerance
            atomic_save_npy(directory / "eigenvalues.npy", eigenvalues)
            atomic_save_npy(directory / "omega.npy", omega)
            atomic_save_npy(directory / "physical_mode.npy", physical)
            atomic_save_npy(directory / "residual_estimates.npy", errors)
            manifest = {"format": "kaldo-eigenvector-row-shards-v1",
                        "n_atoms": harmonic.n_atoms, "n_modes": nconv,
                        "rank_count": size, "dtype": str(np.dtype(dtype)),
                        "source_sha256": harmonic.input_hashes,
                        "shards": shards,
                        "solver": {"name": "SLEPc EPS KrylovSchur interval",
                                   "eigenvalue_interval_rad2_ps2": [lower, upper],
                                   "tolerance": tolerance,
                                   "converged": nconv,
                                   "required_full_spectrum": bool(require_full),
                                   "factor_solver": factor_solver,
                                   "scalar_dtype": str(np.dtype(PETSc.ScalarType)),
                                   "residuals_before_checkpoint_cast": True,
                                   "solve_seconds": elapsed,
                                   "max_residual_estimate": (float(np.max(errors))
                                                             if len(errors) else None),
                                   "zero_eigenvalue_tolerance": zero_tolerance}}
            atomic_write_json(directory / "manifest.json", manifest)
        comm.Barrier()
        return (cls(directory, comm),
                np.load(directory / "omega.npy"),
                np.load(directory / "physical_mode.npy"))

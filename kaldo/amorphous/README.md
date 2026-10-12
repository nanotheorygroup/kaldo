# Amorphous Gamma QHGK prototype

`calculate_qhgk_isotropic_blocked` computes per-mode isotropic conductivity
(W/(m K)) and diffusivity (mm²/s) from sorted angular frequencies (rad/ps),
`diffusivity_bandwidth` (rad/ps), a physical-mode mask, real coordinate-major
eigenvectors, and three real derivative operators. Volume is in Å³. It keeps
the existing `float32` work and `float64` accumulation defaults.
`core_start`/`core_stop` restrict the result to a half-open range of core modes.

The pair kernel and the generalized heat capacity are
`kaldo.controllers.qhgk.diffusivity_kernel` and
`kaldo.controllers.qhgk.generalized_heat_capacity`, the same functions
`Conductivity(method='qhgk')` uses. `test_amorphous_blocked_qhgk_matches_conductivity.py`
checks per-mode agreement with `Conductivity` on `kaldo/tests/si-amorphous`.

The pair width is `2*(bandwidth_i+bandwidth_j)`. When the legacy QHGK path has
no explicit bandwidth, use `phonons.bandwidth/2`. `diffusivity_threshold` keeps
its asymmetric legacy rule: pair `(i,j)` uses partner bandwidth `j`. The
Gaussian and Lorentzian kernels are evaluated over every pair unless this
explicit compatibility threshold is set. A symmetric tail cutoff requires a
separate convergence study.

For triangular broadening, the reducer skips a whole partner block only when
the smallest possible frequency separation is at least twice the sum of the
largest bandwidths in the two blocks. This is a conservative bound on the
kernel's *exact* compact support. The reported `evaluated_pairs` counts pairs
sent to the pair kernel, including some zeros inside retained blocks.

## Sparse harmonic input

`SparseAmorphousHarmonic` stores directed atom-block CSR arrays:

| File | Shape | Meaning |
| --- | --- | --- |
| `row_ptr.npy` | `(N+1,)` | Atom-row offsets |
| `column_atom.npy` | `(n_blocks,)` | Sorted column atoms; repeated entries represent separate images |
| `dynamical_blocks.npy` | `(n_blocks,3,3)` | Real mass-weighted dynamical blocks |
| `displacement.npy` | `(n_blocks,3)` | Pair displacement in Å; derivative = displacement × block |
| `masses.npy` | `(N,)` optional | Positive masses for acoustic-sum diagnostics |
| `positions.npy` | `(N,3)` optional | Atom positions in Å |
| `cell.npy` | `(3,3)` optional | Cell vectors in Å |
| `metadata.json` | object | Units, dtypes, source hashes, validation residuals |

All reverse atom-pair/image blocks and onsite blocks are stored. Onsite displacement
is zero. The three derivative views share the graph and base values; a panel
product creates only one temporary sparse direction's values and no dense
derivative. `validation_report()` checks symmetry, reverse displacement,
onsite displacement, skew derivatives, and an acoustic-sum residual when
masses are present. Reverse blocks are matched by atom pair and, when a pair
is stored more than once, by periodic image (displacement `r` pairs with `-r`).
The checks are vectorized over blocks and run on every `save()` and `load()`.
Direct production producers can write the same directory contract and load it
with `SparseAmorphousHarmonic.load(path)`. Arrays and metadata are written
atomically. Blocks default to `float64`; pass `dtype=np.float32` (or
`--dtype float32`) to halve their storage.

The nonperiodic legacy converter streams a mass-weighted dense `.npy` matrix
or one-replica `Dyn.form` text in three-coordinate row slabs. It uses no dense
intermediate for text input, drops zero offsite blocks, and can verify three
saved derivative matrices:

```bash
python -m kaldo.amorphous.sparse_harmonic D.npy positions.npy sparse_out \
  --masses masses.npy --derivative Dx.npy --derivative Dy.npy --derivative Dz.npy
python -m kaldo.amorphous.sparse_harmonic Dyn.form positions.npy sparse_out \
  --masses masses.npy --apply-acoustic-sum
```

It uses `positions[i]-positions[j]`. If a legacy pair combines periodic images
with different displacements, supply the pair/image data directly; the
converter rejects derivative mismatches instead of guessing an effective
displacement.

For a producer that already has sparse ordinary IFC2 blocks, use
`from_sparse_force_constants(...)` to apply kALDo's onsite acoustic sum rule,
mass weighting, and its `eV/angstrom^2` to dynamical-unit factor without a
dense intermediate. It returns the same atom-block contract for saving.
`symmetrize=True` averages reverse atom blocks before mass weighting; use it
for finite-difference IFC data that need the same Hermitian projection as a
saved dense reference. Set `apply_acoustic_sum=False` when matching a saved
reference that did not apply that correction.

## Bandwidth and eigenvector sources

`DirectBandwidth(values, unit="rad/ps")` accepts scalar or per-mode values.
`TabulatedBandwidth(frequencies, temperatures, values, ...)` interpolates in
frequency then temperature and raises outside its table by default. Values
may be in `rad/ps`, cyclic `THz`, or `lifetime_ps`. Lifetime conversion uses
the explicit convention `bandwidth = 1/(2*lifetime_ps)`; check that this
matches the source's definition before using it. Provider metadata records
units, interpolation, conversion, and source hash.

`NpyEigensystemSource` memory-maps the legacy `(M+1,M)` file and excludes the
eigenvalue row. The reducer receives any source through
`read_modes(start, stop, dtype=...)`; `RowShardedEigenvectorSource` (below)
holds one rank's coordinate rows.

## MPI and restart

The MPI paths need `mpi4py` (`pip install kaldo[mpi]`); the SLEPc solver also
needs `petsc4py` and `slepc4py` (`pip install kaldo[slepc]`).

`run_distributed_qhgk(...)` in `kaldo.amorphous.distributed` is the
coordinate-row distributed path. Each rank memory-maps only its atom rows of
the sparse harmonic graph and the corresponding coordinate rows of every
eigenvector. It exchanges partner-panel rows for off-rank neighbor atoms,
applies the three derivative actions to its local rows, and reduces only
bounded flux blocks to the owner of each core-mode block. No rank loads the
full sparse graph or eigensystem. Core-mode blocks are assigned round-robin to
owner ranks, which write disjoint `rows_*.npz` checkpoints; the manifest records
source hashes, dtypes, units, broadening policy, mode and coordinate ownership,
completed core ranges, per-rank timing, peak RSS, and pair counts. Valid blocks
are reused on restart, and existing results are rejected if the calculation
configuration changes. Row shards fix the rank count. The original `Phonons`, `SecondOrder`, and
`HarmonicWithQ` APIs are unchanged; the distributed harmonic is an adapter to
the same atom-block file contract, and the distributed eigenvector sources
derive from the existing NumPy source interface.

The precomputed-eigenvector workflow is:

```bash
mpiexec -n 4 python -m kaldo.amorphous.cli shard \
  --eigensystem _eigensystem.npy --omega omega.npy \
  --physical physical_mode.npy --output row_shards
mpiexec -n 4 python -m kaldo.amorphous.cli reduce \
  --harmonic sparse_out --eigenvectors row_shards --output qhgk_out \
  --temperature 300 --volume ANGSTROM3 --bandwidth 0.035
```

When `_eigensystem.npy` is memory-mapped, sharding now reads only the
coordinate rows owned by each rank for each mode panel. The one-time sharding
step therefore avoids reading a full global panel on every rank.

`SlepcRowEigenvectorSource` is an optional child source that builds a
distributed PETSc matrix from the same sparse atom blocks and asks SLEPc for
all real-symmetric eigenpairs in a specified eigenvalue interval. Each rank
writes its local PETSc vector rows directly into a row-shard file. PETSc and
SLEPc are imported only for this solve and are not installed in this local
environment. The solve requires real `float64` PETSc scalars. For more than
one MPI rank the adapter selects MUMPS with inertia reporting, so PETSc must
be built with MUMPS; the eigenvector checkpoint may still be `float32`.
A full-spectrum solve uses:

```bash
mpiexec -n 4 python -m kaldo.amorphous.cli solve \
  --harmonic sparse_out --output solved_rows
```

For spectrum slices, the CLI accepts explicit eigenvalue boundaries, an
approximate DOS histogram (`bin_edges` and `density` in an `.npz`), or a sorted
full eigenvalue array for optional exact planning. Boundaries must cover the
full dynamical spectrum, including unstable modes. Supply a verified global
maximum bandwidth to form conservative triangular halos:

```bash
mpiexec -n 4 python -m kaldo.amorphous.cli windows \
  --harmonic sparse_out --dos eigenvalue_dos.npz --windows 16 \
  --max-bandwidth 0.1 --temperature 300 --volume ANGSTROM3 \
  --bandwidth-table linewidths.npz --output sliced_qhgk
```

The interval pipeline checks each window's converged eigenpairs and residual
estimates, reduces only its half-open core, verifies that the merged core count
equals `3*N`, and writes one final sorted mode vector. It resumes completed
window solves and core checkpoints. The local integration test uses a
precomputed interval solver to exercise that entire dataflow; a PETSc/SLEPc
installation and target-machine benchmarks are still needed to verify the
production solver and select full-spectrum versus sliced execution.
Interval halos include a small outward numerical margin so eigenvalues exactly
on a planning boundary remain visible even with zero linewidth.

The MPI integration tests (1, 2, and 4 ranks, interrupted restarts, the
interval pipeline, and the CLI) carry the `mpi` marker and are skipped when
`mpiexec` or `mpi4py` is unavailable:

```bash
pytest -m mpi kaldo/tests/test_amorphous_distributed_mpi.py
```

The benchmark and saved-data validation scripts live in
`examples/amorphous_gamma_qhgk/`, outside the installed package. The 512-atom
saved-data gate is opt-in because that untracked directory may be absent:

```bash
python examples/amorphous_gamma_qhgk/validate_512.py EXAMPLE_PATH --volume ANGSTROM3
```

Pass `--file STEM=RELATIVE_PATH` for each ambiguous property name. The script
compares saved flux and heat-capacity blocks, saved conductivity, block sizes
16/64/257/512/1536, CSR versus dense operators, mixed precision, and peak
process RSS. Keep the example out of git.

For a synthetic precomputed-eigenvector memory and I/O probe, run:

```bash
python examples/amorphous_gamma_qhgk/benchmark.py --atoms 32 64 128 --block 32
```

For distributed synthetic resource measurements, run
`mpiexec -n 2 python examples/amorphous_gamma_qhgk/distributed_benchmark.py --atoms 32 64 --block 32`.
Before attempting 122,880 atoms, benchmark the optional SLEPc full solve
against spectrum windows on the target machine. The local synthetic probes
show memory and communication shapes only; they do not validate target-scale
runtime or physical accuracy. A full interval still retains an eigenvector
set of quadratic size during the solve; smaller spectrum windows bound that
solver workspace but repeat sparse shifted factorizations and halo modes.

## Precision policy

Frequencies and linewidths are kept in `float64`. Bose heat capacity is
dimensionless (`C/k_B`); `k_B` and the SI/volume factor enter only when each
finished mode is written. Real Gamma self-mode flux is enforced as exactly
zero, avoiding a false contribution from projected skew-operator roundoff.
The default `float32` panel/operator work with `float64` accumulation is a
performance starting point, not a uniform accuracy guarantee. For narrow
linewidths or near-degenerate modes, use `--compute-dtype float64` and compare
the total and important per-mode contributions with a `float64` reference.
Keeping sparse dynamical blocks in `float64` is cheap relative to storing the
full eigenvector set. A `float32` eigenvector checkpoint with `float64` work
halves checkpoint I/O while retaining double-precision contractions; validate
that storage cast with eigenpair residual and orthogonality checks on each
target-machine pilot.

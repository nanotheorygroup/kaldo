# Colored finite-difference IFC2: design and implementation status

**Scope.** Start with a periodic, Gamma-only `supercell=(1,1,1)` structure
in the calculator's ASE cell. Require `0 < store_radius <= color_radius` and
a positive displacement. Warn when `color_radius` exceeds half a perpendicular
cell width. The estimator's decay profile and optional calculator cutoff hint
are evidence, not proof, of the true force-dependency range. A too-small
coloring radius can contaminate *stored* responses, which outside-pattern
diagnostics alone cannot bound. Test a many-body model needing twice its
descriptor cutoff.

**Components and API.** Put neighbor construction, CPR coloring, and range
estimation in `kaldo/controllers/colored_fd/structure.py` and `estimate.py`;
put immutable plan I/O, sharded force tasks, and sparse assembly in
`plan.py`, `run.py`, and `assembly.py`. Expose
`estimate_interaction_range(atoms, calculator, *, delta_shift=1e-3,
n_probe=12, seed=0, checkpoint_dir=None)`,
`plan_colored_second_order(atoms, color_radius, store_radius, *,
delta_shift=1e-3, plan_dir, seeds=...)`,
`run_colored_second_order(plan_dir, calculator, *, shard=(0, 1), workdir,
calculator_description)`, and
`merge_colored_second_order(plan_dir, shard_dirs, *, apply_acoustic_sum=True,
asr_mode='projection', symmetrize=True, asr_tolerance=1e-10,
asr_max_iter=1000)`.
The result is `SparseSecondOrder` in
`kaldo/observables/sparse_secondorder.py`, holding float64 3×3 SciPy BSR
blocks in eV/Å², with save/load, guarded `to_dense()`, mass-weighted BSR and
`LinearOperator` accessors. The staged sparse path is available through the
functions above and `python -m kaldo.controllers.colored_fd.build_dynmat`.
It does not route through `SecondOrder.calculate`, assign
`SecondOrder.value`, or write `second.npy`. Dense calculation and
`ForceConstants` defaults remain unchanged. A merged result carries its
structure and masses (`atoms.npz`) and loads into the existing API with
`ForceConstants.from_folder(merged_dir, format='colored', only_second=True)`,
which builds a dense one-cell `SecondOrder` for small cells.

**122,880-atom Gamma matrix workflow.** The command has three stages:

```bash
python -m kaldo.controllers.colored_fd.build_dynmat plan \
  --workdir /scratch/colored-si --structure amorphous.extxyz \
  --calculator my_potential:make_calculator --delta-shift 0.01 --shards 16
python -m kaldo.controllers.colored_fd.build_dynmat run \
  --workdir /scratch/colored-si --index 0
# Submit `run` for indices 0 through 15, one index per job.
python -m kaldo.controllers.colored_fd.build_dynmat finalize \
  --workdir /scratch/colored-si
```

`plan` probes force responses and uses the recommendation at `--tolerance`
(default `1e-5`) for `R_store`. It sets `R_color` to at least twice the
potential cutoff. The cutoff may be detected from the calculator or supplied
with `--cutoff-hint`; when both exist, the larger one wins. Planning
stops when the cutoff or probe radius is unavailable. A potential with a
receptive field longer than two cutoffs needs a larger
`--cutoff-multiplier`. The displacement must be selected for the calculator's
force precision; the probe includes a displacement convergence diagnostic.
The factory must be importable and take no arguments. Set
`--calculator-description` to a stable model/version identity when changing a
model file does not change the factory path.

Each shard resumes from its own checkpoint. `finalize` checks merge diagnostics
and writes `dynmat_bsr.npz`, a float64 SciPy 3×3 BSR dynamical matrix in
kALDo units, plus `dynmat_metadata.json`. It retains the plan, force chunks,
merged sparse IFC2, and diagnostic report. It also writes `harmonic/`, the
mass-weighted atom-block input of the distributed QHGK reducer. Its pair
displacements are minimum-image vectors `r_a - r_b`: planning already
requires the coloring radius to stay below half the narrowest cell width, so
every stored pair has exactly one shortest image, and the derivative operators
then equal the existing Wigner-Seitz amorphous Gamma derivatives.

**Layout and conventions.** The dense calculation in
`kaldo/controllers/displacement.py` returns
`second[0, displaced_atom, displaced_axis, replica, responding_atom,
responding_axis]` with shape `(1,n_unit,3,n_rep,n_unit,3)`. It computes
`-ΔF/(2 delta_shift)` in eV/Å²; mass weighting and the factor
`ase.units.mol/(10*ase.units.J)` occur later in
`SecondOrder.calculate_dynmat()`. For Gamma, define BSR row `a` as the
responding atom and column `b` as the displaced atom, so
`to_dense()[0,b,alpha,0,a,beta] == bsr[3*a+beta,3*b+alpha]`, including
unsymmetrized data. The existing `acoustic_sum_rule` subtracts the complete
response-atom sum from each displaced atom's onsite block. Thus
`match_dense` explicitly subtracts the corresponding **BSR column** sum from
its onsite block, after any sparse transpose symmetrization. Its result need
not remain exactly symmetric. The default `projection` mode enforces both
transpose symmetry and zero row sums on the fixed block graph. It solves six
signless-graph and three graph-Laplacian dual systems by conjugate gradients,
with one gauge anchor per connected component. This computes the same minimum
change as alternating orthogonal projections without their slow convergence
on long sparse graphs. The merge logs options, residuals, iterations, and
algorithm version to stdout, including when a cached result is reused.
Dense `symmetrize=True` is a space-group average, distinct from transpose
averaging in the proposed sparse path.

**Execution and storage.** Build atom-level CSR structure using ASE's neighbor
list in the calculator cell, include the diagonal, and cache the validated
coloring and `R_store` row-neighbor lists in a hashed plan. Wrap SciPy's
private `group_columns` with a validated greedy fallback; run multiple fixed
orderings and verify that each structure row contains at most one atom per
group. Tasks are `(group, Cartesian axis)` in a fixed order, assigned by task
ID modulo shard count. Each shard writes decompressed block chunks and a
manifest atomically, with plan/calculator hashes and completed task IDs;
merge rejects gaps and duplicates and builds BSR without a dense intermediate.
The plan records structure and plan hashes, cell, PBC, radii, units, calculator
description, seeds, and software versions; progress uses kALDo's logger.

**QHGK integration and later replicas.** `SparseSecondOrder` stores *ordinary*
IFC2 plus its `atoms`; `dynmat`, `dynmat_operator`-style accessors,
`derivative_operators()`, and `to_sparse_harmonic()` derive the mass-weighted
QHGK inputs, using explicit per-block displacements when given and minimum
images otherwise. A replicated cell does
have a clean rectangular raw finite-difference map of shape
`(3*n_rep*n_unit, 3*n_unit)` in response-row order, but a direct transpose
does not implement periodic replica symmetry, and `SecondOrder.value` has a
separate translation-support axis. Keep replicated cells out of the first
colored API with a clear error; extend only after translation-aware symmetry,
ASR, and pair-image mapping are defined.

**Validation and status.** Focused tests cover the structure, coloring, range
estimator, sharded runs, sparse merge, transpose symmetrization, sparse ASR,
and the staged `build_dynmat` command; shared toy calculators live in
`kaldo/tests/colored_fd_helpers.py`. A synthetic 46×46×46 periodic
nearest-neighbor graph (97,336 atoms, 681,352 stored blocks) completed
symmetry and projection ASR in 1.13 s with 170.6 MB peak RSS. The six
symmetric solves took 31 iterations each and the three skew solves 249–255;
final relative row-sum and transpose residuals were about 1.0×10⁻¹¹ and zero.
That is an algorithm benchmark, not a force-calculator or production run.
Still open: a representative 100k-atom production benchmark.

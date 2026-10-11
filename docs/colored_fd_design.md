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
`LinearOperator` accessors. The remaining API integration will add
`method='dense', color_radius=None,
store_radius=None` to `SecondOrder.calculate(...)`. `method='colored'` uses
`scratch_dir` for plan/run/merge, defaults to projection ASR, and returns
`SparseSecondOrder` without assigning it
to the dense `SecondOrder.value`. Dense calculation, `second.npy`, and
`ForceConstants` defaults remain unchanged.

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

**Amorphous integration and later replicas.** The new object stores *ordinary*
IFC2; `kaldo.amorphous.sparse_harmonic.from_sparse_force_constants` already
provides the mass-weighting/ASR bridge to the amorphous QHGK block contract
(use `apply_acoustic_sum=False` after ASR), provided unique directed pair
displacements can be assigned. That contract
also stores pair vectors, which a bare BSR does not. A replicated cell does
have a clean rectangular raw finite-difference map of shape
`(3*n_rep*n_unit, 3*n_unit)` in response-row order, but a direct transpose
does not implement periodic replica symmetry, and `SecondOrder.value` has a
separate translation-support axis. Keep replicated cells out of the first
colored API with a clear error; extend only after translation-aware symmetry,
ASR, and pair-image mapping are defined.

# Gonze-Lee NAC Velocity Design

## Goal

Implement Gonze-Lee NAC group velocity support in kALDo by matching phonopy's
finite-difference velocity workflow on a fixed diagnostic q-point set, and ship
the public `HarmonicWithQ.velocity` Gonze path once the parity suite passes.

This project is intentionally production-path-first. The same internal helper
chain used by the parity tests should become the shipped Gonze velocity path.
Passing the parity suite should therefore mean the public implementation is
ready to expose, not merely that a separate debug harness matches phonopy.

## Ground Truth and Comparison Policy

- Phonopy velocity debug output is the reference truth for this project.
- The active reference tree is:

  `/home/nwlundgren/data/rephonopy/.worktrees/gonze-velocity-debug-dump/example/nacl-att3/debug-velocity`

- The active kALDo input side remains the same NaCl Gonze test fixture already
  used in the current NAC debugging work:
  - force constants from `examples/nacl_phonopy_v2`
  - NAC metadata injected from `examples/nacl_phonopy/espresso.ifc2`
- Velocity debugging should use the exact tensor layout and runtime ordering
  present in the phonopy dump rather than introducing a kALDo-specific debug
  schema.
- When parity fails, tests should fail on the earliest wrong intermediate in the
  runtime flow. No `xfail` should be used for the new velocity parity layers.

## Scope

This design covers:

- adding non-public Gonze velocity helpers in `HarmonicWithQ`
- adding phonopy-parity reference helpers for the `debug-velocity` tree
- comparing central-q, finite-difference, degeneracy-handling, and final
  velocity tensors on a shared four-point diagnostic set
- routing `HarmonicWithQ(..., nac_method="gonze").velocity` through the new
  implementation once parity passes

This design does not cover:

- widening the diagnostic set beyond four q-points in the first pass
- refactoring the whole NAC architecture into a new abstraction layer
- adding a new persistent cache for velocity derivative tensors
- changing the legacy NAC velocity path except where shared infrastructure
  naturally overlaps

## Diagnostic Q-Point Set

Use the same four diagnostic q-points already established by the att3 acoustic
debugging work:

- `q-00000`
- `q-00013`
- `q-00020`
- `q-00030`

These points are enough for the first velocity rollout because they include:

- Gamma behavior and cutoff handling
- an interior q-point already used as an acoustic-debug anchor
- a mid-path point
- the G->X endpoint used in existing Gonze frequency work

The q-point set should be defined once in shared test helpers and reused by all
velocity parity tests.

## Approaches Considered

### Recommended: Production-Path Layered Parity

Implement the Gonze derivative and velocity workflow directly in
`HarmonicWithQ`, expose detailed debug tensors from that same path, and build
layered phonopy-parity tests around it.

This is preferred because it avoids maintaining a separate test-only velocity
implementation. Once the parity suite passes, the public Gonze velocity path can
be enabled with confidence that the shipped codepath is what was tested.

### Alternative: Test-Only Reimplementation First

Build a standalone test harness that reproduces phonopy velocity assembly, use
that to understand the dump, and port the logic into production code later.

This would be faster for early exploration but would weaken the connection
between parity passing and shipped behavior.

### Rejected: Final-Velocity-Only Checks

Implement public Gonze velocity first and compare only the final scaled
velocities.

This is rejected because it would hide the first wrong intermediate and make the
debugging loop much slower.

## Architecture

The Gonze velocity path should be added as a dedicated internal pipeline inside
`kaldo/observables/harmonic_with_q.py`, but split into small units with clear
responsibilities.

Recommended internal units:

- a top-level Gonze velocity orchestrator that computes and optionally dumps all
  layered intermediates for one q-point
- a helper that computes the central Gonze dynamical matrix and eigensystem once
  for velocity use
- a helper that computes one finite-difference derivative matrix for one of the
  four phonopy directions
- a helper that performs degeneracy-aware projection from derivative matrices to
  raw group-velocity matrix elements
- a helper that applies phonopy-aligned cutoff and prefactor scaling to produce
  final velocities

The production and debug paths must be identical. Debug dumping should observe
the production path rather than branch away from it.

Test-side structure should stay separate from production code:

- extend `kaldo/tests/gonze_debug_reference.py` with velocity-specific reference
  loaders and formatting helpers
- add a dedicated Gonze velocity parity test module rather than folding all new
  coverage into the existing NAC API test file
- keep the current acoustic-debug helpers available where they already fit, but
  avoid mixing acoustic-debug and velocity-debug responsibilities into one large
  helper surface

## Data Flow

For one q-point, the Gonze velocity path should follow phonopy's runtime order:

1. Compute or reuse the Gonze dynamical matrix at the central q-point.
2. Diagonalize it once to obtain `dm_q`, `eigenvalues`, `eigenvectors`, and
   `frequencies`.
3. Build the four finite-difference derivative matrices using phonopy's
   direction set:
   - `d0 = [1, 2, 3] / sqrt(14)`
   - `d1 = [1, 0, 0]`
   - `d2 = [0, 1, 0]`
   - `d3 = [0, 0, 1]`
4. For each direction, compute `dq_cart`, convert to reduced `dq_red`, evaluate
   `dm_minus` and `dm_plus`, then form `delta_dm` and `ddm_fd`.
5. Use `d0` to rotate within degenerate subspaces and use `d1..d3` to extract
   the raw directional matrix elements that become `gv_raw`.
6. Apply the phonopy cutoff mask and scaling prefactor to produce `gv_scaled`.
7. Return `gv_scaled` from the public velocity API.

The central-q eigensystem should be reused within one `HarmonicWithQ` instance
so the public path and debug path cannot diverge through accidental
re-diagonalization differences.

Caching should remain conservative:

- reuse the existing `SecondOrder` Gonze reconstruction cache
- reuse the central-q eigensystem locally in the `HarmonicWithQ` instance
- do not add a new persistent cache for finite-difference derivative tensors in
  the first implementation

## Error Handling

Velocity-specific failures should be explicit and local to the Gonze velocity
pipeline.

- The internal finite-difference helper should validate the supported direction
  indices and raise a targeted error on invalid input.
- The Gonze velocity path should require the same NAC metadata already required
  by the Gonze dynamical-matrix path and fail with a direct message if that data
  is missing.
- Degenerate-subspace handling should fail loudly if non-finite or malformed
  eigensystem data prevents deterministic projection.
- Gamma and near-Gamma behavior should follow the phonopy reference policy
  rather than introducing a kALDo-specific special case.

The debug output should make the active failure boundary visible by emitting the
same runtime layer names used in the parity tests.

## Testing Strategy

The test suite should be layered in runtime order and use the shared four-point
diagnostic set.

### 1. Reference Fixture Coverage

Add helper tests that verify the velocity dump is present and correctly shaped:

- top-level q-point tensors exist
- `d0..d3` directories exist for each diagnostic q-point
- shared helper functions load top-level and per-direction tensors correctly

### 2. Central-Q Parity

At each diagnostic q-point, compare:

- `q_red`
- `q_cart`
- `dm_q`
- `eigenvalues`
- `eigenvectors`
- `frequencies`
- `degenerate_sets`
- `nac_branch`

### 3. Per-Direction Finite-Difference Parity

For each `d0..d3`, compare:

- `direction_cart`
- `dq_cart`
- `dq_red`
- `dm_minus`
- `dm_plus`
- `delta_dm`
- `ddm_fd`

### 4. Velocity Assembly Parity

Compare:

- `gv_raw`
- `gv_scaling_prefactor`
- `gv_cutoff_mask`
- `gv_scaled`

### 5. Public API Parity

Once the internal parity layers pass, replace the existing Gonze
`NotImplementedError` expectation with tests that confirm the public
`HarmonicWithQ.velocity` path returns the same `gv_scaled` reference values.

## Tolerance Policy

- Use exact or near-exact comparisons for bookkeeping tensors such as q-points,
  direction vectors, and masks.
- Use numerical tolerances consistent with the existing att3 debug workflow for
  matrix and tensor parity.
- Failure messages should include q-point, direction when applicable, tensor
  name, maximum absolute difference, and relative difference.
- If a tensor requires a different tolerance, document that exception in the
  test helper rather than ad hoc in individual test bodies.

## Public API Rollout

This project should remove the current Gonze velocity `NotImplementedError` once
the layered parity suite passes.

The final public behavior should be:

- `HarmonicWithQ(..., nac_method="gonze").velocity` uses the new Gonze
  finite-difference path
- `Phonons(..., nac_method="gonze")` receives Gonze velocities through the same
  route without a second implementation
- the legacy NAC path remains available and behaviorally unchanged

This rollout is part of the same project, not a separate follow-up.

## Success Criteria

Success is:

1. the new velocity parity suite fails on the earliest wrong intermediate rather
   than only on final velocity mismatch
2. the same internal Gonze helper chain powers both debug parity tests and the
   shipped public velocity path
3. the four diagnostic q-points pass layered parity against phonopy's
   `debug-velocity` dump
4. the public Gonze velocity API no longer raises `NotImplementedError`
5. existing legacy NAC behavior remains intact while Gonze velocity support is
   added

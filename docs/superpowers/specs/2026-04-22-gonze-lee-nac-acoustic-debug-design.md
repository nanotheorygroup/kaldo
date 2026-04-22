# Gonze-Lee NAC Acoustic Debug Design

## Goal

Close the current testing gaps around the acoustic mismatch in the Gonze-Lee NAC
path by adding deterministic parity tests against phonopy debug output and using
those tests to drive implementation debugging.

This effort combines test-gap filling and implementation debugging. The tests are
not just regression coverage; they are the mechanism for identifying the first
wrong intermediate quantity in kALDo relative to phonopy.

## Ground Truth and Comparison Policy

- Phonopy debug output is the reference truth for this workflow.
- The active reference tree is the `att3` debug output:

  `/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replay/example/nacl-att3/debug`

- The active kALDo input side remains `examples/nacl_phonopy_v2`, with NAC data
  injected from `examples/nacl_phonopy/espresso.ifc2` as already established by
  the current tests.
- Tensor and frequency parity checks use a `2%` tolerance envelope unless a
  tighter existing test is already justified and should be preserved.
- When a mismatch is exposed, the test should fail normally. No `xfail` is used
  for the new diagnostic layers.

## Scope

This design covers:

- adding a shared four-point diagnostic q-point set for att3 debugging
- expanding test coverage for runtime Gonze intermediates and reconstructed
  short-range force constants
- aligning derived-tensor comparisons with phonopy's saved tensor conventions
- using the tests to identify the first failing intermediate before debugging
  downstream frequencies

This design does not cover:

- full 31-point path tests
- velocity or derivative support
- refactoring the Gonze implementation into a new architecture
- removing the legacy NAC path

## Diagnostic Q-Point Set

Use a fixed four-point diagnostic set rather than a full path sweep:

- `q-00000`
- `q-00013`
- `q-00020`
- `q-00030`

These points give:

- Gamma behavior
- an early non-Gamma point
- a mid-path point
- the path endpoint used by the existing `G -> X` checks

The q-point set should be defined in one shared test helper so every parity test
uses the same anchor points.

## Recommended Approach

Use a layered diagnostic-point suite that follows the actual runtime and
reconstruction flow.

This is preferred over frequency-first testing because the current issue is not
that frequencies are merely wrong; it is that the first bad intermediate is not
fully covered. The tests should make the first broken quantity explicit so the
debug target is always the earliest failing layer.

## Test Architecture

Add a shared helper layer in the test area, not in production code.

The helper layer should:

- define the shared four diagnostic q-points
- load phonopy static tensors and q-point tensors
- run `HarmonicWithQ(..., nac_method="gonze", nac_debug=True)` for the same
  q-points
- compare named tensors with consistent reporting
- emit compact failure messages containing:
  - q-point name
  - tensor name
  - shape
  - max absolute difference
  - relative difference

The helper API should be reusable across both runtime-parity and
reconstruction-parity tests so the comparison logic is not duplicated.

## Test Layers

### 1. Static and Mapping Parity

Keep and extend the existing checks for static Gonze setup data and mapping
objects that feed both runtime assembly and short-range reconstruction.

This layer covers:

- static Gonze data such as `born`, `dielectric`, `reciprocal_lattice`,
  `masses`, `G_list`, `dd_q0`, `dd_limiting`
- mapping and geometry tensors such as `p2s_map`, `s2p_map`, `s2pp_map`,
  `p2p_map`, `multi`, `svecs`

This layer is expected to pass and should remain a prerequisite for treating a
downstream failure as actionable.

### 2. Runtime Intermediate Parity

Add explicit parity tests for the tensors produced in
`HarmonicWithQ._calculate_gonze_dynamical_matrix` in runtime order.

At each diagnostic q-point, compare:

- `q_cart`
- `q_direction_cart`
- `dd_recip`
- `dd_real`
- `dd_limiting_expanded`
- `dd_drift`
- `dd_total_mass_weighted`
- `dm_short`
- `dm_final`
- `eigenvalues`
- `frequencies`

This layer should use phonopy's saved tensor convention directly. In particular,
`dd_total_mass_weighted` must be interpreted according to
`dd_total_mass_weight.md`:

- `dd_recip`, `dd_limiting_expanded`, and `dd_real` are combined directly
- `dd_drift` is subtracted only on diagonal atom blocks
- mass-weighting is then applied blockwise
- no extra `nac_factor * (...)` is introduced at that save site

The runtime layer should answer whether the first bad quantity appears in the
NAC assembly path or only later in the short-range branch.

### 3. Reconstruction Parity

Add new parity tests for the reconstructed short-range branch owned by
`SecondOrder.gonze_short_range_force_constants`.

This layer covers:

- parity of reconstructed `gonze_short_range_force_constants` against phonopy's
  `short_range_force_constants.npy`
- parity of `dm_short` rebuilt from those reconstructed FCs at the four
  diagnostic q-points
- any additional reconstruction intermediate that becomes necessary once the
  first failing quantity is isolated

This is the highest-priority gap to close because the current evidence points to
the short-range reconstruction path as the most likely source of the acoustic
mismatch.

### 4. Frequency Parity

Keep frequency tests, but treat them as downstream confirmation rather than the
first diagnostic signal.

Frequency tests should remain on the same four q-points and compare both
acoustic and optical modes against phonopy within the shared tolerance rule.

The purpose of this layer is to confirm that once the first failing intermediate
is fixed, the acoustic mismatch and any remaining final-frequency mismatch move
or disappear in an understandable way.

## Debugging Workflow

The debugging workflow is built into the test rollout.

Implementation and debugging should proceed in this order:

1. Add the shared diagnostic q-point helper and standardize comparison output.
2. Correct the `dd_total_mass_weighted` comparison logic to match the phonopy
   saved convention from `dd_total_mass_weight.md`.
3. Add runtime-intermediate parity tests at the four q-points.
4. Add reconstruction-parity tests for
   `gonze_short_range_force_constants` and `dm_short`.
5. Keep frequency parity tests last and use them only after the first bad
   intermediate is known.

When the suite fails, the active debug target is the earliest failing tensor in
this order, not the final frequency mismatch.

## Success Criteria

Success is:

1. the new four-point diagnostic tests fail normally on the first wrong
   quantity
2. the failure messages identify the tensor and q-point clearly enough to debug
   without rerunning ad hoc scripts first
3. acoustic frequency mismatches are explained by an earlier tensor mismatch,
   not treated as isolated final-output failures
4. once an intermediate parity layer passes, the next failing layer becomes the
   active debug target with no ambiguity about what to work on next

# Gonze-Lee NAC `att3` Debug Design

## Goal

Retarget the current kALDo Gonze-Lee NAC debugging workflow to the new NaCl `att3`
reference so we can compare intermediate phonopy-style tensors in output order and
work upward to `dm_final`, eigenvalues, and frequencies.

This example takes precedence over the earlier `att2` case.

The kALDo input side for this work is `examples/nacl_phonopy_v2`, with the modified
`espresso.ifc2` treated as the active force-constant source. The phonopy reference
side is:

`/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replay/example/nacl-att3/debug`

## Scope

This design covers:

- file-based debug comparison against the `att3` phonopy debug tree
- parity checks for static tensors and selected q-point tensors
- structured reporting of the first wrong tensor in phonopy output order
- frequency comparison only after the intermediate tensors are understood

This design does not cover:

- refactoring the Gonze path into a new helper module first
- velocity or derivative support
- removal of the legacy NAC path
- broad generalization beyond what is needed for the `att3` example

## Current Context

The current checkout already has an inline Gonze implementation in
`kaldo/observables/harmonic_with_q.py`. That path will remain the active execution
path for this phase of work.

The new `att3` reference differs materially from the earlier determinant-32 case:

- `supercell_matrix = diag(8, 8, 8)`
- `p2s_map = [0, 512]`
- `s2p_map.shape = (1024,)`
- `multi.shape = (1024, 2, 2)`

So this second example should be treated as a new target representation, not as a
small variation of `att2`.

## Recommended Approach

Keep the physics path where it is now and add a small reusable reference/debug
harness so every comparison is measurable.

This is preferred over backporting the cleaner helper split first because the next
unknown is physical parity against the new `att3` reference, not code structure.
The fastest way to learn where the mismatch starts is to compare the existing path
step by step against the new debug tree.

## Architecture

The work is divided into three bounded pieces.

### 1. Reference and Debug Harness

Add a small helper layer that can:

- load static arrays from a phonopy debug tree
- load q-point arrays by q-index or q-folder name
- load the corresponding kALDo debug arrays
- compare named tensors with consistent metrics

For each compared tensor, the harness should report:

- tensor name
- source path
- shape and dtype
- max absolute difference
- relative difference against the reference norm
- optional first mismatching index for targeted debugging

The harness should be usable from tests and from an ad hoc debug workflow so the
same comparison logic is not duplicated.

### 2. `att3` Retargeting of the Inline Gonze Path

Use `examples/nacl_phonopy_v2` as the active kALDo input source.

The example, tests, and any debug-path defaults should point to the `att3`
phonopy debug tree instead of `att2` for this workflow. The force constants used by
kALDo must come from `examples/nacl_phonopy_v2/espresso.ifc2`.

The current inline Gonze implementation in
`kaldo/observables/harmonic_with_q.py` remains the active execution path. The goal
here is not an architectural cleanup. The goal is to make the existing path
measurable against the new reference.

### 3. Ordered Tensor-Parity Checkpoints

Drive the investigation in phonopy output order and stop at the first wrong tensor.

Static checkpoint group:

- `primitive_cell`
- `supercell_cell`
- `reciprocal_lattice`
- `born`
- `dielectric`
- `masses`
- `G_cutoff`
- `G_list`
- `Lambda`
- `nac_factor`
- `dd_q0`
- mapping tensors such as `p2s_map`, `s2p_map`, `s2pp_map`, `multi`, `svecs`

Q-point checkpoint group for anchor q-points `q-00000`, `q-00013`, and `q-00030`:

- `q_red`
- `q_cart`
- `q_direction_cart` when present
- reciprocal-space NAC terms
- limiting and drift terms
- `dd_total_mass_weighted`
- `dm_short`
- `dm_final`

Final checkpoint group:

- eigenvalues
- frequencies

At each stage, downstream tensors stay out of scope until the first bad tensor is
understood.

## Data Flow

The active data flow is:

1. Load second-order force constants from `examples/nacl_phonopy_v2/espresso.ifc2`.
2. Run the existing inline Gonze path in `HarmonicWithQ`.
3. Emit file-based kALDo debug arrays under the configured debug folder.
4. Compare those arrays against the `att3` phonopy debug tree with the harness.
5. Record the last matching tensor and the first failing tensor.
6. Only after the intermediate tensor path is consistent, compare final
   eigensystems and frequencies.

## Reporting and Trace Discipline

Maintain a dedicated markdown trace for the `att3` example.

For each checkpointed tensor, classify it as:

- `match`
- `close but convention-scaled`
- `first bad tensor`
- `downstream, not yet actionable`

When a tensor is the first bad one, record:

- how kALDo currently computes it
- the current unit convention
- any scaling, phase, mapping, or representation assumptions that feed it
- concrete theories for why it differs from phonopy

The trace should always make it obvious what the current boundary is without having
to reconstruct the debug session from terminal history.

## Testing Strategy

Testing should support both fast regression checks and deeper investigation.

Primary checks:

- loadability of the `att3` reference tree
- deterministic comparison of selected static tensors
- deterministic comparison of selected q-point tensors at `q-00000`, `q-00013`,
  and `q-00030`
- a final frequency comparison only after intermediate parity is established

The old `att2` reference should remain available as prior context, but `att3`
becomes the active target for this debug workflow.

## Success Criteria

Success is incremental.

1. The harness can compare kALDo and phonopy debug outputs for the `att3` case in a
   reusable way.
2. Static tensors and mapping tensors match the `att3` reference representation.
3. Intermediate q-point tensors match, or the first bad tensor is isolated with a
   documented explanation and concrete hypotheses.
4. `dm_final`, eigenvalues, and frequencies are only treated as active targets after
   the intermediate tensors are on solid ground.

The main success criterion for this phase is not merely a final frequency number. It
is a trustworthy, ordered debugging workflow for the new `att3` example that makes
the first real mismatch explicit.

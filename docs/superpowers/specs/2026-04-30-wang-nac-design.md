# Wang NAC Design

## Goal

Add a production `nac_method="wang"` path to kALDo so `HarmonicWithQ` can
compute Wang-style non-analytic corrections for both frequencies and group
velocities, using the phonopy Wang trace as the parity reference.

The implementation should be validated internally with NumPy debug checkpoints
and externally with a small fixed subset of q-points from the traced 31-point
`G->X` path used by the phonopy reference data.

## Context

The current checkout already contains:

- the legacy NAC path in `HarmonicWithQ`
- an in-progress Gonze path in `HarmonicWithQ`
- existing NaCl validation inputs in `examples/nacl_phonopy_v2`
- a Wang reference trace at
  `/home/nwlundgren/data/rephonopy/.worktrees/wang-nac-trace/example/nacl-att3/debug-wang-band`

That Wang trace contains 31 q-points along `G->X` and includes:

- q-local production dynmat checkpoints
- q-local derivative dynmat checkpoints
- q-local group velocity outputs

The worktree is currently dirty with unrelated Gonze work. Implementation should
therefore happen in a fresh git worktree created from `main`, but that worktree
creation is part of implementation, not part of this design document.

## Scope

This design covers:

- production frequency support for `nac_method="wang"`
- production velocity support for `nac_method="wang"`
- Wang-specific debug checkpoint dumping in NumPy format
- Wang-specific helper and API parity tests
- a 31-point example/script-level comparison for frequencies and velocities

This design does not cover:

- refactoring Wang and Gonze into a shared NAC architecture
- moving Wang logic into a separate module
- requiring all 31 q-points in automated tests
- changing the numerical behavior of `legacy` or `gonze` beyond adding support
  for the new `wang` option in shared dispatch points

## Public API

`HarmonicWithQ` should accept:

```python
HarmonicWithQ(
    q_point=q_red,
    second=second_order,
    storage="memory",
    nac_method="wang",
    nac_debug=False,
    nac_debug_folder="debug",
    nac_q_direction=None,
)
```

Behavior:

- `nac_method="wang"` is a production option, not a debug-only path.
- `frequency` should use the Wang dynamical matrix.
- `velocity` should use the Wang derivative dynamical matrix and velocity
  projection path.
- `Phonons(..., nac_method="wang")` should preserve and pass through the Wang
  option the same way the existing NAC options are propagated.

Error handling:

- Raise `ValueError` if `nac_method="wang"` is requested without both
  `atoms.info["dielectric"]` and `atoms.arrays["charges"]`.
- At Gamma with no `nac_q_direction`, follow the traced Wang behavior and skip
  the Wang NAC contribution instead of raising.
- `nac_debug` must remain optional and must not change numerical results.

## Architecture

The Wang implementation should stay in
`kaldo/observables/harmonic_with_q.py` and remain part of the
`HarmonicWithQ` class.

The class should gain a Wang-specific internal pipeline rather than a shared
NAC abstraction. This keeps Wang isolated from the in-progress Gonze code and
avoids scope growth.

Recommended Wang-specific methods:

- `_build_wang_static_data()`
- `_calculate_wang_dynamical_matrix()`
- `_calculate_wang_derivative_dynamical_matrix()`
- `_calculate_wang_velocity_data()`
- `_save_wang_debug(...)`

Small pure numerical helpers may remain module-level private functions in the
same file, but orchestration stays owned by `HarmonicWithQ`.

`SecondOrder` should not gain Wang-specific preprocessing or reconstruction
logic. Unlike the Gonze path, Wang does not require a separate short-range
force-constant reconstruction stage.

## Wang Data Flow

### Static Context

`HarmonicWithQ` should build Wang inputs directly from the existing `second`
object and the attached ASE atom metadata:

- force constants
- masses
- shortest vectors and multiplicities
- primitive/supercell atom maps
- dielectric tensor
- Born effective charges
- reciprocal lattice

These values are already available to the current Fourier assembly path and are
the natural source of truth for the Wang implementation.

### Frequency Path

For a single q-point:

1. Build or reuse Wang static data from `second`.
2. Convert reduced `q_red` to Cartesian `q_cart`.
3. Determine the NAC vector used in the Wang formula:
   - generic q: use `q_cart`
   - Gamma with `nac_q_direction`: use the provided direction in the Wang
     numerator and denominator while keeping the short-range phase at Gamma
   - Gamma with no direction: skip the Wang NAC contribution
4. Compute the Wang intermediates in the same order as the phonopy trace:
   - `q_cart`
   - `q_norm`
   - dielectric denominator `C`
   - `A` / `q_born`
   - Wang scalar prefactor
   - primitive Wang correction tensor `charge_sum`
5. Inject that primitive correction tensor into each matching supercell image
   during the usual phase-weighted dynamical-matrix assembly, including the
   `1 / n` compensation described in the Wang implementation notes.
6. Hermitian symmetrize only after the full dynamical matrix is assembled.
7. Diagonalize the final Wang dynamical matrix for eigenvalues, eigenvectors,
   and frequencies.

### Velocity Path

For the same q-point:

1. Reuse the central Wang eigensystem from the frequency path when available.
2. Build the analytical Wang derivative terms:
   - derivative q-vector in Cartesian coordinates
   - `A`
   - `dA`
   - `C`
   - `dC`
   - `dnac`
   - `ddnac`
3. Combine the short-range phase derivative with the Wang derivative
   contribution into a three-direction derivative dynamical matrix.
4. Hermitian symmetrize each derivative-direction matrix separately.
5. Project the derivative dynamical matrices with the phonon eigenvectors and
   apply the phonopy-style scaling used to turn eigenvalue derivatives into
   frequency derivatives and final group velocities.

### Caching

The Wang path may use per-instance private caches to avoid recomputing the same
Wang dynamical matrix and eigensystem when both `frequency` and `velocity` are
requested on the same `HarmonicWithQ` instance.

Caching must stay internal and must not change public behavior.

## Debug Artifact Contract

Debug output should be NumPy-only and kALDo-native.

Layout:

- `debug/static/*.npy` for reused Wang static inputs and mapping arrays
- `debug/q-00013/*.npy` for per-q runtime arrays

The checkpoint names should stay close to the phonopy trace vocabulary so parity
failures are easy to diagnose.

Expected static arrays include:

- `born`
- `dielectric`
- `reciprocal_lattice`
- `masses`
- `force_constants`
- `svecs`
- `multi`
- `p2s_map`
- `s2p_map`

Expected per-q frequency arrays include:

- `q_red`
- `q_cart`
- `q_direction_red`
- `q_direction_cart`
- `wang_q_norm`
- `wang_C_term`
- `wang_A_terms`
- `wang_nac_prefactor`
- `wang_charge_sum`
- `wang_dynmat_before_hermitian`
- `wang_dynmat_after_hermitian`
- `eigenvalues`
- `frequencies`

Expected per-q derivative and velocity arrays include:

- `wang_derivative_q_cart`
- `wang_dA_terms`
- `wang_dC_terms`
- `wang_dnac`
- `wang_ddnac`
- `wang_derivative_dynmat_after_hermitian`
- `group_velocity_eigvals`
- `group_velocity_eigvecs`
- `group_velocity_freqs`
- `group_velocities`

Additional convenience arrays are acceptable, but the parity checkpoints above
should remain stable.

## Testing Strategy

### Reference Data

Use the Wang phonopy trace at:

`/home/nwlundgren/data/rephonopy/.worktrees/wang-nac-trace/example/nacl-att3/debug-wang-band`

Automated tests should use these 4 fixed q-points from the existing 31-point
`G->X` trace so the indices line up with the phonopy reference tree:

- `q-00000`
- `q-00010`
- `q-00020`
- `q-00030`

### Helper-Level Tests

Add Wang-specific helper parity tests for:

- `q_cart`
- dielectric denominator `C`
- `A` / `q_born`
- scalar prefactor
- `charge_sum`
- `dnac`
- `ddnac`

These tests should fail at the earliest incorrect intermediate rather than only
at the final observable.

### Runtime Integration Tests

For each diagnostic q-point, run
`HarmonicWithQ(..., nac_method="wang", nac_debug=True)` and compare:

- final dynamical matrix
- eigenvalues
- frequencies
- derivative dynamical matrix
- final group velocities

### Public API Tests

Add Wang-specific API tests to verify:

- `nac_method="wang"` is accepted
- `frequency` works through the Wang path
- `velocity` works through the Wang path
- `Phonons(..., nac_method="wang")` preserves and forwards the option correctly

### Example-Level Validation

Keep the full 31-point `G->X` comparison as an example/script-level validation,
not as the core automated test gate.

The existing example comparison script in
`examples/nacl_phonopy_v2/plot_gx_dispersion_compare.py` should be updated or
extended to run the Wang path and compare both frequencies and velocity
magnitudes along the full traced path.

The script should report the first failing q-index and summary error metrics.

## File Change Plan

Modify:

- `kaldo/observables/harmonic_with_q.py`
  - add support for `nac_method="wang"`
  - add Wang-specific static/runtime builders
  - add Wang dynamical-matrix assembly
  - add Wang derivative dynamical-matrix assembly
  - add Wang velocity projection/scaling path
  - add Wang debug save helpers

Add:

- `kaldo/tests/wang_debug_reference.py`
- `kaldo/tests/test_wang_nac_helpers.py`
- `kaldo/tests/test_wang_nac_api.py`
- `kaldo/tests/test_wang_nac_velocity.py`

Modify:

- `examples/nacl_phonopy_v2/plot_gx_dispersion_compare.py`

## Success Criteria

The project is complete when all of the following are true:

1. `HarmonicWithQ(..., nac_method="wang").frequency` matches the phonopy Wang
   reference on the 4 diagnostic q-points within the chosen test tolerances.
2. `HarmonicWithQ(..., nac_method="wang").velocity` matches the phonopy Wang
   reference on the 4 diagnostic q-points within the chosen test tolerances.
3. The saved Wang debug checkpoints match the selected traced reference arrays
   for the 4 diagnostic q-points.
4. The example-level 31-point `G->X` script compares Wang frequencies and
   velocities against the full traced path and reports usable summary metrics.
5. Existing `legacy` and `gonze` behavior remains unchanged except for shared
   dispatch sites recognizing the new Wang option.

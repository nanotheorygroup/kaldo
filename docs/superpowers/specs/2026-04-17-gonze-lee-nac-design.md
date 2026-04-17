# Gonze-Lee NAC Design

## Goal

Add a `nac_method="gonze"` option so kALDo can calculate full phonopy-style Gonze-Lee non-analytic corrections for harmonic frequencies without importing phonopy. The first implementation targets the NaCl `nacl-att2` debug data and should match reference frequencies within about 2%, allowing for the known charge and force-constant differences between the kALDo-prepared inputs and the phonopy debug run.

The existing NAC implementation remains available as the default `nac_method="legacy"` path.

## Scope

This design covers frequencies only. Group velocities and Gonze-Lee NAC derivatives are out of scope for the first implementation.

The implementation will use kALDo, ASE, and NumPy. It will not import phonopy.

## User-Facing API

`Phonons` and `HarmonicWithQ` will accept these options:

```python
nac_method="legacy"
nac_debug=False
nac_debug_folder="debug"
```

Supported methods:

- `legacy`: preserve current behavior.
- `gonze`: use full Gonze-Lee assembly for frequencies.

Example:

```python
phonons = Phonons(
    forceconstants=force_constants,
    kpts=[7, 7, 7],
    is_unfolding=True,
    nac_method="gonze",
    nac_debug=True,
)
```

When `nac_method="gonze"` is used and `velocity` is requested, kALDo will raise `NotImplementedError` with a clear message that Gonze-Lee velocity derivatives are not implemented yet.

## Architecture

`HarmonicWithQ` remains the integration point. It will choose the NAC path from `nac_method`:

- `legacy`: continue adding `nac_dynmat(qpoint=None) + nac_dynmat(qpoint=self.q_point)` in eigensystem calculations and continue using `nac_derivatives()` for velocities.
- `gonze`: build the full dynamical matrix as `D_short(q) + D_NAC_Gonze(q)` and diagonalize that matrix for frequencies.

The Gonze-Lee arithmetic will be split into small pure NumPy helper functions. For the first implementation, these helpers will be private module-level functions in `kaldo/observables/harmonic_with_q.py`, keeping `HarmonicWithQ` as the integration point while avoiding large methods on the class.

The helper functions will mirror the replay script structure:

- dielectric quadratic form
- G-vector radius and list construction
- reciprocal dipole-dipole tensor
- Born-charge contraction
- q=0 reciprocal drift term
- limiting term
- real-space dipole-dipole term
- mass weighting
- short-range dynamical matrix assembly

## Data Flow

For `nac_method="gonze"`, each q-point calculation builds a phonopy-style full-term matrix from kALDo and ASE inputs.

Static inputs derived from `second` and `atoms`:

- primitive cell in Angstrom
- primitive positions in Angstrom
- reciprocal lattice without an explicit `2*pi`, from ASE cell reciprocal vectors
- atom masses
- Born effective charges from `atoms.get_array("charges")`
- dielectric tensor from `atoms.info["dielectric"]`
- supercell cell
- shortest-vector and multiplicity data needed for the short-range matrix

Static Gonze terms:

- `G_list`
- `dd_q0`
- `dd_limiting`
- `dd_real_q0`
- short-range force constants or an equivalent short-range force-constant view

Per q-point terms:

- `q_cart = reciprocal_lattice @ q_red`
- `q_direction_cart` for Gamma limiting behavior
- `dd_recip`
- `dd_real`
- `dd_limiting_expanded`
- `dd_drift = dd_q0 + N_atom * dd_limiting + dd_real_q0`
- `dd_total`
- `dd_total_mass_weighted`
- `dm_short(q)`
- `dm_final(q) = dm_short(q) + dd_total_mass_weighted`

The key behavioral difference from the legacy path is that the Gonze-Lee path does not add NAC onto kALDo's existing Fourier dynamical matrix. It explicitly constructs the short-range matrix and then adds the full Gonze-Lee correction, matching phonopy's full-term assembly.

## Units And Conventions

The implementation will follow the `gonze_lee_trace.md` conventions:

- reduced q coordinates do not include an explicit `2*pi`
- reciprocal lattice vectors are Cartesian and do not include an explicit `2*pi`
- phase factors use `exp(2j * pi * dot(...))`
- reciprocal phases use `G` in the atom-pair phase, not `G + q`
- `q_cart` is computed using phonopy-style reciprocal vectors
- force-constant-like NAC tensors are mass-weighted by dividing atom blocks by `sqrt(M_i * M_j)`

The NAC force-constant terms will be converted from `eV / Angstrom^2` to kALDo's `10 J / mol / Angstrom^2` convention with:

```python
units.mol / (10 * units.J)
```

This matches the trace's final dynamical-matrix conversion factor of about `9648.53`.

## Short-Range Matrix

Phonopy's full-term Gonze-Lee path computes:

```text
D_final(q) = D_short(q) + D_NAC(q)
```

Therefore kALDo must construct an equivalent short-range matrix instead of using the current full force constants directly.

The first implementation will generate the long-range real-space correction from the same Gonze-Lee terms and subtract it from the loaded second-order force constants to produce a short-range force-constant view. It will then Fourier assemble `dm_short(q)` with shortest-vector and multiplicity logic that mirrors the replay script.

If the needed atom mapping or shortest-vector information cannot be generated from the current `second` object, the Gonze path will raise a diagnostic error naming the missing piece and suggesting `nac_method="legacy"` for that system.

## Debug Output

When `nac_debug=True`, kALDo will write NumPy debug files under `nac_debug_folder`, defaulting to `debug`.

Static files go under:

```text
debug/static/
```

Planned static files:

- `born.npy`
- `dielectric.npy`
- `primitive_cell.npy`
- `primitive_positions.npy`
- `reciprocal_lattice.npy`
- `masses.npy`
- `supercell_cell.npy`
- `G_cutoff.npy`
- `G_list.npy`
- `Lambda.npy`
- `nac_factor.npy`
- `dd_q0.npy`
- `dd_limiting.npy`
- `dd_real_q0.npy`
- `short_range_force_constants.npy`

Q-point files go under deterministic q folders:

```text
debug/q-00000/
debug/q-00001/
```

Planned q-point files:

- `q_red.npy`
- `q_cart.npy`
- `q_direction_cart.npy`
- `dd_recip.npy`
- `dd_real.npy`
- `dd_limiting_expanded.npy`
- `dd_drift.npy`
- `dd_total_mass_weighted.npy`
- `dm_short.npy`
- `dm_final.npy`
- `eigenvalues.npy`
- `frequencies.npy`

`Phonons.frequency` and dispersion plotting will pass a q index into `HarmonicWithQ` so debug folder names line up with band-path or mesh order. Direct one-off `HarmonicWithQ` calls will use a sanitized q-point label when no q index is provided.

## Error Handling

The Gonze path will raise:

- `ValueError` if `nac_method` is unknown.
- `ValueError` if `nac_method="gonze"` is requested without both dielectric and Born charges.
- `NotImplementedError` if velocity is requested with `nac_method="gonze"`.
- A diagnostic runtime error if the short-range mapping machinery cannot be generated for the supplied system.

Legacy NAC behavior remains unchanged and remains the default.

## Testing

Testing will focus on frequencies and intermediate dynamical-matrix terms.

Regression checks will compare kALDo's file-based debug output to the NaCl phonopy debug tree for selected q-points:

- Gamma: `q-00000`
- an interior point, such as `q-00006` or `q-00013`
- endpoint: `q-00030`

The user-facing frequency pass criterion is about 2% relative tolerance. Intermediate tensor tests will use tighter tolerances when inputs match exactly and looser checks where the kALDo-prepared charges and force constants intentionally differ from the phonopy debug data.

Existing tests for default NAC behavior will continue to use `nac_method="legacy"` implicitly, protecting current behavior.

## Non-Goals

- No phonopy runtime dependency.
- No Gonze-Lee velocity or derivative support in the first implementation.
- No removal of the existing legacy NAC implementation.
- No broad refactor of force-constant loading outside what is needed to build the short-range matrix.

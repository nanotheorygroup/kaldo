# Choosing IFC interpolation

Interatomic force constants (IFCs) describe interactions between atoms in
periodic copies of a simulation cell. To calculate a phonon at an arbitrary q
point, kALDo must decide which periodic copy represents each interaction and
which Fourier phase belongs to it. The `is_unfolding` option of `Phonons`
controls that choice.

## The short answer

For normal calculations, keep the default:

```python
from kaldo.phonons import Phonons

phonons = Phonons(
    forceconstants=forceconstants,
    kpts=(9, 9, 9),
    # is_unfolding=False,  # This is the default.
    storage="numpy",
)
```

The default uses information recorded by the force-constant loader to select
the representation appropriate for each input. This is the recommended setting
for crystalline and periodically repeated amorphous systems.

## Available options

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>What kALDo does</th>
      <th>When to use it</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>is_unfolding=False</code></td>
      <td>Uses the representation associated with the loaded IFC source.</td>
      <td>All normal calculations. This is the default and recommended choice.</td>
    </tr>
    <tr>
      <td><code>is_unfolding=True</code></td>
      <td>Assigns every IFC block, from every source, to the shortest periodic image of its atom pair. Geometrically tied images share the block. Literal translations from TDEP and ShengBTE files are first folded into periodic classes, and QE d3q third-order IFCs use these images instead of their native direct-periodic convention. Requires periodic boundary conditions in all three directions.</td>
      <td>Controlled comparisons or an explicit compact-IFC calculation.</td>
    </tr>
  </tbody>
</table>

Despite its name, `is_unfolding` selects the interpolation of real-space force
constants. It does not unfold a phonon band structure.

The private argument `_ifc_interpolation="periodic"` selects kALDo's historical
direct-periodic representation, which uses the translation stored inside the
finite supercell instead of the shortest image of each atom pair. It is meant
for developer diagnostics and comparison with legacy NAC-off results, not for
production calculations.

## Why the default is source aware

Different file formats do not store the translation axes of their IFC tensors
in the same way. Applying one interpolation rule to every source can move,
merge, or discard interactions. The loaders therefore retain enough
provenance for the default to make the following choices.

### Compact periodic IFCs

NumPy, ESKM/LAMMPS, hiphive, GPUMD, and VASP/phonopy-style inputs normally
store one representative of each periodic supercell class. kALDo uses
pair-dependent Wigner--Seitz images for these inputs. The image is chosen from
the complete cell geometry and the positions of the two interacting atoms,
which is important for skew and low-symmetry cells.

### Quantum Espresso q2r IFC2

A Quantum Espresso (QE) q2r harmonic file uses its header geometry to build
the pair-dependent Wigner--Seitz representation. This applies whether or not
the file contains the dielectric tensor and Born charges needed for a
non-analytic correction (NAC).

### Files with explicit translations

TDEP and ShengBTE `FORCE_CONSTANTS_3RD` can store the actual lattice
translation of each interaction. The default preserves those translations rather
than folding them into a compact supercell representation.

### QE d3q IFC3

QE d3q third-order IFCs retain their validated native direct-periodic
convention. It is normal for the second- and third-order IFCs in one workflow
to resolve to different representations when they come from different
sources.

## Wigner--Seitz in this context

Here, Wigner--Seitz means a shortest-image construction in real space. For
each nonzero IFC block, kALDo finds the periodic copy that gives the shortest
Cartesian separation for that atom pair. If symmetry produces several equally
short images, kALDo distributes the block evenly among them.

This makes the result independent of an arbitrary choice of unit-cell origin.
Moving a crystal rigidly and wrapping its atoms back into the cell changes how
interactions are labelled, but it must not change frequencies, velocities,
scattering rates, or thermal conductivity.

This use of Wigner--Seitz is unrelated to the Wigner transport equation or to
the coherence contribution to thermal conductivity.

## Polar materials and NAC

For a polar calculation, keep the default. kALDo then uses the Wigner--Seitz
representation needed to combine the interpolated short-range IFCs with the
long-range dipole term.

The historical direct-periodic representation cannot be combined consistently
with active NAC, and kALDo rejects that combination. A developer who
intentionally wants the legacy short-range diagnostic must also set
`is_nac=False`:

```python
legacy_diagnostic = Phonons(
    forceconstants=forceconstants,
    kpts=(9, 9, 9),
    _ifc_interpolation="periodic",  # private developer option
    is_nac=False,
    storage="numpy",
)
```

For the underlying polar-phonon conventions, see
[Non-analytic corrections for polar crystals](https://github.com/nanotheorygroup/kaldo/blob/main/docs/docsource/non_analytic_corrections.md).

## Periodicity requirements

Pair-dependent Wigner--Seitz interpolation currently requires periodic
boundary conditions in all three directions.

- Three-dimensional crystals are supported.
- A periodically repeated amorphous bulk cell is supported. The entire
  disordered cell is the reference cell, and boundary-crossing pairs receive
  their nearest periodic images without assuming crystal symmetry.
- True slabs, nanowires, and isolated clusters are not yet supported by the
  Wigner--Seitz interpolation kernel. `is_unfolding=True` therefore raises an
  error for them.

For compatibility with earlier kALDo releases, the default instead selects the
historical periodic representation for the harmonic properties of systems that
are not periodic in all three directions. This fallback lets existing workflows
run, but it is not a validated shortest-image treatment for slabs, wires, or
isolated systems. Treat results from such models with care.

The `is_nw=True` option changes the Gamma-point acoustic-mode mask; it does not
provide axis-only IFC interpolation for a nanowire.

## Relation to earlier releases

In kALDo 2.2 and earlier, the default `is_unfolding=False` used the
direct-periodic representation. From 2.3 the default is source aware, which
means pair-dependent Wigner--Seitz images for compact periodic data. The
`is_unfolding` argument and its default are unchanged.

## Interpreting changed results

The corrected interpolation can change off-mesh dispersions, group
velocities, three-phonon matrix elements, scattering rates, and thermal
conductivity. Frequencies sampled on the force-constant supercell's
commensurate q mesh may remain unchanged because periodically equivalent
images have the same phase on that mesh. Unchanged commensurate frequencies
therefore do not prove that derivatives or transport quantities used the same
real-space geometry.

Cache names include the resolved interpolation and translation support, so
results produced by an older representation are not silently reused. Old
cache directories may remain on disk and can be deleted when they are no
longer needed.

## Troubleshooting

### A warning says explicit translations are being folded

The input file stores literal lattice translations, but `is_unfolding=True`
requested a compact representation. Keep the default unless that conversion is
the purpose of the calculation.

### The direct-periodic representation is rejected for a polar input

NAC is active. Use the default for the physical polar calculation. Use
`is_nac=False` only when deliberately constructing a legacy short-range
comparison.

### Wigner--Seitz interpolation rejects the boundary conditions

`is_unfolding=True` was set for a structure that is not periodic in all three
directions. Wigner--Seitz interpolation currently supports only fully periodic
systems.

With the default, kALDo retains the historical periodic representation for the
harmonic properties of slabs, wires, and isolated systems. That compatibility
path is not a validated shortest-image treatment for partially periodic or
nonperiodic geometries. Use a fully periodic model when Wigner--Seitz
interpolation is required, and assess other geometries according to the
assumptions of your calculation.

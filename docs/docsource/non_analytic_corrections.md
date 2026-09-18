# Non-analytic corrections for polar crystals

This guide describes how kALDo reconstructs the long-range electrostatic
contribution to harmonic phonons in three-dimensional polar crystals. It also
explains how to provide the required data, how kALDo selects a numerical
convention, and which parts of the implementation are covered by reference
tests.

## Why a correction is needed

Displacing an ion in a polar crystal produces a macroscopic electric field
through its Born effective-charge tensor. The field is screened by the
high-frequency dielectric tensor. The resulting dipole--dipole interaction is
long-ranged and therefore cannot be represented reliably by truncating
real-space interatomic force constants (IFCs) to a finite supercell.

The long-range contribution also has a direction-dependent limit as
$\mathbf q$ approaches the zone centre. A longitudinal optical mode creates a
different macroscopic field from a transverse optical mode, giving rise to
longitudinal-optical/transverse-optical (LO--TO) splitting near Gamma. The
direction used at exact Gamma is consequently physical input, not a numerical
regularization.

kALDo writes the corrected harmonic dynamical matrix schematically as

```math
D_{\mathrm{full}}(\mathbf q;\hat{\mathbf q}) =
\operatorname{FT}[\Phi_{\mathrm{short}}](\mathbf q) +
D_{\mathrm{dipole}}(\mathbf q;\hat{\mathbf q}).
```

Here, $\Phi_{\mathrm{short}}$ contains the short-range IFCs and
$D_{\mathrm{dipole}}$ is constructed from the Born effective charges and
dielectric tensor. The non-analytic correction (NAC) is the complete,
convention-consistent separation and restoration of this dipole term, rather
than an adjustment applied only to frequencies at Gamma.

## Activating NAC

The public `is_nac` option has three states:

- `None` (default): activate NAC when the loaded structure contains both a
  dielectric tensor and nonzero Born effective charges. Use the ordinary
  harmonic path when no polar-response data are present.
- `True`: require NAC. Missing or zero polar-response data raise an error
  instead of silently evaluating the ordinary harmonic model.
- `False`: disable NAC explicitly. This is useful for isolating the long-range
  contribution or comparing with a short-range model.

The dielectric tensor must be stored in `atoms.info['dielectric']` and the Born
effective charges in `atoms.arrays['charges']`. Supplying only one of these
objects is an error. The standard force-constant loaders populate them when the
selected input format contains the corresponding data.

For example, a phonon-grid calculation normally needs no NAC-specific option:

```python
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons

forceconstants = ForceConstants.from_folder(
    folder="inputs",
    supercell=(4, 4, 4),
    format="vasp-sheng",
    only_second=True,
)
phonons = Phonons(
    forceconstants=forceconstants,
    kpts=(9, 9, 9),
    # is_nac=None,  # This is the default.
    storage="numpy",
)
```

Use `is_nac=True` when the presence of a correction is a requirement of the
calculation. Use `is_nac=False` only when the uncorrected result is intentional.

### Exact-Gamma direction

Finite-q calculations use $\mathbf q$ itself to define the direction of the
long-range term. At exact Gamma,
`HarmonicWithQ(..., nac_q_direction=(h, k, l))` specifies the limiting
direction in reduced reciprocal coordinates:

```python
import numpy as np
from kaldo.observables.harmonic_with_q import HarmonicWithQ

gamma_along_c = HarmonicWithQ(
    q_point=np.zeros(3),
    second=forceconstants.second,
    is_nac=True,
    nac_q_direction=(0, 0, 1),
    storage="numpy",
)
frequencies = gamma_along_c.frequency
```

Both `Phonons(..., nac_q_direction=(h, k, l))` and
`HarmonicWithQ(..., nac_q_direction=(h, k, l))` accept an exact-Gamma
direction. The default is `(1, 0, 0)`.

### Born--von Karman cell

`nac_bvk_supercell_matrix` identifies the Born--von Karman (BvK) cell that
defines the force constants when it cannot be inferred as
`diag(forceconstants.second.supercell)`. It does not request that kALDo remesh
the IFCs and should normally be omitted. In particular, a Quantum Espresso
(QE) q2r IFC body cannot be assigned to a different BvK cell; kALDo rejects
such a request.

### Interpolation choice

For normal calculations, leave `ifc_interpolation` at its default value,
`"auto"`. kALDo then chooses the interpolation that matches the force-constant
source. When NAC is active, kALDo automatically uses the interpolation needed
to combine the short-range force constants with the long-range dipole term.

The `"periodic"` setting is only a developer option for comparing against
legacy NAC-off results; kALDo rejects it when NAC is active. See
[Choosing IFC interpolation](https://github.com/nanotheorygroup/kaldo/blob/main/docs/docsource/ifc_interpolation.md)
for the available modes and guidance.

## Input conventions and provenance

kALDo supports two input conventions. They are not user-selectable NAC
algorithms. The loader records how the force constants were produced and
selects the only compatible reconstruction:

<table>
  <thead>
    <tr>
      <th>Loaded harmonic data</th>
      <th>Selected convention</th>
      <th>Operation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Total finite-supercell IFCs with dielectric and Born tensors, including VASP/phonopy-style inputs</td>
      <td>VASP/phonopy</td>
      <td>Subtract the Gonze dipole term once on the commensurate mesh, interpolate the remaining short-range IFCs, and restore the matching Gonze term at the requested q point.</td>
    </tr>
    <tr>
      <td>A polar QE <code>espresso.ifc2</code> written by q2r with its macroscopic header</td>
      <td>QE q2r</td>
      <td>Recognize that q2r already subtracted its dipole term, interpolate that short-range body in the native q2r gauge, and restore QE's matching rigid-ion term.</td>
    </tr>
  </tbody>
</table>

The central invariant is **one dipole subtraction and one matching
restoration**. Treating a polar q2r body as total IFCs would subtract the dipole
interaction twice. Conversely, applying the QE restoration to VASP/phonopy total
IFCs would mix incompatible lattice, phase, Ewald, and charge conventions.
kALDo therefore provides no `nac_method` or fallback switch that guesses
between the two paths.

### VASP/phonopy total-IFC path

Let $\Phi_{\mathrm{total}}^{\mathrm{SC}}(\mathbf R)$ denote total periodic
IFCs represented by a finite supercell, and let $\{\mathbf q_m\}$ be its
commensurate q mesh. The VASP/phonopy path evaluates

```math
D_{\mathrm{VASP/phonopy}}(\mathbf q;\hat{\mathbf q}) =
\operatorname{FT}_{\mathrm{WS}}\!\left[
  \Phi_{\mathrm{total}}^{\mathrm{SC}}(\mathbf R) -
  \operatorname{IFFT}_{\mathbf q_m}
    [D_{\mathrm{Gonze}}^{\mathrm{dipole}}(\mathbf q_m)]
\right](\mathbf q) +
D_{\mathrm{Gonze}}^{\mathrm{dipole}}
  (\mathbf q;\hat{\mathbf q}).
```

The inverse transform first produces short-range IFCs on the defining BvK
cell. Wigner--Seitz phase averaging then interpolates those IFCs at the
requested q point before the same dipole convention is restored. kALDo applies
the Born-charge acoustic sum rule to a private copy for this path; it does not
modify user-owned arrays.

### QE q2r path

For a polar q2r file, $\Phi_{\mathrm{q2r}}^{\mathrm{short}}$ is already the
result of QE removing its rigid-ion term. kALDo evaluates

```math
\begin{aligned}
D_{\mathrm{QE}}(\mathbf q;\hat{\mathbf q}) ={}&
\operatorname{FT}_{\mathrm{q2r}}[
  \Phi_{\mathrm{q2r}}^{\mathrm{short}}](\mathbf q) +
D_{\mathrm{QE}}^{\mathrm{rigid}}(\mathbf q) \\
&+ \delta_{\mathbf q,\Gamma}
D_{\mathrm{QE}}^{\mathrm{directional}}(\hat{\mathbf q}),
\end{aligned}
```

The factor $\delta_{\mathbf q,\Gamma}$ in the last line is one only at exact
Gamma. The loader preserves the q2r lattice gauge, q grid, Ewald parameter,
dielectric tensor, Born charges, masses, and the flag proving that the IFC body
is already dipole-subtracted. The tensors are not modified by the
VASP/phonopy Born-charge sum rule because they must reproduce the term removed
by QE.

Do not strip the header from a polar `espresso.ifc2` file. If a
workflow intentionally supplies total force constants instead, produce a q2r
file without the polar subtraction and attach the dielectric and Born tensors
to that total-IFC input explicitly.

## Effect on observables

NAC is part of the harmonic eigensystem, not a post-processing correction. The
selected controller therefore feeds frequencies, eigenvectors, group
velocities, heat-flux operators, thermodynamic quantities, and transport
properties. The `is_nac` and `nac_bvk_supercell_matrix` choices are propagated
through these consumers so that they use the same harmonic model.

## Validation

The reference suite tests the correction at several levels. Comparing only a
plausible frequency spectrum can conceal an incorrect tensor convention, so
the tests also exercise raw force-constant tensors and mass-weighted dynamical
matrices.

<table>
  <thead>
    <tr>
      <th>Campaign</th>
      <th>Quantity</th>
      <th>Reference and coverage</th>
      <th>Acceptance criterion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/tests/test_qe_q2r_nac.py">QE rigid-ion kernel</a></td>
      <td>Unmass-weighted finite-q and directional-Gamma tensors in native QE units, plus the unit and mass-weighting bridge</td>
      <td>Arrays emitted by QE 7.6 <code>rigid.f90</code> for a low-symmetry, three-atom cell</td>
      <td><code>rtol=2e-13</code> and <code>atol=3e-16</code></td>
    </tr>
    <tr>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/tests/test_qe_nac_bravais_reference.py">QE Bravais campaign</a></td>
      <td>NAC-on minus NAC-off tensor at Gamma</td>
      <td>QE 7.6 <code>matdyn.x</code> inputs for all 14 three-dimensional Bravais classes</td>
      <td>Relative Frobenius error below <code>5e-6</code></td>
    </tr>
    <tr>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/tests/test_nac_gonze_phonondb26.py">VASP/phonopy campaign</a></td>
      <td>Complete mesh sampling, inverse transform, interpolation, and dipole restoration</td>
      <td>Phonopy calculations for 26 polar materials spanning 13 Bravais classes and 76 matrix comparisons</td>
      <td>Relative Frobenius error at most <code>1e-5</code>; absolute error at most <code>1e-9</code> for cancellation cases</td>
    </tr>
    <tr>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/tests/test_gan_nac.py">Anisotropic GaN integration</a></td>
      <td>Frequencies, LO&ndash;TO splitting, degeneracies, group velocities, and the diagonal heat-flux identity</td>
      <td>Wurtzite GaN compared with Phonopy spectra and finite differences</td>
      <td>Quantity-specific tolerances, including a 1% velocity-ratio check</td>
    </tr>
    <tr>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/tests/test_nac_api.py">Loaded QE integration</a></td>
      <td>Public harmonic frequencies from a real q2r input</td>
      <td>Six pinned MgO frequencies at a finite reduced q point</td>
      <td><code>rtol=atol=2e-7</code></td>
    </tr>
  </tbody>
</table>

Additional contract tests cover q2r-header parsing, input units, Fourier
gauges, BvK-cell rejection, option propagation, and fixture checksums. The
readable reference inputs and their provenance are documented in the
[Phonopy campaign README](https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/tests/data/input/gonze-phonopy/README.md),
[QE Bravais README](https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/tests/data/input/qe76-bravais-reference/README.md),
and [QE rigid-ion fixture README](https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/tests/data/input/qe76-rigid-f90/README.md).

## Scope and limitations

- The validated scientific scope is three-dimensional periodic polar crystals.
  This implementation does not claim two-dimensional LO-TO electrostatics.
- `nac_q_direction` defaults to `(1, 0, 0)` at exact Gamma and can be set on
  either `Phonons` or `HarmonicWithQ`.
- QE q2r force constants cannot be remeshed to a different BvK cell.
- Zero Born charges do not produce a dipole term. With `is_nac=None` the
  ordinary harmonic path is used; with `is_nac=True` the input is rejected.

## Implementation provenance

The VASP/phonopy path follows the separation introduced by Gonze and Lee and is
cross-validated against [Phonopy 2.17.1](https://github.com/phonopy/phonopy/tree/b67269df9ff46149550db06ebb652850bae1d1cc).
The q2r path follows `do_q2r`, `matdyn`, and `rigid.f90` from
[Quantum ESPRESSO 7.6](https://github.com/QEF/q-e/tree/9f93ddec427d2b9a45bb72d828c6d324f62fcabd).
These versions identify the numerical conventions used to create the pinned
reference data; they are not runtime dependencies.

The source correspondence is summarized below. Upstream links are pinned to
the versions used for the validation data; kALDo links target `main` so they
continue to describe the installed library after this documentation is merged.

<table>
  <thead>
    <tr>
      <th>Operation</th>
      <th>Upstream implementation</th>
      <th>kALDo implementation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VASP/phonopy reciprocal-space dipole term</td>
      <td><a href="https://github.com/phonopy/phonopy/blob/b67269df9ff46149550db06ebb652850bae1d1cc/phonopy/harmonic/dynamical_matrix.py#L763-L840">Phonopy <code>_get_Gonze_dipole_dipole</code></a></td>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/controllers/nac.py">NAC controller</a></td>
    </tr>
    <tr>
      <td>VASP/phonopy commensurate-mesh subtraction and inverse transform</td>
      <td><a href="https://github.com/phonopy/phonopy/blob/b67269df9ff46149550db06ebb652850bae1d1cc/phonopy/harmonic/dynamical_matrix.py#L675-L726">Phonopy <code>make_Gonze_nac_dataset</code></a></td>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/observables/secondorder.py">Second-order NAC preparation</a></td>
    </tr>
    <tr>
      <td>q2r short-range IFC generation and provenance</td>
      <td><a href="https://github.com/QEF/q-e/blob/9f93ddec427d2b9a45bb72d828c6d324f62fcabd/PHonon/PH/do_q2r.f90#L257-L336">QE <code>do_q2r</code></a></td>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/interfaces/qe_io.py">QE IFC and header reader</a></td>
    </tr>
    <tr>
      <td>q2r short-range Fourier interpolation</td>
      <td><a href="https://github.com/QEF/q-e/blob/9f93ddec427d2b9a45bb72d828c6d324f62fcabd/PHonon/PH/matdyn.f90#L1134-L1232">QE <code>matdyn::frc_blk</code></a></td>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/controllers/nac.py">NAC controller</a></td>
    </tr>
    <tr>
      <td>QE finite-q rigid-ion restoration</td>
      <td><a href="https://github.com/QEF/q-e/blob/9f93ddec427d2b9a45bb72d828c6d324f62fcabd/PHonon/PH/rigid.f90#L15-L246">QE <code>rigid::rgd_blk</code></a></td>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/controllers/nac.py">NAC controller</a></td>
    </tr>
    <tr>
      <td>QE directional Gamma term</td>
      <td><a href="https://github.com/QEF/q-e/blob/9f93ddec427d2b9a45bb72d828c6d324f62fcabd/PHonon/PH/rigid.f90#L250-L313">QE <code>rigid::nonanal</code></a></td>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/controllers/nac.py">NAC controller</a></td>
    </tr>
    <tr>
      <td>Public harmonic eigensystem and derivatives</td>
      <td>N/A</td>
      <td><a href="https://github.com/nanotheorygroup/kaldo/blob/main/kaldo/observables/harmonic_with_q.py">HarmonicWithQ</a></td>
    </tr>
  </tbody>
</table>

## Reference

X. Gonze and C. Lee, *Dynamical matrices, Born effective charges, dielectric
permittivity tensors, and interatomic force constants from density-functional
perturbation theory*, Physical Review B **55**, 10355 (1997),
[doi:10.1103/PhysRevB.55.10355](https://doi.org/10.1103/PhysRevB.55.10355).

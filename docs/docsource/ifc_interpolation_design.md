# IFC interpolation design and migration

At the highest level, kALDo constructs a harmonic dynamical matrix as $D(\mathbf q)=\mathrm{FT}[\Phi^{\mathrm{short}}](\mathbf q)+D^{\mathrm{long\ range}}(\mathbf q)$. Here $\Phi$ is a real-space interatomic force-constant tensor, $\mathrm{FT}$ is its Fourier sum over lattice translations, and $D^{\mathrm{long\ range}}$ is the optional macroscopic dipole contribution for a polar material. This PR fixes how the first term assigns real-space images and Fourier phases, applies that representation to second-order derivatives (IFC2) and both translation legs of the third-order tensor (IFC3), and preserves the matching source convention required by the optional second term.

## The issue

Moving the origin of a periodic cell and wrapping its atoms back into the cell changes labels, not physics. If atom $i$ crosses lattice vector $\mathbf n_i$, the same interaction is represented by

$$\boldsymbol\tau_i'=\boldsymbol\tau_i-\mathbf n_i,\qquad \mathbf R_{ij}'=\mathbf R_{ij}+\mathbf n_j-\mathbf n_i.$$

Frequencies, velocities, scattering rates, and thermal conductivity must be identical before and after this relabeling. The previous implementation could not satisfy that invariant: it assigned one translation and Fourier phase to an entire replica slot even though the shortest periodic image depends on the interacting atom pair. It also used one “grid” abstraction for reciprocal q points, finite-supercell translation classes, and translations explicitly stored by an IFC file.

This produced two independent failure modes:

1. Compact IFCs used the wrong real-space image or a phase that could not follow the atom-dependent relabeling. The error affected harmonic derivatives and both third-order Fourier legs.
2. Literal IFC translations could be folded into compact slots or dropped. In particular, the ShengBTE IFC3 reader silently lost an out-of-range block when its offset did not match the one representative kept by the old grid.

### Observable evidence

| Case | Physical expectation | Defect signal or committed oracle |
|---|---|---|
| Synthetic skew cell | The image minimizes Cartesian pair distance, not each fractional component independently. | Componentwise wrapping selects a different vector; the committed test compares the selected image with an exhaustive $9\times9\times9$ lattice search to `atol=1e-12`. |
| Every public IFC input route | A wrapped rigid translation preserves the spectrum and the scalar RTA conductivity. | The committed matrix covers ten canonical formats and three legacy aliases, requires nonzero moved IFC3 weight, and checks the invariant after source-aware loading. |
| Periodically repeated amorphous Si at Gamma | A wrapped origin change leaves both the zeroth IFC2 moment and its physical first moment unchanged. | Gamma frequencies were unchanged, but the direct-periodic heat-flux operator moved by 161.97% and the QHGK conductivity trace by 6.71%. |
| ShengBTE `FORCE_CONSTANTS_3RD` with a literal out-of-range offset | The block and its translation remain present. | The old lookup returned no matching representative and NumPy assignment became a silent no-op. A maintainer's Si report estimated about 2% of IFC3 weight was lost. |

All geometric and origin-invariance claims in the first three rows are reproducible from fixtures committed here. The amorphous percentages measure the deliberately retained direct-periodic diagnostic; the corrected result is judged by exact origin invariance and an independent finite-difference first moment, not by a self-generated absolute conductivity. The ShengBTE percentage is identified only as the motivating maintainer report; the committed regression proves the data-loss mechanism directly and the retained upstream calculation validates the corrected reader end to end.

### Why ordinary frequency tests missed it

On a q point commensurate with the defining supercell, periodically equivalent translations have the same Fourier phase. A wrong representative can therefore leave every sampled frequency unchanged while changing an off-mesh interpolation, a Cartesian derivative, or a three-phonon matrix element. Cubic cells also hide the replica-only minimum-image error geometrically, and the older ESKM Si IFC3 fixture carried zero force-constant weight on the relevant boundary images.

## Resolution

This PR rebuilds IFC interpolation from explicit lattice topology and source provenance. It adds exact supercell quotient arithmetic, preserves literal file translations, constructs pair-dependent shortest images with correct tie weights, generalizes the IFC3 projection to its actual translation support, fixes the ShengBTE IFC3 data-loss path, and validates the result from geometry-level identities through complete RTA transport.

| Root problem | Why it is wrong | Observable consequence |
|---|---|---|
| Integer replica components were wrapped independently. | The shortest vector in index space is not generally shortest in Cartesian space for a skew cell. | Non-orthogonal crystals used incorrect interaction vectors and velocity derivatives. |
| One image and phase were assigned to an entire replica. | The nearest image depends on $\mathbf R+\boldsymbol\tau_j-\boldsymbol\tau_i$, so it is specific to an atom pair. | Results could depend on where the unit-cell boundary was drawn. |
| IFC translation axes were assumed to have length $\lvert\det\mathbf M\rvert$. | A file can specify several translations in one periodic class, and compiled IFC3 can require more translations than physical replicas. | Literal blocks were folded, overwritten, or contracted with the wrong phase dimension. |
| Reciprocal q meshes and real-space translation topology shared one grid API. | Equal array sizes do not imply equal physical meaning or ordering. | Diagonal/cubic tests hid matrix, ordering, and provenance errors. |
| Cached observables did not fully identify their IFC representation. | Periodic, literal, and pair-image tensors can describe different off-mesh Fourier sums. | A corrected run could reuse stale results from another interpolation path. |

## User-facing API

The public control is now `ifc_interpolation`:

```python
phonons = Phonons(
    forceconstants=forceconstants,
    kpts=(5, 5, 5),
    ifc_interpolation="auto",
)
```

| Value | Meaning | Intended use |
|---|---|---|
| `"auto"` | Respect the input representation: use pair-dependent Wigner–Seitz images for compact periodic IFCs, retain translations that a file specifies directly, and honor a format's validated native Fourier convention. | Normal calculations. |
| `"wigner-seitz"` | Explicitly fold translations by periodic class, then redistribute each IFC block over the shortest images of its atom pair. | Controlled comparison or an explicit compact-IFC choice. |
| `"periodic"` | Explicitly fold to one direct translation per periodic class without pair-image redistribution. | Diagnostic comparison with the historical periodic representation. |

The former `is_unfolding` name is deprecated. This operation interpolates real-space IFCs; it does not unfold a phonon band structure. For one release `Phonons` still accepts `is_unfolding` with a `DeprecationWarning`, mapping `True` to `ifc_interpolation="wigner-seitz"` and `False` to `"auto"`; combining it with an explicit `ifc_interpolation` raises. The historical folded numbers are the explicit `ifc_interpolation="periodic"` diagnostic, not the `is_unfolding=False` default.

`"periodic"` is incompatible with active non-analytic correction (NAC), the polar long-range term in the opening equation, because the interpolated short-range body and restored long-range term must use matching image weights. To inspect a polar file through the direct-periodic diagnostic, disable the long-range correction explicitly with `is_nac=False`.

Wigner–Seitz interpolation currently requires periodic boundary conditions in all three directions. A periodically repeated amorphous cell is supported: the whole disordered cell is the reference cell, and boundary-crossing atom pairs receive their nearest periodic images without assuming crystal symmetry. True nanowires, slabs, and isolated clusters are not yet supported by this interpolation kernel. The explicit `"periodic"` mode remains a diagnostic representation; it is not a physical substitute for an image search restricted to the periodic subspace.

## Notation

| Symbol | Meaning | kALDo representation |
|---|---|---|
| $\mathbf A$ | Primitive-cell matrix, with lattice vectors stored as rows. | `atoms.cell` |
| $\boldsymbol\tau_i$ | Fractional basis position of atom $i$. | `atoms.positions @ inv(atoms.cell)` |
| $\mathbf M$ | Integer supercell matrix. | `SupercellGrid.matrix` |
| $\mathbf R$ | Integer primitive-lattice translation stored on an IFC axis. | `TranslationSupport.translations` |
| $\Phi^{(2)}_{i\alpha,j\beta}(\mathbf R)$ | Second-order IFC: force response in Cartesian direction $\alpha$ on atom $i$ to displacement in direction $\beta$ of atom $j$ translated by $\mathbf R$. | IFC2 block with shape `(i, alpha, R, j, beta)` |
| $\Phi^{(3)}_{i\alpha,j\beta,k\gamma}(\mathbf R_j,\mathbf R_k)$ | Third-order IFC with two independently translated partner atoms. | IFC3 block with shape `(i, alpha, Rj, j, beta, Rk, k, gamma)` |
| $N=\lvert\det\mathbf M\rvert$ | Number of physical periodic translation classes in the finite supercell. | `n_replicas` |
| $S$ | Number of translations actually indexing an IFC tensor; $S$ can differ from $N$. | `n_translations` |
| $\mathbf q$ | Reduced reciprocal coordinate, so the integer-translation phase is $e^{2\pi i\mathbf q\cdot\mathbf R}$. | `QGrid.fractional_points` or a `HarmonicWithQ` q point |
| $D^{\mathrm{long\ range}}$ | Macroscopic dipole contribution restored for a polar model; it is zero for an ordinary nonpolar model. | NAC controller selected by IFC provenance |
| $\mathrm{FT}$ | The real-to-reciprocal-space Fourier sum over IFC translations. | Harmonic and anharmonic interpolation plans |

## The physics problem

### One periodic class has many equivalent translations

In a finite supercell, two primitive translations are periodically equivalent when they differ by a supercell lattice vector:

$$\mathbf R' = \mathbf R + \mathbf n\mathbf M,\qquad \mathbf n\in\mathbb Z^3.$$

All such translations belong to one quotient class. There are exactly $N=\lvert\det\mathbf M\rvert$ classes, but there is no universally “best” representative of a class until the cell geometry and the interacting atom pair are known.

### The physical separation belongs to an atom pair

For an IFC block connecting atom $i$ in the reference cell to atom $j$ in translated cell $\mathbf R$, the Cartesian pair displacement is

$$\mathbf d_{ij}(\mathbf R)=\left(\mathbf R+\boldsymbol\tau_j-\boldsymbol\tau_i\right)\mathbf A.$$

The basis offset $\boldsymbol\tau_j-\boldsymbol\tau_i$ changes with the pair. Therefore a translation that is shortest for $(i,j)$ need not be shortest for $(i,k)$, even when both blocks carry the same stored replica label $\mathbf R$. Wrapping the three integer components of $\mathbf R$ independently only finds a minimum image in index space; for skew, hexagonal, monoclinic, or triclinic cells it need not minimize Cartesian distance at all.

The shortest distance and the complete set of tied shortest translations are

$$d^{\min}_{ij}(\mathbf R)=\min_{\mathbf n\in\mathbb Z^3}\left\|\left(\mathbf R+\mathbf n\mathbf M+\boldsymbol\tau_j-\boldsymbol\tau_i\right)\mathbf A\right\|,$$

$$\mathcal I_{ij}(\mathbf R)=\left\{\mathbf R+\mathbf n\mathbf M:\left\|\left(\mathbf R+\mathbf n\mathbf M+\boldsymbol\tau_j-\boldsymbol\tau_i\right)\mathbf A\right\|=d^{\min}_{ij}(\mathbf R)\right\}.$$

`Pair image` means one member of $\mathcal I_{ij}(\mathbf R)$: a periodic copy of the same IFC block chosen using the full geometry of the specific atom pair. It does not create a new interaction or change the force constant. If several images are tied, the block is partitioned among them with normalized weights $w=1/\lvert\mathcal I_{ij}\rvert$. Selecting only one tied image would introduce an arbitrary symmetry and origin dependence.

“Wigner–Seitz” in this PR refers only to this real-space shortest-image construction for a finite supercell. It is unrelated to the Wigner transport equation or the coherence contribution to thermal conductivity.

### Direct periodic and pair-image Fourier sums

The old direct-periodic representation used one stored translation for an entire replica slot:

$$D^{\mathrm{periodic}}_{i\alpha,j\beta}(\mathbf q)=\frac{1}{\sqrt{m_i m_j}}\sum_{\mathbf R}\Phi^{(2)}_{i\alpha,j\beta}(\mathbf R)e^{2\pi i\mathbf q\cdot\mathbf R}.$$

For compact finite-supercell IFCs, the pair-image representation instead uses every shortest image of that block:

$$D^{\mathrm{pair}}_{i\alpha,j\beta}(\mathbf q)=\frac{1}{\sqrt{m_i m_j}}\sum_{\mathbf R}\Phi^{(2)}_{i\alpha,j\beta}(\mathbf R)\sum_{\mathbf R'\in\mathcal I_{ij}(\mathbf R)}w_{ij}(\mathbf R')e^{2\pi i\mathbf q\cdot\mathbf R'}.$$

In kALDo's eigenvector gauge, only the integer translation $\mathbf R'$ appears in this exponential; the basis positions are already carried by the phonon eigenvectors. The full Cartesian displacement $\mathbf d_{ij}(\mathbf R')$ is nevertheless required for distance cutoffs, elastic moments, and the Cartesian dynamical-matrix derivative used for group velocity. Adding the basis offset to the exponential as well would mix two eigenvector gauges.

### Why commensurate frequencies did not reveal the bug

On a q point commensurate with the defining supercell, $\mathbf q\cdot\mathbf n\mathbf M$ is an integer. Equivalent images then have the same phase:

$$e^{2\pi i\mathbf q\cdot(\mathbf R+\mathbf n\mathbf M)}=e^{2\pi i\mathbf q\cdot\mathbf R}.$$

This makes the periodic and pair-image sums identical on the Born–von Karman mesh even when their off-mesh interpolation and Cartesian derivatives are different. It explains how a commensurate frequency suite can remain unchanged to numerical precision while velocities, three-phonon matrix-element magnitudes, or conductivity are wrong. The older crystalline ESKM Si fixture also carried zero IFC3 weight on the relevant boundary images, so it exercised the code path without exercising the defect.

### Why a rigid origin change is a decisive test

Move every atom by the same constant vector and wrap the basis back into the unit cell. The crystal has not changed, but different atoms can cross different cell faces. The same physical pair is then relabeled by

$$\boldsymbol\tau_i' = \boldsymbol\tau_i-\mathbf n_i,\qquad \mathbf R'_{ij}=\mathbf R_{ij}+\mathbf n_j-\mathbf n_i.$$

A correct Fourier representation gives the same spectrum, velocity, three-phonon matrix-element magnitudes, scattering rates, and conductivity before and after this relabeling. A representation with one phase per replica cannot follow the atom-dependent change $\mathbf n_j-\mathbf n_i$, so origin dependence exposes the missing pair geometry directly.

### Crystals and amorphous bulk expose different moments

For a periodically repeated amorphous cell sampled only at Gamma, every translation phase is one:

$$D(0)=\sum_{\mathbf R}\Phi^{(2)}(\mathbf R).$$

Changing the periodic representative therefore leaves the Gamma frequencies unchanged. The Allen--Feldman/QHGK heat-flux operator instead contains the first Cartesian moment, up to the fixed mass, frequency, and unit factors used when it is projected into the mode basis:

$$M^{(1)}_{ij,\alpha}=\sum_{\mathbf R}\Phi^{(2)}_{ij}(\mathbf R)d_{ij,\alpha}(\mathbf R).$$

Two atoms at $x=9.5$ and $0.5$ Å in a 10 Å cell are separated by 1 Å across the periodic boundary, not by $-9$ Å through the cell. Both vectors give the same zeroth-moment Gamma matrix; they give heat-flux elements with different magnitude and sign. Amorphous frequencies alone therefore cannot validate this PR.

| System | Reciprocal sampling | Pair-image observable most likely to expose the bug |
|---|---|---|
| Crystal | Multiple q points | Off-mesh dispersion, group velocity, IFC3 matrix elements, rates, and RTA conductivity |
| Periodically repeated amorphous bulk | Usually Gamma only | IFC2 first moment, heat-flux operator, diffusivity, and QHGK conductivity |
| Nanowire or slab | Partial periodicity | Not yet validated; images must be restricted to the periodic subspace |

The retained 32-atom amorphous-Si Tersoff fixture makes this distinction measurable in a small CI case. Boundary-crossing pairs carry 22.31% of its IFC2 L1 weight, and entries with at least one boundary-crossing leg carry 33.89% of its IFC3 L1 weight. A fixed wrapped-origin relabeling leaves all Gamma frequencies and the pair-aware result invariant, while the direct-periodic heat-flux operator changes by 161.97% and the resulting QHGK conductivity trace changes by 6.71%. An independently constructed minimum-image Fourier sum also verifies the analytic IFC2 first moment by centered finite difference. The absolute QHGK tensor generated by kALDo is retained only as a numerical-regression snapshot, not as external ground truth for amorphous-Si conductivity.

At Gamma the two IFC3 phases are also one, so the amorphous projection uses the exact translation-axis sum $\sum_{\mathbf R_j,\mathbf R_k}\Phi^{(3)}(\mathbf R_j,\mathbf R_k)$. Loader preservation still matters, but pair-image redistribution cannot change that zeroth moment. Off-Gamma two-leg IFC3 correctness is established by the crystal and direct-Fourier tests.

## Third-order IFCs need two pair-image operations

With atom $i$ fixed in the reference cell, IFC3 has independent translations for partners $j$ and $k$:

$$V^{(3)}(\mathbf q,\mathbf q',\mathbf q'')\ \propto\ \sum_{\mathbf R_j,\mathbf R_k}\Phi^{(3)}(\mathbf R_j,\mathbf R_k)e^{2\pi i(\mathbf q'\cdot\mathbf R_j+\mathbf q''\cdot\mathbf R_k)},$$

where the omitted factors are the normal-mode eigenvectors and mass/frequency normalization. The $(i,j,\mathbf R_j)$ leg and $(i,k,\mathbf R_k)$ leg must be resolved independently. If the first leg has tied images with weights $w_{jm}$ and the second has tied images with weights $w_{kn}$, the original IFC3 entry is distributed over their Cartesian product with weight $w_{jm}w_{kn}$.

The union of all translations used by nonzero pair images becomes the compiled IFC3 support. Its size $S$ can exceed the physical class count $N=\lvert\det\mathbf M\rvert$. The anharmonic projection must therefore shape both translation axes and both Fourier phase arrays with $S$, not with `n_replicas`. Collapsing the expanded tensor back into $N$ slots would discard distinct off-mesh phases and recreate the bug.

## Source-aware behavior

Pair-dependent Wigner–Seitz interpolation is correct for a compact tensor that supplies one block per periodic supercell class. It is not correct to impose that representation blindly on every file format. Some files specify the physical translations directly, while other generators define a validated native Fourier gauge.

| Input representation | IFC2 under `auto` | IFC3 under `auto` | Reason |
|---|---|---|---|
| Compact numpy, ESKM/LAMMPS, hiphive, GPUMD, and VASP/Phonopy-style tensors | Pair-dependent Wigner–Seitz | Pair-dependent Wigner–Seitz | The stored axes represent finite-supercell classes, so pair geometry must choose the images used away from the commensurate mesh. |
| QE q2r harmonic body, with or without polar metadata | Pair-dependent Wigner–Seitz | Depends on the accompanying IFC3 source | q2r's header defines the structure and lattice topology from which pair-specific reconstruction is built; a polar body must also use weights consistent with QE long-range restoration. |
| TDEP | Preserve translations written in the file | Preserve translations written in the file | A TDEP file can contain several translations in one periodic class; folding them would change its literal Fourier sum. |
| ShengBTE `FORCE_CONSTANTS_3RD` | IFC2 follows its VASP or QE source | Preserve translations written in `FORCE_CONSTANTS_3RD` | Cartesian offsets are converted to integer primitive translations and retained, including out-of-cube representatives. |
| QE/thirdorder d3q IFC3 | IFC2 follows its harmonic source | Direct periodic | d3q writes explicit unrecentered cell indices in its native Fourier convention. |

The interface records provenance; it does not expose additional user modes. `auto` resolves that provenance, while an explicit `"periodic"` or `"wigner-seitz"` request overrides it. Overriding a literal file first folds all translations by exact periodic class and emits a warning, so the representation change is visible and deterministic.

## What the implementation changes

### 1. Separate reciprocal sampling, periodic topology, and IFC storage

| Object | Single responsibility | Implementation |
|---|---|---|
| `QGrid` | Integer reciprocal-mesh addresses, reduced q points, time reversal, and exact momentum partners. It contains no real-space geometry. | [`QGrid`](kaldo/grid.py#L62) |
| `SupercellGrid` | Exact quotient of primitive translations by the integer supercell matrix $\mathbf M$, including non-diagonal matrices. | [`SupercellGrid`](kaldo/grid.py#L130) |
| `TranslationSupport` | Ordered translations that actually index an IFC tensor, including multiple translations from the same periodic class. | [`TranslationSupport`](kaldo/grid.py#L227) |
| `WignerSeitzImages` | Lazy, cached pair-specific shortest-image search with every geometric tie and normalized weights. | [`WignerSeitzImages`](kaldo/grid.py#L303), [`image`](kaldo/grid.py#L364) |

The ambiguous legacy `Grid`, `NonDiagonalGrid`, coordinate-wrapping helpers, and implicit direct-grid state are removed. Integer supercell matrices and reciprocal mesh shapes are validated rather than silently truncated from fractional values.

### 2. Make the IFC container own both topologies

`ForceConstant` now keeps the physical supercell (`supercell_grid`, `replica_translations`, and `n_replicas`) separate from the tensor axis (`translation_support` and `n_translations`). Compact inputs normally have $S=N$; literal TDEP/ShengBTE inputs and Wigner–Seitz-compiled IFC3 may have $S\ne N$. See [`ForceConstant`](kaldo/observables/forceconstant.py).

### 3. Use one harmonic interpolation plan for matrices, derivatives, and elastic moments

The harmonic plan resolves source provenance once, folds only on explicit request, skips zero blocks, and reuses pair-image geometry for the dynamical matrix, Cartesian derivatives, distance cutoffs, and elastic moments. This prevents frequency, velocity, and elasticity from using different real-space vectors. See [`_resolve_ordinary_ifc_interpolation`](kaldo/observables/harmonic_with_q.py#L48), [`_HarmonicIFCInterpolation.matrices`](kaldo/observables/harmonic_with_q.py#L148), and [`ForceConstants.elastic_prop`](kaldo/forceconstants.py#L350).

### 4. Compile IFC3 without densifying it

`ThirdOrder.get_interpolation` returns an immutable interpolation object rather than mutating the loaded tensor. Compact IFCs are folded and expanded at nonzero pair-block granularity; tied images receive Cartesian-product weights; duplicate sparse coordinates are coalesced; the per-block Gamma translation sum is checked; and the result is cached by source identity and support digest. See [`ThirdOrder.get_interpolation`](kaldo/observables/thirdorder.py#L139) and [`ThirdOrder._compile_wigner_seitz`](kaldo/observables/thirdorder.py#L241).

The crystal projection obtains the interpolation before building phases and uses its actual support size $S$ in every reshape and contraction. See [`Phonons._project_crystal`](kaldo/phonons.py#L1932) and [`sparse_potential_mu`](kaldo/controllers/anharmonic.py#L112).

### 5. Preserve literal TDEP and ShengBTE translations

The TDEP loaders validate integer lattice translations and retain every distinct vector in deterministic order, including several vectors from one periodic class and true non-diagonal supercell matrices. The ShengBTE IFC3 reader converts each Cartesian cell offset to an integer primitive translation, retains out-of-cube vectors, and builds rank-eight sparse storage directly. Exact duplicate tensor coordinates accumulate; periodically equivalent but physically distinct translations remain separate. See [`parse_tdep_third_forceconstant`](kaldo/interfaces/tdep_io.py#L641) and [`read_third_order_matrix`](kaldo/interfaces/shengbte_io.py#L30).

This fixes the former ShengBTE failure mode in which modulo-wrapped offsets were assigned into compact slots and a later block could overwrite an earlier out-of-cube block without warning.

### 6. Keep caches representation-safe

Harmonic and anharmonic cache namespaces include the requested and resolved interpolation modes, supercell topology, source/native-gauge hint, ordered translation-support digest, and interpolation version. Lazy IFC3 loading refreshes the identity before stored properties are resolved, so results produced from one Fourier representation cannot be reused by another. See [`Phonons.ifc_cache_key`](kaldo/phonons.py#L758) and [`Storable`](kaldo/storable.py).

Legacy IFC3 export formats that cannot encode an arbitrary translation support now reject that export with a targeted error instead of writing an ambiguous file.

## Grounding the equations in the code

| Scientific term | Role in the equations | kALDo code |
|---|---|---|
| $\mathbf q$ and reciprocal-mesh momentum arithmetic | Supplies reduced q points and exact $\mathbf q\pm\mathbf q'$ partners. | [`QGrid`](kaldo/grid.py#L62) |
| $\mathbf R\sim\mathbf R+\mathbf n\mathbf M$ | Classifies finite-supercell periodic equivalence exactly for diagonal or non-diagonal $\mathbf M$. | [`SupercellGrid.class_key`](kaldo/grid.py#L207), [`SupercellGrid.class_id`](kaldo/grid.py#L217) |
| Stored $\mathbf R$ axis | Preserves the translations that the source actually places on IFC2/IFC3. | [`TranslationSupport`](kaldo/grid.py#L227), [`ForceConstant`](kaldo/observables/forceconstant.py) |
| $\mathbf d_{ij}(\mathbf R)$ and $\mathcal I_{ij}(\mathbf R)$ | Finds all shortest periodic copies for one nonzero atom-pair block. | [`WignerSeitzImages.image`](kaldo/grid.py#L364) |
| $D^{\mathrm{periodic}}$ and $D^{\mathrm{pair}}$ | Evaluates the harmonic Fourier sum and its Cartesian derivative in one consistent plan. | [`_HarmonicIFCInterpolation.matrices`](kaldo/observables/harmonic_with_q.py#L148) |
| Two IFC3 Fourier legs | Compiles $(i,j,\mathbf R_j)$ and $(i,k,\mathbf R_k)$ independently and applies $w_{jm}w_{kn}$. | [`ThirdOrder._compile_wigner_seitz`](kaldo/observables/thirdorder.py#L241) |
| $S$-sized anharmonic contraction | Uses the compiled/literal IFC3 support rather than assuming $S=N$. | [`Phonons._project_crystal`](kaldo/phonons.py#L1932), [`sparse_potential_mu`](kaldo/controllers/anharmonic.py#L112) |
| Amorphous Gamma IFC3 contraction | Sums both stored translation axes exactly before the Gamma-only mode projection. | [`ThirdOrder.gamma_contracted_value`](kaldo/observables/thirdorder.py#L200), [`Phonons._project_amorphous`](kaldo/phonons.py#L1817) |
| Long-range polar term | Combines a matching Wigner–Seitz short-range body with the provenance-specific Gonze or QE restoration. | [`nac.dynamical_matrices`](kaldo/controllers/nac.py), [`HarmonicWithQ`](kaldo/observables/harmonic_with_q.py) |
| Elastic long-wavelength moments | Uses the same pair displacements and tie weights as harmonic interpolation. | [`ForceConstants.elastic_prop`](kaldo/forceconstants.py#L350) |

## Validation

The tests are organized around physical invariants and independent external data, not only around internal arrays.

| Quantity or failure mode | Fixture/oracle | What is asserted | Test |
|---|---|---|---|
| Exact grid topology | Synthetic diagonal and non-diagonal integer matrices | Quotient size, class invariance, C/F ordering, exact momentum partners, rejection of fractional grid data. | [`test_grid_refactor.py`](kaldo/tests/test_grid_refactor.py) |
| Pair-image geometry | Skew and rounded hexagonal cells | True Cartesian shortest vectors, every tied image, normalized weights, lazy cache reuse. | [`test_grid_refactor.py`](kaldo/tests/test_grid_refactor.py) |
| IFC3 tie handling | Synthetic sparse IFC3 | Cartesian-product tie weights, coalescing, and conserved total block weight. | [`test_wigner_seitz_ties_form_cartesian_product_with_conserved_weights`](kaldo/tests/test_thirdorder_interpolation.py#L44) |
| Commensurate equivalence | Synthetic IFC3 on its BvK mesh | Pair-image and periodic Fourier sums agree to `atol=1e-14`. | [`test_wigner_seitz_and_periodic_gauges_agree_at_commensurate_q`](kaldo/tests/test_thirdorder_interpolation.py#L136) |
| Wrapped-origin invariance on QE Si | QE q2r IFC2 plus literal ShengBTE IFC3 | Nonzero boundary IFC3 weight is exercised; an exact relabeling of IFC2 and both IFC3 legs preserves the spectrum and RTA conductivity trace. | [`test_qe_si_rta_is_invariant_to_wrapped_crystal_origin`](kaldo/tests/test_crystal_qe_vasp.py#L148) |
| Every public file route | numpy, ESKM, LAMMPS, VASP/ShengBTE, QE/ShengBTE, VASP/d3q, QE/d3q, hiphive, TDEP, GPUMD, and legacy aliases | Resolved source mode, non-Gamma and Gamma spectra, nonzero moved IFC3 weight, and basis-invariant RTA conductivity trace after an atom-dependent wrapped-origin regauging. | [`test_ifc_format_origin_invariance.py`](kaldo/tests/test_ifc_format_origin_invariance.py) |
| Direct calculator/MLIP boundary | Self-contained three-atom Lennard–Jones Ar crystal | Independently regenerated IFC2/IFC3, complete spectrum, and full RTA tensor remain invariant after a wrapped rigid shift. | [`test_ifc_origin_invariance_lj.py`](kaldo/tests/test_ifc_origin_invariance_lj.py) |
| Periodically repeated amorphous bulk | Regenerable 32-atom amorphous Si cell from an explicitly listed ASE Tersoff potential | A rigid wrapped-origin shift leaves Gamma frequencies unchanged but exposes the historical IFC2 first-moment error; pair-aware heat flux, diffusivity, and the full QHGK tensor remain invariant. A centered finite difference of an independently constructed minimum-image Fourier sum verifies the analytic first moment. Nonzero boundary weight is required on IFC2 and both IFC3 legs. A same-class literal IFC3 representation with $S=2$ and $N=1$ gives the same Gamma bandwidth and transport, proving that the projection contracts the stored translation support rather than the replica count. Absolute kALDo-generated transport values are labeled regression snapshots, not conductivity ground truth. | [`test_amorphous_ifc_interpolation.py`](kaldo/tests/test_amorphous_ifc_interpolation.py), [fixture regeneration](kaldo/tests/data/input/amorphous-si-tersoff-32/README.md) |
| Non-diagonal literal IFC2 | Four TDEP real-material fixtures with determinant 3–32 supercells | Loaded matrices equal an independent literal-record Fourier sum and reproduce external ALAMODE frequencies; repeated periodic classes remain distinct. | [`test_tdep_nondiagonal_reference.py`](kaldo/tests/test_tdep_nondiagonal_reference.py) |
| Compact non-diagonal IFC2/IFC3 integration | Regenerable diamond-Si geometry with an exact analytic bond model, determinant-four tiling, four physical replicas, and seven literal IFC translations | Public TDEP loading, inferred cell mapping, distinct physical/support topology, IFC2 and IFC3 acoustic sum rules, sparse cubic projection, and an end-to-end RTA calculation. IFC4 is explicitly deferred. | [`test_nondiagonal_forceconstants.py`](kaldo/tests/test_nondiagonal_forceconstants.py), [fixture and generator](kaldo/tests/data/input/tdep-si-conventional/README.md) |
| Non-diagonal sparse IFC3 | Synthetic determinant-3 skew supercell | Sparse compilation, commensurate direct Fourier equality, plan reuse, and cache invalidation. | [`test_nondiagonal_sparse_ifc3_projection_is_cached_and_remains_sparse`](kaldo/tests/test_thirdorder_interpolation.py#L217) |
| ShengBTE offset preservation | Synthetic `FORCE_CONSTANTS_3RD` records | Negative, half-box, out-of-range, and same-class literal translations are retained; invalid non-lattice offsets fail; exact duplicates add. | [`test_shengbte_offsets.py`](kaldo/tests/test_shengbte_offsets.py) |
| Independent transport ground truth | Official ShengBTE revision `b0d209068239c37fc86d2021efda131ad854f1c1`, VASP and QE Si inputs | All 27 q-point frequencies and velocity norms, VASP mode-resolved anharmonic rates and RTA tensor, and the QE anharmonic-rate distribution. | [`test_shengbte_external_reference.py`](kaldo/tests/test_shengbte_external_reference.py), [inputs and regeneration guide](kaldo/tests/data/input/shengbte-si-reference/README.md) |
| Elastic response | Synthetic relabeling and skew-cell basis images | Elastic constants use the same pair geometry and remain invariant. | [`test_elastic.py`](kaldo/tests/test_elastic.py) |
| Cache separation | Lazy and loaded IFC3 with different matrices, support orders, provenance, and explicit modes | No interpolation representation can read another representation's cached artifacts. | [`test_ifc_cache_identity.py`](kaldo/tests/test_ifc_cache_identity.py) |

The retained ShengBTE references include the exact upstream revision, raw inputs and outputs, run script, calculation settings, and SHA-256 manifest. The VASP case reproduces the mode-resolved rate distribution and RTA tensor within the documented coarse-grid tolerances. For the QE case, the test deliberately claims only the validated quantities: full-grid harmonic data and the anharmonic-rate distribution. The comparison also uncovered a fixture defect: the tracked `FORCE_CONSTANTS_3RD` was written in the standard diamond basis (second atom at $+a/4\,(1,1,1)$) while `POSCAR` and `espresso.ifc2` use the x-mirrored setting ($a/4\,(-1,1,1)$). The fixture file has been corrected by the exact mirror transformation, which restores cubic isotropy of the RTA conductivity diagonal to $0.04\%$. The retained ShengBTE run consumed the original, internally inconsistent pair, so the reference directory keeps its own byte-exact copies of those inputs and the rate comparison runs on them: the contract is same inputs, same rates, against the pinned upstream revision.

The final exact-commit run on two `debug-cpu` cores completed with `477 passed, 1 skipped` in 50 minutes 41 seconds and 1.90 GiB peak resident memory. The compact TDEP fixture replaces nine former IFC2/IFC3 production-data skips with mandatory tests; the sole remaining skip is the explicitly deferred external non-diagonal IFC4 case.

## Rebaselined regression values

The pair-image interpolation changes the absolute numbers pinned by several regression tests. These pins are labeled snapshots at deliberately under-converged settings (coarse q meshes, fixed broadening); they are regression anchors, not converged conductivities. The headline movements, all at each test's own settings:

| Test | Quantity | 2.2.1 pin | This PR | Cause |
|---|---|---|---|---|
| `test_crystal_qe_vasp.py` | QE/Sheng Si RTA trace | 4.500 | 14.864 | pair-image interpolation, then the fixture basis correction above |
| `test_crystal_qe_vasp.py` | QE/Sheng Si inverse trace | 5.049 | 17.667 | same |
| `test_crystal_qe_vasp.py` | QE/Sheng Si QHGK trace | 2.241 | 1.476 | interpolation only: the constant `diffusivity_bandwidth` decouples this pin from the IFC3 basis |
| `test_crystal_vasp.py` | VASP/Sheng Si RTA trace | 16.406 | 12.841 | pair-image interpolation |
| `test_crystal_vasp.py` | VASP/Sheng Si inverse trace | 16.576 | 12.895 | same |
| `test_crystal_vasp.py` | VASP/Sheng Si QHGK trace | 1.694 | 1.616 | same |

Values are W/(m K) traces averaged as each test defines them. The QE movement is dominated by the fixture correction: with a consistent basis the interpolation-only change is the modest kind seen in the VASP row. Per-pin comments in each test file record the same provenance.

Stored artifacts from earlier releases are not silently reused: harmonic and anharmonic cache labels now include the resolved interpolation mode and the translation-support digest, so a 2.2.1 cache never matches. Old `storage='numpy'` or `storage='formatted'` folders simply go stale on disk; delete them to reclaim space and to avoid comparing new runs against stale files by hand.

The harmonic assembly is now a vectorized scatter over a flattened per-image plan rather than a Python loop over atom pairs. On the 216-atom amorphous fixture one dynamical-matrix assembly drops from 0.28 s to 0.02 s, and the amorphous results are bit-for-bit identical; crystal derivative kernels move by at most one floating-point ulp.

## Scope boundaries and current limitations

- This PR fixes real-space translation topology, pair-image interpolation, source preservation, and the resulting harmonic/anharmonic/elastic integration. It does not change the Gaussian-width convention or the three-phonon delta-function integration scheme.
- Wigner–Seitz interpolation is currently limited to fully three-dimensionally periodic cells. The code raises rather than inventing images through a nonperiodic direction. `is_nw=True` changes only the four-mode Gamma acoustic mask; true axis-only nanowire interpolation and effective-area normalization are deferred and diagnosed separately.
- q-symmetry replication is not yet validated when compiled IFC3 support has $S\ne N$. That combination logs a warning and falls back to the full q-point grid for the anharmonic projection: the result is correct, only slower, and the cleared flag is what cache labels record.
- The direct-calculator test exercises the same callable ASE-calculator boundary used by MLIPs with a self-contained Lennard–Jones model. CI does not depend on a particular external MLIP framework or model file, so this should not be described as framework-specific MLIP validation.
- Most cross-format origin tests regauge a successfully loaded object so that every parser feeds the same invariant calculation. Loader-level shifted-representation parsing is additionally tested directly for ShengBTE offsets, TDEP literal translations, and QE header/auxiliary geometry.

## Suggested review order

1. Read the four data-model classes in [`kaldo/grid.py`](kaldo/grid.py): they establish the distinction between q points, periodic classes, stored translations, and pair images.
2. Review the harmonic formula and derivative together in [`kaldo/observables/harmonic_with_q.py`](kaldo/observables/harmonic_with_q.py), then the IFC3 Cartesian-product compilation in [`kaldo/observables/thirdorder.py`](kaldo/observables/thirdorder.py).
3. Check that IFC3 support size is propagated without falling back to `n_replicas` in [`Phonons._project_crystal`](kaldo/phonons.py#L1932) and [`sparse_potential_mu`](kaldo/controllers/anharmonic.py#L112).
4. Review source boundaries in [`tdep_io.py`](kaldo/interfaces/tdep_io.py), [`shengbte_io.py`](kaldo/interfaces/shengbte_io.py), and [`qe_io.py`](kaldo/interfaces/qe_io.py).
5. Finish with the origin-invariance, external ShengBTE, non-diagonal TDEP, and cache-identity tests listed above; these are the integration contracts that motivated the refactor.

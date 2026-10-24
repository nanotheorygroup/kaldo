# Amorphous third-order projection

Scalar amorphous calculations automatically reduce three-phonon phase space
and anharmonic linewidths during projection. They avoid constructing and
retaining a sparse tensor for every allowed triplet. The calculation uses
float64 throughout, without permutation symmetry or additional IFC
thresholding; changes in contraction and summation order permit roundoff
differences from the previous projector.

## When the projection is selected

The route applies to `Phonons.phase_space` and `anharmonic_bandwidth` when
`kpts=(1, 1, 1)`, the harmonic supercell contains one cell,
`broadening_shape="gauss"`, and `is_balanced=False`. Both quantum and classical
statistics are supported. It requires a finite, nonnegative scalar
`third_bandwidth` in THz, finite sorted frequencies, and strictly positive
frequencies on physical modes. A zero width or empty physical selection
returns zeros.

The source IFC3 must be a finite real float64 NumPy array or zero-filled
`sparse.COO`, with the supported rank-three or rank-eight shape. Eigenvectors,
frequencies, and populations must use the supported real float64 NumPy
representations; eigenvectors and physical populations must be finite.
Other broadening shapes, balanced or collision-tensor calculations,
unsupported array types or shapes, and non-float64 inputs retain their
previous projection behavior. The crystal route is unchanged.

## Contraction, shapes, and units

For `N` atoms, `n=3*N` Cartesian modes, and `S` translations, IFC3 has shape
`(n, S*n, S*n)` or `(N, 3, S, N, 3, S, N, 3)` in eV/Angstrom cubed.
Translations are summed at Gamma into owned, chunked float64 CSR rows ordered
by the Cartesian pair `(j, i)`. The authoritative source remains unchanged.
Mass-rescaled eigenvectors have shape `(n, n)` and inverse square-root atomic
mass units. Frequencies `(1, n)` and the supplied Gaussian width are in THz;
the reducer converts them to angular frequencies in rad/ps. Populations
`(n,)` and the boolean physical mask are dimensionless.

The third Cartesian index is contracted first in column blocks, then the
initial modes in blocks. Sorted frequencies restrict the final contraction
to tiles enclosing each energy window. The exact existing selection
`abs(delta_omega) < 2*sigma` is applied inside each tile using angular units.
The reduced float64 array has shape `(n, 2)`: phase space followed by
linewidth in THz. Public scalar properties retain shape `(1, n)`.
Phase space has inverse-angular-frequency units, or ps with dimensionless
radians, and retains the package's occupation and Gaussian normalization.
Classical values include the internal inverse-hbar population scaling and
cannot be compared directly with quantum phase-space magnitudes.

## Memory and CPU workers

The private planner starts with at most 512 columns and 256 initial modes.
Its estimated scratch budget in bytes is
`min(0.25*available_memory, max(128 MiB, 8*(CSR_bytes + 8*n**2), minimum_feasible_scratch))`.
Here `CSR_bytes` counts CSR data, indices, and row pointers, and
`minimum_feasible_scratch` is the estimate for one column and one mode with
the requested workers. Columns are reduced first, then mode blocks; a budget
too small for the minimum raises `MemoryError`.
This bounds estimated projection scratch, not total process RSS. The source
IFCs remain resident, and IFC loading, CSR packing, and other retained arrays
can set the peak. No persistent CSR cache is created.

`n_workers=1` remains the default. On this route, larger values use CPU
threads, and `None` selects the CPU affinity count when available, otherwise
the CPU count. It uses neither worker processes nor a GPU. BLAS limits are
managed through `threadpoolctl`; these limits are process-global, so
simultaneous projections in one process can interfere with each other's
thread limits.

The first eligible calculation compiles a Numba reducer and can be slower
for small systems. If the optional acceleration cannot be imported, including
an unsupported SciPy BLAS bridge, a NumPy reducer performs the same operation.
Arithmetic and execution failures propagate. TensorFlow remains a dependency
for harmonic, crystal, explicit sparse, and tensor calculations.

## Release behavior and caches

Public constructor and method signatures, defaults, and existing attributes
remain unchanged.
The existing `_ps_and_gamma` cache keys, result layout, loading order, and
`memory`, `numpy`, `hdf5`, and `formatted` storage formats are unchanged.
An existing collision-tensor cache can still supply its first two columns.
Existing reduced results therefore load before projection is attempted.

Eligible scalar calculations no longer implicitly populate the per-triplet
sparse cache. Explicit `sparse_phase` and `sparse_potential` requests retain
their TensorFlow sparse types, persistence formats, and loading behavior.
Temperature or statistics sweeps without cached reduced results may repack
and reproject rather than cheaply reweighting a retained sparse projection.

## Reproducing timings

Run the following from the repository root in a fresh Python process. It
uses the public scalar properties and memory storage. Harmonic setup and IFC
loading precede the timer. The cold time includes CSR packing and first-use
JIT compilation. Warm repeats explicitly remove the in-memory reduced and
sparse caches, so they measure recomputation, including repacking, rather
than cache hits. The cache attribute names used for invalidation are private.

```python
import os
os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1")
import resource
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons

with threadpool_limits(limits=1, user_api="blas"):
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/data/input/amorphous-si-tersoff-32",
        format="numpy", supercell=(1, 1, 1), is_acoustic_sum=True,
    )
    phonons = Phonons(forceconstants=forceconstants, temperature=300.0,
                      kpts=(1, 1, 1), third_bandwidth=0.05 / 4.135,
                      broadening_shape="gauss", is_balanced=False,
                      is_classic=False, n_workers=1, storage="memory")
    for name in ("frequency", "physical_mode", "population", "eigenvectors"):
        getattr(phonons, name)
    timings = []
    for repeat in range(3):
        for name in ("phase_space", "anharmonic_bandwidth", "_ps_and_gamma",
                     "_sparse_phase_and_potential"):
            vars(phonons).pop("_lazy__" + name, None)
        started = time.perf_counter()
        phase = phonons.phase_space
        linewidth = phonons.anharmonic_bandwidth
        timings.append(time.perf_counter() - started)
        if repeat == 0:
            reference = (phase.copy(), linewidth.copy())
        np.testing.assert_allclose(phase, reference[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(linewidth, reference[1], rtol=1e-12, atol=1e-13)
print("cold seconds:", timings[0], "warm median seconds:", np.median(timings[1:]))
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("process peak MiB:", peak / (1024**2 if sys.platform == "darwin" else 1024))
```

`resource` reports a process-lifetime peak on Linux/macOS, including imports
and IFC loading. For the 216-atom fixture, use
`folder="kaldo/tests/si-amorphous"` and `format="eskm"`; select classical
statistics with `is_classic=True`. Keep CPU affinity and worker/thread counts
fixed when comparing revisions.

### Integrated small-system measurements

Measured on an AMD Threadripper 3970X, 32 physical/64 logical cores, 251 GiB
RAM, with Python 3.11.13, NumPy 1.26.4, SciPy 1.17.1, TensorFlow 2.17.1,
`sparse` 0.17.0, and Numba 0.61.2. All runs used CPU only, BLAS 1 thread,
TensorFlow 2 intra-op/1 inter-op threads, `n_workers=1`, 300 K, Gaussian
width `0.05/4.135` THz, and memory result storage. Warm values are medians of
two actual recomputations; process RSS is reported in KiB. The before column
uses the original `Phonons` projector from revision `d041c9b2`.

<table>
  <thead><tr><th>Atoms / statistics</th><th>Cold seconds, before / after</th><th>Warm seconds, before / after</th><th>Process peak KiB, before / after</th></tr></thead>
  <tbody>
    <tr><td>32 / quantum</td><td>0.875143 / 1.014457</td><td>0.459318 / 0.024936</td><td>674364 / 682956</td></tr>
    <tr><td>32 / classical</td><td>0.869812 / 1.004792</td><td>0.453406 / 0.026521</td><td>673128 / 681048</td></tr>
    <tr><td>216 / quantum</td><td>36.229568 / 5.764523</td><td>not measured / 4.777326</td><td>899440 / 849500</td></tr>
    <tr><td>216 / classical</td><td>40.174935 / 5.831055</td><td>not measured / 4.791794</td><td>900292 / 848688</td></tr>
  </tbody>
</table>

Validation requires exact masks, selected-pair counts, cache keys, shapes,
and dtypes. Float64 comparisons use `rtol=1e-12`, with `atol=1e-12` for phase
space and `atol=1e-13` THz for linewidths, allowing reordered reductions.
These runs differed by less than `6.8e-16` relative in phase space and
`7.5e-15` in linewidths. Classical phase-space absolute differences reached
`1.91e-6` because of the occupation scaling; the relative tolerance applies.
Float32 QHGK checks use `rtol=5e-7`, `atol=1e-8` to allow float32 rounding:
conductivity `(n, 3, 3)` is in W/(m K) and diffusivity `(1, n)` in mm squared/s.
All small-system QHGK arrays were bitwise equal.

### Integrated 1920-atom carbon measurement

Both quantum and classical calculations used all 5760 modes, including 5757
physical modes, at 300 K, Gaussian width `0.5/4.136` THz, `n_workers=32`, and
memory result storage. They used the same hardware/software, CPU only,
BLAS 1 thread, and TensorFlow 32 intra-op/1 inter-op threads. Automatic blocks were 512 columns
and 256 modes. No large-system warm repeat was measured.

The harmonic eigenvectors were pinned, and original reference frequencies
and populations were installed to match the trusted reference inputs exactly.
The preparation time below measures that installation, not a fresh harmonic
recomputation. Timed totals exclude Python imports.

<table>
  <thead><tr><th>Integrated stage</th><th>Quantum seconds</th><th>Classical seconds</th></tr></thead>
  <tbody>
    <tr><td>Source IFC loading</td><td>332.710777</td><td>332.102758</td></tr>
    <tr><td>Pinned harmonic/reference preparation</td><td>0.158391</td><td>0.031916</td></tr>
    <tr><td>Projection, including packing and first-use JIT</td><td>427.050243</td><td>447.740163</td></tr>
    <tr><td>CSR packing within projection</td><td>78.366589</td><td>78.332847</td></tr>
    <tr><td>Projection kernel within projection, including JIT</td><td>348.478925</td><td>369.201588</td></tr>
    <tr><td>QHGK transport</td><td>199.868253</td><td>243.532185</td></tr>
    <tr><td>Total loading, preparation, projection, and transport</td><td>959.787671</td><td>1023.407029</td></tr>
  </tbody>
</table>

Process-lifetime peak RSS, including loading and packing, was 87,563,448 KiB
(83.507 GiB) for quantum and 87,565,212 KiB (83.509 GiB) for classical.
In both runs, the authoritative source retained 24,323,877,720 bytes; CSR
data, indices, and row pointers occupied 14,602,872,852 bytes for
1,216,193,886 entries and 2,136,402 pair rows.

The trusted earlier production report measured 51,695 s of legacy projection
plus 1439 s of setup, including 331 s of IFC loading, and 242.8 GiB peak RSS.
That historical run was not repeated. Relative to it, the integrated quantum
projection is 121.05 times faster and peak RSS 2.91 times lower. New quantum
loading, pinned preparation, and projection total 759.919411 s versus the
earlier 53,134 s, a 69.92-fold pre-transport reduction with the harmonic
preparation qualification above. No separate historical classical wall time
was measured for this comparison.

For each statistics choice, scalar comparisons against 40 original reference
modes have exact inputs and selected-pair counts. Quantum/classical maximum
relative differences were `9.82e-16`/`1.41e-15` for phase space and
`1.01e-15`/`1.86e-15` for linewidths. Maximum absolute differences were
`5.82e-11`/`4.58e-5` ps and `8.88e-16`/`1.55e-15` THz respectively, within
the scalar tolerances above. All 5760-mode outputs in both runs were finite
and nonphysical modes were zero.

Full QHGK quantum diffusivity was bitwise equal; conductivity differed by at
most `1.82e-12` W/(m K), or `1.77e-7` relative. Classical diffusivity differed
by at most `4.66e-10` mm squared/s, or `1.14e-7` relative; conductivity by
`1.46e-11` W/(m K). The largest classical conductivity relative difference,
`3.17e-6`, occurs in near-zero off-diagonal components. All comparisons pass
the original `atol=1e-8` plus `rtol=5e-7` times the reference magnitude;
neither tolerance is relaxed. Total conductivity diagonals in W/(m K) are
`(1.9136855602, 1.9244325161, 1.9094816446)` for quantum and
`(3.5563695431, 3.5900201797, 3.5835828781)` for classical, matching production.

The standalone 261 s/27.1 GiB prototype used permutation symmetry and float32
IFC storage. Those numbers do not measure this float64 implementation, which
retains its authoritative source and assumes no permutation symmetry.

# Amorphous Gamma eigensystem

Amorphous calculations (`kpts=(1, 1, 1)` and a one-cell harmonic supercell)
diagonalize a single real dynamical matrix at Gamma. kALDo assembles that
matrix once, directly in the float64 buffer that will hold the eigensystem,
and diagonalizes it in place with a multithreaded LAPACK divide-and-conquer
driver. No complex Fourier matrix, derivative kernels or TensorFlow tensor is
created on this route.

## When the route is selected

The route applies to `Phonons` and `HarmonicWithQ` requests with
`is_amorphous=True` at the Gamma point, without an active non-analytic
correction, with `distance_threshold=None`, and with a dense float64 NumPy
IFC2 of shape `(1, N, 3, S, N, 3)`. Other requests keep the TensorFlow route.

The dynamical matrix element for atoms `i`, `j` and Cartesian components
`a`, `b` is the Gamma sum over the `S` stored translations of
`phi[i, a, s, j, b] / sqrt(m_i m_j)`, in (rad/ps)^2. As before, only the lower
triangle in C order is read, so finite-difference IFCs that are slightly
asymmetric give the same eigenvalues as the historical solver. The returned
eigensystem keeps its `(3N+1, 3N)` float64 C-order layout: eigenvalues in the
first row and eigenvector `k` in column `k`.

## Solvers, memory and threads

For `n = 3N` modes, the eigensystem buffer holds `8 n^2` bytes; the IFC2
source holds another `8 S n^2` bytes. Divide and conquer (`dsyevd`) needs a
further `16 n^2` bytes of workspace. When that workspace does not fit in 90%
of the available memory, the solver switches to MRRR (`dsyevr`), which needs
`8 n^2` bytes for its separate eigenvector array but is slower. Eigenvalue-only
requests (`frequency` before any eigenvector property) use one `8 n^2` scratch
copy and `O(n)` workspace.

SciPy's LAPACK is always available. If Intel oneMKL is installed (for example
`pip install kaldo[mkl]`, a conda `mkl` package, or an `intel-oneapi-mkl`
module on `LD_LIBRARY_PATH`), its 64-bit integer interface is loaded
privately and used instead. MKL is 2 to 6 times faster than SciPy's bundled
OpenBLAS here (see below). SciPy's 32-bit integer `dsyevd` workspace limits it
to `n <= 32766` (about 10,900 atoms); larger SciPy requests use `dsyevr`.
SciPy above that size has not been benchmarked; install MKL for large systems.

`Phonons(n_workers=...)` sets the eigensolver threads. The default `1` is
serial; `None` uses the CPU affinity count.

An eigensystem that is still referenced, for example by `Phonons._eigensystem`,
is shared with later Gamma requests for the same IFC2 array through a weak
reference, so frequencies, participation ratios and flux operators do not
diagonalize again. Requesting `frequency` first performs a separate
eigenvalue-only solve; request `eigenvectors` first when both are needed.

## Measurements

Measured on an AMD Threadripper 3970X, 32 physical/64 logical cores, 251 GiB
RAM, with Python 3.11.13, NumPy 1.26.4, SciPy 1.17.1, TensorFlow 2.17.1,
Numba 0.61.2 and oneMKL 2025.2.0 from pip. All runs were CPU only. Peaks are
resident-set high-water marks per stage, including the resident IFC2.

### Integrated 6480-atom amorphous carbon

`Phonons(kpts=(1, 1, 1), storage="memory")` on a 6480-atom amorphous carbon
cell, `n = 19440` modes, 3.0 GiB IFC2. The historical route ran with
`OMP_NUM_THREADS=1`; its TensorFlow CPU `eigh` does not use the BLAS thread
pool. The new route used `n_workers=30`. "Frequency" is the first property
requested and "eigensystem" the next (`eigenvectors`).

<table>
  <thead><tr><th>Route</th><th>Frequency seconds / peak GiB</th><th>Eigensystem seconds / peak GiB</th></tr></thead>
  <tbody>
    <tr><td>Historical TensorFlow</td><td>1854 / 54.6</td><td>12918 / 51.8</td></tr>
    <tr><td>New, SciPy OpenBLAS</td><td>171 / 6.3</td><td>246 / 12.0</td></tr>
    <tr><td>New, MKL</td><td>28 / 6.4</td><td>129 / 12.2</td></tr>
    <tr><td>New, MKL, eigenvectors first</td><td>reused</td><td>124 / 12.2</td></tr>
  </tbody>
</table>

The historical route also re-diagonalizes for `participation_ratio` with
memory storage; the new route reuses the live eigensystem (about 1 s,
blocked). Against a separate production run of the historical code on the
same system, the new frequencies agree within `5.6e-11` THz on all 19437
physical modes. sp2/sp3 group participation ratios and mode weights,
computed from the new eigenvectors with that run's script, agree within
`9e-10` and `4e-9`.

### Solver scaling

Larger periodic matrices were built by tiling the 6480-atom dynamical matrix
(each atom-pair block placed at its minimum-image neighbour cell) and
diagonalized with MKL `dsyevd_64`, 30 threads, as the route does. "Extra" is
memory beyond the matrix itself.

<table>
  <thead><tr><th>Atoms</th><th>n</th><th>Eigensystem seconds</th><th>Matrix GiB</th><th>Extra GiB</th></tr></thead>
  <tbody>
    <tr><td>6480</td><td>19440</td><td>117</td><td>2.8</td><td>5.9</td></tr>
    <tr><td>12960</td><td>38880</td><td>958</td><td>11.3</td><td>22.8</td></tr>
    <tr><td>25920</td><td>77760</td><td>7019</td><td>45.1</td><td>90.7</td></tr>
  </tbody>
</table>

At 25,920 atoms, the largest residual `|D v - w v|` was `6.7e-10` and the
largest orthogonality error `3.9e-14`. Each tiling is block circulant with
phases of plus or minus one, so its exact spectrum is the union of the
spectra of a few real 19440-mode blocks. The 77760 eigenvalues agree with
that union within `6.4e-14` relative.

Eigenvalues alone were computed for a 51,840-atom tiling (`n = 155520`,
180.2 GiB matrix) in 12041 s, with 0.5 GiB of extra memory and a 191 GiB
process peak. They agree with the union of the eight block spectra within
`3.8e-13` relative. This run used the solver directly: the route itself would
also hold a 180 GiB IFC2, which does not fit beside the matrix on this machine.

For the same matrices, ELPA 2026.02 (one node, two-stage) was 1.2 times faster
than MKL at 12,960 atoms with 25% more extra memory. A JAX CPU `eigh` took
253 s at 6480 atoms, and float64 JAX did not fit on an 11 GB GPU. Neither is
used.

### Limits

Time grows as `n^3` and the full eigensystem needs `3 x 8 n^2` bytes with
`dsyevd` or `2 x 8 n^2` with `dsyevr`, plus the IFC2. At 50,000 atoms
(`n = 150000`) that is 503 or 335 GiB, plus 168 GiB for a resident one-cell
IFC2, and about 14 hours with MKL on 30 cores (scaling the 25,920-atom run by `n^3`). Such runs need a large-memory
node, or a distributed solver (for example ELPA or ScaLAPACK over MPI),
which this route does not provide. Eigenvalues alone need one `8 n^2`
matrix and `O(n)` workspace.

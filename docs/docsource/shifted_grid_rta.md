# RTA conductivity on a shifted q-grid

## Purpose

For a disordered cell treated as one large unit cell, a Gamma-centred 2x2x2 q-grid is the smallest grid that samples
more than the zone centre. This page describes the half-step shifted 2x2x2 grid, which gives nonzero group
velocities and therefore a nonzero relaxation-time-approximation (RTA) conductivity at the same cost scale.

## Why the Gamma-centred 2x2x2 grid gives zero velocities

Every point q0 of the Gamma-centred 2x2x2 grid satisfies 2 q0 = G for a reciprocal lattice vector G, so it is
time-reversal invariant. Near such a point the dispersion obeys omega(q0 + d) = omega(-q0 - d) = omega(q0 - d), where
the first equality is time-reversal symmetry and the second is periodicity. The frequency is therefore even in d, its
gradient vanishes at d = 0, and the group velocity of every nondegenerate mode is exactly zero. The RTA conductivity,
which is quadratic in velocity, vanishes with it.

## Supported configuration

Shifting the grid by half a grid step, q = (n + 1/2) / 2 along each axis (equivalently (+-1/4)^3 in fractional
coordinates), removes the time-reversal invariance and gives nonzero velocities. Anharmonic properties on the shifted
grid are supported for exactly this configuration:

- `kpts=(2, 2, 2)` and `kpts_shift=(0.5, 0.5, 0.5)`;
- `forceconstants.supercell_grid.size == 1` (a single-cell force-constant set);
- a third-order translation support of size 1;
- Gaussian broadening with a fixed `third_bandwidth`;
- `is_balanced=False`, `include_isotopes=False`, `use_q_symmetry=False`;
- no active non-analytic correction (NAC).

Any other combination raises `NotImplementedError`. The shifted path uses no TensorFlow operations.

## Only RTA is supported

The shifted grid is not closed under q +- q'. The three-phonon channels couple q to points that are not on the grid,
so the scattering matrix needed for the full Boltzmann solution (`method='inverse'`, `'sc'`, relaxon) cannot be
formed. Only the diagonal linewidths are defined, so only `Conductivity(method='rta')` is available; requesting a
collision matrix raises `ValueError`.

## Usage

```python
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity

fc = ForceConstants.from_folder(folder="kaldo/tests/si-amorphous", format="eskm")
phonons = Phonons(forceconstants=fc, kpts=(2, 2, 2), kpts_shift=(0.5, 0.5, 0.5), temperature=300,
                  third_bandwidth=0.05, storage="memory", n_workers=8)
kappa = Conductivity(phonons=phonons, method="rta", storage="memory").conductivity.sum(axis=0)
```

`phonons.bandwidth` and `phonons.phase_space` are also available on the shifted grid.

## API note

`kpts_shift` is a new `Phonons` argument. Its default, `(0, 0, 0)`, leaves every existing calculation unchanged.
Each component must be 0 or 0.5. Results for different shifts are cached in separate folders.

## Measurements

All values below were measured on 2026-11-12, none are estimates.
Driver: `examples/amorphous_carbon/shifted_rta_1920.py`.

**Setup**

- Hardware: one AMD Ryzen Threadripper 3970X (32 cores, 64 hardware threads, 128 MB L3) with 251 GB RAM.
- Run settings: `n_workers=64`, `OMP_NUM_THREADS=1`, CPU only.
- System: amorphous carbon, 1920 atoms (n = 5760 modes), with single-cell IFCs from `numpy` files. The IFC3 has 1.216e9
  nonzero entries in 2,136,402 Gamma pair rows.
- Physics settings: `kpts=(2, 2, 2)`, `kpts_shift=(0.5, 0.5, 0.5)`, `third_bandwidth=0.5/4.136` THz (Gaussian),
  300 K, quantum statistics.

**Wall time and memory**

| Step | Wall time |
|---|---|
| Force-constant loading | 320 s |
| Shifted harmonics (4 complex + 8 real eigensolves of n = 5760, velocities) | 188 s |
| Projection: planning (pair layout, leg tables, 7.7e6 tiles) | 167 s |
| Projection: stage 1 (sparse IFC3 contraction) | 339 s |
| Projection: stage 2 | 15,307 s |
| Projection: energy-window tiles | 7,592 s |
| Projection total (32 passes, 4.87e10 selected triplets) | 23,556 s |
| RTA conductivity | < 0.1 s |
| Total | 24,064 s (6 h 41 min) |

- Peak RSS was 83.6 GiB (87.7 GB). It was reached while loading the IFC3, before the projection started.
- A harmonic-only run (no IFC3) took 186 s and peaked at 9.0 GiB.

**Results**

- RTA conductivity (W/m/K):

  ```
  [[ 4.2048  0.0433 -0.2715]
   [ 0.0433  3.9710  0.2774]
   [-0.2715  0.2774  4.2854]]
  ```

  trace/3 = 4.154 W/m/K.
- Group velocity |v| over all 46,080 modes (all physical; frequencies 1.53 to 60.39 THz):
  - median 1.085 Angstrom/ps (108.5 m/s);
  - mean 1.506 Angstrom/ps.
- Three-phonon linewidth (`phonons.bandwidth`, THz): median 0.802, minimum 0.019, maximum 45.5. The 28 modes with
  linewidths above 5 THz lie between 33 and 43 THz.

Median and mean linewidth by frequency bin, all 8 shifted points:

| Frequency (THz) | Modes | Median linewidth (THz) | Mean linewidth (THz) |
|---|---|---|---|
| 0-5 | 258 | 0.119 | 0.135 |
| 5-10 | 2398 | 0.348 | 0.374 |
| 10-15 | 5272 | 0.413 | 0.452 |
| 15-20 | 6898 | 0.561 | 0.593 |
| 20-25 | 7370 | 0.714 | 0.758 |
| 25-30 | 7298 | 0.947 | 1.005 |
| 30-35 | 7324 | 0.967 | 1.072 |
| 35-40 | 5814 | 0.977 | 1.049 |
| 40-45 | 2920 | 0.888 | 1.115 |
| 45-50 | 520 | 0.782 | 0.992 |
| 60.39 (one mode per point) | 8 | 1.611 | 1.611 |

These linewidths could not be compared with Gamma-only linewidths for this cell, because no stored Gamma-only
bandwidth array was found in the source folder.

**Effect of the optimizations**

On the a-Si 216-atom test cell with 8 workers, the projection took 80.4 s before the stage-2 and tile optimizations
and 49.3 s after them. The bandwidths agree to 7e-16 relative.

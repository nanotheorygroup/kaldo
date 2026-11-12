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

To be added.

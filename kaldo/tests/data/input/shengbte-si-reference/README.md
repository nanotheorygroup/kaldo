# ShengBTE Si transport references

These cases provide an independent interpretation of the two ShengBTE-format
third-order Si fixtures in `kaldo/tests/si-crystal`. They are evaluated with
the official ShengBTE `master` revision
`b0d209068239c37fc86d2021efda131ad854f1c1` (`v1.5.1-3-gb0d2090`).

Both calculations use a `3 x 3 x 3` Gamma-centred q mesh, 300 K,
`scalebroad=1`, RTA only, and no isotope or non-analytic scattering. These
choices isolate the harmonic and third-order IFC conventions. The VASP case
uses the fixture's Phonopy `FORCE_CONSTANTS`; the QE case uses its native
`espresso.ifc2`. Both use their existing `FORCE_CONSTANTS_3RD` files without
rewriting cell offsets.

The external calculation is intended to ground:

- the full-grid harmonic frequencies and group velocities;
- three-phonon phase space and anharmonic rates;
- the final bulk RTA conductivity tensor.

The two codes must still be compared with care: a conductivity comparison is
meaningful only when the q mesh, temperature, isotope/NAC flags, adaptive
ShengBTE broadening, and tensor normalization are matched. The raw ShengBTE
outputs are retained under each case's `reference/` directory.

## What the comparison established

The VASP/Phonopy route is the clean external transport oracle. Across the full
mesh, kALDo's pair-specific Wigner--Seitz route differs by at most
`8.64e-4 rad/ps` in frequency and `0.150 angstrom/ps` in sorted velocity norm.
For nonzero irreducible-mode anharmonic rates, the kALDo/ShengBTE ratio spans
`0.9869` to `1.0132`, with median `1.00005`. The scalar RTA conductivities are
`12.95545 W/(m K)` in kALDo and `12.739 W/(m K)` in ShengBTE, a `1.70%`
difference on this very coarse mesh.

The QE route is the regression for q2r interpolation. ShengBTE's
`phonon_espresso` routine and kALDo's automatic q2r route both reconstruct
pair-specific Wigner--Seitz images; a direct periodic transform is now only an
explicit diagnostic in kALDo. The comparison compensates for one documented
input convention: ShengBTE replaces the `28.08 amu` mass stored in the q2r
file with its `28.0855099896 amu` natural-isotope average. Frequencies and
velocities are rescaled analytically for that mass difference, so their tight
tolerances continue to test interpolation rather than atomic-mass policy.
On the matched adaptive-broadening calculation, the median anharmonic-rate
ratio is `0.99993` (range `0.8590` to `1.1494`). The coarse-mesh scalar RTA
conductivities are `0.8130 W/(m K)` in kALDo and `0.9662 W/(m K)` in
ShengBTE. This larger final difference is reported explicitly rather than
hidden by the close median mode rate.

This remains a ShengBTE-compatibility oracle. Independent QE 7.6 off-grid
frequency and Cartesian-velocity references are retained with the GaN q2r
tests. Together they reveal why commensurate-frequency-only tests miss phase
and transport defects without allowing one external program to define every
input convention.

The external job was Slurm job `5416312`; the convention-matched audit is in
`agents/compare_shengbte_kaldo.py` and its final log is
`agents/slurm-compare-qe-ws-transport-5416412.out`.

## Regeneration

Build ShengBTE as described in `agents/SHENGBTE_GUIDE.md`, allocate the desired
number of Slurm MPI tasks, and run one case into a new empty directory:

```bash
module load mpi/openmpi-4.1.8
SHENGBTE_MPI_RANKS=2 \
  kaldo/tests/data/input/shengbte-si-reference/run_reference.sh vasp /tmp/si-vasp-shengbte
SHENGBTE_MPI_RANKS=2 \
  kaldo/tests/data/input/shengbte-si-reference/run_reference.sh qe /tmp/si-qe-shengbte
```

Audit fresh results before replacing any retained file. After an intentional
update, regenerate `manifest.json` with:

```bash
python kaldo/tests/data/input/shengbte-si-reference/generate_manifest.py
```

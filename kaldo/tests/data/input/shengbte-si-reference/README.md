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

The QE route exposes a representation distinction rather than a simple FC3
error. ShengBTE's `phonon_espresso` routine reconstructs pair-specific
Wigner--Seitz images for q2r IFC2. kALDo's current automatic nonpolar-q2r path
uses the direct periodic representation. Both give the same frequencies on
the commensurate `3 x 3 x 3` mesh (maximum difference `0.00196 rad/ps`), but
the automatic kALDo velocity norms differ from ShengBTE by as much as
`30.92 angstrom/ps`. An explicit kALDo Wigner--Seitz override reduces that to
`0.193 angstrom/ps` and moves the median anharmonic-rate ratio from `1.150` to
`1.0013`.

That QE result is a ShengBTE-compatibility oracle, not by itself a universal
QE oracle: QE `matdyn.x` and ShengBTE can choose different off-mesh q2r
interpolations. The raw data pin what ShengBTE computes and reveal why
commensurate-frequency-only tests miss velocity and transport defects; they
do not silently redefine the intended QE API.

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

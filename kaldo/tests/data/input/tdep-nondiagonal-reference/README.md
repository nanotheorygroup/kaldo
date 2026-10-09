# Non-diagonal TDEP reference cases

These four small fixtures exercise translation handling for real non-diagonal
TDEP supercells. Each directory contains `infile.ucposcar`,
`infile.ssposcar`, `infile.forceconstant`, and reference harmonic frequencies
in `expected.json`.

| case | supercell determinant | role |
| --- | ---: | --- |
| `mp-7` | 9 | odd-SNF per-pair phase regression |
| `mp-1000` | 32 | even-SNF control |
| `mp-9947` | 3 | well-resolved compact-cell control |
| `mp-1221485` | 24 | force constants extend beyond one periodic representative per pair |

The force constants and frequency references are derived from
[Phonix](https://phonix-db.org/) entries for the named Materials Project IDs
and are distributed under CC-BY-4.0. The `expected.json` files retain the
per-case source attribution. Frequencies come from the corresponding ALAMODE
`S.evec` data with `nonanalytic=0`. The compact phonopy IFC2 arrays retained for
`mp-9947` and `mp-1221485` were independently generated from the same source
force constants.

License text: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

The tests intentionally evaluate both:

1. the literal Fourier sum read independently from `infile.forceconstant`; and
2. the external ALAMODE reference frequencies.

The first check isolates kALDo's import and phase bookkeeping. The second
checks the resulting physical spectrum against a separate implementation.

# Compact non-diagonal TDEP IFC fixture

This fixture uses the two-atom diamond-Si primitive geometry and the
determinant-four integer tiling

```text
[[ 1, -1,  1],
 [ 1,  1, -1],
 [-1,  1,  1]]
```

to produce one eight-atom conventional cubic cell. It replaces the external
216-atom production fixture formerly required by the non-diagonal loader
tests.

The IFCs are an exact analytic nearest-neighbour model. Harmonic bonds use
`k |u_B - u_A|^2 / 2`; cubic bonds use
`g [e . (u_B - u_A)]^3 / 6`. Consequently, the IFC2 and IFC3 acoustic sum
rules hold by construction. The model is intended to test translation
support, Fourier phases, sparse IFC3 projection, and end-to-end code paths.
It is not a prediction of silicon frequencies or thermal conductivity.

The files use TDEP's documented text layouts, but no TDEP executable is
needed to generate them. Regenerate all data from the readable source with:

```bash
python generate_fixture.py
```

`expected.json` records the topology, model constants, record counts, and
literal translation support so the generated files can be audited without
reverse-engineering a binary reference.

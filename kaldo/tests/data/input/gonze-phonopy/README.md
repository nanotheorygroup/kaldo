# Gonze/Phonopy regression inputs

These 26 pinned, plain-text `phonopy_params.yaml` files validate kALDo's Gonze
NAC correction directly against Phonopy's NAC-on minus NAC-off dynamical
matrices. The set covers 13 of the 14 three-dimensional Bravais lattices (`oF`
is absent) and includes general non-diagonal supercell matrices.

[manifest.json](manifest.json) records the commit-pinned PhononDB source index and, for every
case, the official NIMS MDR dataset page and direct `download_all/*.zip` URL.
Each NIMS archive contains `phonopy_params.yaml.xz`; the manifest pins the
SHA-256 of its decompressed YAML together with the primitive-cell policy,
effective supercell matrix, and space group.

There is no upstream `phonondb-selected-26.yaml.zip`. That name referred to a
locally assembled convenience bundle and is not a reproducible download
location. Download the 26 authoritative ZIP files using the `source_archive`
URLs recorded in [manifest.json](manifest.json), extract
`phonopy_params.yaml.xz` from each, and decompress it
to the checked-in `<material-id>/phonopy_params.yaml` path.

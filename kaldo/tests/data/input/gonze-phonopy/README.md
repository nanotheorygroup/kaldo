# Gonze/Phonopy regression inputs

These pinned `phonopy_params.yaml` files are the seven-lattice inputs used
to validate the Gonze path against Phonopy. They cover cubic, hexagonal,
tetragonal, orthorhombic, trigonal, monoclinic, and triclinic cells, including
non-diagonal supercell metadata. `manifest.json` records the NIMS MDR
PhononDB archive names and SHA-256 checksums.

The fixtures are offline test inputs, not generated numerical outputs. To
recreate them from the original downloaded archives, place the seven archives
named in `manifest.json` in one directory and run:

```bash
python generate_inputs.py /path/to/archives
```

The script verifies each source archive and the extracted `phonopy_params.yaml`
before writing it. No network access is performed.

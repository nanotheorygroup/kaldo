# 32-atom amorphous-Si Tersoff fixture

This directory is a small, regenerable real-potential example for the IFC
pair-image regression. It replaces a high-symmetry or boundary-inert example
with a disordered periodic cell whose nonzero IFC2 and IFC3 interactions cross
the unit-cell boundary.

## Physical model

The structure contains 32 silicon atoms in a cubic periodic cell at a target
density of 2330 kg/m³. It is generated with ASE's implementation of the
elemental-Si parameters from J. Tersoff, *Phys. Rev. B* **37**, 6991 (1988).
Those parameters have a 3.2 Å outer cutoff. The retained cell is wider than
twice that cutoff, so an atom cannot interact twice with the same periodic
neighbor.

Starting from a deterministic random packing, the generator removes the
largest initial forces, melts at 3500 K, quenches linearly to 300 K, and
relaxes the fixed cell. The seed, time step, trajectory lengths, relaxation
criterion, finite-difference displacements, software version, and artifact
checksums are recorded in `generation.json`. The parameter values are listed
directly in `generate_fixture.py`; they match the Si fixture shipped with ASE
3.29 and cite the original publication there.

## Regeneration

Use ASE 3.29 or newer and the kALDo test environment. From the repository root:

```bash
python kaldo/tests/data/input/amorphous-si-tersoff-32/generate_fixture.py --n-workers 2
python kaldo/tests/data/input/amorphous-si-tersoff-32/finalize_reference.py
```

The structure-only stage can be audited separately:

```bash
python kaldo/tests/data/input/amorphous-si-tersoff-32/generate_fixture.py --structure-only --force-structure --output /tmp/kaldo-amorphous-si-32-audit
```

The generator refuses to replace a structure while leaving existing IFC
artifacts beside it. Use a fresh output directory for a structure-only audit,
or run the complete regeneration so the structure and tensors stay paired.

`generate_fixture.py` always recomputes the IFCs instead of silently accepting
files already present in the output directory. `finalize_reference.py` then
measures boundary weight and transport using the retained tensor and writes
`expected.json`. The physical oracle is equality before and after an exact
origin relabeling; the recorded numerical values only pin regression drift.
Both scripts bind kALDo imports to the checkout that contains them, so an
unrelated editable installation cannot generate or validate the artifacts.

The melt--quench and IFC3 calculation are reference-generation steps, not CI
requirements. Tests load the retained compact IFC2 and sparse IFC3 through the
public numpy interface.

## What is tested

At Gamma every lattice-translation phase is one. Selecting a wrong periodic
image can therefore leave all frequencies unchanged while changing the first
real-space moment of IFC2, the heat-flux operator, mode diffusivity, and QHGK
conductivity. The fixture tests that contrast directly.

The test also evaluates the two IFC3 translation legs at selected nonzero q
pairs by contracting the compiled sparse translations independently from the
production projection. It checks the full atom-dependent complex gauge phase,
not only a magnitude that a broken zero-phase compiler could preserve. A
second real-IFC3 representation splits every entry across two literal
translations in the same periodic class; identical Gamma bandwidth,
diffusivity, and QHGK conductivity prove that the projection uses the tensor's
translation support rather than the physical replica count.

The larger `kaldo/tests/si-amorphous` ESKM fixture remains valuable scale and
compatibility coverage. This smaller fixture exists so the decisive invariant
and complete transport calculation can run in ordinary CI with transparent
provenance.

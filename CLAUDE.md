# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project overview

κALDo (kaldo) is a Python package for computing vibrational, elastic, and thermal transport properties of crystalline, disordered, and amorphous materials. It implements the Boltzmann Transport Equation (BTE) for crystals and the Quasi-Harmonic Green-Kubo (QHGK) method for disordered systems, with CPU and TensorFlow-GPU code paths. It interfaces with ab initio codes (Quantum ESPRESSO, VASP), MD packages (LAMMPS), ML potentials (NEP, MACE, MatterSim, Orb, DeepMD via ASE), and external phonon codes (ShengBTE, phono3py, hiPhive, TDEP).

## Tech stack

- Python 3.10+
- numpy, scipy, ase, sparse, opt_einsum, h5py, pandas, scikit-learn, seekpath, hiphive
- TensorFlow >= 2.13 (GPU acceleration; CPU fallback)
- pytest + pytest-cov for tests; codecov for coverage upload

## Common commands

- `pytest`: run the test suite (from repo root)
- `pytest kaldo/tests/test_crystal.py`: run a single test file
- `pytest kaldo/tests/test_crystal.py::test_name`: run one test
- `pip install -e .`: editable install for development
- CI is CircleCI (`.circleci/config.yml`)

## Project structure

- `kaldo/`: package source
  - `phonons.py`, `conductivity.py`, `quasiharmonic.py`, `forceconstants.py`, `grid.py`, `storable.py`: top-level modules for the public API
  - `controllers/`: anharmonic scattering, dirac kernel, displacement, isotopic, plotting, sigma2
  - `observables/`: force constants, harmonic-with-q, second/third-order tensors
  - `interfaces/`: I/O for eskm, hiPhive, ShengBTE, TDEP
  - `parallel/`: process-pool executor for ML calculators
  - `helpers/`: logger and small utilities
  - `tests/`: pytest suite with reference data per material (`si-crystal/`, `si-amorphous/`, `mgo/`, etc.)
- `cli/`: command-line entry points
- `docs/`: documentation source
- `examples/`: small runnable examples
- `dockerfiles/`, `Dockerfile`, `doc.Dockerfile`: container builds

## Conventions

- **Style**: line length 119 (see `setup.cfg` `[flake8]` and `[yapf]`). Indent 4 spaces, no tabs.
- **Linting/formatting**: flake8 and yapf are the configured tools. Don't introduce ruff/black without an explicit decision — it would touch every file.
- **Testing**: tests live under `kaldo/tests/`, named `test_*.py`. Use `tmp_path` for filesystem isolation. Reference materials sit in subdirectories alongside the test files (e.g. `kaldo/tests/si-crystal/`).
- **Numerical code**: prefer `np.einsum` / `opt_einsum.contract` over chained `np.dot`/`np.tensordot`. Be explicit about dtype (`float64` vs `float32`). Don't compare floats with `==`.
- **GPU dispatch**: TensorFlow paths must degrade to CPU on machines without a GPU. Don't import TF at module top level if the module also has to load on CPU-only environments without TF available.
- **Versioning**: `versioneer` manages versions from git tags. Don't edit `kaldo/_version.py` by hand.
- **Avoid**: editing `versioneer.py` or `_version.py`; restructuring directories without a clear reason; changing public API names without a migration note in the docs.

## What is in scope

This is a research library used by external groups for thermal-transport calculations. Changes should keep that scope. Avoid:

- Adding heavyweight new frameworks unless `requirements.txt` already pulls them.
- Breaking the CPU code path in pursuit of a GPU optimization.
- Reformatting unrelated files in the same PR as a logic change.

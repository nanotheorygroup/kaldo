<img src="https://raw.githubusercontent.com/nanotheorygroup/kaldo/main/docs/docsource/_resources/logo.png" width="450">

[//]: # (Badges)
[![CircleCI](https://img.shields.io/circleci/build/github/nanotheorygroup/kaldo/main)](https://app.circleci.com/pipelines/github/nanotheorygroup/kaldo)
[![codecov](https://img.shields.io/codecov/c/gh/nanotheorygroup/kaldo)](https://codecov.io/gh/nanotheorygroup/kaldo)
[![licence](https://img.shields.io/github/license/nanotheorygroup/kaldo)](https://github.com/nanotheorygroup/kaldo/blob/master/LICENSE)
[![documentation](https://img.shields.io/badge/docs-github%20pages-informational)](https://nanotheorygroup.github.io/kaldo/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

**κALDo** (kALDo) is an open-source Python package for computing vibrational, elastic, and thermal transport properties of crystalline, disordered, and amorphous materials from first principles and machine-learned interatomic potentials.

Built on the anharmonic lattice dynamics (ALD) framework, κALDo provides GPU- and CPU-accelerated implementations of:
- **Boltzmann Transport Equation (BTE)** for crystals
- **Quasi-Harmonic Green-Kubo (QHGK)** method for disordered and amorphous systems

The QHGK formalism uniquely extends thermal transport predictions beyond crystals to materials lacking long-range order, including glasses, alloys, and complex nanostructures.

## Key Features

### Transport Methods
| Method | Use Case | Solvers |
|--------|----------|---------|
| **BTE** | Crystalline materials | RTA, Self-consistent iteration, Full matrix inversion, Eigendecomposition |
| **QHGK** | Amorphous/disordered materials | Diffuson, locon, propagon decomposition |

### Force Constant Sources
κALDo interfaces with diverse computational tools:
- **Ab initio codes**: Quantum ESPRESSO, VASP (DFPT)
- **MD packages**: LAMMPS (with USER-PHONON)
- **Machine-learned potentials**: NEP, MACE, MatterSim, Orb, DeepMD (via ASE)
- **External phonon codes**: ShengBTE, phono3py, HiPhive
- **TDEP**: Temperature-dependent effective potentials from MD trajectories

### Physical Corrections
- **Isotopic scattering** via Tamura perturbation theory
- **Non-analytical corrections (NAC)** for polar materials (LO-TO splitting)
- **Finite-size effects** and boundary scattering
- **Anharmonicity quantification** (σ_A score)

### Performance & Scalability
- **GPU acceleration** via TensorFlow (5-10× speedup for N > 50 atoms)
- **Sparse tensor operations** for memory efficiency
- **Multiple storage backends**: formatted text, NumPy, HDF5, memory-only
- **Scales to 10,000+ atom systems** for QHGK calculations

## Quickstart

Run κALDo interactively on Google Colab:

| Tutorial | Description |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nanotheorygroup/kaldo/blob/master/docs/docsource/crystal_presentation.ipynb) | Thermal transport in crystalline silicon (BTE) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nanotheorygroup/kaldo/blob/master/docs/docsource/amorphous_presentation.ipynb) | Thermal transport in amorphous silicon (QHGK) |

## Installation

```bash
pip install kaldo
```

Docker deployment:
```bash
docker pull gbarbalinardo/kaldo:latest
```

## Basic Usage

Example calculating thermal conductivity of SiC using the MatterSim machine-learned potential:

```python
# Import kALDo classes and ASE
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import kaldo.controllers.plotter as plotter
from ase.build import bulk
from ase.optimize import BFGS
from ase.constraints import StrainFilter
from mattersim.forcefield import MatterSimCalculator

# Stage 1: Structure optimization
atoms = bulk('SiC', 'zincblende', a=4.35)
calc = MatterSimCalculator(device='cuda')
atoms.calc = calc

# Optimize lattice parameters and atomic positions
sf = StrainFilter(atoms)
opt = BFGS(sf)
opt.run(fmax=0.001)

# Stage 2: Compute force constants using finite differences
fc = ForceConstants(
    atoms=atoms,
    supercell=[10, 10, 10],
    third_supercell=[5, 5, 5],
    folder='fd_SiC_MatterSim'
)
fc.second.calculate(calc, delta_shift=0.03)
fc.third.calculate(calc, delta_shift=0.03)

# Stage 3: Calculate phonon properties
phonons = Phonons(
    forceconstants=fc,
    kpts=[15, 15, 15],
    temperature=300,
    is_classic=False,
    folder='ALD_SiC_MatterSim'
)

# Plot phonon dispersion
plotter.plot_dispersion(phonons, n_k_points=300)

# Stage 4: Calculate thermal conductivity
cond = Conductivity(phonons=phonons, method='inverse')

# Access results
kappa = cond.conductivity.sum(axis=0)
print(f"Thermal conductivity: {kappa.trace()/3:.1f} W/m/K")
```

## Supported File Formats

| Source | Format String | 2nd-Order Files | 3rd-Order Files |
|--------|---------------|-----------------|-----------------|
| NumPy | `numpy` | `second.npy` | `third.npz` or `third.npy` |
| ESKM | `eskm` | `Dyn.form` | `THIRD` |
| LAMMPS | `lammps` | `Dyn.form` | `THIRD` |
| VASP / ShengBTE | `vasp` | `FORCE_CONSTANTS_2ND` or `FORCE_CONSTANTS` | `FORCE_CONSTANTS_3RD` |
| QE + VASP | `qe-vasp` | `espresso.ifc2` | `FORCE_CONSTANTS_3RD` |
| VASP + d3q | `vasp-d3q` | `FORCE_CONSTANTS_2ND` or `FORCE_CONSTANTS` | `FORCE_CONSTANTS_3RD_D3Q` |
| QE + d3q | `qe-d3q` | `espresso.ifc2` | `FORCE_CONSTANTS_3RD_D3Q` |
| HiPhive | `hiphive` | `model2.fcs` | `model3.fcs` |
| TDEP | `tdep` | `infile.forceconstant` | `infile.forceconstant_thirdorder` |

## Theory Background

For detailed theoretical background on anharmonic lattice dynamics, the Boltzmann Transport Equation, and the Quasi-Harmonic Green-Kubo method, see the [documentation](https://nanotheorygroup.github.io/kaldo/).

## Examples

Detailed examples for various materials and workflows are available in the [`examples`](examples/) folder.

## Citations

If you use κALDo, please cite:

| Reference | When to Cite |
|-----------|--------------|
| [1] | Any work using κALDo |
| [2] | QHGK method for disordered materials |
| [3] | Participation ratio analysis |
| [4] | Finite-size thermal conductivity with BTE |
| [5] | TDEP + path-integral MD workflows |
| [6] | Isotopic scattering and hydrodynamic extrapolation |

### References

[1] G. Barbalinardo, Z. Chen, N.W. Lundgren, D. Donadio, *Efficient anharmonic lattice dynamics calculations of thermal transport in crystalline and disordered solids*, [J. Appl. Phys. **128**, 135104 (2020)](https://aip.scitation.org/doi/10.1063/5.0020443).

[2] L. Isaeva, G. Barbalinardo, D. Donadio, S. Baroni, *Modeling heat transport in crystals and glasses from a unified lattice-dynamical approach*, [Nat. Commun. **10**, 3853 (2019)](https://www.nature.com/articles/s41467-019-11572-4).

[3] N.W. Lundgren, G. Barbalinardo, D. Donadio, *Mode Localization and Suppressed Heat Transport in Amorphous Alloys*, [Phys. Rev. B **103**, 024204 (2021)](https://doi.org/10.1103/PhysRevB.103.024204).

[4] G. Barbalinardo, Z. Chen, H. Dong, Z. Fan, D. Donadio, *Ultrahigh convergent thermal conductivity of carbon nanotubes from comprehensive atomistic modeling*, [Phys. Rev. Lett. **127**, 025902 (2021)](https://doi.org/10.1103/PhysRevLett.127.025902).

[5] D.A. Folkner, Z. Chen, G. Barbalinardo, F. Knoop, D. Donadio, *Elastic moduli and thermal conductivity of quantum materials at finite temperature*, [J. Appl. Phys. **136**, 221101 (2024)](https://pubs.aip.org/aip/jap/article/136/22/221101/3325173).

[6] A. Fiorentino, P. Pegolo, S. Baroni, D. Donadio, *Effects of colored disorder on the heat conductivity of SiGe alloys from first principles*, [Phys. Rev. B **111**, 134205 (2025)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.111.134205).

## Publications Using κALDo

See the [publications page](https://github.com/nanotheorygroup/kaldo/tree/main/docs/publications).

## Documentation

Full API documentation: [nanotheorygroup.github.io/kaldo](https://nanotheorygroup.github.io/kaldo/)

## Contributing

We welcome contributions! Please visit our [discussions page](https://github.com/nanotheorygroup/kaldo/discussions) for questions, feature requests, and workflow sharing.

## Copyright

Copyright (c) 2022-2025, The kALDo Developers

## Acknowledgements

We gratefully acknowledge support by the Investment Software Fellowships (grant No. ACI-1547580-479590) of the NSF Molecular Sciences Software Institute (grant No. ACI-1547580) at Virginia Tech.

<a href="https://molssi.org">
<img src="https://raw.githubusercontent.com/nanotheorygroup/kaldo/main/docs/docsource/_resources/molssi-logo.png" height="120">
<img src="https://raw.githubusercontent.com/nanotheorygroup/kaldo/main/docs/docsource/_resources/nsf_logo.png" height="120">
</a>

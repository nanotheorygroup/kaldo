<img src="https://raw.githubusercontent.com/nanotheorygroup/kaldo/main/docs/docsource/_resources/logo.png" width="450">

[//]: # (Badges)
[![CircleCI](https://img.shields.io/circleci/build/github/nanotheorygroup/kaldo/main)](https://app.circleci.com/pipelines/github/nanotheorygroup/kaldo)
[![codecov](https://img.shields.io/codecov/c/gh/nanotheorygroup/kaldo)](https://codecov.io/gh/nanotheorygroup/kaldo)
[![licence](https://img.shields.io/github/license/nanotheorygroup/kaldo)](https://github.com/nanotheorygroup/kaldo/blob/master/LICENSE)
[![documentation](https://img.shields.io/badge/docs-github%20pages-informational)](https://nanotheorygroup.github.io/kaldo/)

## Introduction

kALDo is a modern Python-based software that implements both the Boltzmann Transport equation (BTE) and the Quasi-Harmonic Green Kubo (QHGK) method, which runs on GPUs and CPUs using Tensorflow.
More details can be found on the kALDo website [here](https://nanotheorygroup.github.io/kaldo/).

## Quickstart
You can run kALDo on Google Colab:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nanotheorygroup/kaldo/blob/master/docs/docsource/crystal_presentation.ipynb) Thermal transport calculation for the silicon crystal using the BTE.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nanotheorygroup/kaldo/blob/master/docs/docsource/amorphous_presentation.ipynb) Thermal transport calculation for the silicon amorphous using the QHGK method.

## Software features
Below we illustrate the main features of the code.
- **Forcefields**. Using the Atomic Simulation Environment, kALDo can calculate the interatomic force constants using several _ab-initio_ and molecular dynamics codes. A native LAMMPS interface is also available in the USER-PHONON package. Finally, through a seamless integration with the Hiphive package, the IFC calculation can take advantage of compressing sensing machine learning algorithms.
- **CPUs** and **GPUs**. Multithread implementation on CPUs and GPUs. The algorithms are implemented using linear algebra
operations on tensors, to take advantage of multithreading on GPU and CPU using Numpy, Tensorflow and optimized tensor libraries.
- **Scalable**. In a system of N atoms and N_k k points. kALDo uses (3N)^2 floating point numbers to save the state of the system when using QHGK, (Nk x 3N)^2 for the full solution of the BTE and Nk x 3N^2 when using BTE-RTA.
- **Performance**. The bottleneck, and slow part of ALD simulations is the calculation of the phonons lifetimes and the scattering matrix. This step requires projecting the interatomic potential on 3N phonons modes and the algorithm scales like (Nk x 3N)^3, because of the 3 projections on phonons modes. In kALDo such algorithm is implemented as 2 Nk x 3N tensor multiplications of size Nk x (3N)^2 for BTE calculations while 3N^2 for QHGK.

- **Open-Source**. Free for the community to use and contribute with edits and suggestion. It is designed on modern software best practices, and we hope to provide a development platform to implement new theory and methods.

## Citations

| Reference             | How to cite                       |
| --------------------- | --------------------------------- |
| [1]                   | for any work that used `kALDo`    |
| [2]                   | fundamental theory and implementations on Quasi-Harmonic Green Kubo (QHGK) |
| [3]                   | participation ratio               |
| [4]                   | finite size thermal conductivity calculations with ALD-BTE|
| [5]                   | path-integral MD (from GPUMD) + TDEP + kALDo work flow and elastic moduli calculations|
| [6]                   | isotopic scattering based on Tamura formula and hydrodynamic extrapolation|

## References

[1] Giuseppe Barbalinardo, Zekun Chen, Nicholas W. Lundgren, Davide Donadio, [Efficient anharmonic lattice dynamics calculations of thermal transport in crystalline and disordered solids](https://aip.scitation.org/doi/10.1063/5.0020443), J. Appl. Phys. **128**, 135104 (2020).

[2] L Isaeva, G Barbalinardo, D Donadio, S Baroni, [Modeling heat transport in crystals and glasses from a unified lattice-dynamical approach](https://www.nature.com/articles/s41467-019-11572-4), Nat. Commun ***10***: 3853 (2019)

[3] Nicholas W. Lundgren, Giuseppe Barbalinardo, Davide Donadio, [Mode Localization and Suppressed Heat Transport in Amorphous Alloys](https://doi.org/10.1103/PhysRevB.103.024204), Phys. Rev. B **103**, 024204 (2021).

[4] Giuseppe Barbalinardo, Zekun Chen, Haikuan Dong, Zheyong Fan, Davide Donadio,
[Ultrahigh convergent thermal conductivity of carbon nanotubes from comprehensive atomistic modeling](https://doi.org/10.1103/PhysRevLett.127.025902), Phys. Rev. Lett. **127**, 025902 (2021).

[5] Dylan A. Folkner, Zekun Chen, Giuseppe Barbalinardo, Florian Knoop, Davide Donadio, [Elastic moduli and thermal conductivity of quantum materials at finite temperature](https://pubs.aip.org/aip/jap/article/136/22/221101/3325173), J. Appl. Phys. **136**,  221101 (2024).

[6] Alfredo Fiorentino,  Paolo Pegolo,  Stefano Baroni,  Davide Donadio, [Effects of colored disorder on the heat conductivity of SiGe alloys from first principles](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.111.134205), Phys. Rev. B. ***111***, 134205 (2025).


## Publications using kALDo

Please visit the [publications page](https://github.com/nanotheorygroup/kaldo/tree/main/docs/publications).

## Copyright

Copyright (c) 2022, Giuseppe Barbalinardo, Zekun Chen, Dylan Folkner, Nicholas W. Lundgren, Davide Donadio

## Acknowledgements

We gratefully acknowledge support by the Investment Software Fellowships (grant No. ACI-1547580-479590) of the NSF Molecular Sciences Software Institute (grant No. ACI-1547580) at Virginia Tech. 

<a href="https://molssi.org">
<img src="https://raw.githubusercontent.com/nanotheorygroup/kaldo/main/docs/docsource/_resources/molssi-logo.png" height="120">  
<img src="https://raw.githubusercontent.com/nanotheorygroup/kaldo/main/docs/docsource/_resources/nsf_logo.png" height="120">    
</a>
 
MolSSI builds open source software and data which serves the computational molecular science community. [Explore MolSSI’s software infrastructure projects.](https://molssi.org/software/)

<img src="docs/docsource/_resources/logo.png" width="450">

[//]: # (Badges)
[![CircleCI](https://img.shields.io/circleci/build/github/nanotheorygroup/kaldo/main)](https://app.circleci.com/pipelines/github/nanotheorygroup/kaldo)
[![codecov](https://img.shields.io/codecov/c/gh/nanotheorygroup/kaldo)](https://codecov.io/gh/nanotheorygroup/kaldo)
[![licence](https://img.shields.io/github/license/nanotheorygroup/kaldo)](https://github.com/nanotheorygroup/kaldo/blob/master/LICENSE)
[![documentation](https://img.shields.io/badge/docs-github%20pages-informational)](https://nanotheorygroup.github.io/kaldo/)

## Introduction

kALDo is a modern Python-based software that implements both the Boltzmann Transport equation (BTE) and the Quasi-Harmonic Green Kubo (QHGK) method, which runs on GPUs and CPUs using Tensorflow.
More details can be found on the kALDo website [here](https://nanotheorygroup.github.io/kaldo/).

## Quickstart
You can run kALDO on Google Colab:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nanotheorygroup/kaldo/blob/master/docs/docsource/crystal_presentation.ipynb) Thermal transport calculation for the silicon crystal using the BTE.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nanotheorygroup/kaldo/blob/master/docs/docsource/amorphous_presentation.ipynb) Thermal transport calculation for the silicon amorphous using the QHGK method.

## Software features
Below we illustrate the main features of the code.
- **Forcefields**. Using the Atomic Simulation Environment, kALDo can calculate the interatomic force constants using several _ab-initio_ and molecular dynamics codes. A native LAMMPS interface is also available in the USER-PHONON package. Finally, through a seamless integration with the Hiphive package, the IFC calculation can take advantage of compressingsensing machine learning algorithms.
- **CPUs** and **GPUs**. Multithread implementation on CPUs and GPUs. The algorithms are implemented using linear algebra
operations on tensors, to take advantage of multithreading on GPU and CPU using Numpy, Tensorflow and optimized tensor libraries.
- **Scalable**. In a system of N atoms and N_k k points. kALDo uses (3N)^2 floating point numbers to save the state of the system when using QHGK, (Nk x 3N)^2 for the full solution of the BTE and Nk x 3N^2 when using BTE-RTA.
- **Performance**. The bottleneck, and slow part of ALD simulations is the calculation of the phonons lifetimes and the scattering matrix. This step requires projecting the interatomic potential on 3N phonons modes and the algorithm scales like (Nk x 3N)^3, because of the 3 projections on phonons modes. In kALDo such algorithm is implemented as 2 Nk x 3N tensor multiplications of size Nk x (3N)^2 for BTE calculations while 3N^2 for QHGK.

- **Open-Source**. Free for the community to use and contribute with edits and suggestion. It is designed on modern software best practices, and we hope to provide a development platform to implement new theory and methods.

## How to cite
Barbalinardo, G.; Chen, Z.; Lundgren, N. W.; Donadio, D. Efficient Anharmonic Lattice Dynamics Calculations of Thermal Transport in Crystalline and Disordered Solids. J Appl Phys 2020, 128 (13), 135104–135112. https://doi.org/10.1063/5.0020443

also available open access on ArXiv: https://arxiv.org/abs/2009.01967

## Publications using kALDo

Please visit the [publications page](https://github.com/nanotheorygroup/kaldo/tree/main/docs/publications).

## Copyright

Copyright (c) 2022, Giuseppe Barbalinardo, Zekun Chen, Dylan Folkner, Nicholas W. Lundgren, Davide Donadio

## Acknowledgements

We gratefully acknowledge support by the Investment Software Fellowships (grant No. ACI-1547580-479590) of the NSF Molecular Sciences Software Institute (grant No. ACI-1547580) at Virginia Tech. 

<a href="https://molssi.org">
<img src="docs/docsource/_resources/molssi-logo.png" height="120">  
<img src="docs/docsource/_resources/nsf_logo.png" height="120">    
</a>
 
MolSSI builds open source software and data which serves the computational molecular science community. [Explore MolSSI’s software infrastructure projects.](https://molssi.org/software-projects/)

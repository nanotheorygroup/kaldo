<img src="docs/docsource/_resources/logo.png" width="450">

[//]: # (Badges)
[![CircleCI](https://img.shields.io/circleci/build/github/nanotheorygroup/kaldo/master)](https://app.circleci.com/pipelines/github/nanotheorygroup/kaldo)
[![codecov](https://img.shields.io/codecov/c/gh/nanotheorygroup/kaldo)](https://codecov.io/gh/nanotheorygroup/kaldo)
[![licence](https://img.shields.io/github/license/nanotheorygroup/kaldo)](https://github.com/nanotheorygroup/kaldo/blob/master/LICENSE)
[![documentation](https://img.shields.io/badge/docs-github%20pages-informational)](https://nanotheorygroup.github.io/kaldo/)

kALDo is a modern Python-based software that implements both the Boltzmann Transport equation (BTE) and the Quasi-Harmonic Green Kubo (QHGK) method, which runs on GPUs and CPUs using Tensorflow.
More details can be found on the kALDo website [here](https://nanotheorygroup.github.io/kaldo/).

You can run kALDO on Google Colab as a playground:
- Thermal transport calculation for the silicon crystal using the BTE: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nanotheorygroup/kaldo/blob/master/docs/docsource/crystal_presentation.ipynb) 
- Thermal transport calculation for the silicon amorphous using the QHGK method: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nanotheorygroup/kaldo/blob/master/docs/docsource/amorphous_presentation.ipynb)

Below we illustrate the main features of the code.

<img src="docs/docsource/_resources/features.png" width="650">

## Reference Paper
Barbalinardo, G.; Chen, Z.; Lundgren, N. W.; Donadio, D. Efficient Anharmonic Lattice Dynamics Calculations of Thermal Transport in Crystalline and Disordered Solids. J Appl Phys 2020, 128 (13), 135104–135112. https://doi.org/10.1063/5.0020443
also available open access on ArXiv: https://arxiv.org/abs/2009.01967

## Copyright

Copyright (c) 2020, Giuseppe Barbalinardo, Zekun Chen, Nicholas W. Lundgren, Davide Donadio

## Acknowledgements

We gratefully acknowledge support by the Investment Software Fellowships (grant No. ACI-1547580-479590) of the NSF Molecular Sciences Software Institute (grant No. ACI-1547580) at Virginia Tech. 
<a href="https://molssi.org">
<img src="docs/docsource/_resources/acknowledgement.png" width="450">    
</a>
 
MolSSI builds open source software and data which serves the computational molecular science community. [Explore MolSSI’s software infrastructure projects.](https://molssi.org/software-projects/)

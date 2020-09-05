<img src="docs/docsource/_resources/logo.png" width="450">

[//]: # (Badges)
[![CircleCI](https://img.shields.io/circleci/build/github/nanotheorygroup/kaldo/master)](https://app.circleci.com/pipelines/github/nanotheorygroup/kaldo)
[![codecov](https://img.shields.io/codecov/c/github/nanotheorygroup/kaldo)](https://github.com/nanotheorygroup/kaldo/blob/master/LICENSE)
[![licence](https://img.shields.io/github/license/nanotheorygroup/kaldo)](https://img.shields.io/github/license/nanotheorygroup/kaldo)
[![documentation](https://img.shields.io/badge/docs-github%20pages-informational)](https://nanotheorygroup.github.io/kaldo/)

kALDo is a modern Python-based software that implements both the Boltzmann Transport equation and the Quasi-Harmonic Green Kubo method, which runs on GPUs and CPUs using Tensorflow.
More details can be found on the kALDo website [here](https://nanotheorygroup.github.io/kaldo/).

You can run kALDO on Google Colab as a playground:
- Silicon Crystal using the BTE, [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nanotheorygroup/kaldo/blob/master/docs/docsource/crystal_presentation.ipynb) 
- Silicon Amorphous using QHGK, [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nanotheorygroup/kaldo/blob/master/docs/docsource/amorphous_presentation.ipynb)

kALDo features real space QHGK calculations and three different solvers of the linearized BTE: direct inversion, self-consistent cycle and RTA.

Below we illustrate the main features of the code.

<img src="docs/docsource/_resources/features.png" width="650">

## Copyright

Copyright (c) 2020, Giuseppe Barbalinardo, Zekun Chen, Nicholas W. Lundgren, Davide Donadio

## Acknowledgements

We gratefully acknowledge support by the Investment Software Fellowships (grant No. ACI-1547580-479590) of the NSF Molecular Sciences Software Institute (grant No. ACI-1547580) at Virginia Tech. 
<a href="https://molssi.org">
<img src="docs/docsource/_resources/acknowledgement.png" width="450">    
</a>
 
MolSSI builds open source software and data which serves the computational molecular science community. [Explore MolSSI’s software infrastructure projects.](https://molssi.org/software-projects/)

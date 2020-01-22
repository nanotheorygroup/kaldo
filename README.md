Ballistico: Anharmonic Lattice Dynamics
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/gbarbalinardo/ballistico.svg?token=EFWyhyp9aQcQnteZBpEr&branch=master)](https://travis-ci.com/gbarbalinardo/ballistico)
[![codecov](https://codecov.io/gh/gbarbalinardo/ballistico/branch/master/graphs/badge.svg?token=tiC2xj2OQG)](https://codecov.io/gh/gbarbalinardo/ballistico/branch/master)



### Install
We recommend creating a new environment with Python 3.7.
```bash
conda create -n ballistico python=3.7
```
Alternatively you can use Python 3.6. The Tensorflow module doesn't seem to be compatible with Python 3.8, yet.

Ballistico installation can be done using `pip`
```bash
pip install ballistico
```


### Documentation

A draft of the documentation can be found [here](http://169.237.38.203/ballistico/).

### Examples

Examples are currently written to run on Google Colab, and can be found in the examples folder.

### Other configurations
#### LAMMPS setup

Compile LAMMPS as a lib
```bash
cd path/to/lammps/src
make yes-manybody
make yes-molecule
make mpi mode=shlib
make install-python
```

#### Quantum Espresso setup

Set the environment variable:
```bash
export ASE_ESPRESSO_COMMAND="mpirun -np 4 /path/to/pw.x -in PREFIX.pwi > PREFIX.pwo"
```

### Copyright

Copyright (c) 2020, Your name (or your organization/company/team)


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.



Ballistico: Anharmonic Lattice Dynamics
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/gbarbalinardo/ballistico.svg?token=EFWyhyp9aQcQnteZBpEr&branch=master)](https://travis-ci.com/gbarbalinardo/ballistico)
[![codecov](https://codecov.io/gh/gbarbalinardo/ballistico/branch/master/graphs/badge.svg?token=tiC2xj2OQG)](https://codecov.io/gh/gbarbalinardo/ballistico/branch/master)



### Install

Ballistico installation can be done using `pip`
```bash
pip install ballistico
```


### Documentation

A draft of the documentation can be found [here](http://169.237.38.203/ballistico/).

### Examples

Examples are currently written to run on Google Colab, and can be found in [this repo](https://github.com/gbarbalinardo/ballistico-examples).

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

Copyright (c) 2019, Giuseppe Barbalinardo, Zekun Chen, Davide Donadio


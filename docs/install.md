## Quick Start



We recommend creating a new environment with Python 3.7.
```bash
conda create -n kaldo python=3.7
```
Alternatively you can use Python 3.6. The Tensorflow module doesn't seem to be compatible with Python 3.8, yet.

kaldo installation can be done using `pip`
```bash
pip install kaldo
```

### Code Architecture

<img src="_resources/class_diagram.png" width="650">

### Interfacing with Other Codes

Work in progress.
 
#### LAMMPS setup

Compile LAMMPS as a lib
```bash
cd path/to/lammps/src
make yes-manybody
make yes-molecule
make mpi mode=shlib
make install-python
```

### Quantum Espresso setup

Set the environment variable:
```bash
export ASE_ESPRESSO_COMMAND="mpirun -np 4 /path/to/pw.x -in PREFIX.pwi > PREFIX.pwo"
```


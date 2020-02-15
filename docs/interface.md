
# Interfacing with other codes
## LAMMPS setup

Compile LAMMPS as a lib
```bash
cd path/to/lammps/src
make yes-manybody
make yes-molecule
make mpi mode=shlib
make install-python
```

## Quantum Espresso setup

Set the environment variable:
```bash
export ASE_ESPRESSO_COMMAND="mpirun -np 4 /path/to/pw.x -in PREFIX.pwi > PREFIX.pwo"
```

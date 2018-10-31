![alt text](assets/ballistico.png "Ballistico Logo")

# Introduction
Ballistico is a Python app which allows to calculate phonons dispersion relations and density of states using finite difference.

# Installation

1.  Install LAMMPS as a library with the `manybody` option
```
make yes-manybody
make serial mode=shlib
```
When lammps is compiled as a library the `liblammps.so` file is created in the src folder. Add that to your `PYTHON_PATH` using
```
export PYTHONPATH=[PATH_TO_LAMMPS]:$PYTHONPATH
```
If you want to make this change persistent 

2. Install Python 3 and the following dependencies: `scipy`, `numpy`, `matplotlib`, `pandas`

Dependencies can be installed with conda (`sudo` permission may be needed)
```
conda install [PACKAGE-NAME]
```
Install the packages `spglib` and `ase`. They don't belong to the conda main repo. Use `conda-forge`
```
conda install -c conda-forge [PACKAGE-NAME]
```
FInally, install `seekpath` using `pip`
```
pip install seekpath
```



# Usage
## Define the system: `MolecularSystem` object
This is one of the main classes used in Ballistico, 
it allows the specification of the following parameters
* geometry
* forcefield
* replicas 
* temperature
* optimization 

### Define a `geometry`
You will need to use an extended xyz file to create a geometry using
```python
geometry = ath.from_filename('examples/si-bulk.xyz')
```
examples of geometries are in the `examples` folder 

### Structure optimization
The structure can be optionally optimize using one of the `scipy` minimization methods (Nelder-Mead, Powell, CG, BFGS, Newton-CG, L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr, dogleg, trust-ncg, trust-exact, trust-krylov)

### Define a `forcefield`
forcefield are define through LAMMPS commands using 
```python
still_weber_forcefield = ["pair_style sw", "pair_coeff * * forcefields/Si.sw Si"]
```
another example is
```python
tersoff_forcefield = ["pair_style tersoff", "pair_coeff * * forcefields/si.tersoff Si"]
```

examples of forcefields are in the `forcefields` folder 

### Define the number of replicas
You need to specify the number of replicas in the direct space using 
```python
replicas = np.array ([3, 3, 3])
```

### Create the `MolecularSystem`

Once everything is set, create the `MolecularSystem` using

```python 
system = MolecularSystem (geometry, replicas=replicas, temperature=300., optimize=True, lammps_cmd=still_weber_forcefield)
```

## Phonons calculations: `MolecularSystemPhonons` object

To calculate phononic properties of the system use the `MolecularSystemPhonons` object. Here you can specify the `k_mesh` to use to calculate the density of states. Here's an example of Usage
```python
k_mesh = np.array([5, 5, 5])
phonons = MolecularSystemPhonons(system, k_mesh)
phonons.energy_k_plot (method='auto')
```
where the method specify for the `energy_k_plot` can be `auto`, which tries to automatically define a path on the Brillouin zone, or one of the following:
* cubic
* fcc
* bcc
* hexagonal
* tetragonal
* orthorhombic

# Full example
```python
import numpy as np
import ballistico.atoms_helper as ath
from ballistico.MolecularSystemPhonons import MolecularSystemPhonons
from ballistico.MolecularSystemPhonons import MolecularSystem

if __name__ == "__main__":
	geometry = ath.from_filename('examples/si-bulk.xyz')
	still_weber_forcefield = ["pair_style sw", "pair_coeff * * forcefields/Si.sw Si"]
	replicas = np.array ([3, 3, 3])
	system = MolecularSystem (geometry, replicas=replicas, temperature=300., optimize=True, lammps_cmd=still_weber_forcefield)
	k_mesh = np.array([5, 5, 5])
	phonons = MolecularSystemPhonons(system, k_mesh)
	phonons.energy_k_plot (method='fcc')
```

# Output
A working folder with a name like `Si_a40.0_r333_T300` is created with permanent file used to speed up the calculations. The folder needs to be deleted when the system changes.

# Indices
Ballistico has flexible indices, array and matrices can easily be linearized through `numpy` built in functions. 
Any one-body tensor can be indexed in the following way
```
k_axes, k_coord_id, replica_axis, replica_coord_id, atom_axes, atom_coord_id
```
If a tensor is more than one body, the indices structure is a repetition of the previous line for each body.

Indices with dimension 1 can be omitted
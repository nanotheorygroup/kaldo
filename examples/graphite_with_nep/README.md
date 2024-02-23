# Requirements
In addition to kALDo and its base requirements, this will require the installation of pyNEP. pyNEP can be installed using:
```bash
pip install git+https://github.com/bigd4/PyNEP.git
```

# Calculation
To calculate the lattice thermal conductivity using a pyNEP input calculator with ASE, run the python script using the command:
```bash
python 1_graphite_NEP.py
```
The calculation can take approximately several hours on a machine with a GPU or 32 cores. Structure is initially read from **graph.cif**. NEP file is read from **forcefields/C_2022_NEP3.txt**. The output of the command will be the lattice thermal conductivity by direct inversion and by RTA as well as the phonon properties in either **CL_properties.dat** or **QM_properties.dat**, depending on whether the calculation is classical or quantum. Calculation data is saved in the folder **ALD/**, force constants in **fd/**, and plot inpus in **plots/**. 

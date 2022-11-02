# Running this Example

This example is made to repeat the results found in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.024204 by calculating phononic 
and conductivity information.
First a set of data is generated for a 1728 atom system, and then an interpolation fits the
bandwidths of a larger system to match the trendline of the smaller system. A plotter is 
provided to look at some of the output data.

## Requirements

You will need a python module for lammps decribed here: https://lammps.sandia.gov/doc/Python_install.html
Additionally, to compute the bandwidths of the 1728 atom systems can take in excess of eight hours 
using eight threads on a 3.6 GHz machine. It's reccomended that you compile your lammps library to 
support parallel processing, else these calculations will be infeasibly expensive.

## Procedure

First build aSiGe structures based on stochastic atom replacement to ge concentrations listed in 
the file 1_generate_structures.py. Here, we've selected two concentrations of germanium, 10% and 20%.
The structures will be returned to mechanical equilibrium using an LFBGS linesearch which takes about
~2s and then the force constants will be calculated.

Secondly, phononic information will be recovered. Explicit phonon bandwidth calculations will take the bulk
of the computation time for this example. 


## Warning

While computing force constants using LAMMPS with a triclinic (non-orthogonal) cell, it follows a strict right-handed cell convention. More information about the cell conventioncan be found [here](https://docs.lammps.org/Howto_triclinic.html). The unit cell of the structure used in this example has been made based on the right-handed cell convention. Please be aware of this rule if lammps inputs were prepared from scratch. Alternatively, it is welcome to refer our [carbon nanotube example](https://github.com/nanotheorygroup/kaldo/tree/main/examples/carbon_nanotube_Tersoff_LAMMPS) where we imposed this convention and prepared lammps input files via a handy python script.

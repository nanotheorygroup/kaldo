# Running this Example

This example is made to repeat the results found in https://arxiv.org/abs/2011.08318 by calculating phononic 
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



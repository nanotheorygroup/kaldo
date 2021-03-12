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
of the computation time for this example. Other information is output and saved in numpy's .npy file format
to be used later for plotting. Outputs include frequencies, lifetimes, participation ratios, and diffusivity.
To recreate the velocity plot found in the paper, do not use phonons.velocity as this outputs the diagonal terms
from the generalized velocity in x y and z, which would all be zero for an amorphous system. To extract the 
full matrix requires using an undocumented process creating a HarmonicwithQ object and loading the _sij_<x/y/z>
property (Syntax provided below)

The third script runs the interpolation used which generates a set of lifetimes for a larger system. The process
is not intimately tied to kaldo, but provided to promote open-source chemistry and data sharing.

The fourth script can be run independently of the third script and plots the quantities calculated in 
the second script. Note that kALDo does contain plotting scripts demo'd in some of the other example scripts,
though for the sake of replacating the PRB data this example has its own, more specialized, plotting suite.

### Syntax for returning the full generalized velocity matrix
```
harmonic = HarmonicWithQ(q_point=np.array([0,0,0]),
                    second=phonons.forceconstants.second,
                    distance_threshold=phonons.forceconstants.distance_thre$
                    folder=phonons.folder,
                    storage=phonons.storage,
                    is_nw=phonons.is_nw)
vx = harmonic._sij_x
vy = harmonic._sij_y
vz = harmonic._sij_z
full_generalized_velocity_tensor = np.stack((vx, vy, vz))
```
   





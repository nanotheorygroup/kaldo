# Overview `silicon_bulk_ShengBTE_QE`
Example silicon_bulk_ShengBTE_QE shows off the `is_unfolding` flag using supercells of bulk silicon (2 atoms per cell).
Harmonic force constants are calculated directly with [quantum espress](https://www.quantum-espresso.org/) and
[thirdorder](https://www.shengbte.org/) extracts third order force constants from reference quantum espresso force 
calculations on a supercell.

## Comment on ShengBTE's thirdorder package
When using the [thirdorder](https://www.shengbte.org/announcements/thirdorderpyv110released) it occasionally performs 
silent coordinate transformations. Please ensure that there are no negative coordinates in your unit cell basis, or 
alternatively, triple check that the coordinates for the first unit cell in the supercell file.
(default: "BASE.supercell_template.in) exactly match those for any configuration files used for harmonic quantities.

# Script Explanation

## Python

### 1_run_kaldo.py

A script for running simulations using shengbte data.

    1. Command line arguments specify which model to use in the calculations to follow

    2. A ForceConstants object is created with the ShengBTE and QE data

    3. A Phonons object is initiated with the "is_unfolding" flag controlled by the command line arguments

    4. The dispersion is calculated along the path specified.

    5. The conductivity is calculated as long as the "harmonic" argument is not provided on the command line.

### 3_plot_dispersions.py

While it is possible to load kALDo objects everytime you'd like to plot a property, this script presents a handy 
workflow when comparing two simulations. It simply loads the numpy arrays of ALD data and plots them together.
An important note is that this plots along the path set in 1_run_kaldo.py, and you need to re-run kALDo to set a new
path.

    1. Loads dispersion data directly from files created by kALDo

### 4_plot_anharmonic_data.py

This accomplishes the same thing as the last script with data calculated on k-pt grids, rather than paths.
It should be flexible enough to plot just about any phononic property against another. 
Plots generated are found in "kaldo-outputs"

    1. Gathers the data from chosen directories and makes a figure with them.

    2. Style is controlled by plotting dictionaries in the tools/plotting_dictionaries.py script.

## Bash

### 0_batch_run.sh

Using a combination of environment variables and parameters in the script header this file
executes a number of either ALD simulations. To run it, edit the internal parameters to control whether you'd like to 
calculate only harmonic quantities, or the full conductivity calculation. 
In your CLI run the following with/without the -p flag if you'd like parallel/serial calculations. 

`./0_batch_run.sh -p`

### 2_plot.sh

Pulling the same environment variables as 0_batch_run.sh, this plots simulation data for multiple systems.
In your CLI run the following with/without the -p flag if you'd like parallel/serial calculations.

`./2_plot.sh -p`


## Auxillary

### tools/plotting_dictionaries.py

This contains dictionaries to increase flexibility of plotting tool without adding
to much bulk to the plotting script itself.

### tools/si-qe-sheng-dat.tgz

Compressed silicon force constants.

### tools/vars.sh

Please source this to load in environment variables required for the example scripts. All variables begin with the
prefix "kaldo_" so they should not interfere your environment. To undo this set the environment variables to 
empty strings, or close and re-open your shell.

`source tools/var.sh`

## Subdirectory Organization

The "force_inputs" directory contains the QE 2nd order force constants, ShengBTE-thirdorder force constants and 
optimized atom coordinates in a VASP-formatted POSCAR file.

kALDo outputs data in the directory specified by the "kaldo_ald" environment variable.
Default: 'kaldo_raw_data'

kALDo outputs logging info and plots to the directory specified by the "kaldo_outputs" environment variable
Default: 'kaldo_outputs'

Dispersion data will be found in <kaldo_ald>/<system>/dispersion

The rest of the phonon information can be found in <kaldo_ald>/<system>/<k>_<k>_<k> where k is number of k points
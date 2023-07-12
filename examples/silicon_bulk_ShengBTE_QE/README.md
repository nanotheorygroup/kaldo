# `silicon_bulk_ShengBTE_QE`

## Overview
Example silicon_bulk_ShengBTE_QE shows off the `is_unfolding` flag using supercells of bulk silicon (2 atoms per cell).
Harmonic force constants are calculated directly with [quantum espress](https://www.quantum-espresso.org/) and
[thirdorder](https://www.shengbte.org/) extracts third order force constants from reference quantum espresso force 
calculations on a supercell.

## Script Explanation

### Workflow scripts

- 1_run_kaldo.py

    1. Command line arguments specify which model to use in the calculations to follow

    2. A ForceConstants object is created with the ShengBTE and QE data

    3. A Phonons object is initiated with the "is_unfolding" flag controlled by the command line arguments

    4. The dispersion is calculated along the path specified.

    5. The conductivity is calculated as long as the "disp" argument is not provided on the command line.

- 2_plot_dispersions.py
  Using our handy in house plot_dispersion function, this script compares data
  across models.

    1. Loads dispersion data in directly from output files created by kALDo

- 3_plot_anharmonic_data.py
  While it is possible to load kALDo objects everytime you'd like to plot
  a property, this script just loads thing into memory as numpy arrays and
  creates figures with them. It should be flexible enough to plot just about
  any phononic property against another.
  Plots generated are found in "kaldo-outputs"

    1. Set up parameters to match those give to 1_run_kaldo.py

    2. Gathers the data from chosen directories and makes a figure with them. 

### Auxillary scripts

- 0_batch_run.sh
  If you'd like to run everything just like we did, bash this file.

    1. First, it decompresses the silicon data

    2. Then it makes a directory to collect kaldo output for easier debugging

    3. Next it runs in parallel 8 kALDo simulations for silicon
       a. To run this in serial, just remove the "& disown" from line 14

- plotting_dictionaries.py

    1. Contains dictionaries to increase flexibility of plotting tool without adding
      to much bulk to the plotting script itself.

- si-qe-sheng-dat.tgz

    1. This contains the qe+thirdorder data compressed in a tar ball

## Subdirectory Organization

- Directories with NxNxN contain the QE force constants, ShengBTE-thirdorder force constants and optimized atom
  coordinates in a VASP-style POSCAR file

- kALDo outputs data in the directory specified by the "prefix" variable in "1_run_kaldo.py" set to 'data' by default

- Dispersion data will be found in <prefix>/<system>/dispersion

- The rest of the phonon information can be found in <prefix>/<system>/<k>_<k>_<k> where k is number of k points 

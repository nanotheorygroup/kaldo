# Overview of `inse_monolayer_QE_hiphive`
This example demonstrates a workflow using harmonic forces constants calculated from first principles in 
[quantum espress](https://www.quantum-espresso.org/) at a moderately high level of theory (GGA with Grimmes-D3 vdW 
corrections using ) combined with [HiPhive](https://hiphive.materialsmodeling.org/) anharmonic third-order FCs. The 
high quality second-order FCs give us better resolution of vibration in shear and bending modes that are harder for
HiPhive to capture in comparison to the comparatively strong modes that move in the plane of the quasi-2D monolayer.

# Python

## 1_run_kaldo.py

A script for running simulations using shengbte data.

    1. A ForceConstants object is created with the HiPhive data

    2. The harmonic FCs are replaced by creating a SecondOrder object and passing it to the previous instance.

    3. A Phonons object is initiated with the combination of ForceConstants.

    4. The dispersion is calculated along the path specified.

    5. The conductivity is calculated as long as the "harmonic" argument is not provided on the command line.

# Quantum Espresso

## input.*

These are the inputs we used to generate the "espresso.ifc2" file. You can however, modify these for a new target 
system and should be able to continue the kALDo workflow without problem. ASE can convert input.scf into a coordinate
file if you wish to visualize the geometry.

## espresso.ifc2

These are real space force constants matdyn.x can use to generate phonon data.

## qe_timing.dat

Information about how long the ph.x calculation took, and the CPU that ran it.

# Auxillary

## tools/var.sh

Please source this to load in environment variables required for the example scripts. All variables begin with the
prefix "kaldo_" so they should not interfere your environment. To undo this set the environment variables to 
empty strings, or close and re-open your shell.

## tools/qe_work.tgz

Compressed quantum espresso input scripts and with some reference forces.

# Subdirectory Organization

The "hiphive_fcs" directory contains the HiPhive FCs, and "qe_fcs" contains the Quantum Espresso FCs.

kALDo outputs data in the directory specified by the "kaldo_ald" environment variable.
Default: 'kaldo_raw_data'

kALDo outputs logging info and plots to the directory specified by the "kaldo_outputs" environment variable
Default: 'kaldo_outputs'

Dispersion data will be found in <kaldo_ald>/dispersion

The rest of the phonon information can be found in <kaldo_ald>/<k>_<k>_<k> where k is number of k points
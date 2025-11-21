#!/bin/bash


# First install mattersim and cmake in current enviroment
pip install mattersim
pip install cmake

# Clone lammps from advancesoftcorp
git clone https://github.com/advancesoftcorp/lammps.git

# Compile lammps with cmake
# Make sure to subsitute your python path that contains mattersim
cd lammps
mkdir build
cd build
cmake -D PKG_MANYBODY=yes -D PKG_KSPACE=yes -D PKG_PHONON=yes  -D PKG_ML-GNNP=on -D PKG_PYTHON=on -D PYTHON_EXECUTABLE=/path/to/bin/python ../cmake
cmake --build . -j 8

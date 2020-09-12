# `silicon_bulk_Tersoff_ASE_LAMMPS_hiPhive`

Example silicon_bulk_Tersoff_LAMMPS_hiPhive illustrates how to perform thermal transport simulation for a bulk silicon (2 atoms per cell) system using [hiPhive](https://hiphive.materialsmodeling.org/) to extract 
force constants from [ASE and LAMMPS](https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/lammpslib.html) reference force calculations.


External files required: 
		        1). Si.tersoff

 - 1_Si_hiPhive_generate_fcs.py proceeds as follows:
	
    1.  Generate 100 reference structures and calculate forces with standard rattle scheme.

    2. Set up StructureContainer with 2nd and 3rd order cutoffs of 4.0 angstrom.

    3. Develop force constant potential by fitting reference structures and calculated force with least square methods.

    4. Extract force constants from force constant potential 

    5. Set up input files for kALDo with hiPhive format.


- 2_Si_hiPhive_harmonic_properties.py proceeds as follows:

    1. Set up force constant object by loading  in force constants computed from hiPhive.

    2. Set up phonon object (5x5x5 k-point mesh) and perform quantum simulation at 300K.

    3. Compute and visualize harmonic properties (i.e. dispersion relation, group velocity and DOS). 


 - 3_Si_hiPhive_thermal_conductivity.py proceeds as follows:
 
    1. Set up force constant object by loading  in force constants computed from hiPhive.

    2. Set up phonon object (5x5x5 k-point mesh) and perform quantum simulation at 300K.

    3. Set up Conductivity object and compute thermal conductivity with RTA, sc and inverse methods.
					
    4. Visualize cumulative conductivity from  RTA and inversion as functions of frequency. 
		
- To calculate 2<sup>nd</sup> and 3<sup>rd</sup> order force constants with LAMMPS and ASE, after downloading [LAMMPS](https://lammps.sandia.gov/),compile LAMMPS with shlib mode:
```bash
cd path/to/lammps/src
make yes-manybody
make yes-molecule
make mpi mode=shlib
```
- After properly installed κALDo, run the following line in command window to link LAMMPS with Python and ASE:
```bash
cd path/to/lammps/src
make install-python				
```  
- To extract 2<sup>nd</sup> and 3<sup>rd</sup> force constants from force constant potentials, navigate to this directory and execute:
```python			
python 1_Si_hiPhive_generate_fcs.py
```

- To perform thermal transport  after computing force constants, navigate to this directory and execute:
```python
python 2_Si_hiPhive_harmonic_properties.py
python 3_Si_hiPhive_thermal_conductivity.py
```
- To view figures generated during simulations, navigate to this folder: ***plots/5_5_5/***
- To access data computed during simulations, navigate to this folder: ***ALD_si_bulk*** 

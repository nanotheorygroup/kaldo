# `silicon_clathrate_Tersoff_LAMMPS`

Example silicon_clathrate_Tersoff_LAMMPS illustrates how to perform thermal 
transport simulation for a type I clathrate (46 atoms per cell) system using with
[LAMMPS USER-PHONON ](https://lammps.sandia.gov/doc/Packages_details.html#pkg-user-phonon) package as force calculator.

External files required: 
		       1). forcefields/Si.tersoff


- The force constant calculation proceeds as follows:

    in.Si46:
    1.  Compute 2nd, 3rd interatomic force constants with LAMMPS USER-PHONON
			
	or

	get_precalculated_fc.sh:
	1.  Download precalculated force constants computed with LAMMPS USER-PHONON


- 1_Si46_Tersoff_harmonic_properties.py proceeds as follows:

    1. Set up force constant object by loading in computed 2nd, 3rd force constants with LAMMPS.

    2. Set up phonon object (3x3x3 k-point mesh) and perform quantum simulation at 300K.
     
    3. Compute and visualize harmonic properties (i.e. dispersion relation, group velocity and DOS). 

-  2_Si46_Tersoff_thermal_conductivity.py proceeds as follows:

    1. Set up force constant object by loading in computed 2nd, 3rd force constants with LAMMPS.

    2. Set up phonon object (3x3x3 k-point mesh) and perform quantum simulation at 300K.

    3. Set up Conductivity object and compute thermal conductivity with Relaxation Time Approximation (RTA).

-  3_Si46_Tersoff_visualize_anharmonic_properties.py proceeds as follows:

    1. Set up force constant object by loading in computed 2nd, 3rd force constants with LAMMPS.
			
    2. Set up phonon object (3x3x3 k-point mesh) and perform quantum simulation at 300K.

	3. Compute and visualize phase spaces and life times from RTA.


- LAMMPS with speed up force constant calculations for large systems is available in the following repo: [OptimizedDynamicalMatrix](https://github.com/charlessievers/lammps/tree/OptimizedDynamicalMatrix). 
- LAMMPS packages to install/compile include: [MAKE, MANYBODY, MOLECULE, KSPACE, USER-PHONON](https://lammps.sandia.gov/doc/Packages_details.html).


- To compute 2<sup>nd</sup> and 3<sup>rd</sup> order force constants with LAMMPS, navigate to this directory and execute:
```bash
./mpirun -np 8 /path/to/lammps/src/lmp_mpi < in.Si46 > Si46.log 
```
- To comput 2<sup>nd</sup> and 3<sup>rd</sup> order force constants with speed-up LAMMPS, navigate to this directory and execute:
```bash
./mpirun -np 8 /path/to/lammps/src/lmp_mpi < in.Si46_speed_up > Si46_speed_up.log 
```
- To obtain precalculated 2<sup>nd</sup> and 3<sup>rd</sup> order force constants, navigate to this directory and execute:
```bash
chmod +x  get_precalculated_fc.sh
./get_precalculated_fc.sh
```
- To perform thermal transport after computing force constants, navigate to this directory and execute:
```python
python 1_Si46_Tersoff_harmonic_properties.py
python 2_Si46_Tersoff_thermal_conductivity.py
python 3_Si46_Tersoff_visualize_anharmonic_properties.py
```

- To view figures generated during simulations, navigate to this folder: ***plots/3_3_3/***
- To access data computed during simulations, navigate to this folder: ***ALD_Si_46***

 

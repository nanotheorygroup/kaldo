# `carbon_nanotube_Tersoff_LAMMPS`

Example carbon_nanotube_Tersoff_LAMMPS illustrates how to perform thermal 
transport simulation for a 10,0 carbon nanotube (40 atoms per cell) system using
[LAMMPS USER-PHONON ](https://lammps.sandia.gov/doc/Packages_details.html#pkg-user-phonon) package as force calculator.

External files required: 
			1). forcefields/C.optimize.tersoff
			2). unit.xyz : (10,0) Carbon Nanotube unit cell (40 atoms per cell)

- The force constant calculation proceeds as follows:

    in.CNT:
    1.  Compute 2nd, 3rd force constants computed with LAMMPS USER-PHONON
			
	or

	get_precalculated_fc.sh:
	1.  Download precalculated force constants computed with LAMMPS USER-PHONON

- 0_generate_LAMMPS_input_and_supercell_structures.py proceeds as follows:

    1. Loading in 10,0 CNT unit cell structure (unit.xyz) and write it as LAMMPS input format (CNT.lmp).

    2. Replicate the unit cell (1x1x3 supercell) and write it to force constant input folder (fc_CNT).
     
- 1_CNT_Tersoff_thermal_conductivity_quantum.py proceeds as follows:

    1. Set up force constant object by loading in computed 2nd, 3rd force constants computed with LAMMPS USER-PHONON.

    2. Set up phonon object (1x1x151 k-point mesh) and perform quantum simulation at 300K.
     
    3. Set up Conductivity object and compute thermal conductivity for infinite and finite size samples. 

-  2_CNT_Tersoff_thermal_conductivity_classical.py proceeds as follows:

    1. Set up force constant object by loading in computed 2nd, 3rd force constants computed with LAMMPS USER-PHONON.

    2. Set up phonon object (1x1x151 k-point mesh) and perform classical simulation at 300K.

    3. Set up Conductivity object and compute thermal conductivity for infinite and finite size samples. 

- LAMMPS packages to install/compile include: [MAKE, MANYBODY, MOLECULE, KSPACE, USER-PHONON](https://lammps.sandia.gov/doc/Packages_details.html).

- To generate input files for LAMMPS calculations and the supercell structures, navigate to this directory and execute:
```python
python 0_generate_LAMMPS_input_and_supercell_structures.py
```
- To compute 2<sup>nd</sup> and 3<sup>rd</sup> order force constants with LAMMPS USER-PHONON, navigate to this directory and execute:
```bash
./mpirun -np 8 /path/to/lammps/src/lmp_mpi < in.CNT > CNT.log 
```
- To obtain precalculated 2<sup>nd</sup> and 3<sup>rd</sup> order force constants, navigate to this directory and execute:
```bash
chmod +x  get_precalculated_fc.sh
./get_precalculated_fc.sh
```
- To perform thermal transport after computing force constants, navigate to this directory and execute:
```python
python 1_CNT_Tersoff_thermal_conductivity_quantum.py
python 2_CNT_Tersoff_thermal_conductivity_classical.py
```
- To access data computed during simulations, navigate to this folder: ***ALD_CNT***
- Reference conductivity values (1x1x3 supercell,1x1x151 k-point mesh, method: direct-inversion of scattering matrix): 2999.680 W/m-K (classical), 8794.771 W/m-K (Quantum) 
- Reference conductivity values (1x1x5 supercell,1x1x151 k-point mesh, method: direct-inversion of scattering matrix): 3121.130 W/m-K (classical), 8641.888 W/m-K (Quantum) 

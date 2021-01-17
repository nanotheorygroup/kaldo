# `carbon_nanotube_Tersoff_LAMMPS`

Example carbon_nanotube_Tersoff_LAMMPS illustrates how to perform thermal 
transport simulation for a 10,0 carbon nanotube (40 atoms per cell) system using
[LAMMPS USER-PHONON ](https://lammps.sandia.gov/doc/Packages_details.html#pkg-user-phonon) package as force calculator.

External files required: 
			1). forcefields/C.optimize.tersoff

- The force constant calculation proceeds as follows:

    in.CNT:
    1.  Compute 2nd, 3rd force constants computed with LAMMPS USER-PHONON
			
	or

	get_precalculated_fc.sh:
	1.  Download precalculated force constants computed with LAMMPS USER-PHONON


- 1_CNT_Tersoff_thermal_conductivity_quantum.py proceeds as follows:

    1. Set up force constant object by loading in computed 2nd, 3rd force constants computed with LAMMPS USER-PHONON.

    2. Set up phonon object (1x1x151 k-point mesh) and perform quantum simulation at 300K.
     
    3. Set up Conductivity object and compute thermal conductivity for infinite and finite size samples. 

-  2_CNT_Tersoff_thermal_conductivity_classical.py proceeds as follows:

    1. Set up force constant object by loading in computed 2nd, 3rd force constants computed with LAMMPS USER-PHONON.

    2. Set up phonon object (1x1x151 k-point mesh) and perform classical simulation at 300K.

    3. Set up Conductivity object and compute thermal conductivity for infinite and finite size samples. 

- LAMMPS packages to install/compile include: [MAKE, MANYBODY, MOLECULE, KSPACE, USER-PHONON](https://lammps.sandia.gov/doc/Packages_details.html).

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

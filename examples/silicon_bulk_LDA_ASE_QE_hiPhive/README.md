# `silicon_bulk_LDA_ASE_QE_hiPhive`

Example silicon_bulk_LDA_QE_hiPhive illustrates how to perform thermal transport simulation for a bulk silicon (2 atoms per cell) system using [hiPhive](https://hiphive.materialsmodeling.org/) to extract 
force constants from reference force calculations computed with [ASE and Quantum ESPRESSO](https://wiki.fysik.dtu.dk/ase/ase/calculators/espresso.html).


External files required: 
		        1). Si.pz-n-kjpaw_psl.0.1.UPF

 - 1_Si_hiPhive_generate_fcs.py proceeds as follows:
	
    1. Generate 50 reference structures and perform force calculations with standard rattle scheme.

    2. Set up StructureContainer with 2nd and 3rd order cutoffs of 4.0 angstrom.

    3. Develop force constant potentials by fitting reference structures and calculated force with least square method.

    4. Extract force constants from force constant potentials. 

    5. Set up input files for kALDo with hiPhive format.


 - 2_Si_hiPhive_thermal_conductivity.py proceeds as follows:
 
    1. Set up force constant object by loading in the 2nd, 3rd force constants computed from hiPhive.

    2. Set up phonon object (7x7x7 k-point mesh) and perform quantum simulation at 300K.

    3. Set up Conductivity object and compute thermal conductivity with Relaxation Time Approximation (RTA), self-consistent (sc) and direct inversion of scattering matrix (inverse) methods.


- To set up the force calculator with Quantum ESPRESSO and ASE, set the environment variable as follow:
```bash
export ASE_ESPRESSO_COMMAND="mpirun -np 8 /path/to/pw.x -in PREFIX.pwi > PREFIX.pwo"				
```  
- To extract 2<sup>nd</sup> and 3<sup>rd</sup> force constants from force constant potentials, navigate to this directory and execute:
```python			
python 1_Si_hiPhive_generate_fcs.py
```
- To obtain precalculated 2<sup>nd</sup> and 3<sup>rd</sup> order force constants, navigate to this directory and execute:
```bash
chmod +x  get_precalculated_fc.sh
./get_precalculated_fc.sh
```

- To perform thermal transport  after computing force constants, navigate to this directory and execute:
```python
python 2_Si_hiPhive_thermal_conductivity.py
```

- To access data computed during simulations, navigate to this folder: ***ALD_si_bulk*** 
- Sample conductivity (3x3x3 supercell,7x7x7 k-point mesh): ~57 W/m-K (Quantum)
- Warning: This example only serves the purpose of showing the integration between kALDo and hiPhive. In principle, the well converged thermal conductivity for bulk silicon at 300 K should be ~ 150 W/m-K. To well convergence the calculations presented in this example, one can finetune number of training samples, second and third order force constant potential (fcp) cutoff in the fitting process or exploit different random displacement (rattle) scheme to generate structures and forces for the training sample. All these parameters tuning occur in 1_Si_hiPhive_generate_fcs.py. More details on hiPhive usage can be found [here](https://hiphive.materialsmodeling.org/).

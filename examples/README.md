# Examples
This collection of examples shows how to perform thermal transport with kALDo for cystal and amorphous
systems. Before running simulations with kALDo, download the following third-party packages:  [Quantum ESPRESSO](https://www.quantum-espresso.org/), [LAMMPS](https://lammps.sandia.gov/download.html), [hiPhive](https://hiphive.materialsmodeling.org/)
- To calculate 2<sup>nd</sup> and 3<sup>rd</sup> order force constants with LAMMPS and ASE, after downloading [LAMMPS](https://lammps.sandia.gov/),compile LAMMPS with shlib mode:
```bash
cd path/to/lammps/src
make yes-manybody
make yes-molecule
make mpi mode=shlib
```
- After properly install kALDo, run the following line in command window to link LAMMPS with Python and ASE:
```bash
cd path/to/lammps/src
make install-python				
```
- To calculate 2<sup>nd</sup> and 3<sup>rd</sup>  force constants with Quantum ESPRESSO and ASE, set the environment variable as follow:
```	bash			
export ASE_ESPRESSO_COMMAND="mpirun -np 8 /path/to/pw.x -in PREFIX.pwi > PREFIX.pwo"
```
- In amorphous silicon and silicon clathrate examples, 2<sup>nd</sup> and 3<sup>rd</sup> order force constants are computed by using LAMMPS USER-PHONON package. Since these material systems have relatively large  unit cell, computing force constants using the USER-PHONON package currently available in the official website can take hours. 
To speed up the calculation, one can download and compile LAMMPS with optimized force constants calculation functionality from the following repo: [OptimizedDynamicalMatrix](https://github.com/charlessievers/lammps/tree/OptimizedDynamicalMatrix). 
- To obtain precalculated force constants for each example, navigate to each example folder and execute:
```bash
chmod +x get_precalculated_fc.sh
./get_precalculated.sh
```
- To obtain reference calculations of phonon properties and plots for each example, navigate to each example folder and execute:
```bash
chmod +x get_reference.sh
./get_reference.sh
```
- Precalculated force constant and reference calculation files can be accessed [here](http://sophe.ucdavis.edu/structures/).
- Suggestions of specificing GPU/CPU usage for tensorflow can be accessed [here](https://stackoverflow.com/questions/40069883/how-to-set-specific-gpu-in-tensorflow).
## List and content of examples folder
For each example, more detailed information is provided by the README.md file contained in the corresponding directory.
- `amorphous_silicon_Tersoff_LAMMPS:`
This example illustrates how to perform thermal transport simulation for an amorphous silicon system (512 atoms system) with [LAMMPS USER-PHONON 
package](https://lammps.sandia.gov/doc/Packages_details.html#pkg-user-phonon) as force calculator.
- `carbon_diamond_Tersoff_ASE_LAMMPS:`
This example illustrates how to perform thermal transport simulation for a carbon diamond (2 atoms per cell) system using [ASE and LAMMPS](https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/lammpslib.html) as force calculator.
- `silicon_bulk_LDA_ASE_QE_hiPhive:`
This example illustrates how to perform thermal transport simulation for a bulk silicon (2 atoms per cell) system using hiPhive to extract 
force constants from [ASE and Quantum ESPRESSO](https://wiki.fysik.dtu.dk/ase/ase/calculators/espresso.html) reference force calculations.
- `silicon_bulk_Tersoff_ASE_LAMMPS_hiPhive:`
This example illustrates how to perform thermal transport simulation for a bulk silicon (2 atoms per cell) system using hiPhive to extract 
force constants from [ASE and LAMMPS](https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/lammpslib.html) reference force calculations.
- `silicon_clathrate_Tersoff_LAMMPS:`
This example illustrates how to perform thermal transport simulation for a type I clathrate (46 atoms per cell) system using [LAMMPS USER-PHONON 
package](https://lammps.sandia.gov/doc/Packages_details.html#pkg-user-phonon) as force calculator.

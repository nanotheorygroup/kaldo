==============
Tutorials
==============


All examples explained here can be found in the `examples` folder. There are
also two tutorials available in the documentation that can be run in `Google Colab<https://colab.research.google.com>`.
This page is a collection of tutorials that have been explained in further
detail. If you are new to kaldo and thermal conductivity study, this area
may be a good place to start. Please note that some tutorials also require
that you have LAMMPS built with an mpi exectuable.

1. Conductivity of Crystal Silicon (Requires lmp_mpi)

2. Conductivity of Amorphous Silicon (Requires lmp_mpi)

3. Using Hiphive Force Constants

4. Using ASE-LAMMPS Force Constants

5. Effects of Band Broadening Shapes

6. Using ASE-EMT Force Constants

Conductivity of Crystal Silicon
========================================

To see the results of the example, simply run the bash script `run_all.sh`
on a machine with LAMMPS built with the lmp_mpi command, or keep
reading for a more detailed explanation.

Using kALDo requires an ASE atoms object. Starting in the file labeled
`1_create_structure.py`, first we create a bulk Si crystal with a lattice
parameter of 5.43 $A$ and write the object to a file. Information about its
reciprocal-space cell dimensions are printed out.

.. code-block:: python
	:linenos:

	atoms = bulk('Si', 'diamond', a=5.43)
	write(filename='structures/coords.lmp', images=atoms, format='lammps-data')

Run the `in.lmp` file with  `lmp_mpi < in.lmp` to get the derivatives of the force constant
matrix labeled `third.bin` and `dynmat.dat`.

Moving on, `2_calculate_phonons.py` constructs a :ref:`ForceConstants<forceconstants-api>`
object is created with the atoms from the previous step along the dynamical
matrix and the third order derivative of the force constant matrix found from the
LAMMPS script.

.. code-block:: python
	:linenos:

	supercell = np.array([3, 3, 3])
	folder = '.'
	config_file = str(folder) + '/replicated_coords.lmp'
	dynmat_file = str(folder) + '/dynmat.dat'
	third_file = str(folder) + '/third.bin'
	atoms = read(config_file, format='lammps-data', style='atomic')

	atomic_numbers = atoms.get_atomic_numbers()
	atomic_numbers[atomic_numbers == 1] = 14
	atoms.set_atomic_numbers(atomic_numbers)

	forceconstants = ForceConstant.from_files(atoms, dynmat_file, third_file, folder, supercell)


	k = 5
	kpts = [k, k, k]
	is_classic = False
	temperature = 300

	phonons = Phonons(forceconstants=forceconstants,
					  kpts=kpts,
					  is_classic=is_classic,
					  temperature=temperature)

	print('AF conductivity')
	print(phonons.conductivity(method='qhgk').sum(axis=0))

-----------------------------------
More Tutorials Coming Soon
-----------------------------------

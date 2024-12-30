.. _phonons_frontmatter

.. currentmodule:: kaldo.forceconstants

##############
ForceConstants
##############


The ForceConstants class creates objects that store the system information and load (or calculate) force constant
matrices (IFCs) to be used by an instance of the :ref:`Phonon <phonons-api>` class. This is normally the first thing you
should initialize when working with kALDo and we'll walk you through how to do that in the following sections on this
page. Initializing a ForceConstants object will also initialize SecondOrder and ThirdOrder containers that you can
access as attributes. Refer to the section on :ref:`loading IFCs`.

.. _calculating IFCs:

*************************
Calculating IFCs in kALDo
*************************

We use the finite difference approach, or sometimes called the frozen-phonon method. This method neglects temperature
effects on phonon frequency and velocities, but it's a good approximation if your system is far from melting (well-below
the Debye Temperature).

The basic idea is we explicitly calculate the change in forces on atoms when they are displaced from their equilibrium
and use this to approximate the true derivative of the potential energy at equilibrium. You should keep aim to keep the
displacements small, but not so small that the potential you're using can't resolve the change in forces. On empirical
potentials like Tersoff, you can try pretty small displacements (1e-5 Angstroms) but on more complex potentials like
machine learning potentials you may need to use larger displacements (1e-3 Angstroms).

For a system with N atoms, the number of second order IFCs is :math:`(3N)^2` but, because each frame returns :math:`3N`
forces, we only need to calculate :math:`\frac{(3N)^2}{3N} = 3N` frames. However, the finite difference method uses 2
displacements (+/-) to approximate the derivative so total number is actually :math:`2\times 3N = 6N`.

Each term of the third order FCs are calculated with 2 derivatives calculated with 2 displacements, so the total number
of frames is :math:`2 \times 2 \times 3N \times 3N = (6N)^2`.


Should I use kALDo to calculate my IFCs?
========================================

It's faster to use compiled software like LAMMPS or Quantum Espresso to generate the IFCs when possible. However, if
your system is small ( hundreds to a few thousand atoms ), it may be tractable to calculate them here. The following are
some general cases where you could calculate it all here without a significant performance hit:

* You want to explore the effect of different potentials on only harmonic phonon properties (so no third order needed).
* Small systems with short ranges of interactions
* Users have a custom potential (particularly when it can be calculated within python)
* Systems have no symmetry (which many of the compiled software packages exploit to greatly reduce total calculations)

If you're not sure, :py:meth:`SecondOrder.calculate` prints to stdout the atom its currently working on so run the
calculation and stop it after it prints a few atoms. Take the average time per atom and divide by 6 to get the time per
frame :math:`t_{pf}` including overhead of I/O operations, launching tasks, etc. The total times are given by:

.. math::
    t_{2nd Order} = t_{pf} \times {6N}

.. math::
    t_{3rd Order} = t_{pf} \times {6N}^2

Calculation Workflow
====================

.. hint::
   Be sure to minimize the potential energy of your atomic positions before calculating the IFCs. The steps here assume
   you have already done this.

#. Import required packages which will be kALDo, the ASE calculator you want to use, and either a function to build
   atoms (like ase.build.bulk) or their read tool (ase.io.read)::

    from kaldo.forceconstants import ForceConstants
    from ase.io import read
    from ase.build import bulk
    from ase.calculatos.lammpslib import LAMMPSlib

.. _asecalc: https://wiki.fysik.dtu.dk/ase/ase/atoms.html

#. Create your Atoms, Calculator and ForceConstants object. If you don't set a "folder" argument for the ForceConstants
   object the default save location is "displacements". See `the ASE docs <asecalc>`_ for information on their
   calculators::

    # Create the Atoms object
    atoms = bulk('C', 'diamond', a=3.567)
    # Create the ASE calculator object
    calc = LAMMPSlib(lmpcmds=['pair_style tersoff',
                              'pair_coeff * * SiC.tersoff C'],
                     parameters={'control': 'energy',
                                     'log': 'none',
                                    'pair': 'tersoff',
                                    'mass': '1 12.01',
                                'boundary': 'p p p'},
                     files=['SiC.tersoff']))
    # Create the ForceConstants object
    fc = ForceConstants(atoms,
                        supercell=[3, 3, 3],
                        folder='path_to_save_IFCS/',)

#. Now use the "calculate" methods of the second and third order objects by passing in the calculator as an argument.
   The "delta_shift" argument is how far the atoms move.::

    # Second Order (for harmonic properties - phonon frequencies, velocities, heat capacity, etc.)
    fc.second.calculate(calc)
    # Third Order (for anharmonic properties - phonon lifetimes, thermal conductivity, etc.)
    fc.third.calculate(calc)

.. _carbon diamond example: https://github.com/nanotheorygroup/kaldo/blob/main/examples/carbon_diamond_Tersoff_ASE_LAMMPS/1_C_Tersoff_fc_and_harmonic_properties.py

Try referencing the `carbon diamond example`_ to see an example where we use kALDo and ASE to control LAMMPS
calculations.

.. hint::
   Some libraries, like LAMMPS, can use either a python wrapper (LAMMPSlib) or a direct call to the binary
   executable (LAMMPSrun). We don't notice a significant performance increase by using the direct call method, because
   the bottleneck is the I/O of data back to python. This is part of the reason our examples use the python wrapper,
   which offers more flexibility without losing access to the full LAMMPS functionality.

.. _loading IFCs:

**************************
Loading Precalculated IFCs
**************************

.. _amorphous silicon example: https://github.com/nanotheorygroup/kaldo/blob/main/examples/amorphous_silicon_Tersoff_LAMMPS/1_aSi512_Tersoff_thermal_conductivity.py

Construct your ForceConstants object by using the :obj:`~kaldo.ForceConstants.from_folder` method. The first step of
the `amorphous silicon example`_ can help you get started.
If you'd like to load IFCs into the already-created instances without the :py:meth:`from_folder` generate the
ForceConstants object and then use the :py:meth:`load` method of the SecondOrder and ThirdOrder objects to pull data as
needed.

.. hint::
   If you just want to check harmonic data first, use the "is_harmonic" argument when creating the ForceConstants object
   to only load the second order IFCs. This can save considerable amounts of time if, for instance, you just need to
   generate a phonon dispersion along a new path.

.. _input-files:

Input Files and Formats
=======================

.. list-table:: Input Files
   :align: center
   :widths: auto
   :header-rows: 1

   * - Format
     - Config filename
     - ASE format
     - 2nd Order FCs
     - 3rd Order FCs
   * - numpy
     - replicated_atoms.xyz
     - xyz
     - second.npy
     - third.npz
   * - eskm
     - CONFIG
     - dlp4
     - Dyn.form
     - THIRD
   * - lammps
     - replicated_atoms.xyz
     - xyz
     - Dyn.form
     - THIRD
   * - shengbte
     - CONTROL [#f1]_
     - None [#f2]_
     - FORCE_CONSTANTS_2ND
     - FORCE_CONSTANTS_3RD
   * - shengbte-qe
     - CONTROL [#f1]_
     - None
     - espresso.ifc2
     - FORCE_CONSTANTS_3RD
   * - hiphive
     - atom_prim.xyz + replicated_atoms.xyz
     - xyz
     - model2.fcs
     - model3.fcs

.. rubric:: Notes on Formats

.. [#f1] The shengbte and shengbte-qe format will look for the "CONTROL" file first, however, if it is not found it will
         look for a "POSCAR" file (ASE format "vasp"). If neither are found, it will raise an error.
.. [#f2] The shengbte format does not have a direct equivalent in ASE. You can be create the atoms object manually or
         use the import_control_file method.

.. _forceconstants-api:

*************
API Reference
*************

.. autoclass:: kaldo.forceconstants::ForceConstants
    :members: from_folder, unfold_third_order, elastic_prop


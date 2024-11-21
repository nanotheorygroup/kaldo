.. _forceconstants-api:

.. currentmodule:: kaldo.forceconstants

##############
ForceConstants
##############

The `ForceConstants` class creates objects that store system information and load (or calculate) force constant
matrices (IFCs) to be used by an instance of the :ref:`Phonon <phonons-api>` class. This is typically the first step
when working with kALDo, and we’ll walk you through initializing a `ForceConstants` object on this page. Initializing a
`ForceConstants` object also initializes `SecondOrder` and `ThirdOrder` containers that you can access as attributes.
Refer to the section on :ref:`loading IFCs`.

.. _calculating IFCs:

*************************
Calculating IFCs in kALDo
*************************

We use the finite difference approach, also called the frozen-phonon method. This method neglects temperature effects
on phonon frequencies and velocities, but it’s a good approximation if your system is far from melting (well below
the Debye temperature).

The basic idea is to explicitly calculate the change in forces on atoms when they are displaced from equilibrium and
use this to approximate the derivative of the potential energy at equilibrium. You should aim to keep displacements
small, but not so small that the potential you’re using can’t resolve the change in forces. For empirical potentials
like Tersoff, you can try small displacements (e.g., 1e-5 Å) but for more complex potentials, like machine learning
potentials, you may need larger displacements (e.g., 1e-3 Å).

For a system with N atoms, the number of second-order IFCs is :math:`(3N)^2`. However, since each frame returns
:math:`3N` forces, we only need :math:`3N` frames. The finite difference method uses 2 displacements (+/-) to
approximate the derivative, so the actual number is :math:`6N`.

Each term of the third-order IFCs requires 2 derivatives, each calculated with 2 displacements, so the total number
of frames is :math:`(6N)^2`.

Should I use kALDo to calculate my IFCs?
========================================

It’s often faster to use compiled software like LAMMPS or Quantum Espresso to generate the IFCs when possible.
However, if your system is small (hundreds to a few thousand atoms), it may be feasible to calculate them here. Some
general cases where you might calculate them in kALDo include:

- Exploring different potentials on harmonic phonon properties (no third order needed).
- Small systems with short interaction ranges.
- Custom potentials (especially if calculated within Python).
- Systems with no symmetry (many compiled packages exploit symmetry to reduce calculations).

To estimate the time required, run the calculation and stop it after it prints for a few atoms. The average time per
atom, divided by 6, gives the time per frame :math:`t_{pf}`. Total times are given by:

.. math::
    t_{2nd Order} = t_{pf} \times 6N

.. math::
    t_{3rd Order} = t_{pf} \times (6N)^2

Calculation Workflow
====================

.. hint::
   Ensure that atomic positions are minimized before calculating the IFCs. The following steps assume this.

1. Import required packages (kALDo, ASE calculator, and tools for creating atoms):

    ```python
    from kaldo.forceconstants import ForceConstants
    from ase.io import read
    from ase.build import bulk
    from ase.calculators.lammpslib import LAMMPSlib
    ```

2. Create your `Atoms`, `Calculator`, and `ForceConstants` objects:

    ```python
    atoms = bulk('C', 'diamond', a=3.567)
    calc = LAMMPSlib(lmpcmds=['pair_style tersoff', 'pair_coeff * * SiC.tersoff C'],\
                     parameters={'control': 'energy', 'log': 'none', 'pair': 'tersoff',\
                                 'mass': '1 12.01', 'boundary': 'p p p'},\
                     files=['SiC.tersoff'])
    fc = ForceConstants(atoms, supercell=[3, 3, 3], folder='path_to_save_IFCS/')
    ```

3. Use the `calculate` methods of `second` and `third` order objects:

    ```python
    fc.second.calculate(calc)
    fc.third.calculate(calc)
    ```

.. _carbon diamond example: https://github.com/nanotheorygroup/kaldo/blob/main/examples/carbon_diamond_Tersoff_ASE_LAMMPS/1_C_Tersoff_fc_and_harmonic_properties.py

For a full example, refer to the `carbon diamond example`_ using kALDo and ASE with LAMMPS.

.. hint::
   Some libraries, like LAMMPS, offer a Python wrapper (LAMMPSlib) or a direct call method (LAMMPSrun). We recommend
   the Python wrapper as the I/O bottleneck makes the performance difference minimal.

.. _loading IFCs:

**************************
Loading Precalculated IFCs
**************************

.. _amorphous silicon example: https://github.com/nanotheorygroup/kaldo/blob/main/examples/amorphous_silicon_Tersoff_LAMMPS/1_aSi512_Tersoff_thermal_conductivity.py

Create a `ForceConstants` object using the :obj:`~kaldo.ForceConstants.from_folder` method. The `amorphous silicon
example`_ can help get you started. To load IFCs into an existing instance without :py:meth:`from_folder`, initialize
the `ForceConstants` object, then use the `load` method of `SecondOrder` and `ThirdOrder`.

.. hint::
   If you only need harmonic data, set `is_harmonic` when creating the `ForceConstants` object. This will only load
   second-order IFCs, saving time if, for example, you’re just generating a phonon dispersion.

*************
API Reference
*************

 .. autoclass:: ForceConstants
        :members:

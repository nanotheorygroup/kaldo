.. _forceconstants-frontmatter:

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

It's generally faster to use compiled software like LAMMPS or Quantum Espresso to generate the IFCs when possible.
However, if you want to calculate the IFCs directly from Python, kALDo has some advantages because both second-
and third-order finite-difference calculations can be parallelized with ``n_workers``. This parallel workflow has
been used successfully on HPC compute clusters, including multi-node runs, with a near-linear speed up (for calculators
that do not use thread-parallelization).
The ``distance_threshold`` option can also greatly reduce the amount of work required for third order IFC generation,
by skipping interactions between atoms that are too far apart. This is especially useful for large systems
(like glasses, or clathrates).

Some general cases where calculating IFCs directly in kALDo is a good fit are:

* You want to explore the effect of different potentials on only harmonic phonon properties (so no third order needed).
* Users have a custom, or Python-based calculator
* The system has no symmetry (which many of the compiled software packages exploit to reduce total calculations)
* Larger third-order calculations where a physically reasonable ``distance_threshold`` can be used to skip distant
  interactions

Serial Runtime Estimate
-----------------------

If you're not sure, :py:meth:`SecondOrder.calculate` prints to stdout the atom it is currently working on, so run the
calculation with ``n_workers=1`` and stop it after it prints a few atoms. Take the average time per atom and divide by
6 to get the time per frame :math:`t_{pf}` including overhead from I/O operations and task setup. For serial runs, the
total times are given by:

.. math::
    t_{\text{2nd Order}} = t_{pf} \times {6N}

.. math::
    t_{\text{3rd Order}} = t_{pf} \times {6N}^2

Parallel Runtime Estimate
-------------------------

For parallel runs, kALDo distributes displaced-atom tasks across workers, so a useful first estimate for wall time is
the serial estimate divided by ``n_workers``:

.. math::
    t_{\text{2nd Order, parallel}} \approx \frac{t_{pf} \times {6N}}{n_{\text{workers}}}

.. math::
    t_{\text{3rd Order, parallel}} \approx \frac{t_{pf} \times {6N}^2}{n_{\text{workers}}}

This is only an approximate guide because real runs also include process startup, I/O, scratch traffic, load imbalance,
and calculator-specific overhead. For third-order calculations, using a nonzero ``distance_threshold`` can reduce the
effective amount of work even further by skipping distant interactions, so the actual runtime may be substantially lower
than the no-cutoff estimate above.

For calculators that use shared-memory parallelization (e.g. Orb), it's likely that the fastest route will be a blend
of thread and process parallelization. Try out different combinations of ``n_workers`` and ``OMP_NUM_THREADS`` for the
best results.

Calculation Workflow
====================

.. hint::
   Be sure to minimize the potential energy of your atomic positions before calculating the IFCs. The steps here assume
   you have already done this.

Both :py:meth:`SecondOrder.calculate` and :py:meth:`ThirdOrder.calculate` can distribute the finite-difference work
across multiple worker processes with the ``n_workers`` argument. In both cases, kALDo parallelizes over displaced
unit-cell atoms: second order assigns one atom's harmonic finite-difference block to each task, while third order
assigns one first-index atom of the anharmonic tensor to each task. This keeps the workflow simple and makes it easy to
resume interrupted runs.

For second- and third-order scratch runs, intermediate results can be written to a ``scratch_dir``. Each completed atom
writes its own scratch artifact together with a ``.done`` sentinel file, so rerunning the calculation with the same
scratch directory skips finished atoms and only recomputes missing work.

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
    fc.second.calculate(calc,
                        delta_shift = 1e-5,
                        n_workers = 2,)
    # Third Order (for anharmonic properties - phonon lifetimes, thermal conductivity, etc.)
    fc.third.calculate(calc
                       delta_shift = 1e-5,
                       n_workers = 2,)

.. _carbon diamond example: https://github.com/nanotheorygroup/kaldo/blob/main/examples/carbon_diamond_Tersoff_ASE_LAMMPS/1_C_Tersoff_fc_and_harmonic_properties.py

Try referencing the `carbon diamond example`_ to see an example where we use kALDo and ASE to control LAMMPS
calculations.

.. hint::
   Some libraries, like LAMMPS, can use either a python wrapper (LAMMPSlib) or a direct call to the binary
   executable (LAMMPSrun). We don't notice a significant performance increase by using the direct call method, because
   the bottleneck is the I/O of data back to python. This is part of the reason our examples use the python wrapper,
   which offers more flexibility without losing access to the full LAMMPS functionality.

.. _parallelism-and-memory-safety:

Parallelism and Memory Safety
=============================

Second- and Third-order force constant calculations can run in parallel via the ``n_workers`` argument to
:py:meth:`ThirdOrder.calculate` (or :py:meth:`SecondOrder.calculate`). kALDo will probe the calculator
memory once before launching workers and caps ``n_workers`` if the estimated per-worker cost would exhaust RAM.
This guards against a failure mode where the OS will let all workers start and then collectively exhaust memory through
silent swap thrashing without triggering the OOM killer. When workers are reduced a ``ResourceWarning`` is emitted.

The maximum number of workers that can be launched is equal to $

The memory check and parallel backend selection can be controlled via environment variables:

.. list-table:: kALDo Environment Variables
   :widths: 30 15 55
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``KALDO_SKIP_MEMORY_CHECK``
     - unset
     - If set to ``1``, disable the pre-parallel memory probe and worker cap. Workers launch exactly as requested.
   * - ``KALDO_MAX_WORKERS``
     - unset
     - Integer hard cap on ``n_workers``. Bypasses memory probing and estimation entirely. Use when you already
       know your memory budget.
   * - ``KALDO_MEMORY_HEADROOM``
     - ``0.10``
     - Float in ``[0, 1]``. Fraction of total RAM reserved for the OS and other processes. The remainder is
       treated as usable for kALDo workers.
   * - ``KALDO_PARALLEL_BACKEND``
     - unset
     - Override the multiprocessing backend. Valid values: ``serial``, ``process``, ``mpi``.

Example — disable the memory check and run with as many workers as the script requests::

    KALDO_SKIP_MEMORY_CHECK=1 python run_thirdorder.py

Example — hard-cap workers to 8 regardless of what the script requests::

    KALDO_MAX_WORKERS=8 python run_thirdorder.py

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
   If you just want to check harmonic data first, use the ``only_second`` argument when running the :py:meth:`from_folder`
   command to only load the second order IFCs. This can save considerable amounts of time if, for instance, you just
   need to generate a phonon dispersion along a new path.


.. _input-files:

Input Files and Formats
=======================

.. list-table:: Input Files
   :align: center
   :widths: auto
   :header-rows: 1

   * - Format
     - Config filename
     - ASE format (for config file)
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
   * - vasp-sheng
     - CONTROL / POSCAR [#f1]_
     - N/A / vasp [#f2]_
     - FORCE_CONSTANTS_2ND
     - FORCE_CONSTANTS_3RD
   * - qe-sheng
     - CONTROL / POSCAR [#f1]_
     - N/A / vasp [#f2]_
     - espresso.ifc2
     - FORCE_CONSTANTS_3RD
   * - vasp-d3q
     - CONTROL / POSCAR [#f1]_
     - N/A / vasp [#f2]_
     - FORCE_CONSTANTS_2ND
     - FORCE_CONSTANTS_3RD_D3Q
   * - qe-d3q
     - CONTROL / POSCAR [#f1]_
     - N/A / vasp [#f2]_
     - espresso.ifc2
     - FORCE_CONSTANTS_3RD_D3Q
   * - hiphive
     - atom_prim.xyz + replicated_atoms.xyz
     - xyz
     - model2.fcs
     - model3.fcs
   * - tdep [#f3]_
     - POSCAR/UCPOSCAR/SSPOSCAR
     - xyz
     - second.npy
     - third.npz
      

.. rubric:: Notes on Formats

.. [#f1] The shengbte and shengbte-qe format will look for the "CONTROL" file first, however, if it is not found it will
         look for a "POSCAR" file (ASE format "vasp"). If neither are found, it will raise an error.
.. [#f2] ASE does not support the shengbte format (CONTROL file). You can create the atoms object manually or
         use the ``kaldo.interfaces.shegbte_io.import_control_file`` method.
.. [#f3] TDEP with kaldo will only work for bulk materials (3D) materials

.. _forceconstants-api:

*************
API Reference
*************


.. autoclass:: ForceConstants
        :members:


Companion Objects
=================

Although :class:`ForceConstants` is the main user-facing entry point, it creates
separate ``SecondOrder`` and ``ThirdOrder`` objects that many users interact
with through ``fc.second`` and ``fc.third``. These classes are mostly internal
containers, but users with non-standard input files or custom finite-difference
workflows may still find their ``load(...)`` and ``calculate(...)`` methods
useful.

SecondOrder
-----------

The harmonic companion object is usually accessed as ``ForceConstants.second``. Direct use
is most helpful when you want to load second-order data in a non-standard way (e.g. not using the
``ForceConstants.from_folder`` method) or to run the harmonic finite-difference calculation.

.. currentmodule:: kaldo.observables.secondorder

.. automethod:: SecondOrder.load

.. automethod:: SecondOrder.calculate

ThirdOrder
----------

The anharmonic companion object is usually accessed as ``ForceConstants.third``. Direct use
is most helpful when you need to load third-order data from a specific format or when launching
the finite-difference calculation.

.. currentmodule:: kaldo.observables.thirdorder

.. automethod:: ThirdOrder.load

.. automethod:: ThirdOrder.calculate

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

.. note::
   ML potentials (Orb, MACE, MatterSim, CPUNEP, ...) need a small extra step for parallel runs because their
   PyTorch-backed instances cannot be pickled across processes. See :ref:`parallel-ml-calculators` for the
   recommended pattern (a top-level factory function plus an ``if __name__ == '__main__':`` guard) and the
   ``delta_shift`` guidance for float32 calculators.

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

.. _parallel-ml-calculators:

Parallel runs with ML calculators
=================================

The serial idiom shown above (``calc = LAMMPSlib(...); fc.second.calculate(calc, ...)``) keeps working for parallel
runs as long as the calculator can be pickled and forked safely. That's true for analytical and shared-library
calculators (EMT, Lennard-Jones, LAMMPS), but typically *not* for ML potentials (Orb, MACE, MatterSim, CPUNEP, ...)
that hold PyTorch models, GPU contexts, or C handles. Two issues come up in that case:

1. **The live calculator instance can't cross a process boundary.** Spawn-based multiprocessing pickles every
   argument; non-picklable instances raise ``TypeError`` (kALDo's parallel validator catches this with a
   copy-pasteable fix message) or, worse, fork into a worker that then crashes with ``BrokenProcessPool`` or
   ``Cannot re-initialize CUDA in forked subprocess``.
2. **Spawn re-imports your script.** On any host with CUDA available, kALDo's executor selects the spawn start
   method to avoid fork-vs-CUDA crashes. Spawn re-executes the entry-point module in every worker, so any
   top-level ``fc.second.calculate(...)`` call would re-fire inside the worker.

The recommended pattern: define a no-argument factory function at module top level and pass the function (not its
return value) as the ``calculator`` argument. Each worker calls the function once to build its own isolated
calculator. Wrap your script's executable code in the standard ``if __name__ == '__main__':`` guard so the worker's
re-import of the script doesn't re-fire your kaldo calls::

    from ase.build import bulk
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    from kaldo.forceconstants import ForceConstants


    def make_calculator():
        """Top-level so spawn-imported workers can reach it by name."""
        orbff = pretrained.orb_v3_conservative_inf_omat(
            device='cpu', precision='float32-highest',
        )
        return ORBCalculator(orbff, device='cpu')


    if __name__ == '__main__':
        atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
        fc = ForceConstants(atoms=atoms, supercell=(3, 3, 3), folder='fc_al')
        fc.second.calculate(make_calculator, delta_shift=1e-2, n_workers=4)
        fc.third.calculate(make_calculator, delta_shift=1e-2, n_workers=4)

Notice that ``make_calculator`` is passed without parentheses. Calling it (``make_calculator()``) would build one
instance in the parent process and try to ship it to workers — exactly the un-picklable case we're avoiding.

Choosing ``delta_shift``
------------------------

Float32 ML calculators produce force noise on the order of ``1e-7 eV/Å``. Finite-difference second derivatives
divide this by ``delta_shift``, so a too-small delta produces FC noise that overwhelms the physics. The kALDo default
``delta_shift=1e-3`` is tuned for analytical calculators (EMT, LAMMPS) where forces are accurate to machine
precision; for ML potentials, prefer ``delta_shift=1e-2`` to ``5e-2``. ``SecondOrder.calculate`` and
``ThirdOrder.calculate`` warn once when ``delta_shift < 1e-2`` and the calculator looks ML-based.

Common pitfalls
---------------

* **Passing the constructed instance instead of the factory.** ``calc = ORBCalculator(...)`` followed by
  ``fc.second.calculate(calc, n_workers=4)`` will fail validation on torch-based calculators. Pass the factory
  function (without parentheses) instead.
* **Forgetting the** ``if __name__ == '__main__':`` **guard.** Without the guard, spawn re-imports your script and
  every worker tries to spawn its own pool, which Python rejects (``RuntimeError: An attempt has been made to
  start a new process before the current process has finished its bootstrapping phase``). kALDo detects this
  before workers spawn and raises with a guard-fix message.
* **Defining the factory inside another function or class.** Closures don't pickle and aren't reachable by name
  from spawn-imported workers. Move the ``def`` to module top level.
* **Setting ``delta_shift=1e-4`` (or smaller) for ML.** This is the right delta for analytical potentials, but
  produces ~1e-3 FC noise on float32 ML potentials, swamping the real signal. Use ``1e-2`` instead.

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


.. hint::
   TDEP with kaldo will only work for bulk materials (3D) materials


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
   * - tdep
     - POSCAR/UCPOSCAR/SSPOSCAR
     - xyz
     - second.npy
     - third.npz
      

.. rubric:: Notes on Formats

.. [#f1] The shengbte and shengbte-qe format will look for the "CONTROL" file first, however, if it is not found it will
         look for a "POSCAR" file (ASE format "vasp"). If neither are found, it will raise an error.
.. [#f2] ASE does not support the shengbte format (CONTROL file). You can create the atoms object manually or
         use the ``kaldo.interfaces.shegbte_io.import_control_file`` method.

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
is most helpful when you need to load third-order data from a specific format or
when launching the finite-difference calculation.

.. currentmodule:: kaldo.observables.thirdorder

.. automethod:: ThirdOrder.load

.. automethod:: ThirdOrder.calculate

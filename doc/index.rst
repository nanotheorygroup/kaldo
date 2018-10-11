.. figure:: ../others/ballistico.png
   :alt: Ballistico Logo

   alt text

Introduction
============

Ballistico is a Python app which allows to calculate phonons dispersion
relations and density of states using finite difference.

Installation
============

1. Install LAMMPS as a library with the ``manybody`` option

::

   make yes-manybody
   make serial mode=shlib

When lammps is compiled as a library the ``liblammps.so`` file is
created in the src folder. Add that to your ``PYTHON_PATH`` using

::

   export PYTHONPATH=[PATH_TO_LAMMPS]:$PYTHONPATH

If you want to make this change persistent

2. Install Python 3 and the following dependencies: ``scipy``,
   ``numpy``, ``matplotlib``, ``pandas``

Dependencies can be installed with conda (``sudo`` permission may be
needed)

::

   conda install [PACKAGE-NAME]

Install the packages ``spglib`` and ``ase``. They don’t belong to the
conda main repo. Use ``conda-forge``

::

   conda install -c conda-forge [PACKAGE-NAME]

FInally, install ``seekpath`` using ``pip``

::

   pip install seekpath

Usage
=====

Define the system: ``MolecularSystem`` object
---------------------------------------------

This is one of the main classes used in Ballistico, it allows the
specification of the following parameters \* geometry \* forcefield \*
replicas \* temperature \* optimization

Define a ``geometry``
~~~~~~~~~~~~~~~~~~~~~

You will need to use an extended xyz file to create a geometry using

.. code:: python

   geometry = ath.from_filename('examples/si-bulk.xyz')

examples of geometries are in the ``examples`` folder

Structure optimization
~~~~~~~~~~~~~~~~~~~~~~

The structure can be optionally optimize using one of the ``scipy``
minimization methods (Nelder-Mead, Powell, CG, BFGS, Newton-CG,
L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr, dogleg, trust-ncg,
trust-exact, trust-krylov)

Define a ``forcefield``
~~~~~~~~~~~~~~~~~~~~~~~

forcefield are define through LAMMPS commands using

.. code:: python

   still_weber_forcefield = ["pair_style sw", "pair_coeff * * forcefields/Si.sw Si"]

another example is

.. code:: python

   tersoff_forcefield = ["pair_style tersoff", "pair_coeff * * forcefields/si.tersoff Si"]

examples of forcefields are in the ``forcefields`` folder

Define the number of replicas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to specify the number of replicas in the direct space using

.. code:: python

   replicas = np.array ([3, 3, 3])

Create the ``MolecularSystem``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once everything is set, create the ``MolecularSystem`` using

.. code:: python

   system = MolecularSystem (geometry, replicas=replicas, temperature=300., optimize=True, lammps_cmd=still_weber_forcefield)

Phonons calculations: ``MolecularSystemPhonons`` object
-------------------------------------------------------

To calculate phononic properties of the system use the
``MolecularSystemPhonons`` object. Here you can specify the ``k_mesh``
to use to calculate the density of states. Here’s an example of Usage

.. code:: python

   k_mesh = np.array([5, 5, 5])
   phonons = MolecularSystemPhonons(system, k_mesh)
   phonons.energy_k_plot (method='auto')

where the method specify for the ``energy_k_plot`` can be ``auto``,
which tries to automatically define a path on the Brillouin zone, or one
of the following: \* cubic \* fcc \* bcc \* hexagonal \* tetragonal \*
orth
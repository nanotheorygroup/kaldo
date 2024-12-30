.. _phonons_frontmatter

.. currentmodule:: kaldo.phonons

#######
Phonons
#######

Building the Phonons class object is the intermediate step between finishing your ForceConstants object, and
initializing the Conductivity object. The keyword arguments given to this class help kALDo make the correct
approximations regarding the periodicity of your system and the conditions of your material when calculating the
thermal conductivity (e.g. temperature of the material).

************************
Periodicity and k-points
************************

The primary way that kALDo handles low-symmetry systems is through the grid of points to sample in reciprocal space,
adjusted through the `kpts` argument. The argument here should be an array or list of three natural numbers,
such as (2, 2, 2). You will need to converge the final thermal conductivity of the material against increasingly
dense k-grids to ensure the reliability of your calculation. Increasing the number of `kpts` along a lattice vector
is the equivalent to increasing the wavelength of a vibrational normal mode along that axis. See our
ref:`Basic Concepts` section for more details.

Your choice of `kpts` should strive to sample each direction in reciprocal space equally by scaling the number of
points in that direction according to the magnitude of the reciprocal lattice vector. A quick way to do this is by
taking the inverse of the magnitude of the real space lattice vector. For a concrete example, imagine a cubic lattice
with lengths :math:`a_{1} < a_{2} < a_{3}`. We know our `kpts` should roughly follow :math:`n_{1} > n_{2} > n_{3}`.
More specifically, we want to minimize the differences of the following terms:

..math::
    \frac{n_{1}}{a_{1}} \approx \frac{n_{2}}{a_{2}} \approx \frac{n_{1}}{a_{1}}

Which leads to us solving a series of inequalities. We will demonstrate the idea here by solving for the bounds on
:math:`n_{2}`, and then using the known relationship between the magnitudes of the lattice vectors to setup an
inequality. The result yields:

..math::
    \frac{n_{3} \cdot a_{2}}{a_{3}} \leq n_{2} \leq \frac{n_{1} \cdot a_{2}}{a_{1}}

The example was given for a sample that's periodic in each direction, however it will be true for less symmetric
samples as well with the modification that you can ignore any direction lacking symmetry. The non-periodic direction
should be set to 1, so for amorphous samples the `kpts` argument should be (1,1,1). For 2D materials, set the
out-of-plane direction to 1, and for nanowires use a value greater than 1 for direction along the wire.

***************
Classical Limit
***************

Probably the second most important setting is the `is_classic` argument. By defaut, kALDo uses the Bose-Einstein
distribution to describe the phonon population at a given temperature which assumes quantized energy.
The :ref:`Basic Concepts` section will provide more details, however because of this population choice the heat capacity
also becomes affected. Close to, or above the Debye Temperature you should switch the `is_classic` Boolean to True which
changes the calculation of mode population to the classical limit as described in the theory section. In general, the
heat capacities and scattering will both increase, shortening the lifetimes and increasing the energy per mode

***********************************
General vs. Case-Specific Arguments
***********************************

This section is intended to clarify which arguments need to be set for all users, and which should be used in
more specific situations. If you are a beginner to intermediate user, avoid the case-specific arguemnts. Advanced users
can read more on the details of each case-specific argument in :ref:`the phonons api <phonons-api>` section, and in some
cases the :ref:`Basic Concepts` section.

.. list-table:: General Arguments
   :align: center
   :widths: auto

    * - ForceConstants
      - Temperature
      - is_classic
      - folder
    * - kpts
      - broadening_shape
      - third_bandwidth
      - storage

.. list-table:: Case-Specific Arguments
   :align: center
   :widths: auto

    * - min_frequency
      - max_frequency
      - is_symmetrizing_frequency
      - is_antisymmetrizing_velocity
      - is_balanced
    * - is_unfolding
      - is_nw
      - g_factor
      - include_isotopes
      - iso_speed_up

.. _phonons-api:

*************
API Reference
*************

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: API/Phonons
   :template: minimal_module.rst

   Phonons.pdos

.. rubric:: Attributes

.. autosummary::
  :nosignatures:
  :toctree: API/Phonons
  :template: minimal_module.rst

   Phonons.physical_mode
   Phonons.participation_ratio
   Phonons.bandwidth
   Phonons.isotopic_bandwidth
   Phonons.anharmonic_bandwidth
   Phonons.frequency
   Phonons.omega
   Phonons.velocity
   Phonons.heat_capacity
   Phonons.heat_capacity_2d
   Phonons.population
   Phonons.phase_space
   Phonons.eigenvalues
   Phonons.eigenvectors

.. autoclass:: Phonons
   :members:


   

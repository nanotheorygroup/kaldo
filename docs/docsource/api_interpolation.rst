.. _bandwidth-interpolation:

Interpolation
=============

Large disordered systems can require many thousands of vibrational modes.  In
that regime, explicitly calculating third-order force constants and
anharmonic linewidths for the final production model may be prohibitively
expensive.  kALDo therefore provides a frequency-dependent bandwidth
interpolator for QHGK workflows in which the anharmonic linewidth is modeled
primarily as a function of phonon frequency.

The typical workflow is:

.. code-block:: python

   from kaldo.conductivity import Conductivity
   from kaldo.controllers.interpolator import BandwidthInterpolator
   from kaldo.phonons import Phonons

   phonons_small = Phonons(
       forceconstants=forceconstants_small,
       temperature=300,
       third_bandwidth=0.1,
   )

   interpolator = BandwidthInterpolator(
       frequency=phonons_small.frequency,
       bandwidth=phonons_small.bandwidth,
       sigma=0.1,
       low_frequency_cutoff=2.0,
       max_bandwidth=10.0,
       folder="bandwidth_fit",
   )

   phonons_large = Phonons(
       forceconstants=forceconstants_large,
       temperature=300,
       interpolator=interpolator,
   )

   conductivity = Conductivity(
       phonons=phonons_large,
       method="qhgk",
   )

   kappa = conductivity.conductivity

The source ``frequency`` and ``bandwidth`` arrays should use kALDo's native
units: THz for :attr:`kaldo.phonons.Phonons.frequency` and THz for
:attr:`kaldo.phonons.Phonons.bandwidth`.  The Gaussian smoothing width
``sigma`` is also in THz.

Method
------

Given source modes :math:`\nu` with frequencies :math:`\omega_\nu` and
linewidths :math:`\Gamma_\nu`, the interpolator builds a smoothed curve

.. math::

   \Gamma_{\mathrm{smooth}}(\omega)=
   \frac{
   \sum_\nu \Gamma_\nu
   \exp[-(\omega-\omega_\nu)^2/(2\sigma^2)]
   }{
   \sum_\nu
   \exp[-(\omega-\omega_\nu)^2/(2\sigma^2)]
   }.

The normalization prefactor cancels.  The implementation evaluates this
expression in chunks so that large amorphous models do not require one dense
``(N_eval, N_modes)`` array.

The smoothed data are interpolated with a shape-preserving cubic interpolator
to reduce unphysical spline overshoot.  Below ``low_frequency_cutoff`` the
final prediction explicitly uses

.. math::

   \Gamma(\omega)=A\omega^2.

The coefficient is

.. math::

   A=\Gamma(\omega_c)/\omega_c^2,

where :math:`\omega_c` is ``low_frequency_cutoff``.  This gives continuous
matching at the crossover.  The low-frequency cutoff is in THz.

If the source calculation contains clearly unphysical linewidth outliers, pass
``max_bandwidth`` in THz.  Source modes with bandwidth larger than this value
are excluded from the smoothing and interpolation, logged, and recorded in
``interpolation_parameters.json``.  The default is ``None`` so that kALDo does
not discard data unless the user makes that scientific choice explicitly.

Extrapolation and Storage
-------------------------

High-frequency extrapolation is controlled by ``extrapolation``:

``"error"``
   Raise an error outside the fitted high-frequency range.  This is the
   default and safest option.

``"constant"``
   Use the nearest fitted endpoint value.

``"spline"``
   Allow the cubic interpolator to extrapolate.

When ``folder`` is provided, the interpolator writes reproducibility data to:

.. code-block:: text

   folder/
       fit/
           source_frequency.npy
           source_bandwidth.npy
           smoothed_frequency.npy
           smoothed_bandwidth.npy
           interpolation_parameters.json
       predictions/
           qhgk/
               frequency.npy
               bandwidth.npy

Calling ``predict_bandwidth(target_frequency, label="large_model")`` stores
that prediction under ``predictions/large_model``.  The prediction has the same
shape as ``target_frequency``.  Calling ``plot()`` saves a bandwidth-versus-
frequency figure in ``folder`` if it is defined, otherwise in the current
working directory.  For source data with a few very large linewidth outliers,
``plot(y_percentile=99)`` or ``plot(y_max=...)`` can zoom the y-axis for
display without removing any points from the interpolation.

QHGK Integration
----------------

Attaching an interpolator to a :class:`kaldo.phonons.Phonons` object affects
only the QHGK bandwidth pathway.  It does not replace, overwrite, or invalidate
``Phonons.bandwidth``.

For QHGK, kALDo chooses linewidths in this order:

1. an explicit ``Conductivity(diffusivity_bandwidth=...)`` value,
2. ``phonons.interpolator.predict_bandwidth(phonons.frequency) / 2``,
3. the normal explicit anharmonic ``phonons.bandwidth / 2``.

Literature
----------

This feature follows the linewidth-interpolation strategy motivated by:

* Fiorentino, P. Pegolo, S. Baroni, and D. Donadio, "Hydrodynamic finite-size
  scaling of the thermal conductivity in glasses," *npj Computational
  Materials* **9**, 157 (2023).
* "Mode localization and suppressed heat transport in amorphous alloys,"
  *Physical Review B* (2021).
* "Modeling heat transport in crystals and glasses from a unified
  lattice-dynamical approach," *Nature Communications* (2019).

API
---

.. autoclass:: kaldo.controllers.interpolator.BandwidthInterpolator
   :members:

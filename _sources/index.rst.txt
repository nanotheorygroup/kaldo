.. kaldo documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: docsource/_resources/logo.png
   :width: 400

Scalable Thermal Transport from First Principles and Machine Learning Potentials
=================================================================================

**kALDo** is an open-source Python package for computing thermal transport properties of crystalline, disordered, and amorphous materials. It implements the **Boltzmann Transport Equation (BTE)** for crystals and the **Quasi-Harmonic Green-Kubo (QHGK)** method for systems lacking long-range order.

Read the paper on `arXiv <https://arxiv.org/abs/2009.01967>`_.


.. toctree::
   :glob:
   :caption: Getting Started
   :maxdepth: 1

   docsource/getting_started.md
   docsource/introduction.ipynb


.. toctree::
   :glob:
   :caption: Tutorials
   :maxdepth: 1

   docsource/crystal_presentation.ipynb
   docsource/amorphous_presentation.ipynb


.. toctree::
   :glob:
   :caption: API Reference
   :maxdepth: 2

   docsource/api_forceconstants
   docsource/api_phonons
   docsource/api_conductivity


.. toctree::
   :glob:
   :caption: Citations
   :maxdepth: 1

   docsource/citations.md
   publications/readme.md


.. toctree::
   :glob:
   :caption: Developer Guide
   :maxdepth: 1

   docsource/contributing.md


Acknowledgements
================

.. image:: docsource/_resources/funding.png
   :width: 650

We gratefully acknowledge support by the Investment Software Fellowships (grant No. ACI-1547580-479590) of the NSF Molecular Sciences Software Institute (grant No. ACI-1547580) at Virginia Tech. `Explore MolSSI's software projects <https://molssi.org/software-projects/>`_.

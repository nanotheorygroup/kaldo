---
title: "kALDo: a software to predict thermal properties using Lattice Dynamics"
tags:
  - Python
  - Materials Science
  - Nanotechnology
  - Computational Physics
  - Lattice Dynamics
  - Heat Transport
  - Quantum Effects
  - Boltzmann Transport Equation
  - Machine Learning Interatomic Potentials
  - Quasi-Harmonic Green-Kubo
  - Phonon Lifetimes
  - Thermal Conductivity
authors:
  - name: Giuseppe Barbalinardo
    orchid: 0000-0002-7829-4403
    affiliation: 1
  - name: Zekun Chen
    orchid: 0000-0002-4183-2941
    affiliation: 1
  - name: Nicholas Lundgren
    orchid: 0000-0003-1914-0954
    affiliation: 1
  - name: Dylan Folkner
    orchid: 0009-0006-5321-0900
    affiliation: 1
  - name: Nathaniel Troup
    orchid: 0000-0000-0000-0000
    affiliation: 1
  - name: Alfredo Fiorentino
    orchid: 0000-0002-3048-5534
    affiliation: 2
  - name: Davide Donadio
    orchid: 0000-0002-2150-4182
    affiliation: 1
affiliations:
  - name: University of California, Davis, USA
    index: 1
  - name: Scuola Internazionale Superiore di Studi Avanzati (SISSA), Trieste, Italy
    index: 2
date: 2024
bibliography: literature.bib
---

# Summary
kALDo is versatile and scalable open-source software for predicting thermal properties using Lattice Dynamics. It simulates the properties of crystalline and non-crystalline solids using various numerical methods, such as the Boltzmann Transport Equation (BTE) and the Quasi-Harmonic Green-Kubo (QHGK) approach. kALDo incorporates simulation tools to simulate elastic properties, isotopic effects, finite-size effects, and non-analytical term corrections and integrates seamlessly with popular MLIP frameworks and ab-initio codes like LAMMPS and Quantum Espresso. 

kALDo integrates modern theories of thermal transport with state-of-the-art software development practices to handle complex calculations efficiently. It leverages GPU acceleration and optimized tensor libraries built on TensorFlow to ensure computational efficiency across multicore and multiprocessor environments. It is designed to be user-friendly and provides extensive documentation, examples, and tutorials alongside several visualization tools to plot key physical properties. Its open-source nature encourages community contributions and continuous development, making it a reliable platform for researchers and developers in chemistry, material science, and physics.


# Introduction

We are currently living in a golden age of material design and discovery propelled by significant and synergistic advancements in machine learning and large-scale simulations. On one hand, scientists and engineers were able to define new descriptors for atomic and molecular systems (cite) and develop novel numerical techniques to train and model machine learning interatomic potentials (MLIP) (cite NEP). Thanks to the MLIP is it now possible to study materials with ab-initio accuracy and transferability with a fraction of the computational cost.
On the other hand, the availability of more computational resources allowed the community to bring ab-initio simulations to unprecedented scales, thus creating large materials datasets (cite Material Project) and foundational models that allow us to explore the composition space, pressure space, and temperature space and accurately predict energy, force, and pressure for existing and new materials (cite MatterSim).
Understanding thermal transport is crucial to designing next-generation microelectronic devices, facilitating thermal management, and achieving efficient energy storage and conversion. 
In insulators and semiconductors, quantized lattice vibrations, known as phonons, provide the main thermal transport mechanism, which is customarily modeled through anharmonic lattice dynamics (ALD) and the Boltzmann transport Equation (BTE) applied to phonons. The ALD|BTE has emerged as a powerful numerical method to predict the thermal properties of crystalline materials described at an atomic or molecular level. It is often preferred to MD because it can natively include quantum effects. Moreover, it scales differently than MD, i.e., not with the number of atoms but with the number of phonons. 
In recent years, there has been another critical development in the ALD|BTE applicability, as the theory has been extended to partially disordered and amorphous systems. While, this approach was previously limited to periodic systems, two theories, the Quasi-Harmonic Green-Kubo (QHGK)[@Isaeva.2019] and the Wigner Boltzmann Transport Equation (WBTE) developed a unified approach valid for both crystalline and disordered systems. These two formalisms have unified the main lattice dynamics approaches for both crystalline and disordered systems, bridging the gap between the ALD|BTE and the purely harmonic Allen-Feldman (AF) formula for glasses. Notably, they have highlighted the role of the interband contributions, due to pairs of phonons with the same wave vector but different band indices, in influencing thermal conductivity.5,6
ALD|BTE has also been applied to materials that are sensitive to temperature changes, like Perovskites. The framework can be helpful to improve the understanding and optimization of these materials for practical applications, such as solar cells, LEDs, and sensors. ALD|BTE can be combined with methods like Temperature Dependent Effective Potentials (TDEP)7,8 and the Stochastic Self-Consistent Harmonic Approximation (SSCHA)9 to estimate the properties of these complex materials, where temperature can affect their structural stability and electronic properties.
The workflow for using ALD to simulate thermal transport properties can be described in three steps.
The first step consists of the calculations of the force constants and their derivatives. Depending on the accuracy desired, this step can be done using ab-initio calculations, MLIP, or semi-empirical potentials.
We obtain the dynamical matrix by scaling the second-order derivatives of the interatomic potential by the atomic mass. The phonon picture emerges from its diagonalization. The eigenvalues, or modes, represent their frequencies, and using the Bose-Einstein distribution, we can calculate their population and related quantities as the heat capacity. Phonons are quasi-particles, and they can be interpreted as both particles and waves. As a consequence, we can define the phonon velocity, as the derivative of the frequency, with respect to the wave-vector.  While the quasi-particle picture tends to break down in the presence of disorder, the concept of phonon velocities can be extended to non-crystalline materials by using generalized velocities as defined in QHGK.
In the second step, the anharmonic effects are included in the scattering term and are calculated from the higher-order derivatives of the interatomic potential. In most cases, the third-order derivatives of the interatomic potential are enough to describe the system's most relevant physics. In the phonon space, they represent the processes of two phonons combining into a single one, known as phonon annihilation, or one phonon splitting into two, i.e. phonon creation. The diagonal elements of the scattering matrix correspond to the inverse of the phonon lifetime. In this second step, we also enforce physical constraints, like the conservation of phonon energy and momentum. 
In the last step, once the scattering matrix or the phonon lifetimes are obtained, the BTE can be solved, and the thermal conductivity is obtained. The BTE can be applied to the estimation of thermal properties for finite-size nanostructures, through several physical approximations, including Matthiessen's rule, the McKelvey-Schockley approach, and the relaxons theory.


# Statement of need
Large-scale chemistry, material science, and physics simulations can be enabled by the ALD|BTE approach, which has become more and more powerful in recent years. There is a need for comprehensive software to implement modern theories of thermal transport, in a scalable and community-driven approach. 
kALDo offers a solution to this need, combining several interfaces to modern ML and large-scale ab-initio engines with state-of-the-art software development practices. The software implements modern theories of thermal transport optimized to run on modern parallel architectures. kALDo's smart memory CPU allocation and GPU acceleration, make it suitable for high-performance computing environments. The goal of this open-source project is to leverage this codebase as a platform for the development and adoption of thermal transport numerical methods.

## Features

### Interfaces and Integration
kALDo aims to seamlessly integrate with modern computational chemistry simulation tools.
It provides custom interfaces to popular MLIP like NEP, to the MD software LAMMPS, and the ab-initio code Quantum Espresso through a native interface to D3Q. Moreover, through the integration with the Atomic Simulation Environment (ASE), kALDo can leverage numerous other engines to calculate the interatomic potential. The software has been tested with several MLIPs through Pynep, and Caroline and comes with a Docker container to easily use it on any environment, in combination with Deepmd, and has been used in conjunction with TDEP and SSCHA 
The repository: …..provides several examples of how to use kALDo software to perform thermal transport calculations by using several of these software. kALDo also includes Google Colab-based tutorials, that can be used in an installation-free environment for users to learn and test the framework.

### Numerical methods 
- **Non-crystalline solids.** To simulate the behavior of heat carriers in non-crystalline solids, kALDo leverages the Quasi-Harmonic Green-Kubo (QHGK) approach. This approach is based on the Green-Kubo theory and generalizes heat capacity and velocity to depend on two phonons instead of one, while the Wigner derivation is based on the Wigner Transform of the Boltzmann Transport Equation.  The two theories have been proven equivalent in the weak anharmonic limit. However, for finite anharmonicity, they present minor differences as the antiresonant terms, which can be evaluated by kALDo through a feature flag.
- **Crystalline solids.** For crystalline solids, kALDo solvers for the BTE, include the relaxation time approximation (RTA), the self-consistent iterative approach, the eigenvalue decomposition (the relaxon picture), and the direct inversion. Each method has pros and cons in terms of accuracy, memory usage, and computational time. 
- **Finite size effects.** The BTE solvers can be used to estimate finite-size effects when combined with one of the finite-size algorithms kALDo offers: the ballistic transport approximation, Matthiessen's rule, the McKelvey-Schockley approach, and the relaxons picture.
- **Isotopic effects.** kALDo can account for isotopic effects with Tamura’s formula, which influences phonon scattering and thermal conductivity. By incorporating different isotopic compositions, kALDo can model how isotopic variations impact the vibrational properties and heat transport in materials.
- **Non-analytical term correction.** kALDo incorporates non-analytical term corrections to account for long-range Coulomb interactions in polar materials. This correction is important for accurately modeling the splitting of longitudinal and transverse optical phonon modes at the Brillouin zone center.
- **Mechanical properties.** kALDo provides insights into material behavior under stress by using the dispersion relations and evaluating strain-induced frequency shifts to derive the elastic constants.

### Tools
kALDo offers several tools to optimize the performance of the simulations at each stage, both in terms of resource logging and in terms of tools for validating the consistency of the physics. 
Regarding the latter, several features help evaluate and improve the intermediate stages of a simulation:
- **Sigma-3.** Depending on the interatomic potential surface, we can approximate the Taylor expansion of the potential to the third-order in the potential coupling or decide to go beyond that. The sigma-3 method offers a good metric to inform this decision.
- **Acoustic sum rule.** A method to enforce that the sum of the acoustic phonon branches at the Gamma point is zero, which is essential for maintaining the physical accuracy of the computed phonon spectrum and ensuring the stability of the crystal lattice.
- **Conservation of the detail balance.**  The detailed balance enforces the conservation of the number of quasiparticles and can be used to evaluate the quality of our calculations and affect important physical properties of the system, for example, the hermiticity of the scattering matrix.
Moreover, kALDo has several visualization tools that help understand the main physics of the system. This includes helpers to plot the dispersion relation, the velocities, the heat capacity, the lifetime, the conductivity, and the partial density of state, at a mode-by-mode resolution. 

### Efficiency and Scalability
Designed with high performance in mind, kALDo optimizes resource usage across memory and CPU, leveraging GPU acceleration to enhance computational efficiency. The framework is built using Tensorflow, to be scalable, handling complex calculations across multicore and multiprocessor computing environments.

### Users and Developers Support 
A core design principle of kALDo is user accessibility. It provides extensive documentation, examples, and tutorials to assist new users in mastering the software. Developers are encouraged to implement more features and numerical methods and contribute to the development of this open-source software. 
In the kALDo repo, several docker containers are ready with software pre-installed and configured to work with popular simulation engines like DeepMD.

# Acknowledgements

G.B. gratefully acknowledge support by the Investment Software Fellowships (grant No. ACI-1547580-479590) of the NSF Molecular Sciences Software Institute (grant No. ACI-1547580) at Virginia Tech.

   
# References

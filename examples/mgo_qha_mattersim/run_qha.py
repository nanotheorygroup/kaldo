"""
Example: MgO QHA calculation using kaldo's functional QHA API.

This example demonstrates how to use the functional QHA interface
to perform QHA calculations on MgO using the MatterSim calculator.
"""

import numpy as np
import torch
from ase.build import bulk
from mattersim.forcefield import MatterSimCalculator
from kaldo.quasiharmonic import (
    calculate_qha,
    get_structure_at_temperature,
    get_volumetric_thermal_expansion,
    save_qha_results,
    load_qha_results
)
from kaldo.controllers.plotter import plot_qha

# Setup calculator
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running MatterSim on {device}")
calculator = MatterSimCalculator(device=device)

# Create MgO structure
atoms = bulk("MgO", "rocksalt", a=4.19)

# Step 1: Initial QHA at 0 K to get ZPE-optimized structure
print("\n" + "="*60)
print("Step 1: Optimizing structure with zero-point energy")
print("="*60)

results_zpe = calculate_qha(
    atoms=atoms,
    calculator=calculator,
    temperatures=[0],
    supercell=(8, 8, 8),
    kpts=(8, 8, 8),
    symmetry='cubic',  # MgO rocksalt is cubic
    lattice_range=0.01,
    n_lattice_points=11,
    storage='numpy',
    folder='qha_zpe_output'
)

# Get ZPE-optimized structure
atoms_zpe = get_structure_at_temperature(results_zpe, atoms, 0)
print(f"\nZPE-optimized lattice constant: {atoms_zpe.cell.lengths()[0]:.4f} Å")
print(f"Original lattice constant: {atoms.cell.lengths()[0]:.4f} Å")

# Optionally save the structure
# from ase.io import write
# write('zpe_opt_MgO.xyz', atoms_zpe, format='extxyz')

# Step 2: Full QHA from 0 to 300 K
print("\n" + "="*60)
print("Step 2: Running QHA from 0 to 300 K")
print("="*60)

results = calculate_qha(
    atoms=atoms_zpe,  # Use ZPE-optimized structure
    calculator=calculator,
    temperatures=np.linspace(0, 300, 11),
    supercell=(8, 8, 8),
    kpts=(8, 8, 8),
    symmetry='cubic',
    lattice_range=0.01,
    n_lattice_points=11,
    storage='numpy',
    folder='qha_output'
)

# Step 3: Display and plot results
print("\n" + "="*60)
print("QHA Results Summary")
print("="*60)

# NOTE: For orthogonal systems (cubic, tetra, ortho), the returned
# 'lattice_constants' are the diagonal elements (a, b, c) of the magnitude
# matrix M in the L = MD decomposition. For primitive orthogonal cells,
# these directly correspond to the lattice constants. To get the full
# structure at a given temperature, use get_structure_at_temperature().

temperatures = results['temperatures']
lattice_constants = results['lattice_constants'].flatten()
free_energies = results['free_energies']
alpha_L = results['thermal_expansion'].flatten()
alpha_V = get_volumetric_thermal_expansion(results['thermal_expansion'])

print(f"\n{'T (K)':<10} {'a (Å)':<12} {'F (meV/atom)':<15} "
      f"{'α_L (10⁻⁶/K)':<15} {'α_V (10⁻⁶/K)':<15}")
print("-" * 75)

for i in range(len(temperatures)):
    print(f"{temperatures[i]:<10.1f} {lattice_constants[i]:<12.4f} "
          f"{free_energies[i]:<15.4f} {alpha_L[i]*1e6:<15.3f} "
          f"{alpha_V[i]*1e6:<15.3f}")

# Save results
save_qha_results(results, 'qha_results.npz')
print("\nResults saved to qha_results.npz")

# Create plots using the plotting module
print("\nGenerating plots...")
plot_qha(results, folder='qha_plots', is_showing=True)

print("\nQHA calculation completed successfully!")
print("\nTo load results later:")
print("  from kaldo.quasiharmonic import load_qha_results")
print("  results = load_qha_results('qha_results.npz')")

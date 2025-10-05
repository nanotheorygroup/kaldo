# Example: Crystal Silicon - Comprehensive Phonon Property Visualization
# Computes: Phonon dispersion, DOS, thermal properties, and thermal conductivity
# Uses: kaldo built-in test data
# Demonstrates: Plotter.plot_crystal() method for crystalline materials

"""
This example demonstrates how to visualize phonon properties for crystalline materials
using the new Plotter class. It generates comprehensive plots including:
- Phonon dispersion and density of states (DOS)
- Heat capacity, group velocity, and phase space vs frequency
- Lifetime, scattering rate, and mean free path vs frequency
- Per-mode and cumulative thermal conductivity
"""

from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.controllers.plotter import Plotter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# Set matplotlib style for better-looking plots
plt.style.use('seaborn-v0_8-poster')

print("="*70)
print("Crystal Silicon - Phonon Property Visualization Example")
print("="*70)

### Load force constants from test data ###
print("\n[1/4] Loading force constants...")
forceconstants = ForceConstants.from_folder(
    folder="kaldo/tests/si-crystal",
    supercell=[3, 3, 3],
    format="eskm"
)

### Configure and create phonon object ###
print("[2/4] Creating phonon object...")
phonons = Phonons(
    forceconstants=forceconstants,
    kpts=[5, 5, 5],           # k-point mesh density
    is_classic=False,          # Use quantum statistics
    temperature=300,           # Temperature in Kelvin
    storage="memory",          # Store in memory for fast computation
)

# Trigger frequency calculation
_ = phonons.frequency
print(f"    - Number of k-points: {phonons.n_k_points}")
print(f"    - Number of modes: {phonons.n_modes}")
print(f"    - Total phonons: {phonons.n_phonons}")

### Note: Conductivity will be calculated during plotting ###
print("[3/4] Ready to generate plots...")
print("    (Thermal conductivity will be calculated during plotting using 'inverse' method)")

### Generate comprehensive plots ###
print("[4/4] Generating comprehensive plots...")
print("    This will create a multi-panel figure showing:")
print("    • Phonon dispersion and density of states")
print("    • Heat capacity, group velocity, phase space")
print("    • Lifetime, scattering rate, mean free path")
print("    • Per-mode and cumulative thermal conductivity")

# Create Plotter instance
plotter = Plotter(phonons)

# Generate all crystal plots
# This creates a comprehensive visualization with all phonon properties
plotter.plot_crystal(
    is_showing=True,           # Display the plot window
    n_k_points=300,            # Number of k-points for dispersion
    bandwidth=0.05,            # Bandwidth for DOS (THz)
    n_points=200,              # Number of points for frequency-based plots
    symprec=1e-3               # Symmetry precision for band structure
)

print("\n" + "="*70)
print("Plots generated successfully!")
print("Close the plot window to exit.")
print("="*70)

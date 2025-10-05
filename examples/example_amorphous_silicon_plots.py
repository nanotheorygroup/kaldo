# Example: Amorphous Silicon - Comprehensive Phonon Property Visualization
# Computes: DOS, thermal properties, diffusivity, and thermal conductivity
# Uses: kaldo built-in test data
# Demonstrates: Plotter.plot_amorphous() method for amorphous materials

"""
This example demonstrates how to visualize phonon properties for amorphous materials
using the new Plotter class. It generates comprehensive plots including:
- Density of states (DOS) - no dispersion for amorphous
- Heat capacity, diffusivity, and phase space vs frequency
- Participation ratio (mode localization metric)
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
print("Amorphous Silicon - Phonon Property Visualization Example")
print("="*70)

### Load force constants from test data ###
print("\n[1/4] Loading force constants...")
forceconstants = ForceConstants.from_folder(
    folder="kaldo/tests/si-amorphous",
    format="eskm"
)

### Configure and create phonon object ###
print("[2/4] Creating phonon object...")
phonons = Phonons(
    forceconstants=forceconstants,
    is_classic=True,           # Use classical statistics for amorphous
    temperature=300,           # Temperature in Kelvin
    third_bandwidth=0.05 / 4.135,  # Bandwidth for third-order (in rad/ps)
    broadening_shape="triangle",   # Shape for broadening
    storage="memory",          # Store in memory for fast computation
)

# Trigger frequency calculation
_ = phonons.frequency
print(f"    - Number of atoms: {phonons.n_atoms}")
print(f"    - Number of modes: {phonons.n_modes}")
print(f"    - Total phonons: {phonons.n_phonons}")

### Note: Conductivity will be calculated during plotting ###
print("[3/4] Ready to generate plots...")
print("    (Thermal conductivity will be calculated during plotting)")

### Generate comprehensive plots ###
print("[4/4] Generating comprehensive plots...")
print("    This will create a multi-panel figure showing:")
print("    • Density of states (no dispersion for amorphous)")
print("    • Heat capacity, diffusivity, phase space")
print("    • Participation ratio (mode localization)")
print("    • Lifetime, scattering rate, mean free path")
print("    • Per-mode and cumulative thermal conductivity")

# Create Plotter instance
plotter = Plotter(phonons)

# Generate all amorphous plots
# This creates a comprehensive visualization tailored for amorphous materials
plotter.plot_amorphous(
    is_showing=True,           # Display the plot window
    bandwidth=0.05,            # Bandwidth for DOS (THz)
    n_points=200,              # Number of points for frequency-based plots
)

print("\n" + "="*70)
print("Plots generated successfully!")
print("Close the plot window to exit.")
print("="*70)
print("\nNote: Amorphous materials show different behavior:")
print("  - No well-defined dispersion (only DOS)")
print("  - Mode localization measured by participation ratio")
print("  - Diffusivity as an important transport property")
print("  - Generally lower thermal conductivity than crystalline")
print("="*70)

"""
Vibrational Free Energy Calculation for Copper using kALDo and MatterSim

This example demonstrates how to calculate the vibrational free energy
of a copper crystal using kALDo with the MatterSim machine learning potential.
The free energy includes both the zero-point energy and thermal contributions.

The calculation uses the harmonic approximation:
    F = k_B * T * ln(1 - exp(-hbar*omega/(k_B*T))) + hbar*omega/2

External files required:
    - cu.cif: Copper crystal structure file

Dependencies:
    - kALDo
    - MatterSim
    - ASE (Atomic Simulation Environment)
    - matplotlib
    - numpy
"""

import numpy as np
from copy import copy
from ase.io import read
from ase import Atoms

# Use non-interactive backend for plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kaldo.phonons import Phonons
from kaldo.forceconstants import ForceConstants
from kaldo.controllers.plotter import plot_dispersion
from mattersim.forcefield.potential import MatterSimCalculator
from mattersim.applications.relax import Relaxer


def relax_structure(atoms: Atoms, calc) -> Atoms:
    """
    Relax the atomic structure to find the equilibrium geometry.

    Parameters
    ----------
    atoms : Atoms
        The input atomic structure
    calc : Calculator
        ASE calculator for forces and energies

    Returns
    -------
    Atoms
        The relaxed atomic structure
    """
    atoms.calc = calc

    relaxer = Relaxer(
        optimizer="BFGS",           # Optimization algorithm
        filter="ExpCellFilter",      # Allow cell shape/volume changes
        constrain_symmetry=True,     # Preserve crystal symmetry
    )

    print("Relaxing atomic structure...")
    converged, relaxed_structure = relaxer.relax(atoms, steps=500)

    if converged:
        print("Structure successfully relaxed!")
    else:
        print("Warning: Relaxation did not fully converge")

    return relaxed_structure


def calculate_vibrational_free_energy(
    atoms: Atoms,
    calc,
    temperatures: np.ndarray,
    supercell: tuple = (10, 10, 10),
    kpts: tuple = (16, 16, 16),
    delta_shift: float = 1e-2
) -> np.ndarray:
    """
    Calculate vibrational free energy across a range of temperatures.

    Parameters
    ----------
    atoms : Atoms
        The atomic structure (should be relaxed)
    calc : Calculator
        ASE calculator for computing forces
    temperatures : np.ndarray
        Array of temperatures (in Kelvin) to calculate free energy
    supercell : tuple, optional
        Supercell size for force constant calculation (default: 10x10x10)
    kpts : tuple, optional
        k-point mesh for Brillouin zone sampling (default: 16x16x16)
    delta_shift : float, optional
        Finite difference step size for force constant calculation (default: 0.01 Angstrom)

    Returns
    -------
    np.ndarray
        Vibrational free energy in eV/atom for each temperature
    """
    n_atoms = len(atoms)
    atoms.calc = copy(calc)

    # Step 1: Calculate force constants
    print(f"\nCalculating 2nd order force constants...")
    print(f"  Supercell: {supercell}")
    print(f"  Finite difference step: {delta_shift} Angstrom")

    forceconstants = ForceConstants(
        atoms=atoms,
        supercell=supercell,
        folder="force_constants"
    )
    forceconstants.second.is_acoustic_sum = True
    forceconstants.second.calculate(calc, delta_shift=delta_shift)

    # Step 2: Plot phonon dispersion
    print(f"\nPlotting phonon dispersion...")
    # Create a phonon object for dispersion plotting
    phonons_for_dispersion = Phonons(
        forceconstants=forceconstants,
        kpts=kpts,
        is_classic=False,
        temperature=300,  # Reference temperature for dispersion plot
        folder="phonons",
        storage="numpy",
    )
    plot_dispersion(phonons_for_dispersion, n_k_points=300, with_velocity=True, is_showing=False, folder=".")
    print(f"  Dispersion plots saved: dispersion_dos.png and velocity.png")

    # Step 3: Calculate free energy at each temperature
    print(f"\nCalculating vibrational free energy...")
    print(f"  k-point mesh: {kpts}")
    print(f"  Temperature range: {temperatures[0]}-{temperatures[-1]} K")

    free_energies = np.zeros(len(temperatures))

    for i, T in enumerate(temperatures):
        # Create phonon object for this temperature
        phonons = Phonons(
            forceconstants=forceconstants,
            kpts=kpts,
            is_classic=False,  # Use quantum statistics
            temperature=T,
            folder="phonons",
            storage="numpy",   # Cache results for faster recomputation
        )

        # Extract free energy for physical modes only
        # (excludes acoustic modes at Gamma point and non-physical modes)
        physical_mode = phonons.physical_mode.reshape(phonons.frequency.shape)

        # free_energy is in meV per mode, so convert to eV per atom
        f_vib = phonons.free_energy[physical_mode].sum() / 1000.0 / n_atoms
        free_energies[i] = f_vib

        print(f"  T = {T:3d} K: F_vib = {f_vib:.6f} eV/atom")

    return free_energies


def plot_free_energy(temperatures: np.ndarray, free_energies: np.ndarray):
    """
    Plot the vibrational free energy as a function of temperature.

    Parameters
    ----------
    temperatures : np.ndarray
        Temperature values in Kelvin
    free_energies : np.ndarray
        Vibrational free energy in eV/atom
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Absolute free energy
    ax1.plot(temperatures, free_energies, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel('Vibrational Free Energy (eV/atom)', fontsize=12)
    ax1.set_title('Free Energy vs Temperature', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Free energy relative to T=0 (entropy contribution)
    f_relative = free_energies - free_energies[0]
    ax2.plot(temperatures, f_relative, 'o-', linewidth=2, markersize=8, color='C1')
    ax2.set_xlabel('Temperature (K)', fontsize=12)
    ax2.set_ylabel('$\Delta F$ (eV/atom)', fontsize=12)
    ax2.set_title('Entropy Contribution (F - F$_0$)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('cu_free_energy.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'cu_free_energy.png'")
    plt.close()


def main():
    """
    Main execution function.
    """
    print("="*70)
    print("Vibrational Free Energy Calculation for Copper")
    print("="*70)

    # Step 1: Load crystal structure
    print("\nLoading copper structure from cu.cif...")
    atoms = read("cu.cif")
    print(f"  Loaded structure: {atoms.get_chemical_formula()}")
    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Cell volume: {atoms.get_volume():.2f} Angstrom^3")

    # Step 2: Set up calculator
    calc = MatterSimCalculator()

    # Step 3: Relax structure
    atoms = relax_structure(atoms, calc)

    # Step 4: Define temperature range
    temperatures = np.arange(0, 101, 10)  # 0 to 100 K in steps of 10 K

    # Step 5: Calculate free energies
    free_energies = calculate_vibrational_free_energy(
        atoms=atoms,
        calc=MatterSimCalculator(),  # Fresh calculator instance
        temperatures=temperatures,
        supercell=(10, 10, 10),
        kpts=(16, 16, 16),
        delta_shift=1e-2
    )

    # Step 6: Print summary
    print("\n" + "="*70)
    print("Summary of Results")
    print("="*70)
    print(f"Zero-point energy (T=0 K): {free_energies[0]:.6f} eV/atom")
    print(f"Free energy at 100 K:      {free_energies[-1]:.6f} eV/atom")
    print(f"Entropy contribution:      {free_energies[-1] - free_energies[0]:.6f} eV/atom")

    # Step 7: Plot results
    plot_free_energy(temperatures, free_energies)

    print("\nCalculation complete!")
    print("="*70)


if __name__ == "__main__":
    main()

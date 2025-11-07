"""
Vibrational Free Energy Calculation: kALDo vs Phonopy Comparison

This example demonstrates how to calculate the vibrational free energy
using both kALDo and Phonopy, and compares the results to validate
the kALDo implementation.

External files required:
    - cu.cif: Copper crystal structure file

Dependencies:
    - kALDo
    - Phonopy
    - MatterSim
    - ASE
    - matplotlib
    - numpy
"""

import numpy as np
from copy import copy
from ase.io import read
from ase import Atoms
from ase import units

# Use non-interactive backend for plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# kALDo imports
from kaldo.phonons import Phonons
from kaldo.forceconstants import ForceConstants

# MatterSim imports
from mattersim.forcefield.potential import MatterSimCalculator
from mattersim.applications.relax import Relaxer
from mattersim.applications.phonon import PhononWorkflow


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
        optimizer="BFGS",
        filter="ExpCellFilter",
        constrain_symmetry=True,
    )

    print("Relaxing atomic structure...")
    converged, relaxed_structure = relaxer.relax(atoms, steps=500)

    if converged:
        print("Structure successfully relaxed!")
    else:
        print("Warning: Relaxation did not fully converge")

    return relaxed_structure


def calculate_kaldo_free_energy(
    atoms: Atoms,
    calc,
    temperatures: np.ndarray,
    supercell: tuple = (10, 10, 10),
    kpts: tuple = (16, 16, 16),
    delta_shift: float = 1e-2
) -> np.ndarray:
    """
    Calculate vibrational free energy using kALDo.

    Parameters
    ----------
    atoms : Atoms
        The atomic structure (should be relaxed)
    calc : Calculator
        ASE calculator for computing forces
    temperatures : np.ndarray
        Array of temperatures (in Kelvin)
    supercell : tuple, optional
        Supercell size for force constant calculation
    kpts : tuple, optional
        k-point mesh for Brillouin zone sampling
    delta_shift : float, optional
        Finite difference step size

    Returns
    -------
    np.ndarray
        Vibrational free energy in eV/atom for each temperature
    """
    n_atoms = len(atoms)
    atoms.calc = copy(calc)

    print("\n" + "="*70)
    print("kALDo Calculation")
    print("="*70)

    # Calculate force constants
    print(f"Calculating 2nd order force constants...")
    print(f"  Supercell: {supercell}")
    print(f"  Finite difference step: {delta_shift} Angstrom")

    forceconstants = ForceConstants(
        atoms=atoms,
        supercell=supercell,
        folder="force_constants_kaldo"
    )
    forceconstants.second.is_acoustic_sum = True
    forceconstants.second.calculate(calc, delta_shift=delta_shift)

    # Calculate free energy at each temperature
    print(f"\nCalculating free energies...")
    print(f"  k-point mesh: {kpts}")
    print(f"  Temperatures: {len(temperatures)} points from {temperatures[0]} to {temperatures[-1]} K")

    free_energies = np.zeros(len(temperatures))

    for i, T in enumerate(temperatures):
        phonons = Phonons(
            forceconstants=forceconstants,
            kpts=kpts,
            is_classic=False,
            temperature=T,
            folder="phonons_kaldo",
            storage="numpy",
        )

        # Sum free energy over all modes and divide by number of atoms
        f_vib = phonons.free_energy.sum() / n_atoms
        free_energies[i] = f_vib

    print(f"  Complete! F(T=0) = {free_energies[0]:.6f} eV/atom")

    return free_energies


def calculate_phonopy_free_energy(
    atoms: Atoms,
    temperatures: np.ndarray,
    supercell: tuple = (10, 10, 10),
    kpts: tuple = (16, 16, 16),
    amplitude: float = 1e-2
) -> np.ndarray:
    """
    Calculate vibrational free energy using Phonopy.

    Parameters
    ----------
    atoms : Atoms
        The atomic structure (should be relaxed)
    temperatures : np.ndarray
        Array of temperatures (in Kelvin)
    supercell : tuple, optional
        Supercell size for force constant calculation
    kpts : tuple, optional
        k-point mesh for Brillouin zone sampling
    amplitude : float, optional
        Displacement amplitude for force constant calculation

    Returns
    -------
    np.ndarray
        Vibrational free energy in eV/atom for each temperature
    """
    print("\n" + "="*70)
    print("Phonopy Calculation")
    print("="*70)

    # Set up Phonopy workflow
    ph = PhononWorkflow(
        atoms=atoms,
        find_prim=True,  # Find primitive cell
        work_dir="./phonopy_workdir",
        amplitude=amplitude,
        supercell_matrix=np.diag(supercell),
        qpoints_mesh=np.array(kpts),
    )

    print("Running Phonopy calculation...")
    has_imag, phonons = ph.run()

    if has_imag:
        print("Warning: Phonopy found imaginary frequencies!")

    # Calculate thermal properties at the requested temperatures
    phonons.run_thermal_properties(temperatures=temperatures)

    # Extract free energies (Phonopy returns: temps, free_energy, entropy, cv)
    _, free_energy, _, _ = phonons.thermal_properties.thermal_properties

    # Convert from kJ/mol to eV/atom
    # Phonopy outputs in kJ/mol per primitive cell
    free_energies = free_energy * units.kJ / units._Nav  # kJ/mol -> eV/primitive_cell
    free_energies /= len(phonons.primitive.positions)    # eV/primitive_cell -> eV/atom

    print(f"  Complete! F(T=0) = {free_energies[0]:.6f} eV/atom")

    return free_energies


def plot_comparison(
    temperatures: np.ndarray,
    kaldo_energies: np.ndarray,
    phonopy_energies: np.ndarray
):
    """
    Create comparison plots between kALDo and Phonopy results.

    Parameters
    ----------
    temperatures : np.ndarray
        Temperature values in Kelvin
    kaldo_energies : np.ndarray
        kALDo free energies in eV/atom
    phonopy_energies : np.ndarray
        Phonopy free energies in eV/atom
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Absolute free energy comparison
    ax1.plot(temperatures, kaldo_energies, 'o-', linewidth=2, markersize=8,
             label='kALDo', color='C0')
    ax1.plot(temperatures, phonopy_energies, 's--', linewidth=2, markersize=7,
             label='Phonopy', color='C1', alpha=0.8)
    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel('Free Energy (eV/atom)', fontsize=12)
    ax1.set_title('Vibrational Free Energy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Relative free energy
    # Note: This is only the entropy contribution if temperatures[0] = 0
    # (since F(0) = ZPE only, with no thermal part)
    kaldo_rel = kaldo_energies - kaldo_energies[0]
    phonopy_rel = phonopy_energies - phonopy_energies[0]
    ax2.plot(temperatures, kaldo_rel, 'o-', linewidth=2, markersize=8,
             label='kALDo', color='C0')
    ax2.plot(temperatures, phonopy_rel, 's--', linewidth=2, markersize=7,
             label='Phonopy', color='C1', alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Temperature (K)', fontsize=12)
    ax2.set_ylabel('$\\Delta F$ (eV/atom)', fontsize=12)
    
    # Set title based on whether we start from T=0
    if temperatures[0] == 0:
        title = 'Entropy Contribution (F - F$_0$)'
    else:
        title = f'Relative Free Energy (F - F({temperatures[0]:.0f}K))'
    ax2.set_title(title, fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cu_free_energy_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'cu_free_energy_comparison.png'")
    plt.close()


def print_comparison_summary(
    temperatures: np.ndarray,
    kaldo_energies: np.ndarray,
    phonopy_energies: np.ndarray
):
    """
    Print summary statistics comparing kALDo and Phonopy results.
    """
    diff = kaldo_energies - phonopy_energies
    rel_diff = np.where(
        np.abs(phonopy_energies) > 1e-6,
        (kaldo_energies - phonopy_energies) / np.abs(phonopy_energies) * 100,
        0
    )

    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"\nZero-point energy (T=0 K):")
    print(f"  kALDo:   {kaldo_energies[0]:.6f} eV/atom")
    print(f"  Phonopy: {phonopy_energies[0]:.6f} eV/atom")
    print(f"  Diff:    {diff[0]*1000:.3f} meV/atom ({rel_diff[0]:.2f}%)")

    print(f"\nFree energy at T={temperatures[-1]} K:")
    print(f"  kALDo:   {kaldo_energies[-1]:.6f} eV/atom")
    print(f"  Phonopy: {phonopy_energies[-1]:.6f} eV/atom")
    print(f"  Diff:    {diff[-1]*1000:.3f} meV/atom ({rel_diff[-1]:.2f}%)")

    print(f"\nStatistics over all temperatures:")
    print(f"  Mean absolute diff:  {np.mean(np.abs(diff))*1000:.3f} meV/atom")
    print(f"  Max absolute diff:   {np.max(np.abs(diff))*1000:.3f} meV/atom")
    print(f"  Mean relative diff:  {np.mean(np.abs(rel_diff)):.2f}%")
    print(f"  Max relative diff:   {np.max(np.abs(rel_diff)):.2f}%")

    print("\n" + "="*70)


def main():
    """
    Main execution function.
    """
    print("="*70)
    print("Vibrational Free Energy: kALDo vs Phonopy Comparison")
    print("="*70)

    # Step 1: Load and relax structure
    print("\nLoading copper structure from cu.cif...")
    atoms = read("cu.cif")
    print(f"  Loaded: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

    calc = MatterSimCalculator()
    atoms = relax_structure(atoms, calc)

    # Step 2: Define parameters
    temperatures = np.arange(0, 101, 10)  # 0 to 100 K
    supercell = (10, 10, 10)
    kpts = (16, 16, 16)
    delta_shift = 1e-2

    # Step 3: Calculate with kALDo
    kaldo_energies = calculate_kaldo_free_energy(
        atoms=atoms,
        calc=MatterSimCalculator(),
        temperatures=temperatures,
        supercell=supercell,
        kpts=kpts,
        delta_shift=delta_shift
    )

    # Step 4: Calculate with Phonopy
    phonopy_energies = calculate_phonopy_free_energy(
        atoms=atoms,
        temperatures=temperatures,
        supercell=supercell,
        kpts=kpts,
        amplitude=delta_shift
    )

    # Step 5: Compare results
    print_comparison_summary(temperatures, kaldo_energies, phonopy_energies)

    # Step 6: Plot comparison
    plot_comparison(temperatures, kaldo_energies, phonopy_energies)

    print("\nComparison complete!")
    print("="*70)


if __name__ == "__main__":
    main()

# `copper_free_energy_mattersim`

This example demonstrates how to calculate the vibrational free energy of a copper crystal using kALDo with the MatterSim machine learning potential.

## Overview

The vibrational free energy is calculated using the harmonic approximation:

```
F = k_B * T * ln(1 - exp(-hbar*omega/(k_B*T))) + hbar*omega/2
```

where:
- `k_B`: Boltzmann constant
- `T`: temperature
- `hbar`: reduced Planck constant
- `omega`: angular frequency

The free energy includes two contributions:
1. **Zero-point energy (ZPE)**: `hbar*omega/2` - the quantum mechanical ground state energy
2. **Thermal contribution**: `k_B*T*ln(1 - exp(-hbar*omega/(k_B*T)))` - temperature-dependent entropy effects

## Requirements

External files required:
- `cu.cif`: Copper crystal structure file (FCC, space group Fm-3m)

Python packages:
- kALDo
- MatterSim (machine learning potential)
- ASE (Atomic Simulation Environment)
- NumPy
- Matplotlib

## Installation

Install MatterSim:
```bash
pip install mattersim
```

Make sure kALDo is properly installed following the main installation instructions.

## Running the Example

### Script 1: Basic Free Energy Calculation

The script `1_cu_free_energy.py` proceeds as follows:

1. **Load structure**: Read copper crystal structure from `cu.cif`

2. **Relax geometry**: Use MatterSim potential to find the equilibrium structure

3. **Calculate force constants**: Compute 2nd order force constants using finite differences
   - Supercell: 10×10×10
   - Finite difference step: 0.01 Angstrom

4. **Plot phonon dispersion**: Visualize phonon bands along high-symmetry path
   - Dispersion relation with density of states (DOS) panel
   - Group velocity along high-symmetry path
   - 300 k-points for smooth band structure

5. **Compute free energies**: Calculate vibrational free energy from 0 to 100 K
   - k-point mesh: 16×16×16
   - Quantum statistics (Bose-Einstein)
   - Physical modes only (excludes acoustic modes at Γ)

6. **Visualize results**: Generate plots showing:
   - Absolute free energy vs temperature
   - Entropy contribution (F - F₀) vs temperature

### Script 2: Comparison with Phonopy

The script `2_cu_free_energy_comparison.py` compares kALDo and Phonopy calculations:

1. **Calculate with kALDo**: Compute free energies using kALDo
2. **Calculate with Phonopy**: Compute free energies using Phonopy workflow
3. **Compare results**: Generate comparison plots showing:
   - Absolute free energy from both methods
   - Entropy contribution from both methods
4. **Statistical summary**: Print mean and maximum differences

To run these examples, navigate to this directory and execute:

```bash
# Basic calculation
python 1_cu_free_energy.py

# Comparison with Phonopy
python 2_cu_free_energy_comparison.py
```

## Output

The script generates:

### Terminal output
- Detailed progress information
- Force constant calculation status
- Free energy values at each temperature
- Summary of results

### Files

From `1_cu_free_energy.py`:
- `cu_free_energy.png`: Plots of free energy vs temperature
- `dispersion_dos.png`: Phonon dispersion relation with DOS panel
- `velocity.png`: Phonon group velocity along high-symmetry path
- `force_constants/`: Directory containing kALDo force constants (cached)
- `phonons/`: Directory containing kALDo phonon data (cached)

From `2_cu_free_energy_comparison.py`:
- `cu_free_energy_comparison.png`: 2-panel comparison plot (kALDo vs Phonopy)
- `force_constants_kaldo/`: kALDo force constants
- `phonons_kaldo/`: kALDo phonon data
- `phonopy_workdir/`: Phonopy working directory and output files

### Example output
```
======================================================================
Vibrational Free Energy Calculation for Copper
======================================================================

Loading copper structure from cu.cif...
  Loaded structure: Cu
  Number of atoms: 1
  Cell volume: 47.24 Angstrom^3

Relaxing atomic structure...
Structure successfully relaxed!

Calculating 2nd order force constants...
  Supercell: (10, 10, 10)
  Finite difference step: 0.01 Angstrom

Calculating vibrational free energy...
  k-point mesh: (16, 16, 16)
  Temperature range: 0-100 K
  T =   0 K: F_vib = 0.030936 eV/atom
  T =  10 K: F_vib = 0.030935 eV/atom
  T =  20 K: F_vib = 0.030930 eV/atom
  ...
  T = 100 K: F_vib = 0.027558 eV/atom

======================================================================
Summary of Results
======================================================================
Zero-point energy (T=0 K): 0.030936 eV/atom
Free energy at 100 K:      0.027558 eV/atom
Entropy contribution:      -0.003378 eV/atom
```

## Physical Interpretation

### Zero-Point Energy (T=0 K)
The free energy at T=0 K is purely the zero-point energy (~0.031 eV/atom for Cu). This represents the minimum quantum mechanical energy of the vibrational modes.

### Temperature Dependence
As temperature increases:
- The thermal contribution becomes more negative (entropy effect)
- The total free energy decreases
- At high temperatures, the system explores more vibrational states

The **entropy contribution** (F - F₀) is negative and increases in magnitude with temperature, representing the -TS term in thermodynamics.

## Comparison with Phonopy

The second script (`2_cu_free_energy_comparison.py`) directly compares kALDo and Phonopy calculations using identical parameters.

### Expected Agreement
The kALDo implementation should give similar results to Phonopy (typically within ~1%) when using:
- Same supercell size
- Same k-point mesh
- Same displacement amplitude/finite difference step

### Key Implementation Differences
- **Internal units**: kALDo uses meV internally and converts to eV; Phonopy uses kJ/mol
- **Mode filtering**: kALDo explicitly filters physical modes to exclude acoustic modes at Γ
- **Primitive cell**: Phonopy automatically finds the primitive cell; kALDo uses the input cell
- **Formula**: Both use the same harmonic free energy formula

### Typical Results
For copper at 300 K with a 16×16×16 k-point mesh:
- Absolute difference: < 0.5 meV/atom
- Relative difference: < 1%
- Best agreement at low temperatures where quantum effects dominate

## Notes

- **Units**: All energies are reported in eV/atom
- **Caching**: Force constants and phonon data are cached in numpy format for faster recomputation
- **Convergence**: Results are converged with respect to the 16×16×16 k-point mesh
- **ML Potential**: MatterSim is a universal machine learning potential trained on diverse materials

## Further Reading

For more information on:
- Free energy calculations: See thermodynamics references
- kALDo: Check the main kALDo documentation
- MatterSim: Visit [MatterSim documentation](https://github.com/Microsoft/mattersim)

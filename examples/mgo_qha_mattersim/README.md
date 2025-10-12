# MgO Quasi-Harmonic Approximation (QHA) Example

Example for calculating thermal expansion of MgO using the Quasi-Harmonic Approximation with the MatterSim force field.

## Files

- **`run_qha.py`** - Main example demonstrating two-step QHA workflow (ZPE optimization + full QHA)
- **`zpe_opt_MgO.xyz`** - Zero-point energy optimized MgO structure

## Quick Start

```bash
python run_qha.py
```

For faster testing, modify the parameters in the script:
- Reduce `supercell` to `(2, 2, 2)`
- Reduce `kpts` to `(3, 3, 3)`
- Reduce `n_lattice_points` to `3`
- Use fewer temperatures: `[0, 300]`

## What is QHA?

QHA calculates thermal expansion by minimizing the free energy:

```
F(V,T) = E(V) + F_vib(V,T)
```

where `E(V)` is the static energy and `F_vib(V,T)` is the vibrational free energy (including zero-point energy).

## Output

The calculation produces:
- Lattice constants vs temperature
- Free energy vs temperature
- Thermal expansion coefficients
- Publication-quality plots in `qha_plots/`

## Supported Symmetries

| Symmetry | Parameters | Example |
|----------|------------|---------|
| `'cubic'` | `a` | MgO, Si, diamond |
| `'tetra'` | `a, c` | TiO₂ rutile |
| `'ortho'` | `a, b, c` | Many minerals |

Symmetry is auto-detected if not specified.

**Note:** The returned parameters are diagonal elements of the magnitude matrix M in the L = MD decomposition. For primitive orthogonal cells, these correspond to lattice constants. Use `get_structure_at_temperature()` to properly reconstruct the full lattice matrix.

## Requirements

- kaldo
- ase
- numpy
- scipy
- scikit-learn
- matplotlib
- mattersim

## References

For QHA methodology:
- P. Pavone et al., *Phys. Rev. B* **48**, 3156 (1993)
- A. Togo et al., *Phys. Rev. B* **91**, 094306 (2015)

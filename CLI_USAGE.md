# kALDO CLI Usage

Simple command-line interface for running kALDO thermal transport calculations.

## Quick Start

```bash
# Run with JSON file
python kaldo_cli.py example.json

# Run with stdin
python kaldo_cli.py < example.json

# Validate input only
python kaldo_cli.py --validate-only example.json

# Verbose output
python kaldo_cli.py -v example.json
```

## Input Format

Create a JSON file with the following structure:

```json
{
  "forceconstants": {
    "folder": "path/to/force/constants",
    "format": "eskm"
  },
  "phonons": {
    "temperature": 300,
    "kpts": [5, 5, 5]
  },
  "conductivity": {
    "method": "inverse",
    "n_iterations": 50
  }
}
```

## Required Parameters

- `forceconstants.folder`: Path to force constants data
- `phonons.temperature`: Temperature in Kelvin

All other parameters use automatic defaults from kALDO function signatures.

## Optional Sections

- `conductivity`: Calculate thermal conductivity
- `dispersion`: Plot phonon dispersion (if plotting enabled)

The CLI automatically parses all available parameters from kALDO class signatures.
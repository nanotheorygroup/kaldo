# kALDo CLI Tool
A command-line interface for running kALDo thermal transport calculations.
## Installation
Make the CLI script executable:
```bash
# Create the bin directory and copy the script to it
mkdir cli/bin
cp cli/kaldo_cli.py cli/bin/kaldo
chmod +x cli/bin/kaldo

# (Optional) Add the bin directory in your PATH
export PATH=$PATH:<path_to_kaldo>/cli/bin
```

## Usage
### Run a Calculation
```bash
kaldo < cli/kaldo.in
```
### Validate Configuration
```bash
kaldo --validate config.json
```
### Verbose Output
```bash
kaldo -v config.json
```

## Configuration
The CLI tool uses JSON configuration files. See the examples below for a complete configuration with all available options.
### Required Sections
- **forceconstants**: Settings for force constants calculation
  - `folder`: Path to force constants files (required)
  - `format`: Format of force constants (shengbte, hiphive, etc.)
  - `supercell`: Supercell dimensions [nx, ny, nz]
### Optional Sections
- **phonons**: Phonon calculation settings
- **conductivity**: Thermal conductivity calculation
- **plotter**: Phonon dispersion plot settings
## Examples
### Basic Thermal Conductivity Calculation
```bash
# Create a basic config file
cat > basic_config.json << EOF
{
  "forceconstants": {
    "folder": "kaldo/tests/si-crystal",
    "format": "eskm",
    "supercell": [3, 3, 3]
  },
  "phonons": {
    "temperature": 300.0,
    "kpts": [5, 5, 5]
  },
  "conductivity": {
    "method": "rta",
    "temperatures": [200, 250, 300, 350, 400]
  }
}
EOF

# Run the calculation
kaldo basic_config.json
```
### Generate Phonon Dispersion Plot
```bash
# Create a config for plotting
cat > plot_config.json << EOF
{
  "forceconstants": {
    "folder": "kaldo/tests/si-crystal",
    "format": "eskm",
    "supercell": [3, 3, 3]
  },
  "phonons": {
    "kpts": [10, 10, 10]
  },
  "plotter": {
    "output": "phonon_dispersion.png",
    "path": "GXWKG",
    "npoints": 200
  }
}
EOF

# Run the calculation
kaldo plot_config.json
```


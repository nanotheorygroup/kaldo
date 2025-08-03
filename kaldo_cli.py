#!/usr/bin/env python3
"""
kALDO CLI Tool - Improved Version

A command-line interface for running kALDO thermal transport calculations
using JSON input files with automatic parameter parsing.

Usage: python kaldo_cli.py input.json > output.log
       python kaldo_cli.py < input.json > output.log
"""

import sys
import json
import inspect
import logging
from typing import Dict, Any, Optional
import argparse


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def get_function_defaults(func) -> Dict[str, Any]:
    """Extract default parameter values from a function signature."""
    sig = inspect.signature(func)
    defaults = {}
    
    for param_name, param in sig.parameters.items():
        if param.default != inspect.Parameter.empty:
            defaults[param_name] = param.default
        elif param_name in ['self', 'args', 'kwargs']:
            continue  # Skip these special parameters
        else:
            # Required parameter (no default)
            defaults[param_name] = None
    
    return defaults


def parse_config_with_defaults(config: Dict[str, Any], defaults: Dict[str, Any], 
                              required_keys: list = None) -> Dict[str, Any]:
    """Parse configuration using function defaults, applying KISS principle."""
    if required_keys is None:
        required_keys = []
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required parameter '{key}' missing from configuration")
    
    # Create final config with defaults
    final_config = {}
    
    # Add all parameters that have defaults or are provided
    for key, default_value in defaults.items():
        if key in config:
            final_config[key] = config[key]
        elif default_value is not None:
            final_config[key] = default_value
    
    # Add any additional config keys not in defaults (for flexibility)
    for key, value in config.items():
        if key not in final_config:
            final_config[key] = value
    
    return final_config


def create_forceconstants(config: Dict[str, Any], logger: logging.Logger):
    """Create ForceConstants object using automatic parameter parsing."""
    try:
        from kaldo.forceconstants import ForceConstants
        
        # Get defaults from the from_folder class method
        defaults = get_function_defaults(ForceConstants.from_folder)
        required_keys = ['folder']  # Only folder is truly required
        
        # Parse config with automatic defaults
        parsed_config = parse_config_with_defaults(config, defaults, required_keys)
        
        # Convert lists to tuples where needed (for supercell parameters)
        for key in ['supercell', 'third_supercell']:
            if key in parsed_config and isinstance(parsed_config[key], list):
                parsed_config[key] = tuple(parsed_config[key])
        
        logger.info("Creating ForceConstants object...")
        logger.info(f"  Folder: {parsed_config['folder']}")
        logger.info(f"  Format: {parsed_config.get('format', 'default')}")
        logger.info(f"  Supercell: {parsed_config.get('supercell', 'default')}")
        
        forceconstants = ForceConstants.from_folder(**parsed_config)
        logger.info("ForceConstants created successfully")
        return forceconstants
        
    except ImportError as e:
        logger.error(f"Failed to import kALDO: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create ForceConstants: {e}")
        raise


def create_phonons(forceconstants, config: Dict[str, Any], logger: logging.Logger):
    """Create Phonons object using automatic parameter parsing."""
    try:
        from kaldo.phonons import Phonons
        
        # Get defaults from Phonons __init__ method
        defaults = get_function_defaults(Phonons.__init__)
        required_keys = ['temperature']  # Temperature is required for meaningful calculation
        
        # Parse config with automatic defaults
        parsed_config = parse_config_with_defaults(config, defaults, required_keys)
        
        # Convert lists to tuples where needed
        if 'kpts' in parsed_config and isinstance(parsed_config['kpts'], list):
            parsed_config['kpts'] = tuple(parsed_config['kpts'])
        
        # Convert g_factor to numpy array if provided and numpy is available
        if parsed_config.get('g_factor') is not None:
            try:
                import numpy as np
                parsed_config['g_factor'] = np.array(parsed_config['g_factor'])
            except ImportError:
                logger.warning("NumPy not available, g_factor will be passed as-is")
        
        logger.info("Creating Phonons object...")
        logger.info(f"  Temperature: {parsed_config['temperature']} K")
        logger.info(f"  K-points: {parsed_config.get('kpts', 'default')}")
        logger.info(f"  Classical statistics: {parsed_config.get('is_classic', 'default')}")
        
        phonons = Phonons(forceconstants=forceconstants, **parsed_config)
        logger.info("Phonons created successfully")
        return phonons
        
    except Exception as e:
        logger.error(f"Failed to create Phonons: {e}")
        raise


def calculate_conductivity(phonons, config: Dict[str, Any], logger: logging.Logger):
    """Calculate thermal conductivity using automatic parameter parsing."""
    try:
        from kaldo.conductivity import Conductivity
        
        # Get defaults from Conductivity __init__ method
        defaults = get_function_defaults(Conductivity.__init__)
        
        # Parse config with automatic defaults (no required keys for conductivity)
        parsed_config = parse_config_with_defaults(config, defaults)
        
        # Convert lists to tuples where needed
        if 'length' in parsed_config and isinstance(parsed_config['length'], list):
            parsed_config['length'] = tuple(parsed_config['length'])
        
        logger.info("Calculating thermal conductivity...")
        logger.info(f"  Method: {parsed_config.get('method', 'default')}")
        if parsed_config.get('n_iterations'):
            logger.info(f"  Iterations: {parsed_config['n_iterations']}")
        
        conductivity = Conductivity(phonons=phonons, **parsed_config)
        
        # Display results
        try:
            import numpy as np
            cond_matrix = conductivity.conductivity.sum(axis=0)
            diag = np.diag(cond_matrix)
            offdiag = np.abs(cond_matrix).sum() - np.abs(diag).sum()
            
            logger.info("Thermal conductivity calculation completed")
            logger.info(f"  Conductivity ({parsed_config.get('method', 'default')}): {np.mean(diag):.3f} W/m-K")
            logger.info(f"  Off-diagonal sum: {offdiag:.3f}")
            logger.info("  Full conductivity matrix (W/m-K):")
            for i, row in enumerate(cond_matrix):
                logger.info(f"    [{row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f}]")
        except Exception as e:
            logger.warning(f"Could not display detailed results: {e}")
            logger.info("Thermal conductivity calculation completed")
        
        return conductivity
        
    except Exception as e:
        logger.error(f"Failed to calculate conductivity: {e}")
        raise


def plot_dispersion_if_requested(phonons, config: Optional[Dict[str, Any]], logger: logging.Logger):
    """Plot dispersion relation if requested."""
    if not config or not config.get('plot', True):
        return
    
    try:
        from kaldo.controllers.plotter import plot_dispersion
        from ase.io import read
        import os
        
        # Default values
        npoints = config.get('npoints', 150)
        path_string = config.get('path', 'GXULG')
        folder = config.get('folder')
        
        if folder is None:
            folder = getattr(phonons, 'folder', 'output') + '/dispersion'
        
        logger.info("Plotting dispersion relation...")
        
        # Try to find structure file
        possible_files = ['POSCAR', 'atoms.xyz', 'structure.xyz', 'geometry.xyz']
        atoms_file = None
        fc_folder = getattr(phonons.forceconstants, 'folder', '.')
        
        for filename in possible_files:
            test_path = os.path.join(fc_folder, filename)
            if os.path.exists(test_path):
                atoms_file = test_path
                break
        
        if atoms_file:
            atoms = read(atoms_file)
            cell = atoms.cell
            path = cell.bandpath(path_string, npoints=npoints)
            
            plot_dispersion(phonons, 
                          is_showing=False,
                          manually_defined_path=path, 
                          folder=folder)
            logger.info(f"  Dispersion plot saved to: {folder}")
        else:
            logger.warning("Could not find structure file for dispersion plot")
    
    except Exception as e:
        logger.warning(f"Failed to plot dispersion: {e}")


def run_kaldo_calculation(input_config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Run the complete kALDO calculation pipeline using automatic parsing."""
    results = {}
    
    # Validate required sections
    required_sections = ['forceconstants', 'phonons']
    for section in required_sections:
        if section not in input_config:
            raise ValueError(f"Required section '{section}' missing from input")
    
    try:
        # Create ForceConstants
        forceconstants = create_forceconstants(input_config['forceconstants'], logger)
        results['forceconstants'] = forceconstants
        
        # Create Phonons
        phonons = create_phonons(forceconstants, input_config['phonons'], logger)
        results['phonons'] = phonons
        
        # Plot dispersion if requested
        if 'dispersion' in input_config:
            plot_dispersion_if_requested(phonons, input_config['dispersion'], logger)
        
        # Calculate conductivity if requested
        if 'conductivity' in input_config:
            conductivity = calculate_conductivity(phonons, input_config['conductivity'], logger)
            results['conductivity'] = conductivity
        
        logger.info("kALDO calculation completed successfully!")
        
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        raise
    
    return results


def load_input(input_file: Optional[str] = None) -> Dict[str, Any]:
    """Load input configuration from file or stdin."""
    if input_file:
        with open(input_file, 'r') as f:
            return json.load(f)
    else:
        return json.load(sys.stdin)


def main():
    """Main CLI entry point following KISS principles."""
    parser = argparse.ArgumentParser(
        description='kALDO CLI Tool with automatic parameter parsing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example JSON input file:

{
  "forceconstants": {
    "folder": "path/to/force/constants",
    "format": "shengbte",
    "supercell": [3, 3, 3]
  },
  "phonons": {
    "temperature": 300,
    "kpts": [5, 5, 5]
  },
  "conductivity": {
    "method": "inverse"
  }
}

Parameters are automatically parsed from kALDO class signatures.
Only required parameters need to be specified.
        """
    )
    
    parser.add_argument('input_file', nargs='?', 
                       help='JSON input file (default: read from stdin)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate input file, do not run calculation')
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    try:
        # Load configuration
        input_config = load_input(args.input_file)
        source = args.input_file if args.input_file else "stdin"
        logger.info(f"Loaded configuration from: {source}")
        
        if args.validate_only:
            # Basic validation
            required_sections = ['forceconstants', 'phonons']
            for section in required_sections:
                if section not in input_config:
                    raise ValueError(f"Required section '{section}' missing from input")
            
            logger.info("Input validation successful!")
            return 0
        
        # Run calculation
        run_kaldo_calculation(input_config, logger)
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
#!/usr/bin/env python
"""
kALDo CLI Tool

A command-line interface for running kALDo thermal transport calculations.

Example usage:
  # Show example configuration
  python kaldo_cli.py --example

  # Run a calculation
  python kaldo_cli.py config.json

  # Validate configuration file
  python kaldo_cli.py --validate config.json

  # Read from stdin
  cat config.json | python kaldo_cli.py
"""

import sys
import json
import logging
import argparse
import os
from typing import Dict, Any, Optional, Tuple, Union, List


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration for both CLI and kALDo."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure kALDo logger if available
    try:
        from kaldo.helpers.logger import get_logger
        kaldo_logger = get_logger()
        kaldo_logger.setLevel(level)
    except ImportError:
        # kALDo not installed or helpers.logger not available
        pass
    
    return logging.getLogger('kALDo-CLI')


def load_input(input_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and validate input configuration from file or stdin.
    
    Args:
        input_file: Path to input JSON file or '-' for stdin
        
    Returns:
        dict: Parsed configuration
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If input is not valid JSON
        ValueError: If configuration is empty or invalid
    """
    try:
        # Read from file or stdin
        if input_file and input_file != '-':
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
            with open(input_file, 'r') as f:
                config = json.load(f)
        else:
            if sys.stdin.isatty():
                raise ValueError("No input provided. Use --help for usage information.")
            config = json.load(sys.stdin)
            
        # Basic validation
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")
            
        if not config:
            raise ValueError("Empty configuration provided")
            
        return config
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            "Invalid JSON in input",
            getattr(e, "doc", "") if getattr(e, "doc", None) is not None else "",
            getattr(e, "pos", 0) if getattr(e, "pos", None) is not None else 0
        ) from None


def validate_config_section(config: Dict[str, Any], section: str, required_keys: List[str] = None) -> None:
    """Validate a configuration section.
    
    Args:
        config: Full configuration dictionary
        section: Section name to validate
        required_keys: List of required keys in the section
        
    Raises:
        ValueError: If validation fails
    """
    if section not in config:
        if required_keys:
            raise ValueError(f"Missing required section: {section}")
        return
        
    section_config = config[section]
    if not isinstance(section_config, dict):
        raise ValueError(f"{section} configuration must be a dictionary")
        
    if required_keys:
        missing = [k for k in required_keys if k not in section_config]
        if missing:
            raise ValueError(f"Missing required keys in {section}: {', '.join(missing)}")


def run_kaldo_calculation(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Run a kALDo calculation based on the provided configuration.
    
    Args:
        config: Dictionary containing the calculation configuration
        logger: Configured logger instance
        
    Returns:
        Dictionary containing calculation results
        
    Raises:
        ImportError: If required kALDo modules are not available
        ValueError: If configuration is invalid
        RuntimeError: If calculation fails
    """
    results = {}
    
    try:
        # Import kALDo modules with proper error handling
        try:
            from kaldo.forceconstants import ForceConstants
            from kaldo.phonons import Phonons
            from kaldo.conductivity import Conductivity
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "Failed to import kALDo modules. Make sure kALDo is properly installed. "
                f"Error: {e}"
            ) from None
        
        # 1. ForceConstants
        validate_config_section(config, 'forceconstants', ['folder'])
        fc_config = config.get('forceconstants', {})
        
        logger.info("Creating ForceConstants...")
        try:
            fc = ForceConstants.from_folder(**fc_config)
            results['forceconstants'] = fc
            logger.debug(f"ForceConstants created with config: {fc_config}")
        except Exception as e:
            raise RuntimeError(f"Failed to create ForceConstants: {e}") from None
        
        # 2. Phonons
        phonons_config = config.get('phonons', {})
        if not phonons_config:
            logger.warning("No phonons configuration provided, using defaults")
        
        logger.info("Creating Phonons...")
        try:
            phonons = Phonons(forceconstants=fc, **phonons_config)
            results['phonons'] = phonons
            logger.debug(f"Phonons created with config: {phonons_config}")
        except Exception as e:
            raise RuntimeError(f"Failed to create Phonons: {e}") from None
        
        # 3. Conductivity (optional)
        if 'conductivity' in config:
            cond_config = config['conductivity']
            logger.info("Calculating conductivity...")
            try:
                cond = Conductivity(phonons=phonons, **cond_config)
                results['conductivity'] = cond
                
                # Log results
                kappa = cond.conductivity.sum(axis=0)
                logger.info("Thermal conductivity (W/mK):")
                logger.info(f"{kappa}")
                    
            except Exception as e:
                logger.error(f"Failed to calculate conductivity: {e}")
                if logger.level <= logging.DEBUG:
                    import traceback
                    logger.debug(traceback.format_exc())
        
        # 4. Plotting (optional)
        if 'plotter' in config:
            plot_config = config['plotter']
            logger.info("Generating plots...")
            
            try:
                from kaldo.controllers import plotter
                import matplotlib.pyplot as plt
                
                # Handle plot_dispersion
                if 'plot_dispersion' in plot_config:
                    dispersion_config = plot_config['plot_dispersion']
                    logger.info("Generating dispersion plot...")
                    plotter.plot_dispersion(phonons, **dispersion_config)
                    logger.info("Dispersion plot completed")
                
                # Handle plot_dos
                if 'plot_dos' in plot_config:
                    dos_config = plot_config['plot_dos']
                    logger.info("Generating DOS plot...")
                    plotter.plot_dos(phonons, **dos_config)
                    logger.info("DOS plot completed")
                
                # Handle plot_vs_frequency (if needed)
                if 'plot_vs_frequency' in plot_config:
                    for plot_spec in plot_config['plot_vs_frequency']:
                        observable_name = plot_spec['observable']
                        plot_params = {k: v for k, v in plot_spec.items() if k != 'observable'}
                        logger.info(f"Generating {observable_name} vs frequency plot...")
                        
                        # Get the observable from phonons object
                        observable = getattr(phonons, observable_name)
                        plotter.plot_vs_frequency(phonons, observable, observable_name, **plot_params)
                        logger.info(f"{observable_name} vs frequency plot completed")
                
            except ImportError:
                logger.warning("Matplotlib not available, skipping plotting")
            except Exception as e:
                logger.error(f"Failed to generate plots: {e}")
                if logger.level <= logging.DEBUG:
                    import traceback
                    logger.debug(traceback.format_exc())
        
        return results
        
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        if logger.level <= logging.DEBUG:
            import traceback
            logger.debug(traceback.format_exc())
        raise



def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description='kALDo Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run calculation with config file
  python kaldo_cli.py config.json

  # Validate config file without running
  python kaldo_cli.py --validate config.json

  # Run with verbose output
  python kaldo_cli.py -v config.json

  # Read config from stdin
  cat config.json | python kaldo_cli.py
"""
    )
    
    parser.add_argument(
        'input', 
        nargs='?', 
        help='Input JSON configuration file (use - for stdin)'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        help='Enable verbose output (debug level)'
    )
    parser.add_argument(
        '--validate', 
        action='store_true',
        help='Validate configuration file without running the calculation'
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the kALDo CLI.
    
    Handles command line arguments, sets up logging, and orchestrates
    the calculation workflow.
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set up logging
        logger = setup_logging(args.verbose)
        logger.debug("kALDo CLI started")
        
        
        # Check if input is provided
        if args.input is None and sys.stdin.isatty():
            logger.error("No input provided. Use --help for usage information.")
            return 1
            
        try:
            # Load and validate configuration
            logger.info("Loading configuration...")
            config = load_input(args.input)
            
            # Validate configuration
            if not config:
                raise ValueError("Empty configuration provided")
                
            logger.debug("Configuration loaded successfully")
            
            # If only validation is requested
            if args.validate:
                logger.info("✓ Configuration is valid")
                return 0
                
            # Run the calculation
            logger.info("Starting kALDo calculation...")
            results = run_kaldo_calculation(config, logger)
            
            # Final success message
            logger.info("✓ Calculation completed successfully!")
            return 0
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return 1
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in input: {e}")
            return 1
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            if args.verbose:
                logger.debug("Detailed error:", exc_info=True)
            return 1
        except ImportError as e:
            logger.error(f"Import error: {e}")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if args.verbose:
                logger.debug("Detailed error:", exc_info=True)
            return 1
            
    except KeyboardInterrupt:
        logger.info("\nCalculation interrupted by user")
        return 1
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

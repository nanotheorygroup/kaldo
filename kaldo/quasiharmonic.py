"""
Quasi-Harmonic Approximation (QHA) module for kaldo.

This module provides functions for calculating thermodynamic properties
including thermal expansion using the quasi-harmonic approximation.
"""

import numpy as np
from ase import Atoms
from scipy.interpolate import LinearNDInterpolator
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.helpers.logger import get_logger

logging = get_logger()


# Helper functions for QHA calculations

def generate_lattice_grid(initial_lattice_matrix, symmetry, lattice_range=0.01,
                         n_lattice_points=10):
    """
    Generate a grid of lattice parameters to sample.

    Parameters
    ----------
    initial_lattice_matrix : ndarray, shape (3, 3)
        Initial lattice matrix
    symmetry : str
        Crystal symmetry: 'cubic', 'tetra', 'ortho', or 'general'
    lattice_range : float, optional
        Fractional range for lattice scan (default: 0.01)
    n_lattice_points : int, optional
        Number of points per dimension (default: 10)

    Returns
    -------
    grid : ndarray
        Grid of lattice parameters to sample
    grid_metadata : dict
        Metadata about the grid (shape, symmetry info, etc.)
    """
    if symmetry == 'cubic':
        initial_a = initial_lattice_matrix[0, 0]
        lattice_values = initial_a * np.linspace(
            1 - lattice_range,
            1 + 2 * lattice_range,
            n_lattice_points
        )
        grid = lattice_values[:, np.newaxis]
        metadata = {
            'degrees_of_freedom': 1,
            'initial_params': [initial_a],
            'param_names': ['a']
        }

    elif symmetry == 'tetra':
        initial_a = initial_lattice_matrix[0, 0]
        initial_c = initial_lattice_matrix[2, 2]
        a_values = initial_a * np.linspace(1 - lattice_range,
                                           1 + 2 * lattice_range,
                                           n_lattice_points)
        c_values = initial_c * np.linspace(1 - lattice_range,
                                           1 + 2 * lattice_range,
                                           n_lattice_points)
        grid = np.array(np.meshgrid(a_values, c_values)).T
        shape = grid.shape
        grid = grid.reshape(shape[0] * shape[1], shape[2])
        metadata = {
            'degrees_of_freedom': 2,
            'initial_params': [initial_a, initial_c],
            'param_names': ['a', 'c']
        }

    elif symmetry == 'ortho':
        initial_a = initial_lattice_matrix[0, 0]
        initial_b = initial_lattice_matrix[1, 1]
        initial_c = initial_lattice_matrix[2, 2]
        a_values = initial_a * np.linspace(1 - lattice_range,
                                           1 + 2 * lattice_range,
                                           n_lattice_points)
        b_values = initial_b * np.linspace(1 - lattice_range,
                                           1 + 2 * lattice_range,
                                           n_lattice_points)
        c_values = initial_c * np.linspace(1 - lattice_range,
                                           1 + 2 * lattice_range,
                                           n_lattice_points)
        grid = np.array(np.meshgrid(a_values, b_values, c_values)).T
        shape = grid.shape
        grid = grid.reshape(shape[0] * shape[1] * shape[2], shape[3])
        metadata = {
            'degrees_of_freedom': 3,
            'initial_params': [initial_a, initial_b, initial_c],
            'param_names': ['a', 'b', 'c']
        }

    else:
        raise ValueError(f"Unknown symmetry: {symmetry}. "
                        "Must be 'cubic', 'tetra', or 'ortho'")

    return grid, metadata


def grid_point_to_lattice_matrix(grid_point, initial_lattice_matrix,
                                 symmetry, metadata):
    """
    Convert grid point to lattice matrix using L = MD decomposition.

    L = MD where M = diag(a, b, c) and D contains lattice directions.
    QHA minimizes over M while keeping D fixed. For orthogonal systems,
    D = I, so L = M.

    Parameters
    ----------
    grid_point : ndarray
        Lattice parameters (diagonal elements of M)
    initial_lattice_matrix : ndarray, shape (3, 3)
        Initial lattice matrix
    symmetry : str
        Crystal symmetry: 'cubic', 'tetra', or 'ortho'
    metadata : dict
        Grid metadata from generate_lattice_grid

    Returns
    -------
    lattice_matrix : ndarray, shape (3, 3)
        Lattice matrix L = MD
    """
    if symmetry == 'cubic':
        a_value = grid_point[0]
        return np.eye(3) * a_value

    elif symmetry == 'tetra':
        a_value, c_value = grid_point
        lattice_matrix = np.eye(3) * a_value
        lattice_matrix[2, 2] = c_value
        return lattice_matrix

    elif symmetry == 'ortho':
        a_value, b_value, c_value = grid_point
        lattice_matrix = np.eye(3) * a_value
        lattice_matrix[1, 1] = b_value
        lattice_matrix[2, 2] = c_value
        return lattice_matrix

    else:
        raise ValueError(f"Unknown symmetry: {symmetry}")


def fit_and_minimize_polynomial(grid, free_energies, degree=4, alpha=1e-3,
                                n_fine_points=1000):
    """
    Fit a polynomial to free energy data and find the minimum.

    Parameters
    ----------
    grid : ndarray
        Grid of lattice parameters
    free_energies : ndarray
        Free energies at each grid point
    degree : int, optional
        Polynomial degree (default: 4)
    alpha : float, optional
        Regularization parameter (unused, kept for API compatibility)
    n_fine_points : int, optional
        Number of points for fine grid search (default: 1000)

    Returns
    -------
    optimized_params : ndarray
        Lattice parameters at minimum
    min_free_energy : float
        Minimum free energy value
    """
    if grid.shape[1] == 1:
        # 1D case: use numpy polyfit
        coeffs = np.polyfit(grid.flatten(), free_energies, degree)
        poly_func = np.poly1d(coeffs)

        # Generate fine grid and evaluate
        grid_min, grid_max = grid.min(), grid.max()
        fine_grid = np.linspace(grid_min, grid_max, n_fine_points)
        fine_free_energies = poly_func(fine_grid)

        # Find minimum
        min_idx = np.argmin(fine_free_energies)
        optimized_params = np.array([fine_grid[min_idx]])
        min_free_energy = fine_free_energies[min_idx]

    else:
        # Multi-dimensional case: use scipy interpolation
        # Interpolate to create smooth function
        interpolator = LinearNDInterpolator(grid, free_energies)

        # Generate fine grid
        if grid.shape[1] == 2:
            x1_min, x1_max = grid[:, 0].min(), grid[:, 0].max()
            x2_min, x2_max = grid[:, 1].min(), grid[:, 1].max()
            n_pts = int(np.sqrt(n_fine_points))
            x1_fine = np.linspace(x1_min, x1_max, n_pts)
            x2_fine = np.linspace(x2_min, x2_max, n_pts)
            X1, X2 = np.meshgrid(x1_fine, x2_fine)
            fine_grid_points = np.column_stack([X1.ravel(), X2.ravel()])
        else:  # 3D case
            x1_min, x1_max = grid[:, 0].min(), grid[:, 0].max()
            x2_min, x2_max = grid[:, 1].min(), grid[:, 1].max()
            x3_min, x3_max = grid[:, 2].min(), grid[:, 2].max()
            n_pts = int(np.cbrt(n_fine_points))
            x1_fine = np.linspace(x1_min, x1_max, n_pts)
            x2_fine = np.linspace(x2_min, x2_max, n_pts)
            x3_fine = np.linspace(x3_min, x3_max, n_pts)
            X1, X2, X3 = np.meshgrid(x1_fine, x2_fine, x3_fine, indexing='ij')
            fine_grid_points = np.column_stack([X1.ravel(), X2.ravel(), X3.ravel()])

        # Interpolate on fine grid
        fine_free_energies = interpolator(fine_grid_points)

        # Handle NaN values from extrapolation (use nearest neighbor)
        nan_mask = np.isnan(fine_free_energies)
        if nan_mask.any():
            fine_free_energies[nan_mask] = np.nanmax(fine_free_energies)

        # Find minimum
        min_idx = np.argmin(fine_free_energies)
        optimized_params = fine_grid_points[min_idx]
        min_free_energy = fine_free_energies[min_idx]

    return optimized_params, min_free_energy


def calculate_thermal_expansion(lattice_params, temperatures):
    """
    Calculate thermal expansion coefficient from lattice vs temperature.

    Parameters
    ----------
    lattice_params : ndarray
        Lattice parameter(s) vs temperature
    temperatures : ndarray
        Temperature values

    Returns
    -------
    alpha : ndarray
        Linear thermal expansion coefficient(s) (1/K)
    """
    # Need at least 2 points to calculate thermal expansion
    if len(temperatures) < 2:
        # Return zeros with same shape as input
        return np.zeros_like(lattice_params)

    if lattice_params.ndim == 1:
        # Single parameter (e.g., cubic)
        alpha = np.gradient(lattice_params, temperatures) / lattice_params
    else:
        # Multiple parameters
        alpha = np.zeros_like(lattice_params)
        for i in range(lattice_params.shape[1]):
            alpha[:, i] = (np.gradient(lattice_params[:, i], temperatures) /
                          lattice_params[:, i])

    return alpha


def create_structure_at_grid_point(atoms, grid_point, initial_lattice_matrix,
                                   symmetry, metadata):
    """
    Create an ASE Atoms object at a specific grid point.

    Parameters
    ----------
    atoms : ase.Atoms
        Original atomic structure
    grid_point : ndarray
        Point in the lattice parameter grid
    initial_lattice_matrix : ndarray, shape (3, 3)
        Initial lattice matrix
    symmetry : str
        Crystal symmetry
    metadata : dict
        Grid metadata

    Returns
    -------
    atoms_scaled : ase.Atoms
        Structure with scaled lattice
    """
    lattice_matrix = grid_point_to_lattice_matrix(
        grid_point, initial_lattice_matrix, symmetry, metadata
    )

    atoms_scaled = Atoms(
        symbols=atoms.symbols,
        scaled_positions=atoms.get_scaled_positions(),
        cell=lattice_matrix,
        pbc=atoms.pbc
    )

    return atoms_scaled


def detect_symmetry(lattice_matrix, tolerance=1e-5):
    """
    Automatically detect crystal symmetry from lattice matrix.

    Parameters
    ----------
    lattice_matrix : ndarray, shape (3, 3)
        Lattice matrix
    tolerance : float, optional
        Numerical tolerance for comparison (default: 1e-5)

    Returns
    -------
    symmetry : str
        Detected symmetry: 'cubic', 'tetra', or 'ortho'

    Raises
    ------
    ValueError
        If lattice is not orthogonal (non-orthogonal systems not supported)
    """
    # Check if lattice is diagonal (orthogonal)
    off_diagonal = lattice_matrix - np.diag(np.diagonal(lattice_matrix))
    is_orthogonal = np.allclose(off_diagonal, 0, atol=tolerance)

    if not is_orthogonal:
        raise ValueError(
            "Non-orthogonal lattice detected. QHA currently only supports "
            "orthogonal crystal systems (cubic, tetragonal, orthorhombic). "
            "Please use an orthogonal unit cell."
        )

    a, b, c = np.diagonal(lattice_matrix)

    # Check degeneracies
    if np.allclose(a, b, rtol=tolerance) and np.allclose(b, c, rtol=tolerance):
        return 'cubic'
    elif (np.allclose(a, b, rtol=tolerance) or
          np.allclose(b, c, rtol=tolerance) or
          np.allclose(a, c, rtol=tolerance)):
        return 'tetra'
    else:
        return 'ortho'


def calculate_qha(atoms, calculator, temperatures,
                  supercell=(3, 3, 3), kpts=(12, 12, 12),
                  symmetry=None, lattice_range=0.01, n_lattice_points=10,
                  storage='numpy', folder='qha_output', is_classic=False):
    """
    Calculate thermal expansion using Quasi-Harmonic Approximation.

    This function performs QHA calculations by minimizing the free energy
    F(V,T) = E(V) + F_vib(V,T) at different temperatures, where E(V) is
    the static potential energy and F_vib(V,T) is the harmonic vibrational
    free energy (including zero-point energy).

    Parameters
    ----------
    atoms : ase.Atoms
        Input atomic structure
    calculator : ase.calculators.calculator.Calculator
        ASE calculator for energy and force calculations
    temperatures : array-like
        Array of temperatures (K) to calculate properties
    supercell : tuple, optional
        Supercell dimensions for force constant calculation (default: (3,3,3))
    kpts : tuple, optional
        k-point grid for phonon calculations (default: (12,12,12))
    symmetry : str or None, optional
        Crystal symmetry constraint: 'cubic', 'tetra', or 'ortho'.
        If None, will be auto-detected (default: None)
    lattice_range : float, optional
        Fractional range for lattice scan: scans (1-p)*a to (1+2*p)*a
        (default: 0.01)
    n_lattice_points : int, optional
        Number of lattice points per dimension (default: 10)
    storage : str, optional
        Storage format for phonon calculations (default: 'numpy')
    folder : str, optional
        Base folder for storing calculation data (default: 'qha_output')
    is_classic : bool, optional
        Use classical statistics instead of quantum (default: False)

    Returns
    -------
    dict
        Dictionary containing:
        - 'temperatures' : ndarray
            Input temperature array
        - 'lattice_constants' : ndarray
            Optimized lattice parameters for each temperature.
            Shape: (n_temperatures, n_params) where n_params depends on symmetry.

            Note: These are diagonal elements of magnitude matrix M in L = MD
            decomposition. For primitive cells, these are the lattice constants.
            Use get_structure_at_temperature() to reconstruct full lattice.
        - 'free_energies' : ndarray
            Minimized free energies (meV/atom) for each temperature
        - 'thermal_expansion' : ndarray
            Linear thermal expansion coefficient(s) α(T) in 1/K
        - 'symmetry' : str
            Crystal symmetry used
        - 'grid' : ndarray
            Lattice parameter grid used
        - 'free_energy_matrix' : ndarray
            Full free energy matrix (n_temperatures, n_grid_points)
        - 'grid_metadata' : dict
            Metadata about the grid generation including the direction matrix D

    Examples
    --------
    >>> from kaldo.quasiharmonic import calculate_qha
    >>> from ase.build import bulk
    >>> from mattersim.forcefield import MatterSimCalculator
    >>>
    >>> atoms = bulk("MgO", "rocksalt", a=4.19)
    >>> calc = MatterSimCalculator()
    >>>
    >>> results = calculate_qha(
    ...     atoms=atoms,
    ...     calculator=calc,
    ...     temperatures=np.linspace(0, 300, 11),
    ...     supercell=(8, 8, 8),
    ...     kpts=(8, 8, 8),
    ...     symmetry='cubic'
    ... )
    >>>
    >>> print(results['lattice_constants'])
    >>> print(results['thermal_expansion'])
    """
    temperatures = np.asarray(temperatures)

    # Auto-detect symmetry if not provided
    if symmetry is None:
        symmetry = detect_symmetry(atoms.cell.array)
        logging.info(f"Auto-detected symmetry: {symmetry}")

    logging.info(f"Starting QHA calculation with {len(temperatures)} "
                f"temperatures and {n_lattice_points} lattice points")
    logging.info(f"Symmetry: {symmetry}")

    # Check if structure is primitive (for orthogonal systems)
    n_atoms = len(atoms)
    if symmetry == 'cubic' and n_atoms > 1:
        # For cubic, primitive cell has 1 atom
        logging.warning(
            f"Non-primitive cell detected ({n_atoms} atoms). "
            "Returned parameters are magnitude matrix elements (a, b, c), "
            "not literal lattice constants. Use get_structure_at_temperature() "
            "to reconstruct the full lattice."
        )
    elif symmetry == 'tetra' and n_atoms > 1:
        logging.warning(
            f"Possibly non-primitive cell ({n_atoms} atoms). "
            "Verify that returned parameters correspond to actual lattice constants."
        )
    elif symmetry == 'ortho' and n_atoms > 1:
        logging.warning(
            f"Possibly non-primitive cell ({n_atoms} atoms). "
            "Verify that returned parameters correspond to actual lattice constants."
        )

    # Generate lattice grid
    grid, grid_metadata = generate_lattice_grid(
        atoms.cell.array,
        symmetry,
        lattice_range,
        n_lattice_points
    )

    n_grid_points = len(grid)
    n_temperatures = len(temperatures)
    logging.info(f"Total calculations: {n_grid_points} grid points × "
                f"{n_temperatures} temperatures = "
                f"{n_grid_points * n_temperatures}")

    # Initialize free energy matrix
    free_energy_matrix = np.zeros((n_temperatures, n_grid_points))

    # Calculate free energy for each grid point
    for i_grid, grid_point in enumerate(grid):
        logging.info(f"Grid point {i_grid + 1}/{n_grid_points}: "
                    f"params = {grid_point}")

        # Create structure at this lattice point
        atoms_scaled = create_structure_at_grid_point(
            atoms, grid_point, atoms.cell.array,
            symmetry, grid_metadata
        )
        atoms_scaled.calc = calculator

        # Calculate potential energy (convert to meV/atom)
        n_atoms = len(atoms_scaled)
        potential_energy = (atoms_scaled.get_total_energy() /
                           n_atoms * 1000.0)

        # Calculate force constants
        logging.info(f"  Calculating force constants...")
        fc_config = {
            'atoms': atoms_scaled,
            'supercell': supercell,
            'folder': f'{folder}/fcs_{i_grid}'
        }
        force_constants = ForceConstants(**fc_config)
        force_constants.second.is_acoustic_sum = True
        force_constants.second.calculate(calculator, delta_shift=1e-4)

        # Calculate vibrational free energy for each temperature
        for i_temp, temp in enumerate(temperatures):
            logging.info(f"  Temperature {i_temp + 1}/{n_temperatures}: "
                        f"T = {temp} K")

            phonon_config = {
                'kpts': kpts,
                'is_classic': is_classic,
                'temperature': temp,
                'folder': f'{folder}/ph_{i_grid}',
                'storage': storage
            }
            phonons = Phonons(forceconstants=force_constants,
                             **phonon_config)

            # Get vibrational free energy (convert from eV to meV)
            vibrational_free_energy = (phonons.free_energy.sum() /
                                      n_atoms * 1000.0)

            # Total free energy
            free_energy_matrix[i_temp, i_grid] = (
                potential_energy + vibrational_free_energy
            )

    # Fit and find minimum for each temperature
    logging.info("Fitting polynomials and finding minima...")
    lattice_constants = []
    free_energies = np.zeros(n_temperatures)

    for i_temp, temp in enumerate(temperatures):
        free_energies_at_temp = free_energy_matrix[i_temp, :]

        optimized_params, min_free_energy = fit_and_minimize_polynomial(
            grid,
            free_energies_at_temp,
            degree=4,
            alpha=1e-3
        )

        lattice_constants.append(optimized_params)
        free_energies[i_temp] = min_free_energy

        logging.info(f"T = {temp:6.1f} K: "
                    f"optimized params = {optimized_params}, "
                    f"F = {min_free_energy:.4f} meV/atom")

    lattice_constants = np.array(lattice_constants)

    # Calculate thermal expansion
    thermal_expansion = calculate_thermal_expansion(
        lattice_constants,
        temperatures
    )

    logging.info("QHA calculation completed successfully")

    return {
        'temperatures': temperatures,
        'lattice_constants': lattice_constants,
        'free_energies': free_energies,
        'thermal_expansion': thermal_expansion,
        'symmetry': symmetry,
        'grid': grid,
        'free_energy_matrix': free_energy_matrix,
        'grid_metadata': grid_metadata
    }


def get_volumetric_thermal_expansion(thermal_expansion):
    """
    Calculate volumetric thermal expansion from linear expansion.

    For cubic systems: α_V = 3*α_L
    For general systems: α_V = α_a + α_b + α_c

    Parameters
    ----------
    thermal_expansion : ndarray
        Linear thermal expansion coefficient(s)

    Returns
    -------
    ndarray
        Volumetric thermal expansion in 1/K
    """
    if thermal_expansion.ndim == 1:
        # Single parameter (cubic)
        return 3 * thermal_expansion
    else:
        # Multiple parameters: sum them
        return np.sum(thermal_expansion, axis=1)


def get_structure_at_temperature(qha_results, atoms, temperature):
    """
    Get the optimized structure at a specific temperature.

    Reconstructs the full lattice matrix using L = MD decomposition.

    Parameters
    ----------
    qha_results : dict
        Results dictionary from calculate_qha()
    atoms : ase.Atoms
        Original atomic structure
    temperature : float
        Temperature in K

    Returns
    -------
    ase.Atoms
        Optimized structure at the given temperature
    """
    temperatures = qha_results['temperatures']
    lattice_constants = qha_results['lattice_constants']
    symmetry = qha_results['symmetry']
    grid_metadata = qha_results['grid_metadata']

    # Find closest temperature
    i_temp = np.argmin(np.abs(temperatures - temperature))
    actual_temp = temperatures[i_temp]

    if not np.isclose(temperature, actual_temp):
        logging.warning(f"Requested T = {temperature} K, "
                      f"using closest calculated T = {actual_temp} K")

    # Get lattice parameters at this temperature
    lattice_params = lattice_constants[i_temp]

    # Create structure
    atoms_opt = create_structure_at_grid_point(
        atoms, lattice_params, atoms.cell.array,
        symmetry, grid_metadata
    )

    return atoms_opt


def save_qha_results(qha_results, filename='qha_results.npz'):
    """
    Save QHA results to a numpy compressed file.

    Parameters
    ----------
    qha_results : dict
        Results dictionary from calculate_qha()
    filename : str, optional
        Output filename (default: 'qha_results.npz')
    """
    # Convert grid_metadata dict to saveable format
    metadata = qha_results['grid_metadata']

    np.savez_compressed(
        filename,
        temperatures=qha_results['temperatures'],
        lattice_constants=qha_results['lattice_constants'],
        free_energies=qha_results['free_energies'],
        thermal_expansion=qha_results['thermal_expansion'],
        symmetry=qha_results['symmetry'],
        grid=qha_results['grid'],
        free_energy_matrix=qha_results['free_energy_matrix'],
        grid_metadata=metadata
    )
    logging.info(f"Results saved to {filename}")


def load_qha_results(filename='qha_results.npz'):
    """
    Load QHA results from a file.

    Parameters
    ----------
    filename : str, optional
        Input filename (default: 'qha_results.npz')

    Returns
    -------
    dict
        Dictionary containing loaded results
    """
    data = np.load(filename, allow_pickle=True)
    results = {key: data[key] for key in data.files}

    # Convert grid_metadata back to dict if it was stored as object
    if 'grid_metadata' in results and isinstance(results['grid_metadata'], np.ndarray):
        results['grid_metadata'] = results['grid_metadata'].item()

    logging.info(f"Results loaded from {filename}")
    return results

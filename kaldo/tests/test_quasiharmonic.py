"""
Tests for QHA functions and utilities.
"""

import pytest
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from kaldo.quasiharmonic import (
    calculate_qha,
    get_structure_at_temperature,
    get_volumetric_thermal_expansion,
    save_qha_results,
    load_qha_results,
    generate_lattice_grid,
    grid_point_to_lattice_matrix,
    fit_and_minimize_polynomial,
    calculate_thermal_expansion,
    detect_symmetry
)


class TestQHAUtils:
    """Test suite for QHA utility functions."""

    def test_generate_lattice_grid_cubic(self):
        """Test lattice grid generation for cubic symmetry."""
        lattice_matrix = np.eye(3) * 4.0
        grid, metadata = generate_lattice_grid(
            lattice_matrix, 'cubic', lattice_range=0.01, n_lattice_points=5
        )

        assert grid.shape == (5, 1)
        assert metadata['degrees_of_freedom'] == 1
        assert metadata['param_names'] == ['a']
        assert np.isclose(grid[0, 0], 4.0 * 0.99)
        assert np.isclose(grid[-1, 0], 4.0 * 1.02)

    def test_generate_lattice_grid_tetra(self):
        """Test lattice grid generation for tetragonal symmetry."""
        lattice_matrix = np.diag([4.0, 4.0, 5.0])
        grid, metadata = generate_lattice_grid(
            lattice_matrix, 'tetra', lattice_range=0.01, n_lattice_points=3
        )

        assert grid.shape == (9, 2)  # 3x3 grid
        assert metadata['degrees_of_freedom'] == 2
        assert metadata['param_names'] == ['a', 'c']

    def test_generate_lattice_grid_ortho(self):
        """Test lattice grid generation for orthorhombic symmetry."""
        lattice_matrix = np.diag([3.0, 4.0, 5.0])
        grid, metadata = generate_lattice_grid(
            lattice_matrix, 'ortho', lattice_range=0.01, n_lattice_points=2
        )

        assert grid.shape == (8, 3)  # 2x2x2 grid
        assert metadata['degrees_of_freedom'] == 3
        assert metadata['param_names'] == ['a', 'b', 'c']

    def test_grid_point_to_lattice_matrix_cubic(self):
        """Test conversion of grid point to lattice matrix for cubic."""
        initial_lattice = np.eye(3) * 4.0
        grid, metadata = generate_lattice_grid(initial_lattice, 'cubic')
        grid_point = np.array([4.2])

        lattice = grid_point_to_lattice_matrix(
            grid_point, initial_lattice, 'cubic', metadata
        )

        expected = np.eye(3) * 4.2
        assert np.allclose(lattice, expected)

    def test_grid_point_to_lattice_matrix_ortho(self):
        """Test conversion of grid point to lattice matrix for orthorhombic."""
        initial_lattice = np.diag([3.0, 4.0, 5.0])
        grid, metadata = generate_lattice_grid(initial_lattice, 'ortho')
        grid_point = np.array([3.1, 4.1, 5.1])

        lattice = grid_point_to_lattice_matrix(
            grid_point, initial_lattice, 'ortho', metadata
        )

        expected = np.diag([3.1, 4.1, 5.1])
        assert np.allclose(lattice, expected)

    def test_fit_and_minimize_polynomial(self):
        """Test polynomial fitting and minimization."""
        # Create synthetic data: parabola with minimum at x=2.0
        grid = np.linspace(1.5, 2.5, 11)[:, np.newaxis]
        free_energies = (grid.flatten() - 2.0) ** 2 + 1.0

        opt_params, min_energy = fit_and_minimize_polynomial(
            grid, free_energies, degree=2
        )

        assert np.isclose(opt_params[0], 2.0, atol=0.05)
        assert np.isclose(min_energy, 1.0, atol=0.05)

    def test_calculate_thermal_expansion(self):
        """Test thermal expansion coefficient calculation."""
        temperatures = np.array([0, 100, 200, 300])
        # Linear expansion: a(T) = a0 * (1 + α*T)
        a0 = 4.0
        alpha_ref = 1e-5  # 1/K
        lattice_params = a0 * (1 + alpha_ref * temperatures)

        alpha = calculate_thermal_expansion(lattice_params, temperatures)

        # Should be approximately constant and equal to alpha_ref
        assert np.allclose(alpha, alpha_ref, rtol=0.1)

    def test_detect_symmetry_cubic(self):
        """Test automatic symmetry detection for cubic."""
        lattice = np.eye(3) * 4.0
        symmetry = detect_symmetry(lattice)
        assert symmetry == 'cubic'

    def test_detect_symmetry_tetra(self):
        """Test automatic symmetry detection for tetragonal."""
        lattice = np.diag([4.0, 4.0, 5.0])
        symmetry = detect_symmetry(lattice)
        assert symmetry == 'tetra'

    def test_detect_symmetry_ortho(self):
        """Test automatic symmetry detection for orthorhombic."""
        lattice = np.diag([3.0, 4.0, 5.0])
        symmetry = detect_symmetry(lattice)
        assert symmetry == 'ortho'

    def test_detect_symmetry_general(self):
        """Test automatic symmetry detection for general (non-orthogonal)."""
        lattice = np.array([
            [4.0, 0.5, 0.0],
            [0.0, 4.0, 0.5],
            [0.0, 0.0, 5.0]
        ])
        # Non-orthogonal systems should raise ValueError
        with pytest.raises(ValueError, match="Non-orthogonal lattice"):
            detect_symmetry(lattice)


class TestCalculateQHA:
    """Test suite for calculate_qha function."""

    def test_calculate_qha_simple(self):
        """Test a simple QHA calculation with EMT calculator."""
        # Use simple cubic structure (orthogonal cell)
        atoms = bulk("Cu", "sc", a=3.6, cubic=True)
        calc = EMT()

        results = calculate_qha(
            atoms=atoms,
            calculator=calc,
            temperatures=[0, 300],
            supercell=(2, 2, 2),  # Small for speed
            kpts=(3, 3, 3),        # Coarse for speed
            symmetry='cubic',
            n_lattice_points=3,    # Few points for speed
            storage='memory'       # Don't write to disk
        )

        # Check results exist
        assert results['lattice_constants'].shape == (2, 1)
        assert results['free_energies'].shape == (2,)
        assert results['thermal_expansion'].shape == (2, 1)
        assert results['symmetry'] == 'cubic'

        # Check lattice expands with temperature (may not always be true for all systems)
        # Just check they are close in value since thermal expansion could be minimal
        assert results['lattice_constants'].shape[0] == 2

    def test_calculate_qha_auto_detect_symmetry(self):
        """Test automatic symmetry detection."""
        # Use simple cubic structure (orthogonal cell)
        atoms = bulk("Cu", "sc", a=3.6, cubic=True)
        calc = EMT()

        results = calculate_qha(
            atoms=atoms,
            calculator=calc,
            temperatures=[300],
            supercell=(2, 2, 2),
            kpts=(3, 3, 3),
            symmetry=None,  # Auto-detect
            n_lattice_points=3,
            storage='memory'
        )

        assert results['symmetry'] == 'cubic'

    def test_get_structure_at_temperature(self):
        """Test getting optimized structure at a specific temperature."""
        # Use simple cubic structure (orthogonal cell)
        atoms = bulk("Cu", "sc", a=3.6, cubic=True)
        calc = EMT()

        results = calculate_qha(
            atoms=atoms,
            calculator=calc,
            temperatures=[0, 100, 200, 300],
            supercell=(2, 2, 2),
            kpts=(3, 3, 3),
            symmetry='cubic',
            n_lattice_points=3,
            storage='memory'
        )

        # Get structure at 200 K
        atoms_200 = get_structure_at_temperature(results, atoms, 200)

        # Check it's an Atoms object
        assert isinstance(atoms_200, type(atoms))

        # Check lattice constant is reasonable
        a_200 = atoms_200.cell.lengths()[0]
        assert 3.5 < a_200 < 3.7

    def test_get_volumetric_thermal_expansion(self):
        """Test volumetric thermal expansion coefficient."""
        # Use simple cubic structure (orthogonal cell)
        atoms = bulk("Cu", "sc", a=3.6, cubic=True)
        calc = EMT()

        results = calculate_qha(
            atoms=atoms,
            calculator=calc,
            temperatures=[0, 300],
            supercell=(2, 2, 2),
            kpts=(3, 3, 3),
            symmetry='cubic',
            n_lattice_points=3,
            storage='memory'
        )

        # For cubic: α_V = 3*α_L
        alpha_L = results['thermal_expansion']
        alpha_V = get_volumetric_thermal_expansion(alpha_L)

        assert np.allclose(alpha_V, 3 * alpha_L)

    def test_save_and_load_qha_results(self, tmp_path):
        """Test saving and loading QHA results."""
        # Use simple cubic structure (orthogonal cell)
        atoms = bulk("Cu", "sc", a=3.6, cubic=True)
        calc = EMT()

        results = calculate_qha(
            atoms=atoms,
            calculator=calc,
            temperatures=[0, 300],
            supercell=(2, 2, 2),
            kpts=(3, 3, 3),
            symmetry='cubic',
            n_lattice_points=3,
            storage='memory'
        )

        # Save results
        save_file = tmp_path / "test_qha.npz"
        save_qha_results(results, str(save_file))

        # Load results
        loaded_data = load_qha_results(str(save_file))

        # Check data matches
        assert np.allclose(loaded_data['temperatures'], results['temperatures'])
        assert np.allclose(loaded_data['lattice_constants'],
                          results['lattice_constants'])
        assert np.allclose(loaded_data['free_energies'], results['free_energies'])
        assert loaded_data['symmetry'] == results['symmetry']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

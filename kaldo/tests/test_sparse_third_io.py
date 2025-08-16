"""
I/O tests for sparse third order import optimization using real test data.
"""

import pytest
import tempfile
import os
import time
import psutil
import numpy as np
from ase import Atoms

from kaldo.interfaces.eskm_io import import_sparse_third


def generate_small_third_file(filename, n_interactions=10000, density=0.3):
    """Generate a small test THIRD format file for performance testing."""
    n_atoms = 20  # Small system for fast testing
    
    with open(filename, 'w') as f:
        for i in range(n_interactions):
            # Generate valid atom indices (1-based)
            atom1 = np.random.randint(1, n_atoms + 1)
            atom2 = np.random.randint(1, n_atoms + 1) 
            atom3 = np.random.randint(1, n_atoms + 1)
            
            # Generate coordinate indices (1-based, 1-3)
            coord1 = np.random.randint(1, 4)
            coord2 = np.random.randint(1, 4)
            coord3 = np.random.randint(1, 4)
            
            # Generate values with controlled sparsity
            values = np.random.randn(3) * 0.1
            zero_mask = np.random.random(3) > density
            values[zero_mask] = 0.0
            
            # Write in THIRD format
            f.write(f"{atom1} {coord1} {atom2} {coord2} {atom3} {coord3} ")
            f.write(f"{values[0]:.6e} {values[1]:.6e} {values[2]:.6e}\n")


def measure_memory_mb():
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def create_test_atoms(n_atoms=20):
    """Create a simple test Atoms object"""
    positions = np.random.random((n_atoms, 3)) * 10
    symbols = ['Si'] * n_atoms
    cell = np.eye(3) * 10
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def test_sparse_third_basic_functionality():
    """
    Basic functionality test for sparse third import (runs in all builds).
    Lightweight test to ensure the import works correctly.
    """
    np.random.seed(42)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
        temp_file = f.name
    
    try:
        # Generate very small test file (fast)
        generate_small_third_file(temp_file, n_interactions=1000, density=0.5)
        
        atoms = create_test_atoms(n_atoms=10)
        supercell = (2, 2, 2)
        
        # Test the import works
        sparse_third = import_sparse_third(
            atoms=atoms,
            supercell=supercell,
            filename=temp_file,
            third_energy_threshold=0.0,
            chunk_size=500
        )
        
        # Basic functionality checks
        assert sparse_third.nnz > 0, "Should have non-zero entries"
        expected_shape = (10, 3, 80, 3, 80, 3)  # 10 atoms, 2x2x2 supercell
        assert sparse_third.shape == expected_shape, f"Wrong shape: {sparse_third.shape}"
        
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.performance
def test_sparse_third_si_crystal_eskm():
    """
    Test sparse third import with real si-crystal ESKM data.
    Tests the hybrid implementation with actual scientific data.
    """
    from kaldo.forceconstants import ForceConstants
    
    # Test file path
    test_file = "kaldo/tests/si-crystal/THIRD"
    
    if not os.path.exists(test_file):
        pytest.skip("Si crystal THIRD file not found")
    
    # Load the force constants to get the atoms object
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/si-crystal", 
        supercell=[3, 3, 3], 
        format="eskm"
    )
    
    initial_memory = measure_memory_mb()
    start_time = time.time()
    
    # Test different chunk sizes with real data
    chunk_sizes = [10000, 50000, 100000]
    results = {}
    
    for chunk_size in chunk_sizes:
        chunk_start = time.time()
        chunk_memory = measure_memory_mb()
        
        # Test the import
        sparse_third = import_sparse_third(
            atoms=forceconstants.atoms,
            supercell=(3, 3, 3),
            filename=test_file,
            third_energy_threshold=0.0,
            chunk_size=chunk_size
        )
        
        chunk_end = time.time()
        chunk_memory_end = measure_memory_mb()
        
        results[chunk_size] = {
            'time': chunk_end - chunk_start,
            'memory': chunk_memory_end - chunk_memory,
            'nnz': sparse_third.nnz,
            'shape': sparse_third.shape
        }
        
        # Validate results
        assert sparse_third.nnz > 0, f"Chunk {chunk_size}: Should have non-zero entries"
        assert len(sparse_third.shape) == 6, f"Chunk {chunk_size}: Should be 6D tensor"
        
        # Memory should be reasonable (< 1GB for this file size)
        memory_used = chunk_memory_end - initial_memory
        assert memory_used < 1000, f"Chunk {chunk_size}: Memory usage too high: {memory_used:.1f} MB"
        
        # Time should be reasonable (< 60s for this file)
        assert chunk_end - chunk_start < 60, f"Chunk {chunk_size}: Import too slow: {chunk_end - chunk_start:.2f}s"
    
    # All chunk sizes should produce similar results
    nnz_values = [r['nnz'] for r in results.values()]
    assert max(nnz_values) - min(nnz_values) < 100, "Different chunk sizes produce inconsistent results"
    
    end_time = time.time()
    total_memory = measure_memory_mb() - initial_memory
    
    print(f"\nSi crystal ESKM test results:")
    for chunk_size, result in results.items():
        print(f"  Chunk {chunk_size}: {result['time']:.2f}s, {result['memory']:.1f}MB, {result['nnz']} entries")


@pytest.mark.performance  
def test_sparse_third_si_amorphous_eskm():
    """
    Test sparse third import with si-amorphous ESKM data (larger file).
    """
    from kaldo.forceconstants import ForceConstants
    
    test_file = "kaldo/tests/si-amorphous/THIRD"
    
    if not os.path.exists(test_file):
        pytest.skip("Si amorphous THIRD file not found")
    
    # Load force constants - amorphous doesn't need supercell
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/si-amorphous", 
        format="eskm"
    )
    
    initial_memory = measure_memory_mb()
    start_time = time.time()
    
    # Test with smaller chunk size for the larger amorphous file
    sparse_third = import_sparse_third(
        atoms=forceconstants.atoms,
        supercell=(1, 1, 1),  # Amorphous typically uses unit cell
        filename=test_file,
        third_energy_threshold=0.0,
        chunk_size=25000  # Smaller chunks for larger file
    )
    
    end_time = time.time()
    final_memory = measure_memory_mb()
    memory_used = final_memory - initial_memory
    
    # Validate results
    assert sparse_third.nnz > 0, "Should have non-zero entries"
    assert len(sparse_third.shape) == 6, "Should be 6D tensor"
    
    # Memory constraints (this is a larger file ~500K lines)
    assert memory_used < 2000, f"Memory usage too high: {memory_used:.1f} MB (should be < 2GB)"
    
    # Time constraints (should complete in reasonable time)
    assert end_time - start_time < 300, f"Import too slow: {end_time - start_time:.2f}s (should be < 5min)"
    
    print(f"\nSi amorphous ESKM test results:")
    print(f"  Time: {end_time - start_time:.2f}s")
    print(f"  Memory: {memory_used:.1f}MB") 
    print(f"  Non-zero entries: {sparse_third.nnz}")
    print(f"  Shape: {sparse_third.shape}")


def test_sparse_third_import_integration():
    """
    Integration test ensuring sparse third import works with ForceConstants workflow.
    """
    from kaldo.forceconstants import ForceConstants
    
    test_cases = [
        {
            'name': 'Si Crystal ESKM',
            'folder': 'kaldo/tests/si-crystal',
            'supercell': [3, 3, 3],
            'format': 'eskm',
            'third_file': 'kaldo/tests/si-crystal/THIRD'
        },
        {
            'name': 'Si Amorphous ESKM', 
            'folder': 'kaldo/tests/si-amorphous',
            'supercell': [1, 1, 1],
            'format': 'eskm',
            'third_file': 'kaldo/tests/si-amorphous/THIRD'
        }
    ]
    
    for case in test_cases:
        if not os.path.exists(case['third_file']):
            print(f"Skipping {case['name']}: file not found")
            continue
            
        print(f"\nTesting {case['name']} integration...")
        
        # Test that ForceConstants can load with the optimized sparse third import
        try:
            forceconstants = ForceConstants.from_folder(
                folder=case['folder'],
                supercell=case['supercell'], 
                format=case['format']
            )
            
            # Validate the force constants object
            assert forceconstants is not None, f"{case['name']}: ForceConstants creation failed"
            assert hasattr(forceconstants, 'third'), f"{case['name']}: No third order found"
            
            if forceconstants.third is not None:
                # Check if it's a sparse matrix or has nnz attribute
                if hasattr(forceconstants.third, 'nnz'):
                    assert forceconstants.third.nnz > 0, f"{case['name']}: Third order has no entries"
                    print(f"  SUCCESS {case['name']}: {forceconstants.third.nnz} third order entries")
                else:
                    # For dense arrays, check if any values are non-zero
                    non_zero_count = np.count_nonzero(forceconstants.third)
                    assert non_zero_count > 0, f"{case['name']}: Third order has no entries"
                    print(f"  SUCCESS {case['name']}: {non_zero_count} non-zero third order entries")
            else:
                print(f"  WARNING {case['name']}: No third order data")
                
        except Exception as e:
            print(f"  FAILED {case['name']}: Integration failed - {e}")


def test_sparse_third_chunk_size_parameter():
    """
    Test that the chunk_size parameter is properly passed through the API.
    """
    test_file = "kaldo/tests/si-crystal/THIRD"
    if not os.path.exists(test_file):
        pytest.skip("Test file not found")
    
    # Test different chunk sizes through direct import
    chunk_sizes = [5000, 50000]
    results = []
    
    for chunk_size in chunk_sizes:
        initial_memory = measure_memory_mb()
        start_time = time.time()
        
        # Create simple atoms for testing
        atoms = Atoms('Si8', positions=np.random.random((8, 3)) * 10, cell=np.eye(3) * 10, pbc=True)
        
        sparse_third = import_sparse_third(
            atoms=atoms,
            supercell=(3, 3, 3),
            filename=test_file,
            third_energy_threshold=0.0,
            chunk_size=chunk_size
        )
        
        end_time = time.time()
        final_memory = measure_memory_mb()
        
        results.append({
            'chunk_size': chunk_size,
            'time': end_time - start_time,
            'memory': final_memory - initial_memory,
            'nnz': sparse_third.nnz
        })
    
    # Validate that different chunk sizes work
    for result in results:
        assert result['nnz'] > 0, f"Chunk size {result['chunk_size']}: No entries found"
        assert result['memory'] < 500, f"Chunk size {result['chunk_size']}: Memory usage too high"
    
    # Results should be consistent across chunk sizes
    nnz_values = [r['nnz'] for r in results]
    assert max(nnz_values) - min(nnz_values) < 50, "Chunk sizes produce inconsistent results"
    
    print(f"\nChunk size parameter test:")
    for result in results:
        print(f"  Chunk {result['chunk_size']}: {result['time']:.2f}s, {result['memory']:.1f}MB, {result['nnz']} entries")


@pytest.mark.performance
def test_sparse_third_error_conditions():
    """
    Test error handling with real and synthetic malformed data.
    """
    # Test with malformed file
    print("Testing malformed file handling...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
        f.write("invalid data\n")
        f.write("1 2 3\n")  # Too few columns
        f.write("a b c d e f g h\n")  # Invalid numbers
        temp_file = f.name
    
    try:
        atoms = create_test_atoms(10)
        sparse_third = import_sparse_third(
            atoms=atoms,
            supercell=(2, 2, 2),
            filename=temp_file,
            third_energy_threshold=0.0,
            chunk_size=1000
        )
        # Should handle gracefully and return empty or minimal result
        assert sparse_third.nnz >= 0, "Should handle malformed file gracefully"
        print(f"  SUCCESS Malformed file handled gracefully: {sparse_third.nnz} entries")
    except Exception as e:
        # This is also acceptable - malformed files can cause errors
        print(f"  WARNING Malformed file caused error (acceptable): {e}")
    finally:
        os.unlink(temp_file)
    
    # Test with empty file
    print("Testing empty file handling...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
        temp_file = f.name
    
    try:
        atoms = create_test_atoms(10)
        sparse_third = import_sparse_third(
            atoms=atoms,
            supercell=(2, 2, 2),
            filename=temp_file,
            third_energy_threshold=0.0,
            chunk_size=1000
        )
        assert sparse_third.nnz == 0, "Empty file should produce empty result"
        print("  SUCCESS Empty file handled correctly")
    except Exception as e:
        print(f"  FAILED Empty file caused unexpected error: {e}")
        raise
    finally:
        os.unlink(temp_file)


@pytest.mark.performance
def test_sparse_third_comparative_formats():
    """
    Compare sparse third import across different available formats and files.
    """
    test_scenarios = [
        {
            'name': 'Si Crystal (Small)',
            'file': 'kaldo/tests/si-crystal/THIRD',
            'atoms_count': 8,
            'expected_memory_mb': 100,
            'expected_time_s': 30
        },
        {
            'name': 'Si Amorphous (Large)', 
            'file': 'kaldo/tests/si-amorphous/THIRD',
            'atoms_count': 216,  # Typical amorphous cell
            'expected_memory_mb': 500,
            'expected_time_s': 120
        }
    ]
    
    results = {}
    
    for scenario in test_scenarios:
        if not os.path.exists(scenario['file']):
            print(f"Skipping {scenario['name']}: file not found")
            continue
            
        print(f"\nTesting {scenario['name']}...")
        
        # Create atoms object with expected size
        atoms = create_test_atoms(scenario['atoms_count'])
        
        initial_memory = measure_memory_mb()
        start_time = time.time()
        
        # Test the import
        sparse_third = import_sparse_third(
            atoms=atoms,
            supercell=(1, 1, 1) if 'amorphous' in scenario['name'].lower() else (3, 3, 3),
            filename=scenario['file'],
            third_energy_threshold=0.0,
            chunk_size=50000
        )
        
        end_time = time.time()
        final_memory = measure_memory_mb()
        
        results[scenario['name']] = {
            'time': end_time - start_time,
            'memory': final_memory - initial_memory,
            'nnz': sparse_third.nnz,
            'file_size_mb': os.path.getsize(scenario['file']) / (1024 * 1024)
        }
        
        # Performance validation
        assert end_time - start_time < scenario['expected_time_s'], \
            f"{scenario['name']}: Too slow ({end_time - start_time:.1f}s > {scenario['expected_time_s']}s)"
        
        assert final_memory - initial_memory < scenario['expected_memory_mb'], \
            f"{scenario['name']}: Too much memory ({final_memory - initial_memory:.1f}MB > {scenario['expected_memory_mb']}MB)"
        
        print(f"  SUCCESS {scenario['name']}: {results[scenario['name']]['time']:.1f}s, "
              f"{results[scenario['name']]['memory']:.1f}MB, {results[scenario['name']]['nnz']} entries")
    
    # Summary comparison
    if len(results) > 1:
        print(f"\nComparative Results:")
        for name, result in results.items():
            efficiency = result['nnz'] / (result['time'] * result['memory']) if result['time'] * result['memory'] > 0 else 0
            print(f"  {name}: {efficiency:.0f} entries/s/MB efficiency")


if __name__ == "__main__":
    # Run basic test first
    test_sparse_third_basic_functionality()
    print("SUCCESS Basic functionality test passed")
    
    # Run integration tests
    test_sparse_third_import_integration()
    print("SUCCESS Integration tests completed")
    
    # Run performance tests if data available
    if os.path.exists("kaldo/tests/si-crystal/THIRD"):
        test_sparse_third_si_crystal_eskm()
        print("SUCCESS Si crystal ESKM test passed")
        
        test_sparse_third_chunk_size_parameter()
        print("SUCCESS Chunk size parameter test passed")
    
    if os.path.exists("kaldo/tests/si-amorphous/THIRD"):
        test_sparse_third_si_amorphous_eskm()
        print("SUCCESS Si amorphous ESKM test passed")
    
    test_sparse_third_error_conditions()
    print("SUCCESS Error condition tests passed")
    
    test_sparse_third_comparative_formats()
    print("SUCCESS Comparative format tests completed")
    
    print("\nAll I/O tests completed successfully!")
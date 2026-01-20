import pytest
import numpy as np
import os
import h5py
from unittest.mock import MagicMock

from kaldo.storable import (
    lazy_property,
    is_calculated,
    LAZY_PREFIX,
    Storable,
)


# Simple load/save helper functions for legacy tests
def load(property, folder, instance, format='formatted'):
    """Simple load function for tests - fallback implementation."""
    name = folder + '/' + property
    if format == 'numpy':
        return np.load(name + '.npy', allow_pickle=True)
    elif format == 'hdf5':
        with h5py.File(name.split('/')[0] + '.hdf5', 'r') as storage:
            loaded = storage[name]
            return loaded[()]
    elif format == 'formatted':
        if property == 'physical_mode':
            loaded = np.loadtxt(name + '.dat', skiprows=1)
            return np.round(loaded, 0).astype(bool)
        elif property == 'velocity':
            loaded = []
            for alpha in range(3):
                loaded.append(np.loadtxt(name + '_' + str(alpha) + '.dat', skiprows=1))
            return np.array(loaded).transpose(1, 2, 0)
        elif property == 'mean_free_path':
            loaded = []
            for alpha in range(3):
                loaded.append(np.loadtxt(name + '_' + str(alpha) + '.dat', skiprows=1))
            return np.array(loaded).transpose(1, 2, 0)
        elif property == 'conductivity':
            loaded = []
            for alpha in range(3):
                for beta in range(3):
                    loaded.append(np.loadtxt(name + '_' + str(alpha) + '_' + str(beta) + '.dat', skiprows=1))
            return np.array(loaded).reshape((3, 3, instance.n_phonons)).transpose(2, 0, 1)
        elif '_sij' in property:
            loaded = []
            for alpha in range(3):
                loaded.append(np.loadtxt(name + '_' + str(alpha) + '.dat', skiprows=1, dtype=complex))
            return np.array(loaded).transpose(1, 0)
        else:
            if property == 'diffusivity':
                dt = complex
            else:
                dt = float
            return np.loadtxt(name + '.dat', skiprows=1, dtype=dt)
    elif format == 'memory':
        if hasattr(instance, LAZY_PREFIX + property):
            return getattr(instance, LAZY_PREFIX + property)
        else:
            raise KeyError('Property not found')
    else:
        raise ValueError('Storing format not implemented')


def save(property, folder, loaded_attr, format='formatted'):
    """Simple save function for tests."""
    name = folder + '/' + property
    if format == 'numpy':
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(name + '.npy', loaded_attr)
    elif format == 'hdf5':
        with h5py.File(name.split('/')[0] + '.hdf5', 'a') as storage:
            if not name in storage:
                storage.create_dataset(name, data=loaded_attr, chunks=True, compression='gzip', compression_opts=9)
    elif format == 'formatted':
        if not os.path.exists(folder):
            os.makedirs(folder)
        if property == 'physical_mode':
            fmt = '%d'
        else:
            fmt = '%.18e'
        if property == 'velocity':
            for alpha in range(3):
                np.savetxt(name + '_' + str(alpha) + '.dat', loaded_attr[..., alpha], fmt=fmt, header=str(loaded_attr[..., 0].shape))
        elif property == 'mean_free_path':
            for alpha in range(3):
                np.savetxt(name + '_' + str(alpha) + '.dat', loaded_attr[..., alpha], fmt=fmt,
                           header=str(loaded_attr[..., 0].shape))
        elif '_sij' in property:
            for alpha in range(3):
                np.savetxt(name + '_' + str(alpha) + '.dat', loaded_attr[..., alpha].flatten(), fmt=fmt, header=str(loaded_attr[..., 0].shape))
        elif 'conductivity' in property:
            for alpha in range(3):
                for beta in range(3):
                    np.savetxt(name + '_' + str(alpha) + '_' + str(beta) + '.dat', loaded_attr[..., alpha, beta], fmt=fmt,
                           header=str(loaded_attr[..., 0, 0].shape))
        else:
            np.savetxt(name + '.dat', loaded_attr, fmt=fmt, header=str(loaded_attr.shape))
    elif format == 'memory':
        pass  # Memory storage doesn't need file operations for tests
    else:
        raise ValueError('Storing format not implemented')


def test_load_memory_format_property_not_found():
    property = 'test_property'

    class Instance(object):
        pass

    instance = Instance()
    folder = 'test_folder'
    format = 'memory'

    with pytest.raises(KeyError):
        load(property, folder, instance, format=format)


def test_save_hdf5_format(monkeypatch):
    class MockStorage(dict):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create_dataset(self, name, data, chunks, compression, compression_opts):
            self[name] = data

    storage_instance = MockStorage()

    def mock_h5py_File(filename, mode):
        assert filename == 'test_folder.hdf5'
        assert mode == 'a'
        return storage_instance

    monkeypatch.setattr(h5py, 'File', mock_h5py_File)
    property = 'test_property'
    folder = 'test_folder'
    loaded_attr = np.array([4, 5, 6])
    format = 'hdf5'

    save(property, folder, loaded_attr, format=format)

    # Verify that the dataset was created
    assert 'test_folder/test_property' in storage_instance
    np.testing.assert_array_equal(storage_instance['test_folder/test_property'], loaded_attr)


def test_get_folder_from_label():
    class Instance(Storable):
        def __init__(self):
            self.folder = 'base_folder'
            self.kpts = [2, 2, 2]

    instance = Instance()
    folder = instance.get_folder_from_label()
    assert folder == 'base_folder/2_2_2'


def test_load_formatted_mean_free_path(monkeypatch):
    def mock_loadtxt(filename, skiprows=1):
        if filename == 'test_folder/mean_free_path_0.dat':
            return np.array([[1, 2], [3, 4]])
        elif filename == 'test_folder/mean_free_path_1.dat':
            return np.array([[5, 6], [7, 8]])
        elif filename == 'test_folder/mean_free_path_2.dat':
            return np.array([[9, 10], [11, 12]])
        else:
            raise FileNotFoundError

    monkeypatch.setattr(np, 'loadtxt', mock_loadtxt)
    property = 'mean_free_path'
    folder = 'test_folder'
    instance = MagicMock()
    format = 'formatted'
    result = load(property, folder, instance, format=format)
    expected_loaded = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[9, 10], [11, 12]]),
    ]
    expected_result = np.array(expected_loaded).transpose(1, 2, 0)
    np.testing.assert_array_equal(result, expected_result)


def test_load_formatted_conductivity(monkeypatch):
    def mock_loadtxt(filename, skiprows=1):
        mapping = {
            'test_folder/conductivity_0_0.dat': np.array([1, 2]),
            'test_folder/conductivity_0_1.dat': np.array([3, 4]),
            'test_folder/conductivity_0_2.dat': np.array([5, 6]),
            'test_folder/conductivity_1_0.dat': np.array([7, 8]),
            'test_folder/conductivity_1_1.dat': np.array([9, 10]),
            'test_folder/conductivity_1_2.dat': np.array([11, 12]),
            'test_folder/conductivity_2_0.dat': np.array([13, 14]),
            'test_folder/conductivity_2_1.dat': np.array([15, 16]),
            'test_folder/conductivity_2_2.dat': np.array([17, 18]),
        }
        return mapping[filename]

    monkeypatch.setattr(np, 'loadtxt', mock_loadtxt)
    property = 'conductivity'
    folder = 'test_folder'
    instance = MagicMock()
    instance.n_phonons = 2
    format = 'formatted'
    result = load(property, folder, instance, format=format)
    expected_result = np.array([
        [[1, 3, 5], [7, 9, 11], [13, 15, 17]],
        [[2, 4, 6], [8, 10, 12], [14, 16, 18]],
    ])
    np.testing.assert_array_equal(result, expected_result)


def test_load_formatted_sij_property(monkeypatch):
    def mock_loadtxt(filename, skiprows=1, dtype=complex):
        if filename == 'test_folder/test_sij_0.dat':
            return np.array([1 + 1j, 2 + 2j])
        elif filename == 'test_folder/test_sij_1.dat':
            return np.array([3 + 3j, 4 + 4j])
        elif filename == 'test_folder/test_sij_2.dat':
            return np.array([5 + 5j, 6 + 6j])
        else:
            raise FileNotFoundError

    monkeypatch.setattr(np, 'loadtxt', mock_loadtxt)
    property = 'test_sij'
    folder = 'test_folder'
    instance = MagicMock()
    format = 'formatted'
    result = load(property, folder, instance, format=format)
    expected_result = np.array([
        [1 + 1j, 3 + 3j, 5 + 5j],
        [2 + 2j, 4 + 4j, 6 + 6j],
    ])
    np.testing.assert_array_equal(result, expected_result)


def test_load_missing_file(monkeypatch):
    def mock_loadtxt(filename, skiprows=1, dtype=float):
        raise FileNotFoundError

    monkeypatch.setattr(np, 'loadtxt', mock_loadtxt)
    property = 'nonexistent_property'
    folder = 'test_folder'
    instance = MagicMock()
    format = 'formatted'
    with pytest.raises(FileNotFoundError):
        load(property, folder, instance, format=format)


def test_save_formatted_sij_property(monkeypatch):
    def mock_exists(path):
        return False

    monkeypatch.setattr(os.path, 'exists', mock_exists)

    def mock_makedirs(path):
        pass

    monkeypatch.setattr(os, 'makedirs', mock_makedirs)
    saved_files = {}

    def mock_savetxt(filename, data, fmt, header):
        saved_files[filename] = (data, fmt, header)

    monkeypatch.setattr(np, 'savetxt', mock_savetxt)
    property = 'test_sij'
    folder = 'test_folder'
    loaded_attr = np.array([
        [1 + 1j, 2 + 2j, 3 + 3j],
        [4 + 4j, 5 + 5j, 6 + 6j],
    ])
    format = 'formatted'
    save(property, folder, loaded_attr, format=format)
    for alpha in range(3):
        filename = f'test_folder/test_sij_{alpha}.dat'
        data, fmt, header = saved_files[filename]
        expected_data = loaded_attr[..., alpha].flatten()
        np.testing.assert_array_equal(data, expected_data)
        assert fmt == '%.18e'


def test_get_folder_from_label_with_temperature():
    class Instance(Storable):
        def __init__(self):
            self.folder = 'base_folder'
            self.kpts = [2, 2, 2]
            self.temperature = 300
            self.is_classic = False
        
        def _get_folder_path_components(self, label):
            components = []
            if '<temperature>' in label:
                components.append(str(int(self.temperature)))
            if '<statistics>' in label:
                components.append('quantum' if not self.is_classic else 'classic')
            return components

    label = '<temperature>/<statistics>'
    instance = Instance()
    folder = instance.get_folder_from_label(label)
    assert folder == 'base_folder/2_2_2/300/quantum'


def test_get_folder_from_label_with_method_and_length():
    class Instance(Storable):
        def __init__(self):
            self.folder = 'base_folder'
            self.kpts = [3, 3, 3]
            self.method = 'rta'
            self.length = [10, None, 0]
            self.finite_length_method = 'some_method'
        
        def _get_folder_path_components(self, label):
            components = []
            if '<method>' in label:
                components.append(str(self.method))
                if (self.method == 'rta' or self.method == 'sc' or self.method == 'inverse') \
                        and (self.length is not None):
                    if not (np.array(self.length) == np.array([None, None, None])).all() \
                        and not (np.array(self.length) == np.array([0, 0, 0])).all():
                        if '<length>' in label:
                            length_str = 'l'
                            for alpha in range(3):
                                if self.length[alpha] is not None:
                                    length_str += '_' + str(self.length[alpha])
                                else:
                                    length_str += '_0'
                            components.append(length_str)
                        if '<finite_length_method>' in label and self.finite_length_method is not None:
                            components.append('fs' + str(self.finite_length_method))
            return components

    label = '<method>/<length>/<finite_length_method>'
    instance = Instance()
    folder = instance.get_folder_from_label(label)
    assert folder == 'base_folder/3_3_3/rta/l_10_0_0/fssome_method'


def test_lazy_property_with_storage_formats(monkeypatch):
    # Mock the load and save functions
    def mock_load(property, folder, instance, format):
        return 'loaded_value'

    def mock_save(property, folder, loaded_attr, format):
        pass

    # Mock the instance methods instead of global functions
    def mock_load_property(self, property_name, folder, format):
        return mock_load(property_name, folder, self, format)

    def mock_save_property(self, property_name, folder, data, format):
        return mock_save(property_name, folder, data, format)

    class TestClass(Storable):
        # Define _store_formats at class level
        _store_formats = {
            'test_attr': 'formatted'
        }

        def __init__(self, storage):
            self.storage = storage
            self.folder = 'test_folder'
            self.kpts = [2, 2, 2]

        @lazy_property()
        def test_attr(self):
            return 'computed_value'

    # Apply mocking to the class
    TestClass._load_property = mock_load_property
    TestClass._save_property = mock_save_property

    # Test with 'memory' storage
    instance_memory = TestClass('memory')
    assert instance_memory.test_attr == 'computed_value'
    # Test with 'numpy' storage
    instance_numpy = TestClass('numpy')
    assert instance_numpy.test_attr == 'loaded_value'
    # Test with 'formatted' storage - should use _store_formats
    instance_formatted = TestClass('formatted')
    assert instance_formatted.test_attr == 'loaded_value'


def test_is_calculated_true():
    class Instance(object):
        pass

    instance = Instance()
    setattr(instance, LAZY_PREFIX + 'test_property', 'value')
    result = is_calculated('test_property', instance)
    assert result == True


def test_is_calculated_false(monkeypatch):
    class Instance(object):
        folder = 'test_folder'

    instance = Instance()
    # Non-StorageMixin instance should return False
    result = is_calculated('test_property', instance)
    assert result == False


def test_is_calculated_with_stored_property(monkeypatch):
    class Instance(Storable):
        folder = 'test_folder'
        storage = 'formatted'
        kpts = [2, 2, 2]
        
        def _load_property(self, property_name, folder, format):
            return 'stored_value'

    instance = Instance()
    result = is_calculated('test_property', instance)
    assert result == True
    # Ensure the property is set on the instance
    attr = LAZY_PREFIX + 'test_property'
    assert getattr(instance, attr) == 'stored_value'


def test_is_calculated_property_not_available(monkeypatch):
    class Instance(Storable):
        folder = 'test_folder'
        storage = 'formatted'
        kpts = [2, 2, 2]
        
        def _load_property(self, property_name, folder, format):
            raise FileNotFoundError

    instance = Instance()
    result = is_calculated('test_property', instance)
    assert result == False


def test_lazy_property_format_override():
    """Test that @lazy_property format parameter overrides class defaults"""

    class MockClass(Storable):
        _store_formats = {
            'test_prop': 'formatted'  # Default format
        }

        def __init__(self, storage='formatted'):
            self.storage = storage
            self.folder = 'test'
            self.kpts = [1, 1, 1]

        @lazy_property(format='numpy')  # Override to numpy
        def test_prop(self):
            return np.array([1, 2, 3])

    # Test that decorator format override works
    from kaldo.storable import _get_storage_format

    instance = MockClass()

    # Direct test of the format resolution function
    format = _get_storage_format(instance, 'test_prop', format_override='numpy')
    assert format == 'numpy'

    # Test without override (should use class default)
    format = _get_storage_format(instance, 'test_prop', format_override=None)
    assert format == 'formatted'


def test_lazy_property_storage_integration(monkeypatch, tmp_path):
    """Test improved lazy_property with object-level store formats"""
    import tempfile
    import os

    # Create a mock class similar to Phonons with _store_formats
    class MockPhonons(Storable):
        # Define storage formats at class level (like improved Phonons)
        _store_formats = {
            'frequency': 'formatted',
            'heat_capacity': 'formatted',
            'test_numpy_prop': 'numpy',
            'test_decorator_override': 'formatted'  # will be overridden by decorator
        }

        def __init__(self, storage, kpts, temperature, folder):
            self.storage = storage
            self.kpts = kpts
            self.temperature = temperature
            self.folder = folder
            self.is_classic = False
            self.n_atoms = 2
            self.n_modes = 6
            self.n_k_points = np.prod(kpts)
            self.n_phonons = self.n_k_points * self.n_modes
        
        def _get_folder_path_components(self, label):
            """Get folder path components for test MockPhonons."""
            components = []
            
            if '<temperature>' in label and hasattr(self, 'temperature'):
                components.append(str(int(self.temperature)))
                
            if '<statistics>' in label:
                if self.is_classic:
                    components.append('classic')
                else:
                    components.append('quantum')
                    
            return components

        @lazy_property(label='<temperature>/<statistics>')
        def frequency(self):
            """Mock frequency calculation - returns simple test data"""
            return np.random.rand(self.n_k_points, self.n_modes) * 10

        @lazy_property(label='')
        def heat_capacity(self):
            """Mock heat capacity calculation - returns simple test data"""
            return np.random.rand(self.n_k_points, self.n_modes) * 1e-20

        @lazy_property()
        def test_numpy_prop(self):
            """Test property that should use numpy storage"""
            return np.array([1, 2, 3, 4])

        @lazy_property(format='numpy')  # decorator override
        def test_decorator_override(self):
            """Test decorator format override"""
            return np.array([10, 20, 30])

    # Test with different storage options
    test_folder = str(tmp_path)
    kpts = [2, 2, 2]
    temperature = 300

    # Test 1: Memory storage - should compute and store in memory
    phonons_memory = MockPhonons('memory', kpts, temperature, test_folder)
    freq_memory = phonons_memory.frequency
    assert freq_memory.shape == (8, 6)  # 2*2*2 k-points, 6 modes

    # Access again - should return from memory
    freq_memory_2 = phonons_memory.frequency
    np.testing.assert_array_equal(freq_memory, freq_memory_2)

    # Test 2: Formatted storage with object-level store formats
    phonons_formatted = MockPhonons('formatted', kpts, temperature, test_folder)

    # Mock the instance methods to test file I/O
    saved_data = {}
    loaded_data = {}

    def mock_save_property(self, property_name, folder, data, format):
        saved_data[f"{folder}/{property_name}"] = (data, format)

    def mock_load_property(self, property_name, folder, format):
        key = f"{folder}/{property_name}"
        if key in loaded_data:
            return loaded_data[key]
        else:
            # First time - not found, will trigger calculation and save
            raise FileNotFoundError

    # Apply mocking to the class
    MockPhonons._save_property = mock_save_property
    MockPhonons._load_property = mock_load_property

    # First access - should calculate and save using object's _store_formats
    freq_formatted = phonons_formatted.frequency
    expected_folder = f"{test_folder}/2_2_2/300/quantum"
    assert f"{expected_folder}/frequency" in saved_data
    assert saved_data[f"{expected_folder}/frequency"][1] == 'formatted'  # Check format

    # Store the saved data for loading
    loaded_data[f"{expected_folder}/frequency"] = saved_data[f"{expected_folder}/frequency"][0]

    # Create new instance to test loading
    phonons_formatted_2 = MockPhonons('formatted', kpts, temperature, test_folder)
    freq_loaded = phonons_formatted_2.frequency
    np.testing.assert_array_equal(freq_formatted, freq_loaded)

    # Test 3: Heat capacity with simple label
    hc_formatted = phonons_formatted.heat_capacity
    expected_folder_simple = f"{test_folder}/2_2_2"
    assert f"{expected_folder_simple}/heat_capacity" in saved_data
    assert saved_data[f"{expected_folder_simple}/heat_capacity"][1] == 'formatted'

    # Test 4: Property with numpy format from _store_formats
    numpy_prop = phonons_formatted.test_numpy_prop
    assert f"{expected_folder_simple}/test_numpy_prop" in saved_data
    assert saved_data[f"{expected_folder_simple}/test_numpy_prop"][1] == 'numpy'

    # Test 5: Decorator format override
    override_prop = phonons_formatted.test_decorator_override
    assert f"{expected_folder_simple}/test_decorator_override" in saved_data
    assert saved_data[f"{expected_folder_simple}/test_decorator_override"][1] == 'numpy'  # overridden

    # Test 6: General storage setting (non-formatted)
    # Clear saved_data to test fresh numpy storage behavior
    saved_data.clear()
    loaded_data.clear()

    phonons_numpy = MockPhonons('numpy', kpts, temperature, test_folder)
    freq_numpy = phonons_numpy.frequency
    expected_folder_temp = f"{test_folder}/2_2_2/300/quantum"
    assert f"{expected_folder_temp}/frequency" in saved_data
    # With numpy storage, it should use the general setting, not _store_formats
    assert saved_data[f"{expected_folder_temp}/frequency"][1] == 'numpy'


def test_get_folder_from_label_with_q_point_and_temperature():
    """Test folder path generation for single q-point with temperature.

    Regression test for HarmonicWithQTemp missing _get_folder_path_components().
    """
    class Instance(Storable):
        def __init__(self, temperature, is_classic):
            self.folder = 'base_folder'
            self.q_point = np.array([0., 0., 0.])
            self.temperature = temperature
            self.is_classic = is_classic

        def _get_folder_path_components(self, label):
            components = []
            if '<temperature>' in label:
                components.append(str(int(self.temperature)))
            if '<statistics>' in label:
                components.append('quantum' if not self.is_classic else 'classic')
            return components

    # Test quantum at 300K
    instance_300 = Instance(300, False)
    folder_300 = instance_300.get_folder_from_label('<temperature>/<statistics>/<q_point>')
    assert folder_300 == 'base_folder/single_q/0.0_0.0_0.0/300/quantum'

    # Test classic at 500K - different path
    instance_500 = Instance(500, True)
    folder_500 = instance_500.get_folder_from_label('<temperature>/<statistics>/<q_point>')
    assert folder_500 == 'base_folder/single_q/0.0_0.0_0.0/500/classic'

    # Critical: different temperatures must produce different paths
    assert folder_300 != folder_500

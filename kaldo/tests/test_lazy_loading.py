import pytest
import numpy as np
import os
import h5py
from unittest.mock import MagicMock

from kaldo.helpers.storage import (
    load,
    save,
    get_folder_from_label,
    lazy_property,
    is_calculated,
    LAZY_PREFIX,
    DEFAULT_STORE_FORMATS,
)


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
    class Instance(object):
        folder = 'base_folder'
        kpts = [2, 2, 2]

    instance = Instance()
    folder = get_folder_from_label(instance)
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
    class Instance(object):
        folder = 'base_folder'
        kpts = [2, 2, 2]
        temperature = 300
        is_classic = False

    label = '<temperature>/<statistics>'
    instance = Instance()
    folder = get_folder_from_label(instance, label)
    assert folder == 'base_folder/2_2_2/300/quantum'


def test_get_folder_from_label_with_method_and_length():
    class Instance(object):
        folder = 'base_folder'
        kpts = [3, 3, 3]
        method = 'rta'
        length = [10, None, 0]
        finite_length_method = 'some_method'

    label = '<method>/<length>/<finite_length_method>'
    instance = Instance()
    folder = get_folder_from_label(instance, label)
    assert folder == 'base_folder/3_3_3/rta/l_10_0_0/fssome_method'


def test_lazy_property_with_storage_formats(monkeypatch):
    # Mock the load and save functions
    def mock_load(property, folder, instance, format):
        return 'loaded_value'

    def mock_save(property, folder, loaded_attr, format):
        pass

    monkeypatch.setattr('kaldo.helpers.storage.load', mock_load)
    monkeypatch.setattr('kaldo.helpers.storage.save', mock_save)
    # Add 'test_attr' to DEFAULT_STORE_FORMATS for the test
    DEFAULT_STORE_FORMATS['test_attr'] = 'formatted'

    class TestClass:
        def __init__(self, storage):
            self.storage = storage
            self.folder = 'test_folder'
            self.kpts = [2, 2, 2]

        @lazy_property()
        def test_attr(self):
            return 'computed_value'

    try:
        # Test with 'memory' storage
        instance_memory = TestClass('memory')
        assert instance_memory.test_attr == 'computed_value'
        # Test with 'numpy' storage
        instance_numpy = TestClass('numpy')
        assert instance_numpy.test_attr == 'loaded_value'
        # Test with 'formatted' storage
        instance_formatted = TestClass('formatted')
        assert instance_formatted.test_attr == 'loaded_value'
    finally:
        # Clean up DEFAULT_STORE_FORMATS
        del DEFAULT_STORE_FORMATS['test_attr']


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

    def mock_load(property, folder, instance, format):
        raise FileNotFoundError

    monkeypatch.setattr('kaldo.helpers.storage.load', mock_load)

    result = is_calculated('test_property', instance)
    assert result == False


def test_is_calculated_with_stored_property(monkeypatch):
    class Instance(object):
        folder = 'test_folder'
        storage = 'formatted'
        kpts = [2, 2, 2]

    instance = Instance()

    # Mock load to return a value
    def mock_load(property, folder, instance, format):
        return 'stored_value'

    monkeypatch.setattr('kaldo.helpers.storage.load', mock_load)
    result = is_calculated('test_property', instance)
    assert result == True
    # Ensure the property is set on the instance
    attr = LAZY_PREFIX + 'test_property'
    assert getattr(instance, attr) == 'stored_value'


def test_is_calculated_property_not_available(monkeypatch):
    class Instance(object):
        folder = 'test_folder'
        storage = 'formatted'
        kpts = [2, 2, 2]

    instance = Instance()

    def mock_load(property, folder, instance, format):
        raise FileNotFoundError

    monkeypatch.setattr('kaldo.helpers.storage.load', mock_load)
    result = is_calculated('test_property', instance)
    assert result == False


#TDEP is missing.
#

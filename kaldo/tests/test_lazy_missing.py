# Import pytest for writing test functions and assertions
import pytest
# Import NumPy for array creation and comparison utilities
import numpy as np
# Import h5py for HDF5 file operations in tests
import h5py
# Import os for path manipulations
import os
# MagicMock allows us to create dummy instances for testing
from unittest.mock import MagicMock

# Import the functions and constants we want to test
from kaldo.helpers.storage import load, save, parse_pair, LAZY_PREFIX


def test_load_numpy_format(monkeypatch, tmp_path):
    """
    Test that load(..., format='numpy') correctly uses np.load
    to read a .npy file and returns the contained array.
    """
    # Create a small NumPy array to write to disk
    arr = np.array([1, 2, 3])
    # Define the path for the temporary .npy file
    file = tmp_path / "test_prop.npy"
    # Use NumPy to save our test array to the file
    np.save(str(file), arr)
    # Call our load function pointing to the tmp_path folder
    result = load("test_prop", str(tmp_path), instance=MagicMock(), format="numpy")
    # Assert that the array returned by load matches the original array
    np.testing.assert_array_equal(result, arr)


def test_save_numpy_format(monkeypatch, tmp_path):
    """
    Test that save(..., format='numpy') calls np.save with the correct
    filename and data.
    """
    # Create a 2×2 test array
    arr = np.array([[1, 2], [3, 4]])
    # Define a fresh folder under tmp_path
    folder = str(tmp_path / "fldr")
    # Spy on np.save by intercepting calls in a list
    calls = []
    monkeypatch.setattr(np, "save", lambda fname, data: calls.append((fname, data.copy())))
    # Invoke save for our test array in 'numpy' format
    save("prop", folder, arr, format="numpy")
    # Ensure that np.save was indeed called at least once
    assert calls, "np.save was not called"
    # Extract the filename and data from the spy
    fname, data = calls[0]
    # Confirm that the filename ends with the expected .npy path
    assert fname.endswith(os.path.join("fldr", "prop.npy"))
    # Confirm that the data passed to np.save matches the original array
    np.testing.assert_array_equal(data, arr)


def test_load_hdf5_format(monkeypatch, tmp_path):
    """
    Test that load(..., format='hdf5') can read from an HDF5 file
    and return the stored dataset.
    """
    # Path for our temporary HDF5 file
    h5file = str(tmp_path / "data.hdf5")
    # Create the HDF5 file and add a dataset named "prop"
    with h5py.File(h5file, "w") as f:
        f.create_dataset("prop", data=[9, 8, 7])
    # Create a dummy instance (its attributes are not used by load for HDF5)
    inst = MagicMock()
    # Call load, pointing at the base name 'data' for the folder
    result = load("prop", str(tmp_path / "data"), inst, format="hdf5")
    # Assert that the returned array matches what we wrote
    np.testing.assert_array_equal(result, [9, 8, 7])


def test_save_hdf5_no_duplicate(monkeypatch, tmp_path):
    """
    Test that save(..., format='hdf5') does not recreate an existing
    dataset in the HDF5 file.
    """
    # Define the folder and corresponding .hdf5 filepath
    folder = str(tmp_path / "d")
    h5file = folder + ".hdf5"
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    # Pre-create the HDF5 file with a dataset named "prop"
    with h5py.File(h5file, "w") as f:
        f.create_dataset("prop", data=[1, 2, 3])
    # List to capture any calls to create_dataset
    created = []
    # Spy class that wraps h5py.File and intercepts create_dataset calls
    class SpyFile(h5py.File):
        def create_dataset(self, name, *args, **kwargs):
            created.append(name)
            return super().create_dataset(name, *args, **kwargs)
    # Monkeypatch h5py.File to our spy implementation
    monkeypatch.setattr(h5py, "File", SpyFile)
    # Invoke save; since "prop" already exists, create_dataset should not be called
    save("prop", folder, np.array([4, 5, 6]), format="hdf5")
    # Assert that no new datasets were created
    assert not created, "create_dataset should not be called for existing dataset"


def test_load_formatted_single_real(monkeypatch):
    """
    Test that load for a single .dat file (non-special case) returns a float array.
    """
    # Monkeypatch np.loadtxt to return a known array when called
    monkeypatch.setattr(np, "loadtxt", lambda fn, skiprows, dtype: np.array([[3.14, 2.71]]))
    # Call load for a property handled by the fallback branch
    res = load("heat_capacity", "fld", instance=MagicMock(), format="formatted")
    # Ensure the dtype is float
    assert res.dtype == float
    # Ensure contents match what our fake loadtxt returned
    np.testing.assert_array_equal(res, [[3.14, 2.71]])


@pytest.mark.parametrize("txt,expected", [
    ("(3+4j)", 3 + 4j),
    ("( -1-2j )", -1 - 2j),
])
def test_parse_pair(txt, expected):
    """
    Test that the parse_pair utility correctly converts strings like "(a+bj)" into complex numbers.
    """
    # Assert that parse_pair returns the correct Python complex value
    assert parse_pair(txt) == expected


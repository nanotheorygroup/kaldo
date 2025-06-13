import numpy as np
import os
import h5py
from kaldo.helpers.logger import get_logger

# Logger setup
logger = get_logger()

# Prefix for caching lazy properties
LAZY_PREFIX = '_lazy__'

# Default data storage folder
FOLDER_NAME = 'data'

# Mapping properties to their file storage formats
DEFAULT_STORE_FORMATS = {
    'physical_mode': 'formatted',
    'frequency': 'formatted',
    'participation_ratio': 'formatted',
    'velocity': 'formatted',
    'heat_capacity': 'formatted',
    'population': 'formatted',
    'bandwidth': 'formatted',
    'phase_space': 'formatted',
    'conductivity': 'formatted',
    'mean_free_path': 'formatted',
    'diffusivity': 'numpy',
    'flux': 'numpy',
    '_dynmat_derivatives': 'numpy',
    '_eigensystem': 'numpy',
    '_ps_and_gamma': 'numpy',
    '_ps_gamma_and_gamma_tensor': 'numpy',
    '_generalized_diffusivity': 'numpy'
}

# Load property from disk or memory efficiently
def load(property, folder, instance, format='formatted'):
    path = os.path.join(folder, property)

    if format == 'numpy':
        return np.load(f"{path}.npy", allow_pickle=True)

    elif format == 'hdf5':
        with h5py.File(f"{folder}.hdf5", 'r') as storage:
            return storage[property][()]

    elif format == 'formatted':
        if property == 'physical_mode':
            data = np.loadtxt(f"{path}.dat", skiprows=1)
            return np.round(data, 0).astype(bool)

        if property in ['velocity', 'mean_free_path']:
            data = np.stack([np.loadtxt(f"{path}_{i}.dat", skiprows=1) for i in range(3)], axis=-1)
            return data

        if property == 'conductivity':
            data = np.array([
                np.loadtxt(f"{path}_{i}_{j}.dat", skiprows=1) for i in range(3) for j in range(3)
            ])
            return data.reshape(3, 3, instance.n_phonons).transpose(2, 0, 1)

        if '_sij' in property:
            data = np.stack([
                np.loadtxt(f"{path}_{i}.dat", skiprows=1, dtype=complex)
                for i in range(3)
            ], axis=-1)
            return data

        dtype = complex if property == 'diffusivity' else float
        return np.loadtxt(f"{path}.dat", skiprows=1, dtype=dtype)

    elif format == 'memory':
        attr = LAZY_PREFIX + property
        if hasattr(instance, attr):
            logger.info(f"Loading '{property}' from memory.")
            return getattr(instance, attr)
        else:
            logger.info(f"Property '{property}' not in memory.")
            raise KeyError('Property not found in memory.')

    raise ValueError('Unrecognized format.')

# Save property to disk efficiently
def save(property, folder, loaded_attr, format='formatted'):
    path = os.path.join(folder, property)
    os.makedirs(folder, exist_ok=True)

    if format == 'numpy':
        np.save(f"{path}.npy", loaded_attr)
        logger.info(f"Saved '{property}' in numpy format.")

    elif format == 'hdf5':
        with h5py.File(f"{folder}.hdf5", 'a') as storage:
            if property not in storage:
                storage.create_dataset(property, data=loaded_attr, chunks=True, compression='gzip', compression_opts=9)
        logger.info(f"Saved '{property}' in HDF5 format.")

    elif format == 'formatted':
        fmt = '%d' if property == 'physical_mode' else '%.18e'

        if property in ['velocity', 'mean_free_path']:
            for i in range(3):
                np.savetxt(f"{path}_{i}.dat", loaded_attr[..., i], fmt=fmt)

        elif '_sij' in property:
            for i in range(3):
                np.savetxt(f"{path}_{i}.dat", loaded_attr[..., i], fmt=fmt)

        elif property == 'conductivity':
            for i in range(3):
                for j in range(3):
                    np.savetxt(f"{path}_{i}_{j}.dat", loaded_attr[..., i, j], fmt=fmt)

        else:
            np.savetxt(f"{path}.dat", loaded_attr, fmt=fmt)
        logger.info(f"Saved '{property}' in formatted text files.")

    elif format == 'memory':
        logger.warning(f"'{property}' stored only in memory and will be lost.")

    else:
        raise ValueError('Unrecognized format.')

# Lazy property decorator
def lazy_property(label=''):
    def decorator(fn):
        attr_name = LAZY_PREFIX + fn.__name__

        @property
        def wrapper(self):
            format = DEFAULT_STORE_FORMATS.get(fn.__name__, 'memory')
            folder = get_folder_from_label(self, label)

            if format != 'memory':
                try:
                    return load(fn.__name__, folder, self, format)
                except (FileNotFoundError, OSError, KeyError):
                    logger.info(f"Calculating '{fn.__name__}'.")
                    result = fn(self)
                    save(fn.__name__, folder, result, format)
                    return result

            if not hasattr(self, attr_name):
                setattr(self, attr_name, fn(self))
            return getattr(self, attr_name)

        return wrapper
    return decorator

# Check if calculation exists in memory or storage
def is_calculated(property, instance, label='', format='formatted'):
    attr_name = LAZY_PREFIX + property

    if hasattr(instance, attr_name):
        return True

    folder = get_folder_from_label(instance, label)
    try:
        value = load(property, folder, instance, format=format)
        setattr(instance, attr_name, value)
        return True
    except (FileNotFoundError, OSError, KeyError):
        return False

# Helper: Generate structured folder paths from instance properties
def get_folder_from_label(instance, label='', base_folder=None):
    base_folder = base_folder or getattr(instance, 'folder', FOLDER_NAME)
    components = []

    if hasattr(instance, 'kpts') and np.prod(instance.kpts) > 1:
        components.append(f"{'_'.join(map(str, instance.kpts))}")

    folder_path = os.path.join(base_folder, *components)
    return folder_path

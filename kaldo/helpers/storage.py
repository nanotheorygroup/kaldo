import numpy as np
import os
import h5py
from kaldo.helpers.logger import get_logger

# Setup logger for informational/debug messages
logging = get_logger()

# Prefix used to identify cached in-memory attributes
LAZY_PREFIX = '_lazy__'

# Default folder to store computed data
FOLDER_NAME = 'data'

# Maps property names to their storage format (e.g., 'formatted', 'numpy', 'hdf5')
# 'formatted' refers to text-based `.dat` files
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

# Utility: Parse a complex number stored as a string like "(1+2j)"
def parse_pair(txt):
    return complex(txt.strip("()"))

# Core loader function to read properties from disk or memory
def load(property, folder, instance, format='formatted'):
    name = folder + '/' + property
    
    if format == 'numpy':
        # Load NumPy binary file
        return np.load(name + '.npy', allow_pickle=True)
    
    elif format == 'hdf5':
        # Load from HDF5 file using h5py
        with h5py.File(name.split('/')[0] + '.hdf5', 'r') as storage:
            return storage[name][()]
    
    elif format == 'formatted':
        # Load binary .dat files
        if property == 'physical_mode':
            data = np.loadtxt(name + '.dat', skiprows=1)
            return np.round(data, 0).astype(bool)

        elif property in ['velocity', 'mean_free_path']:
            components = [np.loadtxt(f"{name}_{alpha}.dat", skiprows=1) for alpha in range(3)]
            return np.array(components).transpose(1, 2, 0)

        elif property == 'conductivity':
            # Load 3x3 tensor (9 files)
            components = []
            for alpha in range(3):
                for beta in range(3):
                    components.append(np.loadtxt(f"{name}_{alpha}_{beta}.dat", skiprows=1))
            return np.array(components).reshape((3, 3, instance.n_phonons)).transpose(2, 0, 1)

        elif '_sij' in property:
            # Complex-valued tensor (3 files with complex numbers)
            components = [np.loadtxt(f"{name}_{alpha}.dat", skiprows=1, dtype=complex) for alpha in range(3)]
            return np.array(components).transpose(1, 0)

        else:
            # Fallback to single file (real or complex)
            dtype = complex if property == 'diffusivity' else float
            return np.loadtxt(name + '.dat', skiprows=1, dtype=dtype)
    
    elif format == 'memory':
        # Return property from in-memory attribute
        attr = LAZY_PREFIX + property
        if hasattr(instance, attr):
            logging.info(f'Loading from memory {property}')
            return getattr(instance, attr)
        else:
            logging.info(f'{property} not found in memory')
            raise KeyError('Property not found')

    else:
        raise ValueError('Storing format not implemented')

# Save property to file in given format
def save(property, folder, loaded_attr, format='formatted'):
    name = folder + '/' + property

    if format == 'numpy':
        os.makedirs(folder, exist_ok=True)
        np.save(name + '.npy', loaded_attr)
        logging.info(f'{name} stored')
    
    elif format == 'hdf5':
        with h5py.File(name.split('/')[0] + '.hdf5', 'a') as storage:
            if name not in storage:
                storage.create_dataset(name, data=loaded_attr, chunks=True, compression='gzip', compression_opts=9)
        logging.info(f'{name} stored')
    
    elif format == 'formatted':
        os.makedirs(folder, exist_ok=True)
        fmt = '%d' if property == 'physical_mode' else '%.18e'

        if property in ['velocity', 'mean_free_path']:
            for alpha in range(3):
                np.savetxt(f"{name}_{alpha}.dat", loaded_attr[..., alpha], fmt=fmt, header=str(loaded_attr[..., 0].shape))
        
        elif '_sij' in property:
            for alpha in range(3):
                np.savetxt(f"{name}_{alpha}.dat", loaded_attr[..., alpha].flatten(), fmt=fmt, header=str(loaded_attr[..., 0].shape))

        elif 'conductivity' in property:
            for alpha in range(3):
                for beta in range(3):
                    np.savetxt(f"{name}_{alpha}_{beta}.dat", loaded_attr[..., alpha, beta], fmt=fmt, header=str(loaded_attr[..., 0, 0].shape))

        else:
            np.savetxt(name + '.dat', loaded_attr, fmt=fmt, header=str(loaded_attr.shape))
    
    elif format == 'memory':
        logging.warning(f'Property {property} will be lost when calculation is over.')
    
    else:
        raise ValueError('Storing format not implemented')

# Generate folder name from simulation metadata
def get_folder_from_label(instance, label='', base_folder=None):
    if base_folder is None:
        base_folder = instance.folder if hasattr(instance, 'folder') and instance.folder else FOLDER_NAME

    try:
        if np.prod(instance.kpts) > 1:
            kpts = instance.kpts
            base_folder += f'/{kpts[0]}_{kpts[1]}_{kpts[2]}'
    except AttributeError:
        try:
            q = instance.q_point
            base_folder += f'/single_q/{q[0]}_{q[1]}_{q[2]}'
        except AttributeError:
            pass

    # Interpret user-defined label template and append subfolders
    if label:
        if '<diffusivity_bandwidth>' in label and instance.diffusivity_bandwidth is not None:
            base_folder += f'/db_{np.mean(instance.diffusivity_bandwidth)}'
        if '<diffusivity_threshold>' in label and instance.diffusivity_threshold is not None:
            base_folder += f'/dt_{instance.diffusivity_threshold}'
        if '<temperature>' in label:
            base_folder += f'/{int(instance.temperature)}'
        if '<statistics>' in label:
            base_folder += '/classic' if instance.is_classic else '/quantum'
        if '<third_bandwidth>' in label and instance.third_bandwidth is not None:
            base_folder += f'/tb_{np.mean(instance.third_bandwidth)}'
        if '<include_isotopes>' in label and instance.include_isotopes:
            base_folder += '/isotopes'
        if '<method>' in label:
            base_folder += f'/{instance.method}'
            if instance.method in ['rta', 'sc', 'inverse'] and instance.length is not None:
                length = np.array(instance.length)
                if not ((length == [None, None, None]).all() or (length == [0, 0, 0]).all()):
                    if '<length>' in label:
                        base_folder += '/l' + ''.join([f'_{l if l is not None else 0}' for l in instance.length])
                    if '<finite_length_method>' in label and instance.finite_length_method is not None:
                        base_folder += f'/fs{instance.finite_length_method}'

    return base_folder

# Lazy property decorator: compute-once, cache, load/save if needed
def lazy_property(label=''):
    def _lazy_property(fn):
        @property
        def __lazy_property(self):
            try:
                format = DEFAULT_STORE_FORMATS[fn.__name__] if self.storage == 'formatted' else self.storage
            except KeyError:
                format = 'memory'

            if format != 'memory':
                folder = get_folder_from_label(self, label)
                prop = fn.__name__
                try:
                    # Try loading from disk
                    value = load(prop, folder, self, format=format)
                except (FileNotFoundError, OSError, KeyError):
                    logging.info(f"{folder}/{prop} not found. Calculating...")
                    value = fn(self)
                    save(prop, folder, value, format=format)
                else:
                    logging.info(f'Loaded {folder}/{prop}')
            else:
                # Only use memory
                attr = LAZY_PREFIX + fn.__name__
                if not hasattr(self, attr):
                    value = fn(self)
                    setattr(self, attr, value)
                else:
                    value = getattr(self, attr)
            return value

        __lazy_property.__doc__ = fn.__doc__
        return __lazy_property
    return _lazy_property

# Check if a property has already been calculated or stored
def is_calculated(property, self, label='', format='formatted'):
    attr = LAZY_PREFIX + property
    try:
        return getattr(self, attr) is not None
    except AttributeError:
        pass

    try:
        folder = get_folder_from_label(self, label)
        value = load(property, folder, self, format=format)
        setattr(self, attr, value)
        return value is not None
    except (FileNotFoundError, OSError, KeyError):
        return False


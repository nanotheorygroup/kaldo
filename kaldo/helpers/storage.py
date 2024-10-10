import numpy as np
import os
import h5py
from kaldo.helpers.logger import get_logger

logging = get_logger()

LAZY_PREFIX = '_lazy__'
FOLDER_NAME = 'data'

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
    '_generalized_diffusivity': 'numpy',
}


def parse_pair(txt):
    return complex(txt.strip("()"))


def load(property, folder, instance, format='formatted'):
    name = f"{folder}/{property}"
    if format == 'numpy':
        return np.load(f"{name}.npy", allow_pickle=True)
    elif format == 'hdf5':
        with h5py.File(f"{name.split('/')[0]}.hdf5", 'r') as storage:
            return storage[name][()]
    elif format == 'formatted':
        if property == 'physical_mode':
            loaded = np.loadtxt(f"{name}.dat", skiprows=1)
            return np.round(loaded, 0).astype(bool)
        elif property in ['velocity', 'mean_free_path']:
            loaded = [np.loadtxt(f"{name}_{alpha}.dat", skiprows=1) for alpha in range(3)]
            return np.array(loaded).transpose(1, 2, 0)
        elif property == 'conductivity':
            loaded = [
                np.loadtxt(f"{name}_{alpha}_{beta}.dat", skiprows=1)
                for alpha in range(3) for beta in range(3)
            ]
            return np.array(loaded).reshape(3, 3, instance.n_phonons).transpose(2, 0, 1)
        elif '_sij' in property:
            loaded = [
                np.loadtxt(f"{name}_{alpha}.dat", skiprows=1, dtype=complex)
                for alpha in range(3)
            ]
            return np.array(loaded).transpose(1, 0)
        else:
            dt = complex if property == 'diffusivity' else float
            return np.loadtxt(f"{name}.dat", skiprows=1, dtype=dt)
    elif format == 'memory':
        attr = getattr(instance, f"{LAZY_PREFIX}{property}", None)
        if attr is not None:
            logging.info(f"Loading from memory {property} property.")
            return attr
        else:
            logging.info(f"{property} not found.")
            raise KeyError('Property not found')
    else:
        raise ValueError('Storing format not implemented')


def save(property, folder, loaded_attr, format='formatted'):
    name = f"{folder}/{property}"
    if format == 'numpy':
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(f"{name}.npy", loaded_attr)
    elif format == 'hdf5':
        with h5py.File(f"{name.split('/')[0]}.hdf5", 'a') as storage:
            if name not in storage:
                storage.create_dataset(
                    name, data=loaded_attr, chunks=True,
                    compression='gzip', compression_opts=9
                )
    elif format == 'formatted':
        if not os.path.exists(folder):
            os.makedirs(folder)
        fmt = '%d' if property == 'physical_mode' else '%.18e'
        if property in ['velocity', 'mean_free_path']:
            for alpha in range(3):
                np.savetxt(
                    f"{name}_{alpha}.dat",
                    loaded_attr[..., alpha],
                    fmt=fmt,
                    header=str(loaded_attr[..., 0].shape)
                )
        elif '_sij' in property:
            for alpha in range(3):
                np.savetxt(
                    f"{name}_{alpha}.dat",
                    loaded_attr[..., alpha].flatten(),
                    fmt=fmt,
                    header=str(loaded_attr[..., 0].shape)
                )
        elif 'conductivity' in property:
            for alpha in range(3):
                for beta in range(3):
                    np.savetxt(
                        f"{name}_{alpha}_{beta}.dat",
                        loaded_attr[..., alpha, beta],
                        fmt=fmt,
                        header=str(loaded_attr[..., 0, 0].shape)
                    )
        else:
            np.savetxt(f"{name}.dat", loaded_attr, fmt=fmt, header=str(loaded_attr.shape))
    elif format == 'memory':
        pass
    else:
        raise ValueError('Storing format not implemented')
    logging.info(f"{name} stored")


def get_folder_from_label(instance, label='', base_folder=None):
    base_folder = base_folder or instance.folder or FOLDER_NAME
    try:
        if np.prod(instance.kpts) > 1:
            base_folder += f"/{'_'.join(map(str, instance.kpts))}"
    except AttributeError:
        try:
            q_point = instance.q_point
            base_folder += f"/single_q/{'_'.join(map(str, q_point))}"
        except AttributeError:
            pass
    if label:
        if '<diffusivity_bandwidth>' in label and instance.diffusivity_bandwidth is not None:
            base_folder += f"/db_{np.mean(instance.diffusivity_bandwidth)}"
        if '<diffusivity_threshold>' in label and instance.diffusivity_threshold is not None:
            base_folder += f"/dt_{instance.diffusivity_threshold}"
        if '<temperature>' in label:
            base_folder += f"/{int(instance.temperature)}"
        if '<statistics>' in label:
            base_folder += '/classic' if instance.is_classic else '/quantum'
        if '<third_bandwidth>' in label and instance.third_bandwidth is not None:
            base_folder += f"/tb_{np.mean(instance.third_bandwidth)}"
        if '<include_isotopes>' in label and instance.include_isotopes:
            base_folder += '/isotopes'
        if '<method>' in label:
            base_folder += f"/{instance.method}"
            if instance.method in ['rta', 'sc', 'inverse'] and instance.length:
                length_array = np.array(instance.length)
                if not np.all(length_array == [None, None, None]) and not np.all(length_array == [0, 0, 0]):
                    if '<length>' in label:
                        lengths = [str(l) if l is not None else '0' for l in instance.length]
                        base_folder += f"/l_{'_'.join(lengths)}"
                    if '<finite_length_method>' in label and instance.finite_length_method is not None:
                        base_folder += f"/fs{instance.finite_length_method}"
    return base_folder


def lazy_property(label=''):
    def _lazy_property(fn):
        @property
        def __lazy_property(self):
            format = self.storage if self.storage != 'formatted' else DEFAULT_STORE_FORMATS.get(fn.__name__, 'memory')
            if format != 'memory':
                folder = get_folder_from_label(self, label)
                property = fn.__name__
                try:
                    loaded_attr = load(property, folder, self, format=format)
                    logging.info(f"Loading {folder}/{property}")
                except (FileNotFoundError, OSError, KeyError):
                    logging.info(f"{folder}/{property} not found in {format} format, calculating {property}")
                    loaded_attr = fn(self)
                    save(property, folder, loaded_attr, format=format)
                return loaded_attr
            else:
                attr = f"{LAZY_PREFIX}{fn.__name__}"
                if not hasattr(self, attr):
                    setattr(self, attr, fn(self))
                return getattr(self, attr)
        return __lazy_property
    return _lazy_property


def is_calculated(property, self, label='', format='formatted'):
    attr = f"{LAZY_PREFIX}{property}"
    if getattr(self, attr, None) is not None:
        return True
    try:
        folder = get_folder_from_label(self, label)
        loaded_attr = load(property, folder, self, format=format)
        setattr(self, attr, loaded_attr)
        return loaded_attr is not None
    except (FileNotFoundError, OSError, KeyError):
        return False

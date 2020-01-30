import numpy as np
import os
from ballistico.helpers.logger import get_logger
logging = get_logger()

import h5py
# see bug report: https://github.com/h5py/h5py/issues/1101
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

LAZY_PREFIX = '_lazy__'
FOLDER_NAME = 'data'

def load(property, folder, format='formatted'):
    name = folder + '/' + property
    if format == 'numpy':
        loaded = np.load(name + '.npy')
        return loaded
    elif format == 'hdf5':
        with h5py.File(name.split('/')[0] + '.hdf5', 'r') as storage:
            loaded = storage[name]
            return loaded.value
    elif format == 'formatted':
        if property == 'physical_modes':
            loaded = np.loadtxt(name + '.dat', skiprows=1)
            loaded = np.round(loaded, 0).astype(np.bool)
        elif property == 'velocity':
            loaded = []
            for alpha in range(3):
                loaded.append(np.loadtxt(name + '_' + str(alpha) + '.dat', skiprows=1))
            loaded = np.array(loaded).transpose(1, 2, 0)
        elif property == 'conductivity':
            loaded = []
            for alpha in range(3):
                for beta in range(3):
                    loaded.append(np.loadtxt(name + '_' + str(alpha) + '_' + str(beta) + '.dat', skiprows=1))
            loaded = np.array(loaded).reshape((3, 3, ...)).transpose(2, 3, 0, 1)
        else:
            if property == 'diffusivity':
                dt = np.complex
            else:
                dt = np.float
            loaded = np.loadtxt(name + '.dat', skiprows=1, dtype=dt)
        return loaded
    else:
        raise ValueError('Storing format not implemented')


def save(property, folder, loaded_attr, format='formatted'):
    name = folder + '/' + property
    if format == 'numpy':
        np.save(name + '.npy', loaded_attr)
    elif format == 'hdf5':
        with h5py.File(name.split('/')[0] + '.hdf5', 'a') as storage:
            if not name in storage:
                storage.create_dataset(name, data=loaded_attr, chunks=True)
    elif format == 'formatted':
        loaded_attr = np.nan_to_num(loaded_attr)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if property == 'physical_modes':
            fmt = '%d'
        else:
            fmt = '%.18e'
        if property == 'velocity':
            for alpha in range(3):
                np.savetxt(name + '_' + str(alpha) + '.dat', loaded_attr[..., alpha], fmt=fmt, header=str(loaded_attr[..., 0].shape))
        elif 'conductivity' in property:
            for alpha in range(3):
                for beta in range(3):
                    try:
                        np.savetxt(name + '_' + str(alpha) + '_' + str(beta) + '.dat', loaded_attr[..., alpha, beta], fmt=fmt,
                               header=str(loaded_attr[..., 0, 0].shape))

                    except IndexError:
                        print('error')
        else:
            np.savetxt(name + '.dat', loaded_attr, fmt=fmt, header=str(loaded_attr.shape))
    else:
        raise ValueError('Storing format not implemented')


def get_folder_from_label(phonons, label='', base_folder=None):
    if base_folder is None:
        if phonons.folder:
            base_folder = phonons.folder
        else:
            base_folder = FOLDER_NAME
    if phonons.n_k_points > 1:
        kpts = phonons.kpts
        base_folder += '/' + str(kpts[0]) + '_' + str(kpts[1]) + '_' + str(kpts[2])
    if label != '':
        if '<temperature>' in label:
            base_folder += '/' + str(phonons.temperature)
        if '<statistics>' in label:
            if phonons.is_classic:
                base_folder += '/classic'
            else:
                base_folder += '/quantum'
        if '<third_bandwidth>' in label:
            if phonons.third_bandwidth is not None:
                base_folder += '/' + str(np.mean(phonons.third_bandwidth))
        # logging.info('Folder: ' + str(base_folder))
    return base_folder


def lazy_property(label='', format='formatted'):
    is_storing = (format != 'memory')
    def _lazy_property(fn):
        attr = LAZY_PREFIX + fn.__name__
        @property
        def __lazy_property(self):
            if not hasattr(self, attr):
                if is_storing:
                    folder = get_folder_from_label(self, label)
                    property = fn.__name__
                    try:
                        loaded_attr = load(property, folder, format=format)
                    except (FileNotFoundError, OSError, KeyError):
                        logging.info(str(property) + ' not found in memory, calculating ' + str(fn.__name__))
                        loaded_attr = fn(self)
                        save(property, folder, loaded_attr, format=format)
                    else:
                        logging.info('Loading ' + str(property))
                else:
                    loaded_attr = fn(self)
                setattr(self, attr, loaded_attr)
            return getattr(self, attr)
        __lazy_property.__doc__ = fn.__doc__
        return __lazy_property
    return _lazy_property


def is_calculated(property, self, label='', format='formatted'):
    # TODO: remove this function
    attr = LAZY_PREFIX + property
    try:
        getattr(self, attr)
    except AttributeError:
        try:
            folder = get_folder_from_label(self, label)
            loaded_attr = load(property, folder, format=format)
            setattr(self, attr, loaded_attr)
            return True
        except (FileNotFoundError, OSError, KeyError):
            return False
    else:
        return True
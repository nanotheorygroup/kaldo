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
        loaded = np.loadtxt(name + '.dat', skiprows=1)
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
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savetxt(name + '.dat', loaded_attr, header=str(loaded_attr.shape))
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
        if '<sigma_in>' in label:
            if phonons.sigma_in is not None:
                base_folder += '/' + str(np.mean(phonons.sigma_in))
        logging.info('Folder: ' + str(base_folder))
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
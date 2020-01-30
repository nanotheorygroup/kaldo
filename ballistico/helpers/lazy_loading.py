import numpy as np
import os
from ballistico.helpers.logger import get_logger
logging = get_logger()

LAZY_PREFIX = '_lazy__'
FOLDER_NAME = 'data'


def load(filename, format='formatted'):
    if format == 'numpy':
        loaded = np.load(filename + '.npy')
        return loaded
    elif format == 'formatted':
        loaded = np.loadtxt(filename + '.dat', skiprows=1)
        return loaded
    else:
        raise ValueError('Storing format not implemented')


def save(filename, loaded_attr, format='formatted'):
    if format == 'numpy':
        np.save(filename + '.npy', loaded_attr)
    elif format == 'formatted':
        np.savetxt(filename + '.dat', loaded_attr, header=str(loaded_attr.shape))
    else:
        raise ValueError('Storing format not implemented')


def get_folder_from_label(phonons, label='', folder=None):
    if folder is None:
        if phonons.folder:
            folder = phonons.folder
        else:
            folder = FOLDER_NAME
    if phonons.n_k_points > 1:
        kpts = phonons.kpts
        folder += '/' + str(kpts[0]) + '_' + str(kpts[1]) + '_' + str(kpts[2])
    if label != '':
        if '<temperature>' in label:
            folder += '/' + str(phonons.temperature)

        if '<statistics>' in label:
            if phonons.is_classic:
                folder += '/classic'
            else:
                folder += '/quantum'

        if '<sigma_in>' in label:
            if phonons.sigma_in is not None:
                folder += '/' + str(np.mean(phonons.sigma_in))
        logging.info('Folder: ' + str(folder))
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def lazy_property(label='', format='formatted'):
    is_storing = (format != 'memory')
    def _lazy_property(fn):
        attr = LAZY_PREFIX + fn.__name__
        @property
        def __lazy_property(self):
            if not hasattr(self, attr):
                if is_storing:
                    folder = get_folder_from_label(self, label)
                    filename = folder + '/' + fn.__name__
                    try:
                        loaded_attr, exc = load(filename, format=format)
                    except FileNotFoundError and OSError:
                        logging.info(str(filename) + ' not found in memory, calculating ' + str(fn.__name__))
                        loaded_attr = fn(self)
                        save(filename, loaded_attr, format=format)
                    else:
                        logging.info('Loading ' + str(filename))
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
            loaded_attr = load(folder + '/' + property, format=format)
            setattr(self, attr, loaded_attr)
            return True
        except FileNotFoundError and OSError:
            return False
    else:
        return True
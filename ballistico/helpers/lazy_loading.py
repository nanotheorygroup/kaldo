import numpy as np
import os
from ballistico.helpers.logger import get_logger
logging = get_logger()

LAZY_PREFIX = '_lazy__'
FOLDER_NAME = 'data'


def create_folder(phonons, label):
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


def lazy_property(label, is_storing, is_sparse, is_binary):
    def _lazy_property(fn):
        attr = LAZY_PREFIX + fn.__name__
        @property
        def __lazy_property(self):
            if not hasattr(self, attr):
                if is_storing:
                    folder = create_folder(self, label)
                    filename = folder + '/' + fn.__name__ + '.npy'
                    try:
                        loaded_attr = np.load (filename)
                    except FileNotFoundError:
                        logging.info(str(filename) + ' not found in memory, calculating ' + str(fn.__name__))
                        loaded_attr = fn(self)
                        np.save (filename, loaded_attr)
                    else:
                        logging.info('Loading ' + str(filename))
                else:
                    loaded_attr = fn(self)
                setattr(self, attr, loaded_attr)
            return getattr(self, attr)

        __lazy_property.__doc__ = fn.__doc__
        return __lazy_property
    return _lazy_property


def is_calculated(property, self, label=''):
    attr = LAZY_PREFIX + property
    try:
        getattr(self, attr)
    except AttributeError:
        try:
            folder = create_folder(self, label)
            filename = folder + '/' + property + '.npy'
            loaded_attr = np.load(filename)
            setattr(self, attr, loaded_attr)
            return True
        except FileNotFoundError:
            return False
    else:
        return True
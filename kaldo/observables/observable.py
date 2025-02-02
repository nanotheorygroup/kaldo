import numpy as np
import os
import h5py
from kaldo.helpers.storage import FOLDER_NAME
from kaldo.helpers.logger import get_logger
logging = get_logger()


class Observable:
    def __init__(self, folder: str = FOLDER_NAME, *kargs, **kwargs):
        self.folder = folder


    @classmethod
    def load(cls, *kargs, **kwargs):
        pass


    def save(self, property_name: str | None = None, format: str = 'numpy'):
        folder = self.folder
        loaded_attr = self.value
        if property_name is None:
            property_name = str(self)
        name = folder + '/' + property_name
        if format == 'numpy':
            if not os.path.exists(folder):
                os.makedirs(folder)
            np.save(name + '.npy', loaded_attr)
            logging.info(name + ' stored')
        elif format == 'hdf5':
            with h5py.File(name.split('/')[0] + '.hdf5', 'a') as storage:
                if name not in storage:
                    storage.create_dataset(name, data=loaded_attr, chunks=True, compression='gzip',
                                           compression_opts=9)
            logging.info(name + ' stored')
        elif format == 'formatted':
            # loaded_attr = np.nan_to_num(loaded_attr)
            if not os.path.exists(folder):
                os.makedirs(folder)
            fmt = '%.18e'
            np.savetxt(name + '.dat', loaded_attr, fmt=fmt, header=str(loaded_attr.shape))
        elif format == 'memory':
            logging.warning('Property ' + str(property_name) + ' will be lost when calculation is over.')
        else:
            raise ValueError('Storing format not implemented')


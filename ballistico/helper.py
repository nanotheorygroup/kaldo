import numpy as np
import time
import os

FOLDER_NAME = 'output'
LAZY_PREFIX = '_lazy__'

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def create_folder(phonons, is_reduced_path):
    if phonons.folder:
        folder = phonons.folder
    else:
        folder = FOLDER_NAME
    if phonons.n_k_points > 1:
        kpts = phonons.kpts
        folder += '/' + str(kpts[0]) + '_' + str(kpts[1]) + '_' + str(kpts[2])
    if not is_reduced_path:
        folder += '/' + str(phonons.temperature)
        if phonons.is_classic:
            folder += '/classic'
        else:
            folder += '/quantum'
        if phonons.sigma_in is not None:
            folder += '/' + str(phonons.sigma_in)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def lazy_property(is_storing, is_reduced_path):
    def _lazy_property(fn):
        attr = LAZY_PREFIX + fn.__name__
        @property
        def __lazy_property(self):
            if not hasattr(self, attr):
                if is_storing:
                    folder = create_folder(self, is_reduced_path)
                    filename = folder + '/' + fn.__name__ + '.npy'
                    try:
                        loaded_attr = np.load (filename)
                    except FileNotFoundError:
                        print(filename, 'not found, calculating', fn.__name__)
                        loaded_attr = fn(self)
                        np.save (filename, loaded_attr)
                    else:
                        print('loading', filename)
                else:
                    loaded_attr = fn(self)
                setattr(self, attr, loaded_attr)
            return getattr(self, attr)
        return __lazy_property
    return _lazy_property

def is_calculated(property, self, is_reduced_path=False):
    attr = LAZY_PREFIX + property
    try:
        getattr(self, attr)
    except AttributeError:
        try:
            folder = create_folder(self, is_reduced_path)
            filename = folder + '/' + property + '.npy'
            loaded_attr = np.load(filename)
            setattr(self, attr, loaded_attr)
            return True
        except FileNotFoundError:
            return False
        return False
    else:
        return True


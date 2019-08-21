import numpy as np
import time

def lazy_property(fn):
    attr = '_lazy__' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr):
            filename = self.folder_name + '/' + fn.__name__ + '.npy'
            try:
                loaded_attr = np.load (filename)
            except FileNotFoundError:
                print(filename, 'not found, calculating', fn.__name__)
                loaded_attr = fn(self)
                np.save (filename, loaded_attr)
            else:
                print('loading', filename)
            setattr(self, attr, loaded_attr)
        return getattr(self, attr)
    return _lazy_property

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
"""
Ballistico
Anharmonic Lattice Dynamics
"""
import numpy as np
import time
from itertools import takewhile, repeat
from ballistico.helpers.logger import get_logger
logging = get_logger()



def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            logging.info('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed


def count_rows(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
    return sum(buf.count(b'\n') for buf in bufgen if buf)






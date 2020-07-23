import sys
from time import gmtime, strftime
import psutil
import logging
import numpy as np

datetime = strftime("%a_%d_%b__%Y_%H_%M_%S", gmtime())
logname = 'kaldo_' + str(datetime) + '.log'
logger = logging.getLogger('kaldo')
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_level = logging.INFO

logger.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
formatter = logging.Formatter(format)
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_logger():
    logger = logging.getLogger('kaldo')
    return logger


def log_size(shape, type=np.float, memory_threshold_in_mb=30):
    shape = np.array(shape)
    out = 'Available memory: \n'
    out +=  str(psutil.virtual_memory().available/1e6) + ' MB\n'
    if type == np.float:
        size = 64
    elif type == np.complex:
        size = 128
    out += 'Shape of tensor: ' + str(shape)
    out += 'Type: ' + str(type) + '\n'
    memory_used_in_mb = np.prod(shape) * size / 8 * 1e-6
    out += 'Size of tensor: ' + str(memory_used_in_mb) + ' MB\n'
    if memory_used_in_mb > memory_threshold_in_mb:
        get_logger().info(out)
    else:
        get_logger().debug(out)

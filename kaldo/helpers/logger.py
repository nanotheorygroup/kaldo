import sys
from time import gmtime, strftime
import logging

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


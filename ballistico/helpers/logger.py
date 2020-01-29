import sys
from time import gmtime, strftime
import logging

datetime = strftime("%a_%d_%b__%Y_%H_%M_%S", gmtime())
logname = 'ballistico_' + str(datetime) + '.log'
logger = logging.getLogger('ballistico')
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_level = logging.INFO
# logging.basicConfig(filename=logname,
#                             filemode='a',
#                             format=format,
#                             datefmt='%H:%M:%S',
#                             level=log_level)
logger.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
formatter = logging.Formatter(format)
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_logger():
    logger = logging.getLogger('ballistico')
    return logger


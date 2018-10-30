import logging
import sys

class Logger:
    class __Logger:
        def __init__(self):
            self.val = self.setup_custom_logger()

        def setup_custom_logger(self):
            formatter = logging.Formatter (fmt='%(asctime)s %(levelname)-8s %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S')
            handler = logging.FileHandler ('output.log', mode='a+')
            handler.setFormatter (formatter)
            screen_handler = logging.StreamHandler (stream=sys.stdout)
            screen_handler.setFormatter (formatter)
            logger = logging.getLogger ()
            logger.setLevel (logging.INFO)
            logger.addHandler (handler)
            logger.addHandler (screen_handler)
            return logger
        
    instance = None
    
    def __init__(self):
        if not Logger.instance:
            Logger.instance = Logger.__Logger()
            
    def info(self, message):
        logging.info(message)

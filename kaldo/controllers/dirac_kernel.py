import scipy.special
import numpy as np
from kaldo.helpers.logger import get_logger
logging = get_logger()

IS_DEFAULT_RESCALING = False
DELTA_THRESHOLD = 2

def gaussian_delta(delta_omega, sigma, delta_threshold=None, is_rescaling=IS_DEFAULT_RESCALING):
    gaussian = 1 / np.sqrt(np.pi * sigma ** 2) * np.exp(-delta_omega ** 2 / (sigma ** 2))
    if delta_threshold is None or is_rescaling == False:
        return gaussian
    else:
        return gaussian / (scipy.special.erf(delta_threshold))


def triangular_delta(delta_omega, sigma, delta_threshold=None, is_rescaling=IS_DEFAULT_RESCALING):
    if delta_threshold is not None or is_rescaling == True:
        logging.warning('Calculation with triangular delta do not support delta_threshold or rescaling')
    delta_omega = np.abs(delta_omega)
    deltaa = np.abs(sigma)
    out = np.zeros_like(delta_omega)
    if (delta_omega < deltaa).any():
        out[delta_omega < deltaa] = (1. / deltaa * (1 - delta_omega / deltaa))[delta_omega < deltaa]
    return out



def lorentz_delta(delta_omega, sigma, delta_threshold=None, is_rescaling=IS_DEFAULT_RESCALING):
    lorentzian = 1 / np.pi * 1 / 2 * sigma / (delta_omega ** 2 + (sigma / 2) ** 2)
    if delta_threshold is None or is_rescaling == False:
        return lorentzian
    else:
        return lorentzian / (2 / np.pi * np.arctan(2 * delta_threshold))

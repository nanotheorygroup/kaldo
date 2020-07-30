import scipy.special
import numpy as np
from kaldo.helpers.logger import get_logger
logging = get_logger()

IS_DEFAULT_RESCALING = False
DELTA_THRESHOLD = 2

def gaussian_delta(delta_omega, sigma):
    gaussian = 1 / np.sqrt(np.pi * sigma ** 2) * np.exp(-delta_omega ** 2 / (sigma ** 2))
    return gaussian



def triangular_delta(delta_omega, sigma):
    delta_omega = np.abs(delta_omega)
    deltaa = np.abs(sigma)
    out = np.zeros_like(delta_omega)
    if (delta_omega < deltaa).any():
        out[delta_omega < deltaa] = (1. / deltaa * (1 - delta_omega / deltaa))[delta_omega < deltaa]
    return out



def lorentz_delta(delta_omega, sigma):
    lorentzian = 1 / np.pi * 1 / 2 * sigma / (delta_omega ** 2 + (sigma / 2) ** 2)
    return lorentzian

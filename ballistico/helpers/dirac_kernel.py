import scipy.special
import numpy as np
IS_DELTA_CORRECTION_ENABLED = True
DELTA_THRESHOLD = 2
LORENTZ_CORRECTIONS = [1, 0.704833, 0.844042, 0.894863, 0.920833, 0.936549, 0.947071, 0.954604, 0.960263, 0.964669,  0.968195]

def gaussian_delta(delta_omega, sigma):
    # alpha is a factor that tells whats the ration between the width of the gaussian
    # and the width of allowed phase space
    # allowing processes with width sigma and creating a gaussian with width sigma/2
    # we include 95% (erf(2/sqrt(2)) of the probability of scattering. The erf makes the total area 1
    gaussian = 1 / np.sqrt(np.pi * sigma ** 2) * np.exp(- delta_omega ** 2 / (sigma ** 2))
    if IS_DELTA_CORRECTION_ENABLED:
        correction = scipy.special.erf(DELTA_THRESHOLD / np.sqrt(2))
    else:
        correction = 1
    return gaussian / correction


def triangular_delta(delta_omega, sigma):
    delta_omega = np.abs(delta_omega)
    deltaa = np.abs(sigma)
    out = np.zeros_like(delta_omega)
    out[delta_omega < deltaa] = 1. / deltaa * (1 - delta_omega[delta_omega < deltaa] / deltaa)
    return out


def lorentz_delta(delta_omega, sigma, delta_threshold=DELTA_THRESHOLD):
    if IS_DELTA_CORRECTION_ENABLED:
        # TODO: replace these hardcoded values
        # numerical value of the integral of a lorentzian over +- DELTA_TRESHOLD * sigma
        correction = LORENTZ_CORRECTIONS[delta_threshold]
    else:
        correction = 1
    lorentzian = 1 / np.pi * 1 / 2 * sigma / (delta_omega ** 2 + (sigma / 2) ** 2)
    return lorentzian / correction

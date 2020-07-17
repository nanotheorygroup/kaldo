from opt_einsum import contract
import numpy as np
import ase.units as units
from sparse import COO
from kaldo.controllers.dirac_kernel import lorentz_delta, gaussian_delta, triangular_delta

from kaldo.helpers.logger import get_logger
logging = get_logger()

KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J

# TODO: move these methods inside second order

def calculate_population(phonons):
    frequency = phonons.frequency.reshape((phonons.n_k_points, phonons.n_modes))
    temp = phonons.temperature * KELVINTOTHZ
    density = np.zeros((phonons.n_k_points, phonons.n_modes))
    physical_modes = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    if phonons.is_classic is False:
        density[physical_modes] = 1. / (np.exp(frequency[physical_modes] / temp) - 1.)
    else:
        density[physical_modes] = temp / frequency[physical_modes]
    return density


def calculate_heat_capacity(phonons):
    frequency = phonons.frequency
    c_v = np.zeros_like(frequency)
    physical_modes = phonons.physical_mode
    temperature = phonons.temperature * KELVINTOTHZ
    if (phonons.is_classic):
        c_v[physical_modes] = KELVINTOJOULE
    else:
        f_be = phonons.population
        c_v[physical_modes] = KELVINTOJOULE * f_be[physical_modes] * (f_be[physical_modes] + 1) * phonons.frequency[
            physical_modes] ** 2 / \
                              (temperature ** 2)
    return c_v


def calculate_sij_sparse(phonons):
    logging.info('Start calculation diffusivity sparse')
    diffusivity_threshold = phonons.diffusivity_threshold
    if phonons.diffusivity_bandwidth is not None:
        diffusivity_bandwidth = phonons.diffusivity_bandwidth * np.ones((phonons.n_k_points, phonons.n_modes))
    else:
        diffusivity_bandwidth = phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes)).copy() / 2.

    omega = phonons._omegas.reshape(phonons.n_k_points, phonons.n_modes)
    omegas_difference = np.abs(omega[:, :, np.newaxis] - omega[:, np.newaxis, :])
    condition = (omegas_difference < diffusivity_threshold * 2 * np.pi * diffusivity_bandwidth)
    coords = np.array(np.unravel_index(np.flatnonzero(condition), condition.shape)).T
    s_ij = [COO(coords.T, phonons.flux_dense[..., 0][coords[:, 0], coords[:, 1], coords[:, 2]],
                shape=(phonons.n_k_points, phonons.n_modes, phonons.n_modes)),
            COO(coords.T, phonons.flux_dense[..., 1][coords[:, 0], coords[:, 1], coords[:, 2]],
                shape=(phonons.n_k_points, phonons.n_modes, phonons.n_modes)),
            COO(coords.T, phonons.flux_dense[..., 2][coords[:, 0], coords[:, 1], coords[:, 2]],
                shape=(phonons.n_k_points, phonons.n_modes, phonons.n_modes))]
    return s_ij


def calculate_diffusivity_dense(phonons):
    logging.info('Start calculation diffusivity dense')
    omega = phonons._omegas.reshape((phonons.n_k_points, phonons.n_modes))
    if phonons.diffusivity_bandwidth is not None:
        diffusivity_bandwidth = phonons.diffusivity_bandwidth * np.ones((phonons.n_k_points, phonons.n_modes))
    else:
        diffusivity_bandwidth = phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes)).copy() / 2.

    sigma = 2 * (diffusivity_bandwidth[:, :, np.newaxis] + diffusivity_bandwidth[:, np.newaxis, :])
    if phonons.diffusivity_shape == 'lorentz':
        curve = lorentz_delta
    elif phonons.diffusivity_shape == 'gauss':
        curve = gaussian_delta
    elif phonons.diffusivity_shape == 'triangle':
        curve = triangular_delta
    else:
        logging.error('Diffusivity shape not implemented')

    delta_energy = omega[:, :, np.newaxis] - omega[:, np.newaxis, :]
    kernel = curve(delta_energy, sigma)
    if phonons.is_diffusivity_including_antiresonant:
        sum_energy = omega[:, :, np.newaxis] + omega[:, np.newaxis, :]
        kernel += curve(sum_energy, sigma)
    kernel = kernel * np.pi
    kernel[np.isnan(kernel)] = 0

    sij = phonons.flux.reshape((phonons.n_k_points, phonons.n_modes, phonons.n_modes, 3))

    physical_modes = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    physical_modes_2d = physical_modes[:, :, np.newaxis] & \
                        physical_modes[:, np.newaxis, :]
    sij[np.invert(physical_modes_2d)] = 0

    prefactor = 1 / omega[:, :, np.newaxis] / omega[:, np.newaxis, :] / 4
    diffusivity = contract('knma,knm,knm,knmb->knmab', sij, prefactor, kernel, sij)
    return diffusivity


def calculate_diffusivity_sparse(phonons):
    if phonons.is_diffusivity_including_antiresonant:
        logging.error('is_diffusivity_including_antiresonant not yet implemented for with thresholds and sparse.')
    if phonons.diffusivity_shape == 'lorentz':
        curve = lorentz_delta
    elif phonons.diffusivity_shape == 'gauss':
        curve = gaussian_delta
    elif phonons.diffusivity_shape == 'triangle':
        curve = triangular_delta
    else:
        logging.error('Diffusivity shape not implemented')

    try:
        diffusivity_threshold = phonons.diffusivity_threshold
    except AttributeError:
        logging.error('Please provide diffusivity_threshold if you want to use a sparse diffusivity.')

    if phonons.diffusivity_bandwidth is not None:
        diffusivity_bandwidth = phonons.diffusivity_bandwidth * np.ones((phonons.n_k_points, phonons.n_modes))
    else:
        diffusivity_bandwidth = phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes)).copy() / 2.

    omega = phonons._omegas.reshape(phonons.n_k_points, phonons.n_modes)

    physical_modes = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    physical_modes_2d = physical_modes[:, :, np.newaxis] & \
                        physical_modes[:, np.newaxis, :]
    omegas_difference = np.abs(omega[:, :, np.newaxis] - omega[:, np.newaxis, :])
    condition = (omegas_difference < diffusivity_threshold * 2 * np.pi * diffusivity_bandwidth)

    coords = np.array(np.unravel_index (np.flatnonzero (condition), condition.shape)).T
    sigma = 2 * (diffusivity_bandwidth[coords[:, 0], coords[:, 1]] + diffusivity_bandwidth[coords[:, 0], coords[:, 2]])
    delta_energy = omega[coords[:, 0], coords[:, 1]] - omega[coords[:, 0], coords[:, 2]]
    data = np.pi * curve(delta_energy, sigma, diffusivity_threshold)
    lorentz = COO(coords.T, data, shape=(phonons.n_k_points, phonons.n_modes, phonons.n_modes))
    s_ij = phonons.flux
    prefactor = 1 / (4 * omega[coords[:, 0], coords[:, 1]] * omega[coords[:, 0], coords[:, 2]])
    prefactor[np.invert(physical_modes_2d[coords[:, 0], coords[:, 1], coords[:, 2]])] = 0
    prefactor = COO(coords.T, prefactor, shape=(phonons.n_k_points, phonons.n_modes, phonons.n_modes))

    diffusivity = np.zeros((phonons.n_k_points, phonons.n_modes, phonons.n_modes, 3, 3))
    for alpha in range(3):
        for beta in range(3):
            diffusivity[..., alpha, beta] = (s_ij[alpha] * prefactor * lorentz * s_ij[beta]).todense()
    return diffusivity


def calculate_generalized_diffusivity(phonons):
    if phonons.diffusivity_threshold is not None:
        diffusivity = calculate_diffusivity_sparse(phonons)
    else:
        diffusivity = calculate_diffusivity_dense(phonons)
    return diffusivity


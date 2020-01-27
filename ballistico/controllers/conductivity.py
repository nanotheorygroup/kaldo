"""
Ballistico
Anharmonic Lattice Dynamics
"""
from opt_einsum import contract
import ase.units as units
import numpy as np
import time
from sparse import COO
from ballistico.helpers.dirac_kernel import lorentz_delta, triangular_delta

from ballistico.helpers.logger import get_logger
logging = get_logger()

MAX_ITERATIONS_SC = 200
EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
MAX_LENGTH_TRESHOLD = 1e15


def tau_caltech(lambd, length, velocity,  axis):
    kn = lambd[:, axis] / length
    transmission = (1 - kn * (1 - np.exp(- 1. / kn)))
    tau = transmission / lambd[:, axis] * velocity[:, axis]
    return tau


def tau_matthiesen(lambd, length, velocity, axis):
    tau = (1 / lambd[:, axis] + 1 / length) ** (-1) / velocity[:, axis]
    return tau


def lambd_caltech(lambd, length, axis):
    lambd_out = lambd.copy()
    kn = lambd[axis] / length
    transmission = (1 - kn * (1 - np.exp(- 1. / kn)))
    lambd_out[axis] = transmission / lambd[axis]
    return lambd_out


def lambd_matthiesen(lambd, length, axis):
    lambd_out = lambd.copy()
    lambd_out[:, axis] = (1 / lambd[:, axis] + 1 / length) ** (-1)
    return lambd_out


def scattering_matrix(phonons, gamma=None):
    if gamma is None:
        gamma = phonons.gamma
    scattering_matrix = -1 * phonons._scattering_matrix_without_diagonal
    scattering_matrix = scattering_matrix + np.diag(gamma)
    return scattering_matrix


def calculate_c_v_2d(phonons):
    frequencies = phonons.frequencies
    c_v = np.zeros((phonons.n_k_points, phonons.n_modes, phonons.n_modes))
    temperature = phonons.temperature * KELVINTOTHZ
    physical_modes = phonons._physical_modes.reshape((phonons.n_k_points, phonons.n_modes))

    if (phonons.is_classic):
        c_v[:, :, :] = KELVINTOJOULE
    else:
        f_be = phonons.occupations
        c_v_omega = KELVINTOJOULE * f_be * (f_be + 1) * frequencies / (temperature ** 2)
        c_v_omega[np.invert(physical_modes)] = 0
        freq_sq = (frequencies[:, :, np.newaxis] + frequencies[:, np.newaxis, :]) / 2 * (c_v_omega[:, :, np.newaxis] + c_v_omega[:, np.newaxis, :]) / 2
        c_v[:, :, :] = freq_sq
    return c_v


def calculate_conductivity_qhgk(phonons, gamma_in=None):
    init = time.time()

    volume = np.linalg.det(phonons.atoms.cell)
    if gamma_in is not None:
        gamma = gamma_in * np.ones((phonons.n_k_points, phonons.n_modes))
    else:
        gamma = phonons.gamma.reshape((phonons.n_k_points, phonons.n_modes)).copy() / 2.
    omega = phonons._omegas
    physical_modes = phonons._physical_modes.reshape((phonons.n_k_points, phonons.n_modes))
    physical_modes_2d = physical_modes[:, :, np.newaxis] & \
                        physical_modes[:, np.newaxis, :]
    delta_energy = omega[:, :, np.newaxis] - omega[:, np.newaxis, :]
    sigma = 2 * (gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :])
    params = [delta_energy, sigma]
    lorentz = lorentz_delta(params)
    lorentz = lorentz * np.pi

    # DELTA_THRESHOLD = 10
    # omegas_difference = np.abs(omega[:, :, np.newaxis] - omega[:, np.newaxis, :])
    # condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * gamma_in)
    #
    # coords = np.array(np.unravel_index (np.flatnonzero (condition), condition.shape)).T
    # sigma = 2 * (gamma[coords[:, 0], coords[:, 1]] + gamma[coords[:, 0], coords[:, 2]])
    # delta_energy = omega[coords[:, 0], coords[:, 1]] - omega[coords[:, 0], coords[:, 2]]
    # params = [delta_energy, sigma]
    # data = lorentz_delta(params)
    # data = data * np.pi
    # # data = sigma / (sigma ** 2 + delta_energy ** 2)
    # lorentz_sparse = COO(coords.T, data, shape=(phonons.n_k_points, phonons.n_modes, phonons.n_modes))
    # lorentz = lorentz_sparse.todense()

    lorentz[np.isnan(lorentz)] = 0
    sij = phonons._sij
    sij[np.invert(physical_modes_2d)] = 0

    prefactor = 1 / omega[:, :, np.newaxis] / omega[:,np.newaxis, :] / 4
    diffusivity = contract('knma,knm,knm,knmb->knab', sij, prefactor, lorentz, sij)
    conductivity_per_mode = contract('kn,knab->knab', phonons.heat_capacity, diffusivity)

    conductivity_per_mode = conductivity_per_mode.reshape((phonons.n_phonons, 3, 3))
    conductivity_per_mode = conductivity_per_mode * 1e22 / (volume * phonons.n_k_points)
    diff = 1 / 3 * 1 / 100 * contract('knaa->kn', diffusivity)
    # print('diffusivity', np.sum(diff))
    # cond = conductivity_per_mode.sum(axis=0).diagonal().mean()
    # print('kappa', cond)
    print('time elapsed', time.time() - init)

    return conductivity_per_mode, diff

def calculate_conductivity_inverse(phonons):
    velocities = phonons._keep_only_physical(phonons.velocities.real.reshape((phonons.n_phonons, 3), order='C'))
    scattering_inverse = np.linalg.inv(phonons._scattering_matrix)

    lambd = scattering_inverse.dot(velocities[:, :])
    physical_modes = phonons._physical_modes

    volume = np.linalg.det(phonons.atoms.cell)
    c_v = phonons._keep_only_physical(phonons.heat_capacity.reshape((phonons.n_phonons), order='C'))
    conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))
    conductivity_per_mode[physical_modes, :, :] = c_v[:, np.newaxis, np.newaxis] * \
                                                  velocities[:, :, np.newaxis] * lambd[:, np.newaxis, :]
    neg_diag = (phonons._scattering_matrix.diagonal() < 0).sum()
    logging.info('negative on diagonal : ' + str(neg_diag))
    evals = np.linalg.eigvalsh(phonons._scattering_matrix)
    logging.info('negative eigenvals : ' + str((evals < 0).sum()))

    return conductivity_per_mode * 1e22 / (volume * phonons.n_k_points)


def calculate_conductivity_with_evects(phonons):
    velocities = phonons._keep_only_physical(phonons.velocities.real.reshape((phonons.n_phonons, 3), order='C'))

    evals, evects = np.linalg.eig(phonons._scattering_matrix)
    first_positive = np.argwhere(evals >= 0)[0, 0]
    reduced_evects = evects[first_positive:, first_positive:]
    reduced_evals = evals[first_positive:]
    reduced_scattering_inverse = np.zeros_like(phonons._scattering_matrix)
    reduced_scattering_inverse[first_positive:, first_positive:] = reduced_evects.dot(np.diag(1/reduced_evals)).dot(np.linalg.inv(reduced_evects))
    scattering_inverse = reduced_scattering_inverse
    # e, v = np.linalg.eig(a)
    # a = v.dot(np.diag(e)).dot(np.linalg.inv(v))

    lambd = scattering_inverse.dot(velocities[:, :])
    physical_modes = phonons._physical_modes

    volume = np.linalg.det(phonons.atoms.cell)
    c_v = phonons._keep_only_physical(phonons.heat_capacity.reshape((phonons.n_phonons), order='C'))
    conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))
    conductivity_per_mode[physical_modes, :, :] = c_v[:, np.newaxis, np.newaxis] * \
                                                  velocities[:, :, np.newaxis] * lambd[:, np.newaxis, :]
    neg_diag = (phonons._scattering_matrix.diagonal() < 0).sum()
    logging.info('negative on diagonal : ' + str(neg_diag))
    evals = np.linalg.eigvalsh(phonons._scattering_matrix)
    logging.info('negative eigenvals : ' + str((evals < 0).sum()))

    return conductivity_per_mode * 1e22 / (volume * phonons.n_k_points)


def calculate_conductivity_sc(phonons, tolerance=None, length=None, axis=None, is_rta=False, n_iterations=MAX_ITERATIONS_SC, finite_size_method='matthiesen'):
    if n_iterations is None:
        n_iterations = MAX_ITERATIONS_SC
    if length is not None:
        length_thresholds = np.array([None, None, None])
        length_thresholds[axis] = length
    volume = np.linalg.det (phonons.atoms.cell)
    velocities = phonons.velocities.real.reshape ((phonons.n_k_points, phonons.n_modes, 3), order='C')
    lambd_0 = np.zeros ((phonons.n_k_points * phonons.n_modes, 3))
    velocities = velocities.reshape((phonons.n_phonons, 3), order='C')
    physical_modes = phonons._physical_modes
    if not is_rta:
        scattering_matrix = phonons._scattering_matrix_without_diagonal

    for alpha in range (3):
        gamma = np.zeros (phonons.n_phonons)

        for mu in range (phonons.n_phonons):
            if length:
                if length_thresholds[alpha]:
                    velocity = velocities[mu, alpha]
                    length = length_thresholds[alpha]
                    single_gamma = phonons.gamma.reshape((phonons.n_phonons), order='C')[mu]
                    if finite_size_method == 'matthiessen':
                        gamma[mu] = single_gamma + 2 * abs(velocity) / length
                        # gamma[mu] = phonons.gamma.reshape ((phonons.n_phonons), order='C')[mu] + \
                        #         np.abs (velocities[mu, alpha]) / length_thresholds[alpha]

                    elif finite_size_method == 'caltech':

                        kn = abs(velocity) / (length * single_gamma)
                        transmission = (1 - kn * (1 - np.exp(- 1. / kn))) * kn
                        gamma[mu] = abs(velocity) / length / transmission

                else:
                    gamma[mu] = phonons.gamma.reshape ((phonons.n_phonons), order='C')[mu]
            else:
                gamma[mu] = phonons.gamma.reshape ((phonons.n_phonons), order='C')[mu]
        tau_0 = 1 / gamma
        tau_0[np.invert(physical_modes)] = 0
        lambd_0[:, alpha] = tau_0[:] * velocities[:, alpha]
    c_v = phonons.heat_capacity.reshape ((phonons.n_phonons), order='C')

    lambd_n = lambd_0.copy ()
    conductivity_per_mode = np.zeros ((phonons.n_phonons, 3, 3))
    avg_conductivity = None
    cond_iterations = []
    n_iteration = 0
    for n_iteration in range (n_iterations):
        for alpha in range (3):
            for beta in range (3):
                conductivity_per_mode[physical_modes, alpha, beta] = c_v[physical_modes] * velocities[
                                                                         physical_modes, alpha] * lambd_n[
                                                                         physical_modes, beta]
        new_avg_conductivity = np.diag (np.sum (conductivity_per_mode, 0)).mean ()
        if is_rta:
            break
        if avg_conductivity:
            if tolerance is not None:
                if np.abs (avg_conductivity - new_avg_conductivity) < tolerance:
                    break
        avg_conductivity = new_avg_conductivity

        tau_0 = tau_0.reshape ((phonons.n_phonons), order='C')
        delta_lambd = tau_0[physical_modes, np.newaxis] * scattering_matrix.dot (lambd_n[physical_modes, :])
        lambd_n[physical_modes, :] = lambd_0[physical_modes, :] + delta_lambd[:, :]
        cond_iterations.append(conductivity_per_mode.sum(axis=0))

    for alpha in range (3):
        for beta in range (3):
            conductivity_per_mode[physical_modes, alpha, beta] = c_v[
                physical_modes] * velocities[physical_modes, alpha] * lambd_n[physical_modes, beta]

    if n_iteration == (MAX_ITERATIONS_SC - 1):
        logging.info('Convergence not reached')
    if is_rta:
        return conductivity_per_mode * 1e22 / (volume * phonons.n_k_points)
    else:
        return conductivity_per_mode * 1e22 / (volume * phonons.n_k_points), np.array(cond_iterations) * 1e22 / (volume * phonons.n_k_points)


def conductivity(phonons, method='rta', max_n_iterations=None, length=None, axis=None, finite_length_method='matthiessen', gamma_in=None, tolerance=None):
    # if length is not None:
    if method == 'rta':
        return calculate_conductivity_sc(phonons, length=length, axis=axis, is_rta=True, finite_size_method=finite_length_method, n_iterations=max_n_iterations)
    elif method == 'sc':
        return calculate_conductivity_sc(phonons, length=length, axis=axis, is_rta=False, finite_size_method=finite_length_method, n_iterations=max_n_iterations, tolerance=tolerance)
    elif (method == 'qhgk'):
        return calculate_conductivity_qhgk(phonons, gamma_in=gamma_in)
    elif (method == 'inverse'):
        return calculate_conductivity_inverse(phonons)
    elif (method == 'eigenvectors'):
        return calculate_conductivity_with_evects(phonons)
    else:
        raise TypeError('Conductivity method not implemented')


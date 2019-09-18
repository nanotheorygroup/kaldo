from opt_einsum import contract
import ase.units as units
import numpy as np

MAX_ITERATIONS_SC = 500
EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12



def conductivity(phonons, lambd):
    physical_modes = (phonons.frequencies.reshape(phonons.n_phonons) > phonons.frequency_threshold)

    # TODO: Change units conversion in this method
    volume = np.linalg.det(phonons.atoms.cell) / 1000
    c_v = phonons.keep_only_physical(phonons.c_v.reshape((phonons.n_phonons), order='C') * 1e21)
    velocities = phonons.keep_only_physical(phonons.velocities.real.reshape((phonons.n_phonons, 3), order='C') / 10)
    conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))
    conductivity_per_mode[physical_modes, :, :] = 1 / (volume * phonons.n_k_points) * c_v[:, np.newaxis, np.newaxis] * \
                                                  velocities[:, :, np.newaxis] * lambd[:, np.newaxis, :]
    return conductivity_per_mode

def calculate_conductivity_inverse(phonons):
    velocities = phonons.keep_only_physical(phonons.velocities.real.reshape((phonons.n_phonons, 3), order='C') / 10)
    scattering_inverse = np.linalg.inv(phonons.scattering_matrix)
    lambd = scattering_inverse.dot(velocities[:, :])
    conductivity_per_mode = conductivity(phonons, lambd)

    # TODO: remove this debug info
    neg_diag = (phonons.scattering_matrix.diagonal() < 0).sum()
    print('negative on diagonal : ', neg_diag)
    evals = np.linalg.eigvalsh(phonons.scattering_matrix)
    print('negative eigenvals : ', (evals < 0).sum())
    return conductivity_per_mode

def transmission_caltech(gamma, velocity, length):
    kn = abs(velocity / (length * gamma))
    transmission = (1 - kn * (1 - np.exp(- 1. / kn))) * kn
    return length / abs(velocity) * transmission

def transmission_matthiesen(gamma, velocity, length):
    transmission = (gamma * length / abs(velocity) + 1.) ** (-1)
    return length / abs(velocity) * transmission

def calculate_conductivity_sc(phonons, max_n_iterations=None):
    if not max_n_iterations:
        max_n_iterations = MAX_ITERATIONS_SC

    velocities = phonons.keep_only_physical(phonons.velocities.real.reshape((phonons.n_phonons, 3), order='C') / 10)


    lambda_0 = phonons.tau[:, np.newaxis] * velocities[:, :]
    delta_lambda = np.copy(lambda_0)
    lambda_n = np.copy(lambda_0)
    conductivity_value = np.zeros((3, 3, max_n_iterations))

    for n_iteration in range(max_n_iterations):

        delta_lambda[:, :] = -1 * (phonons.tau[:, np.newaxis] * phonons.scattering_matrix_without_diagonal[:, :]).dot(
            delta_lambda[:, :])
        lambda_n += delta_lambda

        conductivity_value[:, :, n_iteration] = conductivity(phonons, lambda_n).sum(0)

    conductivity_per_mode = conductivity(phonons, lambda_n)
    if n_iteration == (max_n_iterations - 1):
        print('Max iterations reached')

    return conductivity_per_mode, conductivity_value

def calculate_conductivity_rta(phonons):
    volume = np.linalg.det(phonons.atoms.cell)
    gamma = phonons.gamma.reshape((phonons.n_k_points, phonons.n_modes)).copy()
    physical_modes = (phonons.frequencies > phonons.frequency_threshold)
    tau = 1 / gamma
    tau[np.invert(physical_modes)] = 0
    phonons.velocities[np.isnan(phonons.velocities)] = 0
    conductivity_per_mode = np.zeros((phonons.n_k_points, phonons.n_modes, 3, 3))
    conductivity_per_mode[:, :, :, :] = contract('kn,kna,kn,knb->knab', phonons.c_v[:, :], phonons.velocities[:, :, :], tau[:, :], phonons.velocities[:, :, :])
    conductivity_per_mode = 1e22 / (volume * phonons.n_k_points) * conductivity_per_mode
    conductivity_per_mode = conductivity_per_mode.reshape((phonons.n_phonons, 3, 3))
    return conductivity_per_mode


def calculate_c_v_2d(phonons):
    frequencies = phonons.frequencies
    c_v = np.zeros((phonons.n_k_points, phonons.n_modes, phonons.n_modes))
    temperature = phonons.temperature * KELVINTOTHZ
    physical_modes = frequencies > phonons.frequency_threshold

    if (phonons.is_classic):
        c_v[:, :, :] = KELVINTOJOULE
    else:
        f_be = phonons.occupations
        c_v_omega = KELVINTOJOULE * f_be * (f_be + 1) * frequencies / (temperature ** 2)
        c_v_omega[np.invert(physical_modes)] = 0
        freq_sq = (frequencies[:, :, np.newaxis] + frequencies[:, np.newaxis, :]) / 2 * (c_v_omega[:, :, np.newaxis] + c_v_omega[:, np.newaxis, :]) / 2
        c_v[:, :, :] = freq_sq
    return c_v

def calculate_conductivity_AF(phonons, gamma_in=None):
    volume = np.linalg.det(phonons.atoms.cell)
    if gamma_in is not None:
        gamma = gamma_in * np.ones((phonons.n_k_points, phonons.n_modes))
    else:
        gamma = phonons.gamma.reshape((phonons.n_k_points, phonons.n_modes)).copy()
    omega = 2 * np.pi * phonons.frequencies
    physical_modes = (phonons.frequencies[:, :, np.newaxis] > phonons.frequency_threshold) * \
                (phonons.frequencies[:, np.newaxis, :] > phonons.frequency_threshold)
    lorentz = (gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2 / (
                ((gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2) ** 2 +
                (omega[:, :, np.newaxis] - omega[:, np.newaxis, :]) ** 2)

    lorentz[np.invert(physical_modes)] = 0
    conductivity_per_mode = np.zeros((phonons.n_k_points, phonons.n_modes, phonons.n_modes, 3, 3))
    heat_capacity = calculate_c_v_2d(phonons)

    conductivity_per_mode[:, :, :, :, :] = contract('knm,knma,knm,knmb->knmab', heat_capacity,
                                                    phonons.velocities_AF[:, :, :, :], lorentz[:, :, :],
                                                    phonons.velocities_AF[:, :, :, :])
    conductivity_per_mode = 1e22 / (volume * phonons.n_k_points) * conductivity_per_mode
    kappa = contract('knmab->knab', conductivity_per_mode)
    kappa = kappa.reshape((phonons.n_phonons, 3, 3))
    return -1 * kappa

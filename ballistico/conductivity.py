from opt_einsum import contract
import ase.units as units
import numpy as np

MAX_ITERATIONS_SC = 500
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


def calculate_qhgk(phonons, gamma_in=None):
    if gamma_in is not None:
        gamma = gamma_in * np.ones((phonons.n_k_points, phonons.n_modes))
    else:
        gamma = phonons.gamma.reshape((phonons.n_k_points, phonons.n_modes)).copy()
    omega = phonons._omegas
    physical_modes_2d = (phonons.frequencies[:, :, np.newaxis] > phonons.frequency_threshold) * \
                        (phonons.frequencies[:, np.newaxis, :] > phonons.frequency_threshold)
    lorentz = (gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2 / (
            ((gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2) ** 2 +
            (omega[:, :, np.newaxis] - omega[:, np.newaxis, :]) ** 2)

    lorentz[np.invert(physical_modes_2d)] = 0
    heat_capacity = calculate_c_v_2d(phonons)

    conductivity_per_mode = contract('knm,knma,knm,knmb->knmab', heat_capacity,
                                     phonons.velocities_AF[:, :, :, :], lorentz[:, :, :],
                                     phonons.velocities_AF[:, :, :, :])
    conductivity_per_mode = contract('knmab->knab', conductivity_per_mode)
    conductivity_per_mode = conductivity_per_mode.reshape((phonons.n_phonons, 3, 3)).real
    volume = np.linalg.det(phonons.atoms.cell)
    return conductivity_per_mode * 1e22 / (volume * phonons.n_k_points)


def calculate_all(phonons, method, max_n_iterations, gamma_in=None):
    if max_n_iterations and method != 'sc':
        raise TypeError('Only phonons consistent method support n_iteration parameter')

    if method == 'qhgk':
        volume = np.linalg.det(phonons.atoms.cell)
        if gamma_in is not None:
            gamma = gamma_in * np.ones((phonons.n_k_points, phonons.n_modes))
        else:
            gamma = phonons.gamma.reshape((phonons.n_k_points, phonons.n_modes)).copy()
        omega = phonons._omegas
        physical_modes = (phonons.frequencies[:, :, np.newaxis] > phonons.frequency_threshold) * \
                         (phonons.frequencies[:, np.newaxis, :] > phonons.frequency_threshold)
        lorentz = (gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2 / (
                ((gamma[:, :, np.newaxis] + gamma[:, np.newaxis, :]) / 2) ** 2 +
                (omega[:, :, np.newaxis] - omega[:, np.newaxis, :]) ** 2)

        lorentz[np.isnan(lorentz)] = 0
        conductivity_per_mode = np.zeros((phonons.n_k_points, phonons.n_modes, phonons.n_modes, 3, 3))
        heat_capacity = calculate_c_v_2d(phonons)

        conductivity_per_mode[:, :, :, :, :] = contract('knm,knma,knm,knmb->knmab', heat_capacity,
                                                        phonons.velocities_AF[:, :, :, :], lorentz[:, :, :],
                                                        phonons.velocities_AF[:, :, :, :])
        conductivity_per_mode = contract('knmab->knab', conductivity_per_mode)

        conductivity_per_mode = conductivity_per_mode.reshape((phonons.n_phonons, 3, 3))
        conductivity_per_mode[np.invert(phonons._physical_modes), :, :] = 0

    elif method == 'inverse':
        velocities = phonons._keep_only_physical(phonons.velocities.real.reshape((phonons.n_phonons, 3), order='C'))
        scattering_inverse = np.linalg.inv(phonons._scattering_matrix)
        lambd = scattering_inverse.dot(velocities[:, :])
        physical_modes = (phonons.frequencies.reshape(phonons.n_phonons) > phonons.frequency_threshold)

        volume = np.linalg.det(phonons.atoms.cell)
        c_v = phonons._keep_only_physical(phonons.c_v.reshape((phonons.n_phonons), order='C'))
        conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))
        conductivity_per_mode[physical_modes, :, :] = c_v[:, np.newaxis, np.newaxis] * \
                                                      velocities[:, :, np.newaxis] * lambd[:, np.newaxis, :]
        neg_diag = (phonons._scattering_matrix.diagonal() < 0).sum()
        print('negative on diagonal : ', neg_diag)
        evals = np.linalg.eigvalsh(phonons._scattering_matrix)
        print('negative eigenvals : ', (evals < 0).sum())

    elif method == 'rta':
        volume = np.linalg.det(phonons.atoms.cell)
        gamma = phonons.gamma.reshape((phonons.n_k_points, phonons.n_modes)).copy()
        physical_modes = (phonons.frequencies > phonons.frequency_threshold)
        tau = 1 / gamma
        tau[np.invert(physical_modes)] = 0
        phonons.velocities[np.isnan(phonons.velocities)] = 0
        conductivity_per_mode = np.zeros((phonons.n_k_points, phonons.n_modes, 3, 3))
        conductivity_per_mode[:, :, :, :] = contract('kn,kna,kn,knb->knab', phonons.c_v[:, :],
                                                     phonons.velocities[:, :, :], tau[:, :],
                                                     phonons.velocities[:, :, :])
        conductivity_per_mode = conductivity_per_mode.reshape((phonons.n_phonons, 3, 3))
    elif method == 'sc':
        physical_modes = (phonons.frequencies.reshape(phonons.n_phonons) > phonons.frequency_threshold)
        volume = np.linalg.det(phonons.atoms.cell)
        c_v = phonons._keep_only_physical(phonons.c_v.reshape((phonons.n_phonons), order='C'))
        velocities = phonons._keep_only_physical(phonons.velocities.real.reshape((phonons.n_phonons, 3), order='C'))
        conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))

        if not max_n_iterations:
            max_n_iterations = MAX_ITERATIONS_SC

        gamma = phonons._keep_only_physical(phonons.gamma.reshape((phonons.n_phonons), order='C'))
        tau = 1 / gamma

        lambda_0 = np.einsum('i,ia->ia', tau, velocities)
        delta_lambda = np.copy(lambda_0)
        lambda_n = np.copy(lambda_0)

        for n_iteration in range(max_n_iterations):
            delta_lambda = np.einsum('i,ij,ja->ia', tau, phonons._scattering_matrix_without_diagonal, delta_lambda)
            lambda_n += delta_lambda

        conductivity_per_mode[physical_modes, :, :] = np.einsum('i,ia,ib->iab', c_v, velocities ,lambda_n)
        if n_iteration == (max_n_iterations - 1):
            print('Max iterations reached')

    else:
        raise TypeError('Conductivity method not recognized')
    if np.isnan(conductivity_per_mode).sum() != 0:
        print('nan')
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
    frequencies = phonons.frequencies.reshape ((phonons.n_phonons), order='C')
    physical_modes = (frequencies > phonons.frequency_threshold) #& (velocities > 0)[:, 2]
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

                        kn = abs(velocity / (length * single_gamma))
                        transmission = (1 - kn * (1 - np.exp(- 1. / kn))) * kn
                        gamma[mu] = abs(velocity) / length / transmission

                else:
                    gamma[mu] = phonons.gamma.reshape ((phonons.n_phonons), order='C')[mu]
            else:
                gamma[mu] = phonons.gamma.reshape ((phonons.n_phonons), order='C')[mu]
        tau_0 = 1 / gamma
        tau_0[np.invert(physical_modes)] = 0
        lambd_0[:, alpha] = tau_0[:] * velocities[:, alpha]
    c_v = phonons.c_v.reshape ((phonons.n_phonons), order='C')

    lambd_n = lambd_0.copy ()
    conductivity_per_mode = np.zeros ((phonons.n_phonons, 3, 3))
    avg_conductivity = None
    cond_iterations = []
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
        print ('Convergence not reached')
    if is_rta:
        return conductivity_per_mode * 1e22 / (volume * phonons.n_k_points)
    else:
        return conductivity_per_mode * 1e22 / (volume * phonons.n_k_points), np.array(cond_iterations) * 1e22 / (volume * phonons.n_k_points)


def conductivity(self, method='rta', max_n_iterations=None, length=None, axis=None, finite_length_method='matthiessen', gamma_in=None):
    # if length is not None:
    if method == 'rta':
        return calculate_conductivity_sc(self, length=length, axis=axis, is_rta=True, finite_size_method=finite_length_method, n_iterations=max_n_iterations)
    elif method == 'sc':
        return calculate_conductivity_sc(self, length=length, axis=axis, is_rta=False, finite_size_method=finite_length_method, n_iterations=max_n_iterations)
    elif (method == 'qhgk' or method == 'inverse'):
        return calculate_all(self, method=method, max_n_iterations=max_n_iterations,  gamma_in=gamma_in)
    else:
        raise TypeError('Conductivity method not implemented')


"""
Ballistico
Anharmonic Lattice Dynamics
"""
from opt_einsum import contract
import ase.units as units
import numpy as np
from ballistico.helpers.lazy_loading import save, get_folder_from_label
from ballistico.helpers.logger import get_logger
logging = get_logger()

MAX_ITERATIONS_SC = 200
EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
MAX_LENGTH_TRESHOLD = 1e15


def calculate_c_v_2d(phonons):
    frequencies = phonons.frequency
    c_v = np.zeros((phonons.n_k_points, phonons.n_modes, phonons.n_modes))
    temperature = phonons.temperature * KELVINTOTHZ
    physical_modes = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))

    if (phonons.is_classic):
        c_v[:, :, :] = KELVINTOJOULE
    else:
        f_be = phonons.population
        c_v_omega = KELVINTOJOULE * f_be * (f_be + 1) * frequencies / (temperature ** 2)
        c_v_omega[np.invert(physical_modes)] = 0
        freq_sq = (frequencies[:, :, np.newaxis] + frequencies[:, np.newaxis, :]) / 2 * (c_v_omega[:, :, np.newaxis] + c_v_omega[:, np.newaxis, :]) / 2
        c_v[:, :, :] = freq_sq
    return c_v


def conductivity(phonons, method='rta', max_n_iterations=None, length=None, finite_length_method='matthiessen', tolerance=None):
    if method == 'rta':
        conductivity = calculate_conductivity_sc(phonons, length=length, is_rta=True, finite_size_method=finite_length_method, n_iterations=max_n_iterations)
    elif method == 'sc':
        conductivity = calculate_conductivity_sc(phonons, length=length, is_rta=False, finite_size_method=finite_length_method, n_iterations=max_n_iterations, tolerance=tolerance)
    elif (method == 'qhgk'):
        conductivity = calculate_conductivity_qhgk(phonons)
    elif (method == 'inverse'):
        conductivity = calculate_conductivity_inverse(phonons, length=length, finite_size_method=finite_length_method)
    elif (method == 'eigenvectors'):
        conductivity = calculate_conductivity_with_evects(phonons)
    else:
        logging.error('Conductivity method not implemented')
    # TODO: remove this debugging info
    if method == 'eigenvectors':
        neg_diag = (phonons._scattering_matrix.diagonal() < 0).sum()
        logging.info('negative on diagonal : ' + str(neg_diag))
        evals = np.linalg.eigvalsh(phonons._scattering_matrix)
        logging.info('negative eigenvals : ' + str((evals < 0).sum()))

    folder = get_folder_from_label(phonons, '<temperature>/<statistics>/<third_bandwidth>')
    save('conductivity', folder + '/' + method, conductivity.reshape(phonons.n_k_points, phonons.n_modes, 3, 3), \
         format=phonons.store_format['conductivity'])
    sum = (conductivity.imag).sum()
    if sum > 1e-3:
        logging.warning('The conductivity has an immaginary part. Sum(Im(k)) = ' + str(sum))
    return conductivity.real


def calculate_conductivity_qhgk(phonons):
    volume = np.linalg.det(phonons.atoms.cell)
    diffusivity = phonons._generalized_diffusivity
    heat_capacity = phonons.heat_capacity.reshape(phonons.n_k_points, phonons.n_modes)
    conductivity_per_mode = contract('kn,knab->knab', heat_capacity, diffusivity)
    conductivity_per_mode = conductivity_per_mode.reshape((phonons.n_phonons, 3, 3))
    conductivity_per_mode = conductivity_per_mode / (volume * phonons.n_k_points)
    return conductivity_per_mode


def calculate_conductivity_inverse(phonons, length=np.array([None, None, None]), finite_size_method='matthiesen'):
    volume = np.linalg.det(phonons.atoms.cell)
    physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)
    conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))

    for alpha in range (3):
        velocity = phonons.velocity.real.reshape((phonons.n_phonons, 3))
        gamma = phonons.bandwidth.reshape(phonons.n_phonons)
        if length is not None:
            scattering_matrix = - 1 * phonons._scattering_matrix_without_diagonal + \
                            np.diag(gamma_with_matthiessen(gamma, velocity[:, alpha], length[alpha])[physical_modes])
        else:
            scattering_matrix = - 1 * phonons._scattering_matrix_without_diagonal + np.diag(gamma[physical_modes])
        scattering_inverse = np.linalg.inv(scattering_matrix)
        velocity = phonons._keep_only_physical(velocity)
        lambd = scattering_inverse.dot(velocity[:, alpha])
        c_v = phonons._keep_only_physical(phonons.heat_capacity.reshape((phonons.n_phonons)))
        physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)
        conductivity_per_mode[physical_modes, :, alpha] = c_v[:, np.newaxis] * \
                                                      velocity[:, :] * lambd[:, np.newaxis]
    return conductivity_per_mode / (volume * phonons.n_k_points)


def calculate_conductivity_with_evects(phonons):
    velocity = phonons._keep_only_physical(phonons.velocity.real.reshape((phonons.n_phonons, 3)))

    evals, evects = np.linalg.eig(phonons._scattering_matrix)
    new_physical_states = np.argwhere(evals >= 0)
    reduced_evects = evects[new_physical_states, new_physical_states]
    reduced_evals = evals[new_physical_states]
    reduced_scattering_inverse = np.zeros_like(phonons._scattering_matrix)
    reduced_scattering_inverse[new_physical_states, new_physical_states] = reduced_evects.dot(np.diag(1/reduced_evals)).dot(np.linalg.inv(reduced_evects))
    scattering_inverse = reduced_scattering_inverse
    # e, v = np.linalg.eig(a)
    # a = v.dot(np.diag(e)).dot(np.linalg.inv(v))

    lambd = scattering_inverse.dot(velocity[:, :])
    physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)

    volume = np.linalg.det(phonons.atoms.cell)
    c_v = phonons._keep_only_physical(phonons.heat_capacity.reshape((phonons.n_phonons)))
    conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))
    conductivity_per_mode[physical_modes, :, :] = c_v[:, np.newaxis, np.newaxis] * \
                                                  velocity[:, :, np.newaxis] * lambd[:, np.newaxis, :]
    return conductivity_per_mode / (volume * phonons.n_k_points)


def calculate_conductivity_sc(phonons, tolerance=None, length=np.array([None, None, None]),
                              is_rta=False, n_iterations=MAX_ITERATIONS_SC, finite_size_method='matthiesen'):
    if finite_size_method == 'matthiessen':
        mattiessen_length = length
    else:
        mattiessen_length = None
    if n_iterations is None:
        n_iterations = MAX_ITERATIONS_SC
    physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)
    velocity = phonons.velocity.real.reshape ((phonons.n_k_points, phonons.n_modes, 3))
    velocity = velocity.reshape((phonons.n_phonons, 3))
    lambd_n = calculate_sc_mfp(phonons=phonons,
                               tolerance=tolerance,
                               matthiessen_length=mattiessen_length,
                               is_rta=is_rta,
                               n_iterations=n_iterations)

    if finite_size_method == 'caltech':
        lambd_n = mfp_caltech(lambd_n, velocity, length, physical_modes)
    conductivity_per_mode = calculate_conductivity_per_mode(phonons.heat_capacity.reshape((phonons.n_phonons)),
                                                            velocity, lambd_n, physical_modes, phonons.n_phonons)

    volume = np.linalg.det (phonons.atoms.cell)
    return conductivity_per_mode / (volume * phonons.n_k_points)


def calculate_sc_mfp(phonons, tolerance=None, matthiessen_length=np.array([None, None, None]),
                     is_rta=False, n_iterations=MAX_ITERATIONS_SC):
    if n_iterations is None:
        n_iterations = MAX_ITERATIONS_SC
    velocity = phonons.velocity.real.reshape ((phonons.n_k_points, phonons.n_modes, 3))
    velocity = velocity.reshape((phonons.n_phonons, 3))
    physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)
    gamma = phonons.bandwidth.reshape(phonons.n_phonons)
    lambd_0 = mfp_matthiessen(gamma, velocity, matthiessen_length, physical_modes)
    if is_rta:
        return lambd_0
    else:
        lambd_n = np.zeros_like(lambd_0)
        avg_conductivity = None
        n_iteration = 0
        scattering_matrix = phonons._scattering_matrix_without_diagonal
        for n_iteration in range (n_iterations):
            conductivity_per_mode = calculate_conductivity_per_mode(phonons.heat_capacity.reshape((phonons.n_phonons)),
                                                                    velocity, lambd_n, physical_modes, phonons.n_phonons)
            new_avg_conductivity = np.diag (np.sum (conductivity_per_mode, 0)).mean ()
            if avg_conductivity:
                if tolerance is not None:
                    if np.abs (avg_conductivity - new_avg_conductivity) < tolerance:
                        break
            avg_conductivity = new_avg_conductivity
            delta_lambd = 1 / phonons.bandwidth.reshape ((phonons.n_phonons))[physical_modes, np.newaxis] \
                          * scattering_matrix.dot (lambd_n[physical_modes, :])
            lambd_n[physical_modes, :] = lambd_0[physical_modes, :] + delta_lambd[:, :]
        logging.info('Number of self-consistent iterations: ' + str(n_iteration))
        if n_iteration == (MAX_ITERATIONS_SC - 1):
            logging.warning('Max iterations reached. The result may be not converged.')
        return lambd_n


def calculate_conductivity_per_mode(heat_capacity, velocity, mfp, physical_modes, n_phonons):
    conductivity_per_mode = np.zeros ((n_phonons, 3, 3))
    conductivity_per_mode[physical_modes, :, :] = \
        heat_capacity[physical_modes, np.newaxis, np.newaxis] * velocity[physical_modes, :, np.newaxis] * \
        mfp[physical_modes, np.newaxis, :]
    return conductivity_per_mode


def gamma_with_matthiessen(gamma, velocity, length):
    gamma = gamma + abs(velocity) / length
    return gamma


def mfp_matthiessen(gamma, velocity, length, physical_modes):
    lambd_0 = np.zeros_like(velocity)
    for alpha in range(3):
        if length:
            if length[alpha] and length[alpha] != 0:
                gamma = gamma + abs(velocity[:, alpha]) / length[alpha]
        lambd_0[physical_modes, alpha] = 1 / gamma[physical_modes] * velocity[physical_modes, alpha]
    return lambd_0


def mfp_caltech(lambd, velocity, length, physical_modes):
    for alpha in range (3):
        reduced_physical_modes = physical_modes.copy() & (velocity[:, alpha] != 0)
        if length:
            if length[alpha]:
                lambd[reduced_physical_modes, alpha] = lambd[reduced_physical_modes, alpha] * \
                                  (1 - np.abs(lambd[reduced_physical_modes, alpha]) / length[alpha] *
                                  (1 - np.exp(-length[alpha] / np.abs(lambd[reduced_physical_modes, alpha]))))
    return lambd

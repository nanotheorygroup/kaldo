from opt_einsum import contract
from kaldo.grid import wrap_coordinates
from scipy.linalg.lapack import dsyev
from scipy.linalg.lapack import zheev
from ase.units import Rydberg, Bohr
from kaldo.helpers.tools import timeit

import numpy as np
import ase.units as units
from sparse import COO
from kaldo.controllers.dirac_kernel import lorentz_delta, gaussian_delta, triangular_delta

from kaldo.helpers.logger import get_logger
logging = get_logger()

KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J


def chi(qvec, list_of_replicas, cell_inv):
    chi_k = np.exp(1j * 2 * np.pi * list_of_replicas.dot(cell_inv.dot(qvec)))
    return chi_k


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


@timeit
def calculate_frequency(phonons, q_points):
    eigenvals = calculate_eigensystem(phonons, q_points, only_eigenvals=True)
    frequency = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
    return frequency.real


def calculate_dynmat_derivatives(phonons, q_points):
    atoms = phonons.atoms
    list_of_replicas = phonons.forceconstants.second_order.list_of_replicas
    replicated_cell = phonons.forceconstants.second_order.replicated_atoms.cell
    replicated_cell_inv = np.linalg.inv(phonons.forceconstants.second_order.replicated_atoms.cell)

    dynmat = phonons.forceconstants.second_order.dynmat(atoms.get_masses())
    positions = phonons.forceconstants.atoms.positions
    n_unit_cell = atoms.positions.shape[0]
    n_k_points = q_points.shape[0]
    n_replicas = phonons.forceconstants.n_replicas

    if phonons.forceconstants.distance_threshold:
        logging.info('Using folded flux operators')

    ddyn = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3, 3)).astype(np.complex)
    for index_k in range(n_k_points):
        qvec = q_points[index_k]
        if phonons._is_amorphous:
            distance = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distance = wrap_coordinates(distance, replicated_cell, replicated_cell_inv)
            dynmat_derivatives = contract('ija,ibjc->ibjca', distance, dynmat[0, :, :, 0, :, :])
        else:
            distance = positions[:, np.newaxis, np.newaxis, :] - (
                    positions[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])


            distance_to_wrap = positions[:, np.newaxis, np.newaxis, :] - (
                phonons.forceconstants.second_order.replicated_atoms.positions.reshape(n_replicas, n_unit_cell, 3)[np.newaxis, :, :, :])

            list_of_replicas = phonons.forceconstants.second_order.list_of_replicas
            cell_inv = phonons.forceconstants.cell_inv

            if phonons.forceconstants.distance_threshold:
                dynmat_derivatives = np.zeros((n_unit_cell, 3, n_unit_cell, 3, 3), dtype=np.complex)
                for l in range(n_replicas):
                    wrapped_distance = wrap_coordinates(distance_to_wrap[:, l, :, :], replicated_cell,
                                                        replicated_cell_inv)
                    mask = (np.linalg.norm(wrapped_distance, axis=-1) < phonons.forceconstants.distance_threshold)
                    id_i, id_j = np.argwhere(mask).T
                    dynmat_derivatives[id_i, :, id_j, :, :] += np.einsum('fa,fbc->fbca', distance[id_i, l, id_j, :], \
                                                                   dynmat[0, id_i, :, 0, id_j, :] *
                                                                         chi(qvec, list_of_replicas, cell_inv)[l])
            else:

                dynmat_derivatives = contract('ilja,ibljc,l->ibjca', distance, dynmat[0], chi(qvec, list_of_replicas, cell_inv))
        ddyn[index_k] = dynmat_derivatives.reshape((phonons.n_modes, phonons.n_modes, 3))
    return ddyn


def calculate_sij(phonons, q_points):
    dynmat_derivatives = calculate_dynmat_derivatives(phonons, q_points)
    eigenvects = calculate_eigensystem(phonons, q_points)[:, 1:, :]

    if phonons.is_antisymmetrizing_velocity:
        # TODO: Clean up the following logic to make it independent of the system
        if phonons._is_amorphous:
            error = np.linalg.norm(dynmat_derivatives + dynmat_derivatives.swapaxes(0, 1)) / 2
            dynmat_derivatives = (dynmat_derivatives - dynmat_derivatives.swapaxes(0, 1)) / 2
        else:
            error = np.linalg.norm(dynmat_derivatives + dynmat_derivatives.swapaxes(1, 2).conj()) / 2
            dynmat_derivatives = (dynmat_derivatives - dynmat_derivatives.swapaxes(1, 2).conj()) / 2

        logging.info('Velocity anti-symmetrization error: ' + str(error))
    logging.info('Calculating the flux operators')
    if phonons._is_amorphous:
        sij = np.tensordot(eigenvects[0], dynmat_derivatives[0], (0, 1))
        sij = np.tensordot(eigenvects[0], sij, (0, 1))
        sij = sij.reshape((1, sij.shape[0], sij.shape[1], sij.shape[2]))
    else:
        sij = contract('kim,kija,kjn->kmna', eigenvects.conj(), dynmat_derivatives, eigenvects)
    return sij


def calculate_velocity_af(phonons, q_points):
    sij = calculate_sij(phonons, q_points)
    frequency = calculate_frequency(phonons, q_points)
    sij = sij.reshape((q_points.shape[0], phonons.n_modes, phonons.n_modes, 3))
    velocity_AF = contract('kmna,kmn->kmna', sij,
                             1 / (2 * np.pi * np.sqrt(frequency[:, :, np.newaxis]) * np.sqrt(
                                 frequency[:, np.newaxis, :]))) / 2
    return velocity_AF


def calculate_velocity(phonons, q_points):
    velocity_AF = calculate_velocity_af(phonons, q_points)
    velocity = 1j * contract('kmma->kma', velocity_AF)
    return velocity.real



def calculate_eigensystem(phonons, q_points, only_eigenvals=False):
    atoms = phonons.atoms
    n_unit_cell = atoms.positions.shape[0]
    n_k_points = q_points.shape[0]
    n_replicas = phonons.forceconstants.n_replicas
    if phonons.forceconstants.distance_threshold:
        logging.info('Using folded dynamical matrix.')
    if phonons._is_amorphous:
        dtype = np.float
    else:
        dtype = np.complex
    if only_eigenvals:
        esystem = np.zeros((n_k_points, n_unit_cell * 3), dtype=dtype)
    else:
        esystem = np.zeros((n_k_points, n_unit_cell * 3 + 1, n_unit_cell * 3), dtype=dtype)
    for index_k in range(n_k_points):
        qvec = q_points[index_k]
        dynmat = phonons.forceconstants.second_order.dynmat(atoms.get_masses())
        is_at_gamma = (qvec == (0, 0, 0)).all()

        list_of_replicas = phonons.forceconstants.second_order.list_of_replicas
        cell_inv = phonons.forceconstants.cell_inv
        if phonons.forceconstants.distance_threshold:
            distance_threshold = phonons.forceconstants.distance_threshold
            dyn_s = np.zeros((n_unit_cell, 3, n_unit_cell, 3), dtype=np.complex)
            replicated_cell = phonons.forceconstants.second_order.replicated_atoms.cell
            replicated_cell_inv = np.linalg.inv(phonons.forceconstants.second_order.replicated_atoms.cell)

            for l in range(n_replicas):
                distance_to_wrap = atoms.positions[:, np.newaxis, :] - (
                    phonons.forceconstants.second_order.replicated_atoms.positions.reshape(n_replicas, n_unit_cell, 3)[np.newaxis, l, :, :])

                distance_to_wrap = wrap_coordinates(distance_to_wrap, replicated_cell, replicated_cell_inv)

                mask = np.linalg.norm(distance_to_wrap, axis=-1) < distance_threshold
                id_i, id_j = np.argwhere(mask).T

                dyn_s[id_i, :, id_j, :] += dynmat[0, id_i, :, 0, id_j, :] * chi(qvec, list_of_replicas, cell_inv)[l]
        else:
            if is_at_gamma:
                dyn_s = contract('ialjb->iajb', dynmat[0])
            else:
                dyn_s = contract('ialjb,l->iajb', dynmat[0], chi(qvec, list_of_replicas, cell_inv))
        dyn_s = dyn_s.reshape((phonons.n_modes, phonons.n_modes))
        if phonons.is_symmetrizing_frequency:
            dyn_s = 0.5 * (dyn_s + dyn_s.T.conj())
            error = np.sum(np.abs(0.5 * (dyn_s - dyn_s.T.conj())))
            logging.info('Frequency symmetrization error: ' + str(error))

        if only_eigenvals:
            evals = np.linalg.eigvalsh(dyn_s)
            esystem[index_k] = evals
        else:
            if is_at_gamma:
                evals, evects = dsyev(dyn_s)[:2]
            else:
                evals, evects = np.linalg.eigh(dyn_s)
                # evals, evects = zheev(dyn_s)[:2]
            esystem[index_k] = np.vstack((evals, evects))
    return esystem



def calculate_physical_modes(phonons):
    physical_modes = np.ones_like(phonons.frequency.reshape(phonons.n_phonons), dtype=bool)
    if phonons.min_frequency is not None:
        physical_modes = physical_modes & (phonons.frequency.reshape(phonons.n_phonons) > phonons.min_frequency)
    if phonons.max_frequency is not None:
        physical_modes = physical_modes & (phonons.frequency.reshape(phonons.n_phonons) < phonons.max_frequency)
    if phonons.is_nw:
        physical_modes[:4] = False
    else:
        physical_modes[:3] = False
    return physical_modes


def calculate_diffusivity_dense(phonons):
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

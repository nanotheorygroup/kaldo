from opt_einsum import contract
from ballistico.helpers.tools import wrap_positions_with_cell
from scipy.linalg.lapack import dsyev
import numpy as np
import ase.units as units
from sparse import COO
from ballistico.controllers.dirac_kernel import lorentz_delta

from ballistico.helpers.logger import get_logger
logging = get_logger()

KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J
THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
EVTOTENJOVERMOL = units.mol / (10 * units.J)

DELTA_DOS = 1
NUM_DOS = 100
FOLDER_NAME = 'ald-output'


def calculate_occupations(phonons):
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


def calculate_frequency(phonons, q_points=None):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        # TODO: we could do the check on the whole grid instead of the shape
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = wrap_positions_with_cell(q_points)
    eigenvals = calculate_eigensystem(phonons, q_points, only_eigenvals=True)
    frequency = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
    return frequency.real


def calculate_dynmat_derivatives(phonons, q_points=None):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        # TODO: we could do the check on the whole grid instead of the shape
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = wrap_positions_with_cell(q_points)
    atoms = phonons.atoms
    list_of_replicas = phonons.finite_difference.list_of_replicas
    replicated_cell = phonons.finite_difference.replicated_atoms.cell
    replicated_cell_inv = phonons.finite_difference.replicated_cell_inv
    dynmat = phonons.finite_difference.dynmat
    positions = phonons.finite_difference.atoms.positions

    n_unit_cell = atoms.positions.shape[0]
    n_k_points = q_points.shape[0]
    ddyn = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3, 3)).astype(np.complex)
    for index_k in range(n_k_points):
        qvec = q_points[index_k]
        if phonons._is_amorphous:
            dxij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dxij = wrap_positions_with_cell(dxij, replicated_cell, replicated_cell_inv)
            dynmat_derivatives = contract('ija,ibjc->ibjca', dxij, dynmat[:, :, 0, :, :])
        else:
            list_of_replicas = list_of_replicas
            dxij = positions[:, np.newaxis, np.newaxis, :] - (
                    positions[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])
            dynmat_derivatives = contract('ilja,ibljc,l->ibjca', dxij, dynmat, phonons.chi(qvec))
        ddyn[index_k] = dynmat_derivatives.reshape((phonons.n_modes, phonons.n_modes, 3), order='C')
    return ddyn


def calculate_sij(phonons, q_points=None, is_antisymmetrizing=False):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        # TODO: we could do the check on the whole grid instead of the shape
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = wrap_positions_with_cell(q_points)
    if is_main_mesh:
        dynmat_derivatives = phonons._dynmat_derivatives
        eigenvects = phonons._eigensystem[:, 1:, :]
    else:
        dynmat_derivatives = calculate_dynmat_derivatives(phonons, q_points)
        eigenvects = calculate_eigensystem(phonons, q_points)[:, 1:, :]

    if is_antisymmetrizing:
        error = np.linalg.norm(dynmat_derivatives + dynmat_derivatives.swapaxes(0, 1)) / 2
        dynmat_derivatives = (dynmat_derivatives - dynmat_derivatives.swapaxes(0, 1)) / 2
        logging.info('Symmetrization error: ' + str(error))
    if phonons._is_amorphous:
        sij = contract('kim,kija,kjn->kmna', eigenvects, dynmat_derivatives, eigenvects)
    else:
        sij = contract('kim,kija,kjn->kmna', eigenvects.conj(), dynmat_derivatives, eigenvects)
    return sij


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


def calculate_velocity_af(phonons, q_points=None, is_antisymmetrizing=False):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        # TODO: we could do the check on the whole grid instead of the shape
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = wrap_positions_with_cell(q_points)
    if is_main_mesh:
        sij = phonons.flux
        frequency = phonons.frequency
    else:
        sij = calculate_sij(phonons, q_points, is_antisymmetrizing)
        frequency = calculate_frequency(phonons, q_points)
    sij = sij.reshape((q_points.shape[0], phonons.n_modes, phonons.n_modes, 3))
    velocity_AF = contract('kmna,kmn->kmna', sij,
                             1 / (2 * np.pi * np.sqrt(frequency[:, :, np.newaxis]) * np.sqrt(
                                 frequency[:, np.newaxis, :]))) / 2
    return velocity_AF


def calculate_velocity(phonons, q_points=None, is_antisymmetrizing=False):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        # TODO: we could do the check on the whole grid instead of the shape
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = wrap_positions_with_cell(q_points)
    if is_main_mesh:
        velocity_AF = phonons._velocity_af
    else:
        velocity_AF = calculate_velocity_af(phonons, q_points, is_antisymmetrizing=is_antisymmetrizing)
    velocity = 1j * contract('kmma->kma', velocity_AF)
    return velocity.real


def calculate_eigensystem(phonons, q_points=None, only_eigenvals=False):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        # TODO: we could do the check on the whole grid instead of the shape
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = wrap_positions_with_cell(q_points)
    atoms = phonons.atoms
    n_unit_cell = atoms.positions.shape[0]
    n_k_points = q_points.shape[0]
    # Here we store the eigenvalues in the last column
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
        is_at_gamma = (qvec == (0, 0, 0)).all()
        dynmat = phonons.finite_difference.dynmat
        if is_at_gamma:
            dyn_s = contract('ialjb->iajb', dynmat)
        else:
            # TODO: the following espression could be done on the whole main_q_mesh
            dyn_s = contract('ialjb,l->iajb', dynmat, phonons.chi(qvec))
        dyn_s = dyn_s.reshape((phonons.n_modes, phonons.n_modes), order='C')
        if only_eigenvals:
            evals = np.linalg.eigvalsh(dyn_s)
            esystem[index_k] = evals
        else:
            if is_at_gamma:
                evals, evects = dsyev(dyn_s)[:2]
            else:
                evals, evects = np.linalg.eigh(dyn_s)
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
    physical_modes = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    physical_modes_2d = physical_modes[:, :, np.newaxis] & \
                        physical_modes[:, np.newaxis, :]
    delta_energy = omega[:, :, np.newaxis] - omega[:, np.newaxis, :]
    sigma = 2 * (diffusivity_bandwidth[:, :, np.newaxis] + diffusivity_bandwidth[:, np.newaxis, :])
    lorentz = lorentz_delta(delta_energy, sigma)
    lorentz = lorentz * np.pi
    lorentz[np.isnan(lorentz)] = 0

    sij = phonons.flux.reshape((phonons.n_k_points, phonons.n_modes, phonons.n_modes, 3))
    sij[np.invert(physical_modes_2d)] = 0

    prefactor = 1 / omega[:, :, np.newaxis] / omega[:, np.newaxis, :] / 4
    diffusivity = contract('knma,knm,knm,knmb->knab', sij, prefactor, lorentz, sij)
    return diffusivity


def calculate_diffusivity_sparse(phonons):
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
    data = np.pi * lorentz_delta(delta_energy, sigma, diffusivity_threshold)
    lorentz = COO(coords.T, data, shape=(phonons.n_k_points, phonons.n_modes, phonons.n_modes))
    s_ij = phonons.flux
    prefactor = 1 / (4 * omega[coords[:, 0], coords[:, 1]] * omega[coords[:, 0], coords[:, 2]])
    prefactor[np.invert(physical_modes_2d[coords[:, 0], coords[:, 1], coords[:, 2]])] = 0
    prefactor = COO(coords.T, prefactor, shape=(phonons.n_k_points, phonons.n_modes, phonons.n_modes))

    diffusivity = np.zeros((phonons.n_k_points, phonons.n_modes, 3, 3))
    for alpha in range(3):
        for beta in range(3):
            diffusivity[..., alpha, beta] = (s_ij[alpha] * prefactor * lorentz * s_ij[beta]).sum(axis=1).todense()
    return diffusivity


def calculate_generalized_diffusivity(phonons):
    if phonons.diffusivity_threshold is not None:
        diffusivity = calculate_diffusivity_sparse(phonons)
    else:
        diffusivity = calculate_diffusivity_dense(phonons)
    return diffusivity

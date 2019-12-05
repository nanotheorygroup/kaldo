"""
Ballistico
Anharmonic Lattice Dynamics
"""
from opt_einsum import contract
import numpy as np
import ase.units as units
from scipy.linalg.lapack import dsyev
from ballistico.helpers.tools import apply_boundary_with_cell
EVTOTENJOVERMOL = units.mol / (10 * units.J)
DELTA_DOS = 1
NUM_DOS = 100
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J
THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15



def calculate_density_of_states(frequencies, k_mesh, delta=DELTA_DOS, num=NUM_DOS):
    n_modes = frequencies.shape[-1]
    frequencies = frequencies.reshape((k_mesh[0], k_mesh[1], k_mesh[2], n_modes), order='C')
    n_k_points = np.prod(k_mesh)
    # increase_factor = 3
    omega_kl = np.zeros((n_k_points, n_modes))
    for mode in range(n_modes):
        omega_kl[:, mode] = frequencies[..., mode].flatten()
    # Energy axis and dos
    omega_e = np.linspace(0., np.amax(omega_kl) + 5e-3, num=num)
    dos_e = np.zeros_like(omega_e)
    # Sum up contribution from all q-points and branches
    for omega_l in omega_kl:
        diff_el = (omega_e[:, np.newaxis] - omega_l[np.newaxis, :]) ** 2
        dos_el = 1. / (diff_el + (0.5 * delta) ** 2)
        dos_e += dos_el.sum(axis=1)
    dos_e *= 1. / (n_k_points * np.pi) * 0.5 * delta
    return omega_e, dos_e



def calculate_k_points(phonons):
    k_size = phonons.kpts
    n_k_points = phonons.n_k_points
    k_points = np.zeros ((n_k_points, 3))
    for index_k in range (n_k_points):
        k_points[index_k] = np.unravel_index (index_k, k_size, order='C') / k_size
    return k_points

def calculate_dynamical_matrix(phonons):
    atoms = phonons.atoms
    second_order = phonons.finite_difference.second_order.copy()
    n_unit_cell_atoms = phonons.atoms.positions.shape[0]
    geometry = atoms.positions
    n_particles = geometry.shape[0]
    n_replicas = phonons.finite_difference.n_replicas
    is_second_reduced = (second_order.size == n_particles * 3 * n_replicas * n_particles * 3)
    if is_second_reduced:
        dynmat = second_order.reshape((n_particles, 3, n_replicas, n_particles, 3), order='C')
    else:
        dynmat = second_order.reshape((n_replicas, n_particles, 3, n_replicas, n_particles, 3), order='C')[0]
    mass = np.sqrt(atoms.get_masses())
    dynmat /= mass[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    dynmat /= mass[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    dynmat *= EVTOTENJOVERMOL
    return dynmat


def calculate_eigensystem(phonons, k_list=None):
    if k_list is not None:
        k_points = k_list
    else:
        k_points = phonons.k_points
    atoms = phonons.atoms
    n_unit_cell = atoms.positions.shape[0]
    n_k_points = k_points.shape[0]

    # Here we store the eigenvalues in the last column
    if phonons._is_amorphous:
        eigensystem = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3 + 1))
    else:
        eigensystem = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3 + 1)).astype(np.complex)
    for index_k in range(n_k_points):
        eigensystem[index_k, :, -1], eigensystem[index_k, :, :-1] = calculate_eigensystem_for_k(phonons, k_points[index_k])
    return eigensystem


def calculate_second_order_observable(phonons, observable, k_list=None):
    if k_list is not None:
        k_points = k_list
    else:
        k_points = phonons.k_points
    atoms = phonons.atoms
    n_unit_cell = atoms.positions.shape[0]
    n_k_points = k_points.shape[0]
    if observable == 'frequencies':
        tensor = np.zeros((n_k_points, n_unit_cell * 3))
        function = calculate_frequencies_for_k
    elif observable == 'dynmat_derivatives':
        tensor = np.zeros((n_k_points, n_unit_cell * 3,  n_unit_cell * 3, 3)).astype(np.complex)
        function = calculate_dynmat_derivatives_for_k
    elif observable == 'velocities_AF':
        tensor = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3, 3)).astype(np.complex)
        function = calculate_velocities_AF_for_k
    elif observable == 'velocities':
        tensor = np.zeros((n_k_points, n_unit_cell * 3, 3))
        function = calculate_velocities_for_k
    else:
        raise TypeError('Operator not recognized')
    for index_k in range(n_k_points):
        tensor[index_k] = function(phonons, k_points[index_k])
    return tensor


def calculate_eigensystem_for_k(phonons, qvec, only_eigenvals=False):
    dynmat = phonons.dynmat
    atoms = phonons.atoms
    geometry = atoms.positions
    n_particles = geometry.shape[0]
    n_phonons = n_particles * 3
    if phonons._is_amorphous:
        dyn_s = dynmat[:, :, 0, :, :]
    else:
        dyn_s = contract('ialjb,l->iajb', dynmat, phonons._chi(qvec))
    dyn_s = dyn_s.reshape((n_phonons, n_phonons), order='C')
    if only_eigenvals:
        evals = np.linalg.eigvalsh(dyn_s)
        return evals
    else:
        if (qvec == [0, 0, 0]).all():
            evals, evects = dsyev(dyn_s)[:2]
        else:
            evals, evects = np.linalg.eigh(dyn_s)

        return evals, evects

def calculate_dynmat_derivatives_for_k(phonons, qvec):
    dynmat = phonons.dynmat
    atoms = phonons.atoms
    geometry = atoms.positions
    n_particles = geometry.shape[0]
    n_phonons = n_particles * 3
    geometry = atoms.positions
    if phonons._is_amorphous:
        dxij = geometry[:, np.newaxis, :] - geometry[np.newaxis, :, :]
        dxij = apply_boundary_with_cell(dxij, phonons.replicated_cell, phonons.replicated_cell_inv)
        dynmat_derivatives = contract('ija,ibjc->ibjca', dxij, dynmat[:, :, 0, :, :])
    else:
        list_of_replicas = phonons.finite_difference.list_of_replicas
        dxij = geometry[:, np.newaxis, np.newaxis, :] - (geometry[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])
        dynmat_derivatives = contract('ilja,ibljc,l->ibjca', dxij, dynmat, phonons._chi(qvec))
    dynmat_derivatives = dynmat_derivatives.reshape((n_phonons, n_phonons, 3), order='C')
    return dynmat_derivatives

def calculate_frequencies_for_k(phonons, qvec):
    rescaled_qvec = qvec * phonons.kpts
    if (np.round(rescaled_qvec) == qvec * phonons.kpts).all():
        k_index = int(np.argwhere((phonons.k_points == qvec).all(axis=1)).flatten())
        eigenvals = phonons.eigenvalues[k_index]
    else:
        eigenvals = calculate_eigensystem_for_k(phonons, qvec, only_eigenvals=True)
    frequencies = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
    return frequencies

def calculate_velocities_AF_for_k(phonons, qvec):
    rescaled_qvec = qvec * phonons.kpts
    if (np.round(rescaled_qvec) == qvec * phonons.kpts).all():
        k_index = int(np.argwhere((phonons.k_points == qvec).all(axis=1)).flatten())
        dynmat_derivatives = phonons._dynmat_derivatives[k_index]
        frequencies = phonons.frequencies[k_index]
        eigenvects = phonons.eigenvectors[k_index]
        physical_modes = phonons._physical_modes.reshape((phonons.n_k_points, phonons.n_modes))[k_index]
    else:
        dynmat_derivatives = calculate_dynmat_derivatives_for_k(phonons, qvec)
        frequencies = calculate_frequencies_for_k(phonons, qvec)
        _, eigenvects = calculate_eigensystem_for_k(phonons, qvec)
        physical_modes = frequencies > phonons.frequency_threshold

    velocities_AF = contract('im,ija,jn->mna', eigenvects[:, :].conj(), dynmat_derivatives, eigenvects[:, :])
    velocities_AF = contract('mna,mn->mna', velocities_AF,
                                      1 / (2 * np.pi * np.sqrt(frequencies[:, np.newaxis]) * np.sqrt(frequencies[np.newaxis, :])))
    velocities_AF[np.invert(physical_modes), :, :] = 0
    velocities_AF[:, np.invert(physical_modes), :] = 0
    velocities_AF = velocities_AF / 2
    return velocities_AF

def calculate_velocities_for_k(phonons, qvec):
    rescaled_qvec = qvec * phonons.kpts
    if (np.round(rescaled_qvec) == qvec * phonons.kpts).all():
        k_index = int(np.argwhere((phonons.k_points == qvec).all(axis=1)).flatten())
        velocities_AF = phonons._velocities_af[k_index]
    else:
        velocities_AF = calculate_velocities_AF_for_k(phonons, qvec)

    velocities = 1j * np.diagonal(velocities_AF).T
    return velocities

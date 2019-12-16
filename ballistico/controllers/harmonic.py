"""
Ballistico
Anharmonic Lattice Dynamics
"""

import numpy as np
import ase.units as units
from ballistico.helpers.tools import apply_boundary_with_cell
from ballistico.harmonic_single_q import HarmonicSingleQ
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


def calculate_second_order_observable(phonons, observable, q_points=None):
    if q_points is None:
        q_points = phonons._q_vec_from_q_index(np.arange(phonons.n_k_points))
    else:
        q_points = apply_boundary_with_cell(q_points)

    atoms = phonons.atoms
    n_unit_cell = atoms.positions.shape[0]
    n_k_points = q_points.shape[0]

    if observable == 'frequencies':
        tensor = np.zeros((n_k_points, n_unit_cell * 3))
    elif observable == 'dynmat_derivatives':
        tensor = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3, 3)).astype(np.complex)
    elif observable == 'velocities_AF':
        tensor = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3, 3)).astype(np.complex)
    elif observable == 'velocities':
        tensor = np.zeros((n_k_points, n_unit_cell * 3, 3))
    elif observable == 'eigensystem':
        # Here we store the eigenvalues in the last column
        if phonons._is_amorphous:
            tensor = np.zeros((n_k_points, n_unit_cell * 3 + 1, n_unit_cell * 3))
        else:
            tensor = np.zeros((n_k_points, n_unit_cell * 3 + 1, n_unit_cell * 3)).astype(np.complex)
    else:
        raise ValueError('observable not recognized')

    for index_k in range(n_k_points):
        qvec = q_points[index_k]
        hsq = HarmonicSingleQ(qvec=qvec,
                              dynmat=phonons.finite_difference.dynmat,
                              positions=phonons.atoms.positions,
                              replicated_cell=phonons.replicated_cell,
                              replicated_cell_inv=phonons.replicated_cell_inv,
                              cell_inv=phonons.cell_inv,
                              list_of_replicas=phonons.finite_difference.list_of_replicas,
                              frequency_threshold=phonons.frequency_threshold
                              )
        if observable == 'frequencies':
            tensor[index_k] = hsq.calculate_frequencies()
        elif observable == 'dynmat_derivatives':
            tensor[index_k] = hsq.calculate_dynmat_derivatives()
        elif observable == 'velocities_AF':
            tensor[index_k] = hsq.calculate_velocities_AF()
        elif observable == 'velocities':
            tensor[index_k] = hsq.calculate_velocities()
        elif observable == 'eigensystem':
            tensor[index_k] = hsq.calculate_eigensystem()
        else:
            raise ValueError('observable not recognized')

    return tensor


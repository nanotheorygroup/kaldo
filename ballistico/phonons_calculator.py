import numpy as np
import scipy.special
import sparse
import ase.units as units
from opt_einsum import contract

import tensorflow as tf
tf.enable_eager_execution()
#
# def contract(*operands, **kwargs):
#     operands_tf = []
#     is_complex = False
#     for i in range(len (operands)):
#         operand = operands[i]
#         if i==0:
#             operands_tf.append(operand)
#         else:
#             operands_tf.append(tf.convert_to_tensor (operand, operand.dtype))
#             if (operands_tf[i].dtype == tf.complex128):
#                 is_complex = True
#     if is_complex:
#         for i in range (1, len (operands)):
#             operands_tf[i] = tf.dtypes.cast (operands_tf[i], tf.complex128)
#
#     out = tf.einsum(*operands_tf, **kwargs)
#     return np.array(out)


IS_SCATTERING_MATRIX_ENABLED = True
IS_SORTING_EIGENVALUES = False
# DIAGONALIZATION_ALGORITHM = scipy.linalg.lapack.ssytrd
# DIAGONALIZATION_ALGORITHM = np.linalg.eigh
DIAGONALIZATION_ALGORITHM = scipy.linalg.lapack.zheev

IS_DELTA_CORRECTION_ENABLED = True
DELTA_THRESHOLD = 2
DELTA_DOS = 1
NUM_DOS = 100


def calculate_density_of_states(frequencies, k_mesh, delta=DELTA_DOS, num=NUM_DOS):
    n_modes = frequencies.shape[-1]
    frequencies = frequencies.reshape ((k_mesh[0], k_mesh[1], k_mesh[2], n_modes), order='C')
    n_k_points = np.prod (k_mesh)
    # increase_factor = 3
    omega_kl = np.zeros ((n_k_points, n_modes))
    for mode in range (n_modes):
        omega_kl[:, mode] = frequencies[..., mode].flatten ()
    # Energy axis and dos
    omega_e = np.linspace (0., np.amax (omega_kl) + 5e-3, num=num)
    dos_e = np.zeros_like (omega_e)
    # Sum up contribution from all q-points and branches
    for omega_l in omega_kl:
        diff_el = (omega_e[:, np.newaxis] - omega_l[np.newaxis, :]) ** 2
        dos_el = 1. / (diff_el + (0.5 * delta) ** 2)
        dos_e += dos_el.sum (axis=1)
    dos_e *= 1. / (n_k_points * np.pi) * 0.5 * delta
    return omega_e, dos_e


def diagonalize_second_order_single_k(qvec, atoms, second_order, list_of_replicas, replicated_cell, frequencies_threshold):

    geometry = atoms.positions
    cell_inv = np.linalg.inv (atoms.cell)
    kpoint = 2 * np.pi * (cell_inv).dot (qvec)

    n_particles = geometry.shape[0]
    n_replicas = list_of_replicas.shape[0]
    n_phonons = n_particles * 3
    is_second_reduced = (second_order.size == n_particles * 3 * n_replicas * n_particles * 3)
    if is_second_reduced:
        dynmat = second_order.reshape ((n_particles, 3, n_replicas, n_particles, 3), order='C')
    else:
        dynmat = second_order.reshape((n_replicas, n_particles, 3, n_replicas, n_particles, 3), order='C')[0]
    mass = np.sqrt(atoms.get_masses ())
    dynmat /= mass[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    dynmat /= mass[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    
    # TODO: probably we want to move this unit conversion somewhere more appropriate
    dynmat /= (10 * units.J / units.mol)

    chi_k = np.zeros(n_replicas).astype(complex)
    for id_replica in range (n_replicas):
        chi_k[id_replica] = np.exp (1j * list_of_replicas[id_replica].dot (kpoint))
    dyn_s = contract('ialjb,l->iajb', dynmat, chi_k)
    replicated_cell_inv = np.linalg.inv(replicated_cell)
    dxij = apply_boundary_with_cell (replicated_cell, replicated_cell_inv, geometry[:, np.newaxis, np.newaxis] - (
            geometry[np.newaxis, :, np.newaxis] + list_of_replicas[np.newaxis, np.newaxis, :]))
    ddyn_s = 1j * contract('ijla,l,ibljc->ibjca', dxij, chi_k, dynmat)

    out = DIAGONALIZATION_ALGORITHM (dyn_s.reshape ((n_phonons, n_phonons), order='C'))
    eigenvals, eigenvects = out[0], out[1]
    if IS_SORTING_EIGENVALUES:
        idx = eigenvals.argsort ()
        eigenvals = eigenvals[idx]
        eigenvects = eigenvects[:, idx]

    frequencies = np.abs (eigenvals) ** .5 * np.sign (eigenvals) / (np.pi * 2.)

    ddyn = ddyn_s.reshape (n_phonons, n_phonons, 3, order='C')
    velocities = np.zeros ((frequencies.shape[0], 3), dtype=np.complex)
    vel = contract('ki,ija,jq->kqa',eigenvects.conj().T, ddyn, eigenvects)
    for alpha in range (3):
        for mu in range (n_particles * 3):
            if frequencies[mu] > frequencies_threshold:
                velocities[mu, alpha] = vel[mu, mu, alpha] / (2 * (2 * np.pi) * frequencies[mu])

    return frequencies, eigenvals, eigenvects, velocities


def calculate_second_k_list(k_points, atoms, second_order, list_of_replicas, replicated_cell, frequencies_threshold):
    n_unit_cell = atoms.positions.shape[0]
    n_k_points = k_points.shape[0]

    
    frequencies = np.zeros ((n_k_points, n_unit_cell * 3))
    eigenvalues = np.zeros ((n_k_points, n_unit_cell * 3))
    eigenvectors = np.zeros ((n_k_points, n_unit_cell * 3, n_unit_cell * 3)).astype (np.complex)
    velocities = np.zeros ((n_k_points, n_unit_cell * 3, 3))
    for index_k in range (n_k_points):
        freq, eval, evect, vels = diagonalize_second_order_single_k (k_points[index_k], atoms, second_order.copy(),
                                                                     list_of_replicas, replicated_cell,
                                                                     frequencies_threshold)
        frequencies[index_k, :] = freq
        eigenvalues[index_k, :] = eval
        eigenvectors[index_k, :, :] = evect
        velocities[index_k, :, :] = -1 * vels.real

    return frequencies, eigenvalues, eigenvectors, velocities


def calculate_broadening(velocity, cellinv, k_size):
    # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
    delta_k = cellinv / k_size
    base_sigma = ((np.tensordot (velocity, delta_k, [-1, 1])) ** 2).sum (axis=-1)
    base_sigma = np.sqrt (base_sigma / 6.)
    return base_sigma

def gaussian_delta(params):
    # alpha is a factor that tells whats the ration between the width of the gaussian
    # and the width of allowed phase space
    delta_energy = params[0]
    # allowing processes with width sigma and creating a gaussian with width sigma/2
    # we include 95% (erf(2/sqrt(2)) of the probability of scattering. The erf makes the total area 1
    sigma = params[1]
    if IS_DELTA_CORRECTION_ENABLED:
        correction = scipy.special.erf(DELTA_THRESHOLD / np.sqrt(2))
    else:
        correction = 1
    gaussian = 1 / np.sqrt (np.pi * sigma ** 2) * np.exp (- delta_energy ** 2 / (sigma ** 2))
    return gaussian / correction


def triangular_delta(params):
    delta_energy = np.abs(params[0])
    deltaa = np.abs(params[1])
    out = np.zeros_like(delta_energy)
    out[delta_energy < deltaa] = 1. / deltaa * (1 - delta_energy[delta_energy < deltaa] / deltaa)
    return out


def lorentzian_delta(params):
    delta_energy = params[0]
    gamma = params[1]
    if IS_DELTA_CORRECTION_ENABLED:
        # numerical value of the integral of a lorentzian over +- DELTA_TRESHOLD * gamma
        corrections = {
            1 :0.704833,
            2 :0.844042,
            3 :0.894863,
            4 :0.920833,
            5 :0.936549,
            6 :0.947071,
            7 :0.954604,
            8 :0.960263,
            9 :0.964669,
           10 :0.968195}
        correction = corrections[DELTA_THRESHOLD]
    else:
        correction = 1
    lorentzian = 1 / np.pi * 1 / 2 * gamma / (delta_energy ** 2 + (gamma / 2) ** 2)
    return lorentzian / correction


# @profile
def calculate_single_gamma(is_plus, index_k, mu, i_k, i_kp_full, index_kp_full, frequencies, velocities, density, cell_inv, k_size,
                           n_modes, nptk, n_replicas, evect, chi, third_order, sigma_in,
                           frequencies_threshold, is_amorphous, broadening_function):

    omegas = 2 * np.pi * frequencies
    nu = np.ravel_multi_index([index_k, mu], [nptk, n_modes], order='C')
    evect = evect.swapaxes(1, 2).reshape(nptk * n_modes, n_modes, order='C')
    evect_dagger = evect.reshape((nptk * n_modes, n_modes), order='C').conj()

    scaled_potential = sparse.tensordot(third_order, evect[nu, :], [0, 0])
    scaled_potential = scaled_potential.reshape((n_replicas, n_modes, n_replicas, n_modes), order='C')

    i_kpp_vec = i_k[:, np.newaxis] + (int(is_plus) * 2 - 1) * i_kp_full[:, :]
    index_kpp_vec = np.ravel_multi_index(i_kpp_vec, k_size, order='C', mode='wrap')
    # +1 if is_plus, -1 if not is_plus
    second_sign = (int(is_plus) * 2 - 1)
    if sigma_in is None:
        sigma_tensor_np = calculate_broadening(velocities[index_kp_full, :, np.newaxis, :] -
                                               velocities[index_kpp_vec, np.newaxis, :, :], cell_inv, k_size)
        sigma_small = sigma_tensor_np
    else:
        sigma_small = sigma_in

    omegas_difference = np.abs(omegas[index_k, mu] + second_sign * omegas[index_kp_full, :, np.newaxis] -
                          omegas[index_kpp_vec, np.newaxis, :])

    condition = (omegas_difference < DELTA_THRESHOLD * 2 * np.pi * sigma_small) & \
                (frequencies[index_kp_full, :, np.newaxis] > frequencies_threshold) & \
                (frequencies[index_kpp_vec, np.newaxis, :] > frequencies_threshold)
    interactions = np.array(np.where(condition)).T
    # TODO: Benchmark something fast like
    # interactions = np.array(np.unravel_index (np.flatnonzero (condition), condition.shape)).T
    if interactions.size != 0:
        # print('interactions: ' + str (interactions.size))
        index_kp_vec = interactions[:, 0]
        index_kpp_vec = index_kpp_vec[index_kp_vec]
        mup_vec = interactions[:, 1]
        mupp_vec = interactions[:, 2]
        nup_vec = np.ravel_multi_index(np.array([index_kp_vec, mup_vec]),
                                       np.array([nptk, n_modes]), order='C')
        nupp_vec = np.ravel_multi_index(np.array([index_kpp_vec, mupp_vec]),
                                        np.array([nptk, n_modes]), order='C')

        if is_plus:
            dirac_delta = density[nup_vec] - density[nupp_vec]

        else:
            dirac_delta = .5 * (1 + density[nup_vec] + density[nupp_vec])

        dirac_delta /= (omegas[index_kp_vec, mup_vec] * omegas[index_kpp_vec, mupp_vec])
        if sigma_in is None:
            dirac_delta *= broadening_function([omegas_difference[index_kp_vec, mup_vec, mupp_vec], 2 * np.pi * sigma_small[
                index_kp_vec, mup_vec, mupp_vec]])

        else:
            dirac_delta *= broadening_function(
                [omegas_difference[index_kp_vec, mup_vec, mupp_vec], sigma_in])
        dirac_delta_sqrt = np.sqrt(dirac_delta)
        if is_plus:
            first_evect = evect[nup_vec]
            first_chi = chi[index_kp_vec]
        else:
            first_evect = evect_dagger[nup_vec]
            first_chi = chi.conj()[index_kp_vec]
        second_evect = evect_dagger[nupp_vec]
        second_chi = chi.conj()[index_kpp_vec]
        if is_amorphous:
            if is_plus:
                scaled_potential = contract ('litj,aj,ai->a', scaled_potential, second_evect, first_evect)
            else:
                scaled_potential = contract ('litj,aj,ai->a', scaled_potential, second_evect, first_evect)
        else:

            if n_modes > n_replicas:
                # do replicas dirst

                scaled_potential = contract('litj,al,at->ija',
                                            scaled_potential,
                                            first_chi,
                                            second_chi)
                scaled_potential = contract('ija,aj,ai,->a',
                                            scaled_potential,
                                            second_evect,
                                            first_evect)
            else:
                # do modes first

                scaled_potential = contract('litj,aj,ai->lta',
                                                      scaled_potential,
                                                      second_evect,
                                                      first_evect)
                scaled_potential = contract('lta,al,at->a',
                                                      scaled_potential,
                                                      first_chi,
                                                      second_chi)

        # gamma contracted on one index
        pot_times_dirac = np.abs(scaled_potential * dirac_delta_sqrt)
        gamma_coeff = units._hbar * units.mol ** 3 / units.J ** 2 * 1e9 * np.pi / 4.
        pot_times_dirac = pot_times_dirac ** 2 / omegas[index_k, mu] / nptk * gamma_coeff
        return nup_vec, nupp_vec, pot_times_dirac


def calculate_gamma(atoms, frequencies, velocities, density, k_size, eigenvectors, list_of_replicas, third_order,
                    sigma_in, broadening, frequencies_threshold):
    density = density.flatten(order='C')
    nptk = np.prod (k_size)
    n_particles = atoms.positions.shape[0]

    print('Lifetime calculation')
    n_modes = n_particles * 3

    # TODO: We should write this in a better way
    if list_of_replicas.shape == (3,):
        n_replicas = 1
    else:
        n_replicas = list_of_replicas.shape[0]

    cell_inv = np.linalg.inv(atoms.cell)

    is_amorphous = (k_size == (1, 1, 1)).all()

    if is_amorphous:
        chi = 1
    else:
        rlattvec = cell_inv * 2 * np.pi
        chi = np.zeros ((nptk, n_replicas), dtype=np.complex)
        for index_k in range (np.prod (k_size)):
            i_k = np.array (np.unravel_index (index_k, k_size, order='C'))
            k_point = i_k / k_size
            realq = np.matmul (rlattvec, k_point)
            for l in range (n_replicas):
                chi[index_k, l] = np.exp (1j * list_of_replicas[l].dot (realq))
    print('Projection started')
    gamma = np.zeros ((2, nptk, n_modes))
    n_modes = n_particles * 3
    nptk = np.prod (k_size)

    list_of_k = np.arange(np.prod(k_size))

    # print('n_irreducible_q_points = ' + str(int(len(unique_points))) + ' : ' + str(unique_points))
    process = ['Minus processes: ', 'Plus processes: ']
    masses = atoms.get_masses()
    rescaled_eigenvectors = eigenvectors[:, :, :].reshape((nptk, n_particles, 3, n_modes), order='C') / np.sqrt(
        masses[np.newaxis, :, np.newaxis, np.newaxis])
    rescaled_eigenvectors = rescaled_eigenvectors.reshape((nptk, n_particles * 3, n_modes), order='C')
    nu_list = [[], []]
    nup_list = [[], []]
    nupp_list = [[], []]
    pot_times_dirac_list = [[], []]

    index_kp_vec = np.arange(np.prod(k_size))
    i_kp_vec = np.array(np.unravel_index(index_kp_vec, k_size, order='C'))

    is_amorphous = (nptk == 1)
    if broadening == 'gauss':
        broadening_function = gaussian_delta
    elif broadening == 'lorentz':
        broadening_function = lorentzian_delta
    elif broadening == 'triangle':
        broadening_function = triangular_delta

    for is_plus in (1, 0):
        for index_k in (list_of_k):
            i_k = np.array(np.unravel_index(index_k, k_size, order='C'))

            for mu in range(n_modes):
                if frequencies[index_k, mu] > frequencies_threshold:

                    gamma_out = calculate_single_gamma(is_plus, index_k, mu, i_k, i_kp_vec, index_kp_vec, frequencies, velocities, density,
                                                       cell_inv, k_size, n_modes, nptk, n_replicas,
                                                       rescaled_eigenvectors, chi, third_order, sigma_in,
                                                       frequencies_threshold, is_amorphous, broadening_function)
                    if gamma_out:
                        nup_vec, nupp_vec, pot_times_dirac = gamma_out
                        gamma[is_plus, index_k, mu] = pot_times_dirac.sum()
                        nu = np.ravel_multi_index([index_k, mu], [nptk, n_modes], order='C')
                        nu_vec = np.ones(nup_vec.shape[0]).astype(int) * nu

                        nu_list[is_plus].extend(nu_vec)
                        nup_list[is_plus].extend(nup_vec)
                        nupp_list[is_plus].extend(nupp_vec)
                        pot_times_dirac_list[is_plus].extend(pot_times_dirac)
            print(process[is_plus] + 'q-point = ' + str(index_k))
    return nu_list, nup_list, nupp_list, pot_times_dirac_list, gamma




def apply_boundary_with_cell(cell, cellinv, dxij):
    # exploit periodicity to calculate the shortest distance, which may not be the one we have
    sxij = dxij.dot(cellinv)
    sxij = sxij - np.round (sxij)
    dxij = sxij.dot(cell)
    return dxij
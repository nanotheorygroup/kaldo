import numpy as np
from ballistico.logger import Logger
import scipy.special
from opt_einsum import contract
from sparse import COO
import ballistico.constants as constants


IS_SCATTERING_MATRIX_ENABLED = True
IS_SORTING_EIGENVALUES = False
# DIAGONALIZATION_ALGORITHM = scipy.linalg.lapack.zheev
# DIAGONALIZATION_ALGORITHM = scipy.linalg.lapack.ssytrd
# DIAGONALIZATION_ALGORITHM = np.linalg.eigh
IS_DELTA_CORRECTION_ENABLED = False
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


def diagonalize_second_order_single_k(qvec, atoms, second_order, list_of_replicas, replicated_atoms, energy_threshold):

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
    dynmat /= constants.tenjovermol


    DIAGONALIZATION_ALGORITHM = scipy.linalg.lapack.zheev
    chi_k = np.zeros(n_replicas).astype(complex)
    for id_replica in range (n_replicas):
        chi_k[id_replica] = np.exp (1j * list_of_replicas[id_replica].dot (kpoint))
    dyn_s = contract('ialjb,l->iajb', dynmat, chi_k)
    replicated_cell_inv = np.linalg.inv(replicated_atoms.cell)
    dxij = apply_boundary_with_cell (replicated_atoms.cell, replicated_cell_inv, geometry[:, np.newaxis, np.newaxis] - (
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
            if np.abs(frequencies[mu]) > energy_threshold:
                velocities[mu, alpha] = vel[mu, mu, alpha] / (2 * (2 * np.pi) * frequencies[mu])

    if velocities is None:
        velocities = 0
    return frequencies, eigenvals, eigenvects, velocities


def calculate_second_k_list(k_points, atoms, second_order, list_of_replicas, replicated_atoms, energy_threshold):
    n_unit_cell = atoms.positions.shape[0]
    n_k_points = k_points.shape[0]

    frequencies = np.zeros ((n_k_points, n_unit_cell * 3))
    eigenvalues = np.zeros ((n_k_points, n_unit_cell * 3))
    eigenvectors = np.zeros ((n_k_points, n_unit_cell * 3, n_unit_cell * 3)).astype (np.complex)
    velocities = np.zeros ((n_k_points, n_unit_cell * 3, 3))
    for index_k in range (n_k_points):
        freq, eval, evect, vels = diagonalize_second_order_single_k (k_points[index_k], atoms, second_order.copy(),
                                                                     list_of_replicas, replicated_atoms,
                                                                     energy_threshold)
        frequencies[index_k, :] = freq
        eigenvalues[index_k, :] = eval
        eigenvectors[index_k, :, :] = evect
        velocities[index_k, :, :] = vels.real

    # TODO: figure out why we have Nan values
    velocities[np.isnan(velocities)] = 0
    return frequencies, eigenvalues, eigenvectors, -1 * (velocities / 10)


def calculate_broadening(velocity, cellinv, k_size):
    # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
    # 10 = armstrong to nanometers
    delta_k = cellinv / k_size * 2 * np.pi
    base_sigma = ((np.tensordot (velocity * 10., delta_k, [-1, 1])) ** 2).sum (axis=-1)
    base_sigma = np.sqrt (base_sigma / 6.)
    return base_sigma / (2 * np.pi)


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
    domega = np.abs(params[0])
    deltaa = np.abs(params[1])
    out = np.zeros_like(domega)
    out[domega < deltaa] = 1. / deltaa * (1 - domega[domega < deltaa] / deltaa)
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
def calculate_single_gamma(is_plus, index_k, mu, i_k, frequencies, velocities, density, cell_inv, k_size,
                           n_modes, nptk, n_replicas, evect, chi, third_order, sigma_in, broadening,
                           energy_threshold):

    # is_amorphous = nptk == 1

    if broadening == 'gauss':
        broadening_function = gaussian_delta
    elif broadening == 'lorentz':
        broadening_function = lorentzian_delta
    elif broadening == 'triangle':
        broadening_function = triangular_delta

    if np.abs(frequencies[index_k, mu]) > energy_threshold:
        nu = np.ravel_multi_index([index_k, mu], [nptk, n_modes], order='C')
        evect = evect.swapaxes(1, 2).reshape(nptk * n_modes, n_modes, order='C')
        evect_dagger = evect.reshape((nptk * n_modes, n_modes), order='C').conj()

        # TODO: next espression is unreadable and needs to be broken down
        # We need to use the right data structure here, it matters how the sparse matrix is saved
        # (columns, rows, coo, ...)
        scaled_potential = third_order.reshape((n_replicas, n_modes, n_replicas * n_modes * n_replicas * n_modes), order='C')[0]\
            .to_scipy_sparse().T.dot(evect[nu, :]).reshape((n_replicas, n_modes, n_replicas, n_modes), order='C')
        # scaled_potential = COO.from_numpy(scaled_potential)
        index_kp_vec = np.arange(np.prod(k_size))
        i_kp_vec = np.array(np.unravel_index(index_kp_vec, k_size, order='C'))
        i_kpp_vec = i_k[:, np.newaxis] + (int(is_plus) * 2 - 1) * i_kp_vec[:, :]
        index_kpp_vec = np.ravel_multi_index(i_kpp_vec, k_size, order='C', mode='wrap')
        # +1 if is_plus, -1 if not is_plus
        second_sign = (int(is_plus) * 2 - 1)
        if sigma_in is None:
            velocities = velocities.real
            # velocities[0, :3, :] = 0
            sigma_tensor_np = calculate_broadening(velocities[index_kp_vec, :, np.newaxis, :] -
                                                   velocities[index_kpp_vec, np.newaxis, :, :], cell_inv, k_size)
            sigma_small = sigma_tensor_np
        else:
            sigma_small = sigma_in

        freq_diff_np = np.abs(frequencies[index_k, mu] + second_sign * frequencies[index_kp_vec, :, np.newaxis] -
                              frequencies[index_kpp_vec, np.newaxis, :])

        condition = (freq_diff_np < DELTA_THRESHOLD * sigma_small) & (np.abs(frequencies[index_kp_vec, :, np.newaxis]) > energy_threshold) & (np.abs(frequencies[index_kpp_vec, np.newaxis, :]) > energy_threshold)
        interactions = np.array(np.where(condition)).T
        # TODO: Benchmark something fast like
        # interactions = np.array(np.unravel_index (np.flatnonzero (condition), condition.shape)).T
        if interactions.size != 0:
            # Logger ().info ('interactions: ' + str (interactions.size))
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
                dirac_delta = .5 * (
                        1 + density[nup_vec] + density[nupp_vec])

            dirac_delta /= (frequencies[index_kp_vec, mup_vec] * frequencies[index_kpp_vec, mupp_vec])
            if sigma_in is None:
                dirac_delta *= broadening_function([freq_diff_np[index_kp_vec, mup_vec, mupp_vec], sigma_small[
                    index_kp_vec, mup_vec, mupp_vec]])

            else:
                dirac_delta *= broadening_function(
                    [freq_diff_np[index_kp_vec, mup_vec, mupp_vec], sigma_in])

            if is_plus:
                # if (k_size == (1, 1, 1)).any():
                #     potential = contract('litj,aj,ai->alt', scaled_potential,
                #                          evect_dagger[nupp_vec], evect[nup_vec])
                # else:
                potential = contract('litj,aj,ai,al,at->a', scaled_potential,
                                         evect_dagger[nupp_vec], evect[nup_vec], chi[index_kp_vec], chi.conj()
                                         [index_kpp_vec])
            else:
                # if (k_size == (1, 1, 1)).any():
                #     potential = contract('litj,al,at->alt', scaled_potential,  evect_dagger[nupp_vec], evect_dagger[
                #         nup_vec])
                # else:
                potential = contract('litj,aj,ai,al,at->a', scaled_potential, evect_dagger[nupp_vec], evect_dagger[
                        nup_vec], chi.conj()[index_kp_vec], chi.conj()[index_kpp_vec])

            # gamma contracted on one index
            pot_times_dirac = np.abs(potential.flatten(order='C')) ** 2 * dirac_delta.flatten(order='C')
            pot_times_dirac = pot_times_dirac / frequencies[index_k, mu] / nptk * constants.gamma_coeff
            return COO((nup_vec, nupp_vec), pot_times_dirac, (nptk * n_modes, nptk * n_modes))


# @profile
def calculate_gamma(atoms, frequencies, velocities, density, k_size, eigenvectors, list_of_replicas, third_order,
                    sigma_in, broadening, energy_threshold):
    density = density.flatten(order='C')
    nptk = np.prod (k_size)
    n_particles = atoms.positions.shape[0]

    Logger ().info ('Lifetime calculation')
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
    Logger ().info ('Projection started')
    gamma = np.zeros ((2, nptk, n_modes))

    if IS_SCATTERING_MATRIX_ENABLED:
        gamma_tensor_plus = np.zeros((nptk * n_modes, nptk * n_modes))
        gamma_tensor_minus = np.zeros((nptk * n_modes, nptk * n_modes))
    n_modes = n_particles * 3
    nptk = np.prod (k_size)

    list_of_k = np.arange(np.prod(k_size))

    # Logger ().info ('n_irreducible_q_points = ' + str(int(len(unique_points))) + ' : ' + str(unique_points))
    process = ['Minus processes: ', 'Plus processes: ']
    masses = atoms.get_masses()
    rescaled_eigenvectors = eigenvectors[:, :, :].reshape((nptk, n_particles, 3, n_modes), order='C') / np.sqrt(
        masses[np.newaxis, :, np.newaxis, np.newaxis])
    rescaled_eigenvectors = rescaled_eigenvectors.reshape((nptk, n_particles * 3, n_modes), order='C')

    for is_plus in (1, 0):
        for index_k in (list_of_k):
            i_k = np.array(np.unravel_index(index_k, k_size, order='C'))

            for mu in range(n_modes):
                gamma_out = calculate_single_gamma(is_plus, index_k, mu, i_k, frequencies, velocities, density,
                                                   cell_inv, k_size, n_modes, nptk, n_replicas,
                                                   rescaled_eigenvectors, chi, third_order, sigma_in, broadening,
                                                   energy_threshold)
                if gamma_out:
                    # first_contracted_gamma = gamma_out.sum(axis=1)
                    # gamma_scal = first_contracted_gamma.sum(axis=0)
                    gamma[is_plus, index_k, mu] = gamma_out.sum()
                    if IS_SCATTERING_MATRIX_ENABLED:
                        nu = np.ravel_multi_index([index_k, mu], [nptk, n_modes], order='C')
                        # gamma_tensor[nu, nu] += gamma[is_plus, index_k, mu]
                        coords = gamma_out.coords.T
                        for i in range(coords.shape[0]):
                            if is_plus:
                                gamma_tensor_plus[nu, coords[i][0]] -= gamma_out.data[i]
                                gamma_tensor_plus[nu, coords[i][1]] += gamma_out.data[i]
                            else:
                                gamma_tensor_minus[nu, coords[i][0]] += gamma_out.data[i]
                                gamma_tensor_minus[nu, coords[i][1]] += gamma_out.data[i]

                            #
                            # nup_vec = first_contracted_gamma.coords[0]
                            #
                            # second_contracted_gamma = gamma_out.sum(axis=0)
                            # nupp_vec = second_contracted_gamma.coords[0]
                            # if is_plus:
                            #     nu_vec = nu * np.ones(nup_vec.shape[0]).astype(int)
                            #     gamma_tensor[nu_vec, nup_vec] += first_contracted_gamma.data
                            #     # gamma_tensor[nup_vec, nu_vec] += first_contracted_gamma.data
                            #     nu_vec = nu * np.ones(nupp_vec.shape[0]).astype(int)
                            #     gamma_tensor[nu_vec, nupp_vec] -= second_contracted_gamma.data
                            #     # gamma_tensor[nupp_vec, nu_vec] -= second_contracted_gamma.data
                            #
                            # else:
                            #     nu_vec = nu * np.ones(nup_vec.shape[0]).astype(int)
                            #     gamma_tensor[nu_vec, nup_vec] += first_contracted_gamma.data
                            #     # gamma_tensor[nup_vec, nu_vec] += first_contracted_gamma.data
                            #     nu_vec = nu * np.ones(nupp_vec.shape[0]).astype(int)
                            #     gamma_tensor[nu_vec, nupp_vec] += second_contracted_gamma.data
                            #     # gamma_tensor[nupp_vec, nu_vec] += second_contracted_gamma.data

            Logger().info(process[is_plus] + 'q-point = ' + str(index_k))

    # if IS_SCATTERING_MATRIX_ENABLED:
    #     gamma_tensor_copy = np.zeros((nptk, n_modes, nptk, n_modes))
    #     for index_k in range(nptk):
    #         for mu in range(n_modes):
    #             nu = np.ravel_multi_index([index_k, mu], [nptk, n_modes], order='C')
    #             for index_kp in range(nptk):
    #                 for mup in range(n_modes):
    #                     nup = np.ravel_multi_index([index_kp, mup], [nptk, n_modes], order='C')
    #                     gamma_tensor_copy[index_k, mu, index_kp, mup] = gamma_tensor[nu, nup]
    return [gamma[0], gamma[1]], [gamma_tensor_minus, gamma_tensor_plus]



def apply_boundary_with_cell(cell, cellinv, dxij):
    # exploit periodicity to calculate the shortest distance, which may not be the one we have
    sxij = dxij.dot(cellinv)
    sxij = sxij - np.round (sxij)
    dxij = sxij.dot(cell)
    return dxij
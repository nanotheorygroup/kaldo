import sparse
import numpy as np
from opt_einsum import contract
import ase.units as units
from .tools import timeit, lazy_property, is_calculated
from .phasespace import calculate_dirac_delta_amorphous, calculate_dirac_delta
EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12



def calculate_occupations(phonons):
    frequencies = phonons.frequencies
    temp = phonons.temperature * KELVINTOTHZ
    density = np.zeros_like(frequencies)
    physical_modes = frequencies > phonons.frequency_threshold
    if phonons.is_classic is False:
        density[physical_modes] = 1. / (np.exp(frequencies[physical_modes] / temp) - 1.)
    else:
        density[physical_modes] = temp / frequencies[physical_modes]
    return density


def calculate_c_v(phonons):
    frequencies = phonons.frequencies
    c_v = np.zeros_like (frequencies)
    physical_modes = frequencies > phonons.frequency_threshold
    temperature = phonons.temperature * KELVINTOTHZ

    if (phonons.is_classic):
        c_v[physical_modes] = KELVINTOJOULE
    else:
        f_be = phonons.occupations
        c_v[physical_modes] = KELVINTOJOULE * f_be[physical_modes] * (f_be[physical_modes] + 1) * phonons.frequencies[physical_modes] ** 2 / \
                              (temperature ** 2)
    return c_v



@timeit
def calculate_gamma_sparse(phonons, is_gamma_tensor_enabled=False):
    print('Projection started')
    if phonons._is_amorphous:
        ps_and_gamma = project_amorphous(phonons, is_gamma_tensor_enabled)
    else:
        ps_and_gamma = project_crystal(phonons, is_gamma_tensor_enabled)
    return ps_and_gamma


def project_amorphous(phonons, is_gamma_tensor_enabled=False):
    n_particles = phonons.atoms.positions.shape[0]
    n_modes = phonons.n_modes
    masses = phonons.atoms.get_masses()
    rescaled_eigenvectors = phonons.eigenvectors.reshape(
        (n_particles, 3, n_modes), order='C') / np.sqrt(
        masses[:, np.newaxis, np.newaxis])
    rescaled_eigenvectors = rescaled_eigenvectors.reshape((n_modes, n_modes),
                                                          order='C')

    # The ps and gamma matrix stores ps, gamma and then the scattering matrix
    ps_and_gamma = np.zeros((phonons.n_phonons, 2))
    for nu_single in range(phonons.n_phonons):

        print('calculating third', nu_single, np.round(nu_single / phonons.n_phonons, 2) * 100,
              '%')

        if is_gamma_tensor_enabled == True:
            raise ValueError('is_gamma_tensor_enabled=True not supported')
        # ps_and_gamma_sparse = np.zeros(2)
        out = calculate_dirac_delta_amorphous(phonons, nu_single)
        if not out:
            continue
        dirac_delta, mup_vec, mupp_vec = out

        potential_times_evect = sparse.tensordot(phonons.finite_difference.third_order,
                                                 rescaled_eigenvectors[:, nu_single], (0, 0))
        scaled_potential = contract('ij,in,jm->nm', potential_times_evect.real,
                                    rescaled_eigenvectors.real,
                                    rescaled_eigenvectors.real,
                                    optimize='optimal')
        scaled_potential = scaled_potential[np.newaxis, ...]
        scaled_potential = scaled_potential[0, mup_vec, mupp_vec]
        pot_times_dirac = np.abs(scaled_potential) ** 2 * dirac_delta

        gammatothz = 1e11 * units.mol * EVTOTENJOVERMOL ** 2
        pot_times_dirac = units._hbar / 8. * pot_times_dirac / phonons.n_k_points * gammatothz
        ps_and_gamma[nu_single, 0] = dirac_delta.sum()
        ps_and_gamma[nu_single, 1] = pot_times_dirac.sum()
        ps_and_gamma[nu_single, 1:] /= phonons.frequencies.flatten()[nu_single]

        # THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
        # print(phonons.frequencies[0, nu_single], ps_and_gamma[nu_single, 1] * THZTOMEV / (2 * np.pi))
        # print('\n')

    return ps_and_gamma

def project_crystal(phonons, is_gamma_tensor_enabled=False):

    # The ps and gamma matrix stores ps, gamma and then the scattering matrix
    if is_gamma_tensor_enabled:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2 + phonons.n_phonons))
    else:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2))

    n_particles = phonons.atoms.positions.shape[0]
    n_modes = phonons.n_modes
    masses = phonons.atoms.get_masses()
    rescaled_eigenvectors = phonons.eigenvectors[:, :, :].reshape(
        (phonons.n_k_points, n_particles, 3, n_modes), order='C') / np.sqrt(
        masses[np.newaxis, :, np.newaxis, np.newaxis])
    rescaled_eigenvectors = rescaled_eigenvectors.reshape((phonons.n_k_points, n_particles * 3, n_modes),
                                                          order='C')
    rescaled_eigenvectors = rescaled_eigenvectors.reshape((phonons.n_k_points, n_modes, n_modes), order='C')

    for index_k in range(phonons.n_k_points):
        for mu in range(phonons.n_modes):
            nu_single = np.ravel_multi_index([index_k, mu], (phonons.n_k_points, phonons.n_modes), order='C')

            if nu_single % 200 == 0:
                print('calculating third', nu_single, np.round(nu_single / phonons.n_phonons, 2) * 100,
                      '%')
            potential_times_evect = sparse.tensordot(phonons.finite_difference.third_order,
                                                rescaled_eigenvectors.reshape(
                                                    (phonons.n_k_points, phonons.n_modes, phonons.n_modes),
                                                    order='C')[index_k, :, mu], (0, 0))


            for is_plus in (1, 0):


                out = calculate_dirac_delta(phonons, index_k, mu, is_plus)
                if not out:
                    continue
                potential_times_evect = potential_times_evect.reshape(
                    (phonons.n_replicas, phonons.n_modes, phonons.n_replicas, phonons.n_modes),
                    order='C')

                dirac_delta, index_kp_vec, mup_vec, index_kpp_vec, mupp_vec = out

                # The ps and gamma array stores first ps then gamma then the scattering array

                index_kp_full = np.arange(phonons.n_k_points)
                i_kp_vec = np.array(np.unravel_index(index_kp_full, phonons.kpts, order='C'))
                i_k = np.array(np.unravel_index(index_k, phonons.kpts, order='C'))
                i_kpp_vec = i_k[:, np.newaxis] + (int(is_plus) * 2 - 1) * i_kp_vec[:, :]
                index_kpp_full = np.ravel_multi_index(i_kpp_vec, phonons.kpts, order='C', mode='wrap')

                # if is_plus:
                #
                #     scaled_potential = contract('litj,ai,al,aj,at->a', potential_times_evect,
                #                                 phonons.rescaled_eigenvectors[index_kp_vec, mup_vec, :],
                #                                 phonons.chi_k[index_kp_vec, :],
                #                                 phonons.rescaled_eigenvectors[index_kpp_vec, mupp_vec, :].conj(),
                #                                 phonons.chi_k[index_kpp_vec, :].conj()
                #                                 )
                # else:
                #
                #     scaled_potential = contract('litj,ai,al,aj,at->a', potential_times_evect,
                #                                 phonons.rescaled_eigenvectors.conj()[index_kp_vec, mup_vec, :],
                #                                 phonons.chi_k[index_kp_vec, :].conj(),
                #                                 phonons.rescaled_eigenvectors[index_kpp_vec, mupp_vec, :].conj(),
                #                                 phonons.chi_k[index_kpp_vec, :].conj())
                # pot_times_dirac = np.abs(scaled_potential) ** 2 * dirac_delta

                if is_plus:

                    scaled_potential = contract('litj,kim,kl,kjn,kt->kmn', potential_times_evect,
                                                rescaled_eigenvectors,
                                                phonons._chi_k,
                                                rescaled_eigenvectors[index_kpp_full].conj(),
                                                phonons._chi_k[index_kpp_full].conj()
                                                )
                else:

                    scaled_potential = contract('litj,kim,kl,kjn,kt->kmn', potential_times_evect,
                                                rescaled_eigenvectors.conj(),
                                                phonons._chi_k.conj(),
                                                rescaled_eigenvectors[index_kpp_full].conj(),
                                                phonons._chi_k[index_kpp_full].conj())
                pot_times_dirac = np.abs(scaled_potential[index_kp_vec, mup_vec, mupp_vec]) ** 2 * dirac_delta

                gammatothz = 1e11 * units.mol * EVTOTENJOVERMOL ** 2
                pot_times_dirac = units._hbar / 8. * pot_times_dirac / phonons.n_k_points * gammatothz

                if is_gamma_tensor_enabled:
                    # We need to use bincount together with fancy indexing here. See:
                    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
                    nup_vec = np.ravel_multi_index(np.array([index_kp_vec, mup_vec], dtype=int),
                                                   np.array([phonons.n_k_points, phonons.n_modes]), order='C')
                    nupp_vec = np.ravel_multi_index(np.array([index_kpp_vec, mupp_vec], dtype=int),
                                                    np.array([phonons.n_k_points, phonons.n_modes]), order='C')

                    result = np.bincount(nup_vec, pot_times_dirac, phonons.n_phonons)
                    if is_plus:
                        ps_and_gamma[nu_single, 2:] -= result
                    else:
                        ps_and_gamma[nu_single, 2:] += result

                    result = np.bincount(nupp_vec, pot_times_dirac, phonons.n_phonons)
                    ps_and_gamma[nu_single, 2:] += result
                ps_and_gamma[nu_single, 0] += dirac_delta.sum()
                ps_and_gamma[nu_single, 1] += pot_times_dirac.sum()

            ps_and_gamma[nu_single, 1:] /= phonons.frequencies.flatten()[nu_single]

    return ps_and_gamma



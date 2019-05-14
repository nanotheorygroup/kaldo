import numpy as np
import sparse

from scipy.optimize import minimize

MAX_ITERATIONS_SC = 500

def conductivity_integral_fn(f_n, a, b):
    # we are changing sign here, to minimize
    out = -1 * f_n.T.dot(2 * b - (a.dot (f_n)))
    print(out)
    return out

def conductivity_value_fn(f_n, a, b):
    out = -1 * (b - a.dot (f_n))
    # print(out)
    return out

def conductivity(phonons, mfp):
    volume = np.linalg.det(phonons.atoms.cell) / 1000
    frequencies = phonons.frequencies.reshape((phonons.n_phonons), order='C')
    physical_modes = (frequencies > phonons.energy_threshold)
    c_v = phonons.c_v.reshape((phonons.n_phonons), order='C')
    velocities = phonons.velocities.real.reshape((phonons.n_phonons, 3), order='C') / 10
    conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))
    conductivity_per_mode[physical_modes, :, :] = c_v[physical_modes, np.newaxis, np.newaxis] * \
                                                  velocities[physical_modes, :, np.newaxis] * mfp[physical_modes,
                                                                                              np.newaxis, :]
    return 1 / (volume * phonons.n_k_points) * conductivity_per_mode

def conductivity_small(phonons, mfp):
    volume = np.linalg.det(phonons.atoms.cell) / 1000
    frequencies = phonons.frequencies.reshape((phonons.n_phonons), order='C')
    physical_modes = (frequencies > phonons.energy_threshold)
    c_v = phonons.c_v.reshape((phonons.n_phonons), order='C')
    velocities = phonons.velocities.real.reshape((phonons.n_phonons, 3), order='C') / 10
    conductivity_per_mode = np.zeros(phonons.n_phonons)
    conductivity_per_mode[physical_modes] = c_v[physical_modes] * velocities[physical_modes, 2] * mfp[:]
    return 1 / (volume * phonons.n_k_points) * conductivity_per_mode


def conj_grad(phonons, a, a_out, b):
    f_i = 1 / a_out * b
    r_i = b - a.dot(f_i)
    p_i = r_i.copy()
    for i in range(100):
        alpha_i = r_i.T.dot(r_i) / (p_i.T.dot(a.dot(p_i)))
        f_i = f_i + alpha_i * p_i
        r_ip = r_i - alpha_i * a.T.dot(p_i)
        beta_i = r_ip.T.dot(r_ip) / (r_i.T.dot(r_i))
        p_i = r_ip + beta_i * p_i
        print(conductivity_small(phonons, f_i))
    return f_i


class ConductivityController(object):
    def __init__(self, phonons):
        self.phonons = phonons

    def calculate_conductivity_inverse(self):
        phonons = self.phonons
        n_kpoints = np.prod(phonons.kpts)
        gamma = (phonons.full_scattering[0].sum(axis=2).sum(axis=1).reshape((n_kpoints, phonons.n_modes)) + \
                 phonons.full_scattering[1].sum(axis=2).sum(axis=1).reshape((n_kpoints, phonons.n_modes))).todense()

        gamma_tensor_plus = (phonons.full_scattering[1].sum(axis=1) - phonons.full_scattering[1].sum(axis=2)).todense()
        gamma_tensor_minus = (phonons.full_scattering[0].sum(axis=1) + phonons.full_scattering[0].sum(axis=2)).todense()

        scattering_matrix = (gamma_tensor_minus + gamma_tensor_plus)

        velocities = phonons.velocities.real.reshape ((phonons.n_phonons, 3), order='C') / 10
        frequencies = phonons.frequencies.reshape((phonons.n_k_points * phonons.n_modes), order='C')
        physical_modes = (frequencies > phonons.energy_threshold) #& (velocities > 0)[:, 2]
        gamma = gamma.reshape ((phonons.n_phonons), order='C')
        a_in = - 1 * scattering_matrix.reshape ((phonons.n_phonons, phonons.n_phonons), order='C')
        a_in = np.einsum ('a,ab,b->ab', 1 / frequencies, a_in, frequencies)
        a_out = np.zeros_like (gamma)
        a_out[physical_modes] = gamma[physical_modes]
        a_out_inverse = np.zeros_like (gamma)
        a_out_inverse[physical_modes] = 1 / a_out[physical_modes]
        a = np.diag (a_out) + a_in

        # Let's remove the unphysical modes from the matrix
        index = np.outer(physical_modes, physical_modes)
        a = a[index].reshape ((physical_modes.sum (), physical_modes.sum ()), order='C')
        a_inverse = np.linalg.inv (a)
        lambd = np.zeros ((phonons.n_phonons, 3))
        lambd[physical_modes, :] = a_inverse.dot(velocities[physical_modes, :])
        conductivity_per_mode = conductivity(phonons, lambd)
        evals = np.linalg.eigvalsh (a)
        print('negative eigenvals : ', (evals<0).sum())
        return conductivity_per_mode

    def transmission_caltech(self, gamma, velocity, length):
        kn = abs(velocity / (length * gamma))
        transmission = (1 - kn * (1 - np.exp(- 1. / kn))) * kn
        return length / abs(velocity) * transmission


    def transmission_matthiesen(self, gamma, velocity, length):
#        gamma + np.abs(velocity) / length
        transmission = (gamma * length / abs(velocity) + 1.) ** (-1)
        return length / abs(velocity) * transmission


    def calculate_conductivity_variational(self, n_iterations=MAX_ITERATIONS_SC):
        phonons = self.phonons
        frequencies = phonons.frequencies.reshape ((phonons.n_phonons), order='C')
        physical_modes = (frequencies > phonons.energy_threshold)

        velocities = phonons.velocities.real.reshape ((phonons.n_phonons, 3), order='C') / 10
        physical_modes = physical_modes #& (velocities > 0)[:, 2]

        n_kpoints = np.prod(phonons.kpts)
        gamma = (phonons.full_scattering[0].sum(axis=2).sum(axis=1).reshape((n_kpoints, phonons.n_modes)) + \
                 phonons.full_scattering[1].sum(axis=2).sum(axis=1).reshape((n_kpoints, phonons.n_modes))).todense()
        gamma = gamma.reshape ((phonons.n_phonons), order='C')
        gamma_tensor_plus = (phonons.full_scattering[1].sum(axis=1) - phonons.full_scattering[1].sum(axis=2)).todense()
        gamma_tensor_minus = (phonons.full_scattering[0].sum(axis=1) + phonons.full_scattering[0].sum(axis=2)).todense()

        scattering_matrix = (gamma_tensor_minus + gamma_tensor_plus)

        a_in = - 1 * scattering_matrix.reshape ((phonons.n_phonons, phonons.n_phonons), order='C')
        a_in = np.einsum ('a,ab,b->ab', 1 / frequencies, a_in, frequencies)
        a_out = np.zeros_like (gamma)
        a_out[physical_modes] = gamma[physical_modes]
        a_out_inverse = np.zeros_like(gamma)
        a_out_inverse[physical_modes] = 1 / a_out[physical_modes]
        a = np.diag(a_out) + a_in
        b = a_out_inverse[:, np.newaxis] * velocities[:, :]
        a_out_inverse_a_in_to_n_times_b = np.copy(b)
        f_n = np.copy(b)
        conductivity_value = np.zeros((3, 3, n_iterations))
        conductivity_integral = np.zeros((3, 3, n_iterations))

        # f_0 = a_out_inverse[:, np.newaxis] * b
        # index = np.outer(physical_modes, physical_modes)
        # a_small = a[index].reshape ((physical_modes.sum (), physical_modes.sum ()), order='C')
        # a_out_small = a_out[physical_modes]
        # b_small = b[physical_modes, 2]
        #
        # f_small = conj_grad (phonons, a_small, a_out_small, b_small)
        # print(conductivity(phonons, f_0))
        # out = minimize (conductivity_integral_fn, f_0, args=(a_small, b_small), method='BFGS', jac=conductivity_value_fn,
        #                 options={'maxiter': 100, 'disp': True, 'gtol':500})
        # mimized_f = out.x
        # print(out.success)
        # print(out)


        for n_iteration in range(n_iterations):
            a_out_inverse_a_in_to_n_times_b[:, :] = -1 * (a_out_inverse[:, np.newaxis] * a_in[:, physical_modes]).dot (a_out_inverse_a_in_to_n_times_b[physical_modes, :])
            f_n += a_out_inverse_a_in_to_n_times_b

            conductivity_integral[:, :, n_iteration] = - 1 * f_n.T.dot (2 * b - a.dot (f_n))
            conductivity_value[:, :, n_iteration] = conductivity(phonons, f_n).sum(0)

        conductivity_per_mode =  conductivity(phonons, f_n)
        if n_iteration == (MAX_ITERATIONS_SC - 1):
            print ('Max iterations reached')
        # print((f_n[physical_modes, 2] - mimized_f).sum())
        return conductivity_per_mode, conductivity_value, conductivity_integral


    def calculate_conductivity_sc(self, tolerance=0.01, length_thresholds=None, is_rta=False, n_iterations=MAX_ITERATIONS_SC, finite_size_method='matthiesen'):
        phonons = self.phonons
        volume = np.linalg.det (phonons.atoms.cell) / 1000
        velocities = phonons.velocities.real.reshape ((phonons.n_k_points, phonons.n_modes, 3), order='C') / 10
        lambd_0 = np.zeros ((phonons.n_k_points * phonons.n_modes, 3))
        velocities = velocities.reshape((phonons.n_phonons, 3), order='C')
        frequencies = phonons.frequencies.reshape ((phonons.n_phonons), order='C')
        n_kpoints = np.prod(phonons.kpts)
        gamma = (phonons.full_scattering[0].sum(axis=2).sum(axis=1).reshape((n_kpoints, phonons.n_modes)) + \
                 phonons.full_scattering[1].sum(axis=2).sum(axis=1).reshape((n_kpoints, phonons.n_modes))).todense()

        gamma = gamma.reshape((phonons.n_phonons), order='C').copy()
        physical_modes = (frequencies > phonons.energy_threshold) #& (velocities > 0)[:, 2]
        if not is_rta:
            index = np.outer(physical_modes, physical_modes)
            gamma_tensor_plus = (phonons.full_scattering[1].sum(axis=1) - phonons.full_scattering[1].sum(axis=2)).todense()
            gamma_tensor_minus = (phonons.full_scattering[0].sum(axis=1) + phonons.full_scattering[0].sum(axis=2)).todense()

            scattering_matrix = (gamma_tensor_minus + gamma_tensor_plus)


            scattering_matrix = scattering_matrix.reshape((phonons.n_phonons,
                                                                   phonons.n_phonons), order='C')
            # scattering_matrix = scattering_matrix[index].reshape((physical_modes.sum(), physical_modes.sum()), order='C')
            scattering_matrix = np.einsum ('a,ab,b->ab', 1 / frequencies, scattering_matrix, frequencies)

        for alpha in range (3):
            if length_thresholds:
                if length_thresholds[alpha]:
                    if finite_size_method == 'matthiesen':
                        gamma[physical_modes] += abs(velocities[physical_modes, alpha]) / (1/2 * length_thresholds[alpha])
                            # gamma[mu] = phonons.gamma.reshape ((phonons.n_phonons), order='C')[mu] + \
                            #         np.abs (velocities[mu, alpha]) / length_thresholds[alpha]

        tau_0 = np.zeros_like (gamma)
        tau_0[physical_modes] = 1 / gamma[physical_modes]

        
        lambd_0[physical_modes, alpha] = tau_0[physical_modes] * velocities[physical_modes, alpha]
        c_v = phonons.c_v.reshape ((phonons.n_phonons), order='C')
        lambd_n = lambd_0.copy ()
        delta_lambd = np.zeros_like(lambd_n)
        conductivity_per_mode = np.zeros ((phonons.n_phonons, 3, 3))
        avg_conductivity = 0

        for n_iteration in range (n_iterations):
            for alpha in range (3):
                for beta in range (3):
                    conductivity_per_mode[physical_modes, alpha, beta] = 1 / (volume * phonons.n_k_points) * \
                                                                         c_v[physical_modes] * velocities[
                                                                             physical_modes, alpha] * lambd_n[
                                                                             physical_modes, beta]
            if is_rta:
                return conductivity_per_mode, lambd_0

            # new_avg_conductivity = np.diag (np.sum (conductivity_per_mode, 0)).mean ()
            # if avg_conductivity:
            #     if np.abs (avg_conductivity - new_avg_conductivity) < tolerance:
            #         return conductivity_per_mode, lambd_n
            # avg_conductivity = new_avg_conductivity

            # If the tolerance has not been reached update the state
            tau_0 = tau_0.reshape ((phonons.n_phonons), order='C')

            # calculate the shift in mft
            # scattering_matrix = scattering_matrix * frequencies
            delta_lambd = tau_0[:, np.newaxis] * scattering_matrix.dot (lambd_n)
            lambd_n = lambd_0 + delta_lambd

        for alpha in range(3):
    
            for mu in np.argwhere(physical_modes):
                if length_thresholds:
                    if length_thresholds[alpha]:
                        
                        if finite_size_method == 'caltech':
                            # if lambd_n[mu, alpha] > 0:
                            transmission = (1 - np.abs(lambd_n[mu, alpha]) / length_thresholds[alpha] * (1 - np.exp(-length_thresholds[alpha] / np.abs(lambd_n[mu, alpha]))))
                            lambd_n[mu] = lambd_n[mu] * transmission
                            
        for alpha in range (3):
            for beta in range (3):
                conductivity_per_mode[physical_modes, alpha, beta] = 1 / (volume * phonons.n_k_points) * c_v[
                    physical_modes] * velocities[physical_modes, alpha] * lambd_n[physical_modes, beta]

        if n_iteration == (MAX_ITERATIONS_SC - 1):
            print ('Convergence not reached')
            
        
        return conductivity_per_mode, lambd_n


    def calculate_conductivity_sheng(self, n_iterations=20):
        phonons = self.phonons
        velocities = phonons.velocities.real.reshape((phonons.n_phonons, 3), order='C') / 10
        frequencies = phonons.frequencies.reshape((phonons.n_k_points * phonons.n_modes), order='C')
        physical_modes = (frequencies > phonons.energy_threshold)  # & (velocities > 0)[:, 2]
        tau = np.zeros(frequencies.shape)
        n_kpoints = np.prod(phonons.kpts)
        gamma = (phonons.full_scattering[0].sum(axis=2).sum(axis=1).reshape((n_kpoints, phonons.n_modes)) + \
                 phonons.full_scattering[1].sum(axis=2).sum(axis=1).reshape((n_kpoints, phonons.n_modes))).todense()

        tau[physical_modes] = 1 / gamma.reshape((phonons.n_phonons), order='C')[physical_modes]
        gamma_out = phonons.full_scattering
        volume = np.linalg.det(phonons.atoms.cell) / 1000
        c_v = phonons.c_v.reshape((phonons.n_phonons), order='C')

        F_0 = tau * velocities[:, 2] * frequencies
        F_n = F_0.copy()
        list_k = []
        for iteration in range(n_iterations):
            DeltaF = 0
            for is_plus in (1, 0):
                if is_plus:
                    DeltaF -= sparse.tensordot(gamma_out[is_plus], F_n, (1, 0))
                else:
                    DeltaF += sparse.tensordot(gamma_out[is_plus], F_n, (1, 0))
                DeltaF += sparse.tensordot(gamma_out[is_plus], F_n, (2, 0))
            F_n = F_0 + tau * DeltaF.sum(axis=1)

            conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))
            conductivity_per_mode[physical_modes, :, :] = c_v[physical_modes, np.newaxis, np.newaxis] * \
                                                          velocities[physical_modes, :, np.newaxis] * F_n[
                                                              physical_modes,
                                                              np.newaxis, np.newaxis] / frequencies[
                                                              physical_modes, np.newaxis, np.newaxis]
            conductivity_per_mode = 1 / (volume * phonons.n_k_points) * conductivity_per_mode

            conductivity = conductivity_per_mode.sum(axis=0)[2, 2]
            list_k.append(conductivity)

        return conductivity
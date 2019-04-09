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

        velocities = phonons.velocities.real.reshape ((phonons.n_phonons, 3), order='C') / 10
        frequencies = phonons.frequencies.reshape((phonons.n_k_points * phonons.n_modes), order='C')
        physical_modes = (frequencies > phonons.energy_threshold) #& (velocities > 0)[:, 2]
        gamma = phonons.gamma.reshape ((phonons.n_phonons), order='C')
        a_in = - 1 * phonons.scattering_matrix.reshape ((phonons.n_phonons, phonons.n_phonons), order='C')
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

    def calculate_conductivity_variational(self, n_iterations=MAX_ITERATIONS_SC):
        phonons = self.phonons
        frequencies = phonons.frequencies.reshape ((phonons.n_phonons), order='C')
        physical_modes = (frequencies > phonons.energy_threshold)

        velocities = phonons.velocities.real.reshape ((phonons.n_phonons, 3), order='C') / 10
        physical_modes = physical_modes #& (velocities > 0)[:, 2]

        gamma = phonons.gamma.reshape ((phonons.n_phonons), order='C')

        a_in = - 1 * phonons.scattering_matrix.reshape ((phonons.n_phonons, phonons.n_phonons), order='C')
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


    def calculate_conductivity_sc(self, tolerance=0.01, length_thresholds=None, is_rta=False, n_iterations=MAX_ITERATIONS_SC):
        phonons = self.phonons
        volume = np.linalg.det (phonons.atoms.cell) / 1000
        velocities = phonons.velocities.real.reshape ((phonons.n_k_points, phonons.n_modes, 3), order='C') / 10
        lambd_0 = np.zeros ((phonons.n_k_points * phonons.n_modes, 3))
        velocities = velocities.reshape((phonons.n_phonons, 3), order='C')
        frequencies = phonons.frequencies.reshape ((phonons.n_phonons), order='C')
        physical_modes = (frequencies > phonons.energy_threshold) #& (velocities > 0)[:, 2]
        if not is_rta:
            # TODO: clean up the is_rta logic
            scattering_matrix = phonons.scattering_matrix.reshape ((phonons.n_phonons,
                                                                    phonons.n_phonons), order='C')
            scattering_matrix = np.einsum ('a,ab,b->ab', 1 / frequencies, scattering_matrix, frequencies)

        for alpha in range (3):
            gamma = np.zeros (phonons.n_phonons)

            for mu in range (phonons.n_phonons):
                if length_thresholds:
                    if length_thresholds[alpha]:
                        gamma[mu] = phonons.gamma.reshape ((phonons.n_phonons), order='C')[mu] + \
                                    np.abs (velocities[mu, alpha]) / length_thresholds[alpha]
                    else:
                        gamma[mu] = phonons.gamma.reshape ((phonons.n_phonons), order='C')[mu]
                else:
                    gamma[mu] = phonons.gamma.reshape ((phonons.n_phonons), order='C')[mu]
            tau_0 = np.zeros_like (gamma)
            tau_0[gamma > phonons.gamma_cutoff] = 1 / gamma[gamma > phonons.gamma_cutoff]
            lambd_0[:, alpha] = tau_0[:] * velocities[:, alpha]
        c_v = phonons.c_v.reshape ((phonons.n_phonons), order='C')
        lambd_n = lambd_0.copy ()
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
                return conductivity_per_mode

            new_avg_conductivity = np.diag (np.sum (conductivity_per_mode, 0)).mean ()
            if avg_conductivity:
                if np.abs (avg_conductivity - new_avg_conductivity) < tolerance:
                    return conductivity_per_mode
            avg_conductivity = new_avg_conductivity

            # If the tolerance has not been reached update the state
            tau_0 = tau_0.reshape ((phonons.n_phonons), order='C')

            # calculate the shift in mft
            # scattering_matrix = scattering_matrix * frequencies
            delta_lambd = tau_0[:, np.newaxis] * scattering_matrix.dot (lambd_n)
            lambd_n = lambd_0 + delta_lambd

        for alpha in range (3):
            for beta in range (3):
                conductivity_per_mode[physical_modes, alpha, beta] = 1 / (volume * phonons.n_k_points) * c_v[
                    physical_modes] * velocities[physical_modes, alpha] * lambd_n[physical_modes, beta]

        if n_iteration == (MAX_ITERATIONS_SC - 1):
            print ('Convergence not reached')
        return conductivity_per_mode


    def calculate_conductivity_sheng(self):
        phonons = self.phonons
        velocities = phonons.velocities.real.reshape((phonons.n_phonons, 3), order='C') / 10
        frequencies = phonons.frequencies.reshape((phonons.n_k_points * phonons.n_modes), order='C')
        physical_modes = (frequencies > phonons.energy_threshold)  # & (velocities > 0)[:, 2]
        tau = np.zeros(frequencies.shape)
        tau[physical_modes] = 1 / phonons.gamma.reshape((phonons.n_phonons), order='C')[physical_modes]
        gamma_out = phonons.full_scattering
        volume = np.linalg.det(phonons.atoms.cell) / 1000
        c_v = phonons.c_v.reshape((phonons.n_phonons), order='C')

        F_0 = tau * velocities[:, 2] * frequencies
        F_n = F_0.copy()
        list_k = []
        for iteration in range(71):
            DeltaF = 0
            for is_plus in (1, 0):
                if is_plus:
                    DeltaF -= sparse.tensordot(gamma_out[is_plus][0], F_n, (1, 0))
                else:
                    DeltaF += sparse.tensordot(gamma_out[is_plus][0], F_n, (1, 0))
                DeltaF += sparse.tensordot(gamma_out[is_plus][0], F_n, (2, 0))
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
"""
Ballistico
Anharmonic Lattice Dynamics
"""
from opt_einsum import contract
import ase.units as units
import numpy as np
from ballistico.helpers.storage import lazy_property, DEFAULT_STORE_FORMATS
from ballistico.helpers.logger import get_logger
logging = get_logger()

MAX_ITERATIONS_SC = 50
EVTOTENJOVERMOL = units.mol / (10 * units.J)
KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
MAX_LENGTH_TRESHOLD = 1e15


def calculate_conductivity_per_mode(heat_capacity, velocity, mfp, physical_modes, n_phonons):
    conductivity_per_mode = np.zeros((n_phonons, 3, 3))
    conductivity_per_mode[physical_modes, :, :] = \
        heat_capacity[physical_modes, np.newaxis, np.newaxis] * velocity[physical_modes, :, np.newaxis] * \
        mfp[physical_modes, np.newaxis, :]
    return conductivity_per_mode


def gamma_with_matthiessen(gamma, velocity, length):
    gamma = gamma + 2 * np.abs(velocity) / length
    return gamma


def mfp_matthiessen(gamma, velocity, length, physical_modes):
    lambd_0 = np.zeros_like(velocity)
    for alpha in range(3):
        if length is not None:
            if length[alpha] and length[alpha] != 0:
                gamma = gamma + 2 * abs(velocity[:, alpha]) / length[alpha]
        lambd_0[physical_modes, alpha] = 1 / gamma[physical_modes] * velocity[physical_modes, alpha]
    return lambd_0


def mfp_caltech(lambd, velocity, length, physical_modes):
    reduced_physical_modes = physical_modes.copy() & (velocity[:] != 0)
    lambd[reduced_physical_modes] = lambd[reduced_physical_modes] * \
                                           (1 - np.abs(lambd[reduced_physical_modes]) / (length / 2) *
                                            (1 - np.exp(- length / 2 / np.abs(
                                                lambd[reduced_physical_modes]))))
    return lambd


class Conductivity:

    def __init__(self, **kwargs):
        self.phonons = kwargs.pop('phonons')
        self.method = kwargs.pop('method', 'rta')
        self.storage = kwargs.pop('storage', 'formatted')
        if self.method == 'rta':
            self.n_iterations = 0
        else:
            self.n_iterations = kwargs.pop('n_iterations', None)
        self.length = kwargs.pop('length', np.array([None, None, None]))
        self.finite_length_method = kwargs.pop('finite_length_method', 'matthiessen')
        self.tolerance = kwargs.pop('tolerance', None)
        self.folder = self.phonons.folder
        self.kpts = self.phonons.kpts
        self.n_k_points = self.phonons.n_k_points
        self.n_modes = self.phonons.n_modes
        self.n_phonons = self.phonons.n_phonons
        self.temperature = self.phonons.temperature
        self.is_classic = self.phonons.is_classic
        self.third_bandwidth = self.phonons.third_bandwidth
        self.store_format = {}
        for observable in DEFAULT_STORE_FORMATS:
            self.store_format[observable] = DEFAULT_STORE_FORMATS[observable] \
                if self.storage == 'default' else self.storage


    @lazy_property(label='<temperature>/<statistics>/<third_bandwidth>/<method>/<length>/<finite_length_method>')
    def conductivity(self):
        method = self.method
        if method == 'rta':
            cond = self.calculate_conductivity_sc()
        elif method == 'sc':
            cond = self.calculate_conductivity_sc()
        elif (method == 'qhgk'):
            cond = self.calculate_conductivity_qhgk()
        elif (method == 'inverse'):
            cond = self.calculate_conductivity_inverse()
        elif (method == 'relaxons'):
            cond = self.calculate_conductivity_with_evects()
        else:
            logging.error('Conductivity method not implemented')

        # folder = get_folder_from_label(phonons, '<temperature>/<statistics>/<third_bandwidth>')
        # save('cond', folder + '/' + method, cond.reshape(phonons.n_k_points, phonons.n_modes, 3, 3), \
        #      format=phonons.store_format['conductivity'])
        sum = (cond.imag).sum()
        if sum > 1e-3:
            logging.warning('The conductivity has an immaginary part. Sum(Im(k)) = ' + str(sum))
        return cond.real


    @property
    def _scattering_matrix_without_diagonal(self):
        frequency = self._keep_only_physical(self.phonons.frequency.reshape((self.n_phonons)))
        _ps_gamma_and_gamma_tensor = self.phonons._ps_gamma_and_gamma_tensor
        gamma_tensor = self._keep_only_physical(_ps_gamma_and_gamma_tensor[:, 2:])
        scattering_matrix_without_diagonal = contract('a,ab,b->ab', 1 / frequency, gamma_tensor, frequency)
        return scattering_matrix_without_diagonal


    def _keep_only_physical(self, operator):
        physical_modes = self.phonons.physical_mode.reshape(self.n_phonons)
        if operator.shape == (self.n_phonons, self.n_phonons):
            index = np.outer(physical_modes, physical_modes)
            return operator[index].reshape((physical_modes.sum(), physical_modes.sum()))
        elif operator.shape == (self.n_phonons, 3):
            return operator[physical_modes, :]
        else:
            return operator[physical_modes]


    def calculate_c_v_2d(self):
        phonons = self.phonons
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


    def calculate_conductivity_qhgk(self):
        phonons = self.phonons
        volume = np.linalg.det(phonons.atoms.cell)
        diffusivity = phonons._generalized_diffusivity
        heat_capacity = phonons.heat_capacity.reshape(phonons.n_k_points, phonons.n_modes)
        conductivity_per_mode = contract('kn,knab->knab', heat_capacity, diffusivity)
        conductivity_per_mode = conductivity_per_mode.reshape((phonons.n_phonons, 3, 3))
        conductivity_per_mode = conductivity_per_mode / (volume * phonons.n_k_points)
        return conductivity_per_mode
    
    
    def calculate_conductivity_inverse(self):
        length = self.length
        phonons = self.phonons
        finite_size_method = self.finite_length_method
        volume = np.linalg.det(phonons.atoms.cell)
        physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)
        conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))
        velocity = phonons.velocity.real.reshape((phonons.n_phonons, 3))

        for alpha in range (3):
            gamma = phonons.bandwidth.reshape(phonons.n_phonons)
            scattering_matrix = - 1 * self._scattering_matrix_without_diagonal
            if finite_size_method == 'matthiessen':
                if length is not None:
                    if length[alpha]:
                        gamma = gamma_with_matthiessen(gamma, velocity[:, alpha],
                                                                            length[alpha])

            scattering_matrix += np.diag(gamma[physical_modes])
            scattering_inverse = np.linalg.inv(scattering_matrix)
            lambd = np.zeros_like(gamma)
            lambd[physical_modes] = scattering_inverse.dot(velocity[physical_modes, alpha])
            if finite_size_method == 'caltech':
                if length is not None:
                    if length[alpha]:
                        lambd = mfp_caltech(lambd, velocity[:, alpha], length[alpha], physical_modes)
            c_v = phonons.heat_capacity.reshape((phonons.n_phonons))
            physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)
            conductivity_per_mode[physical_modes, :, alpha] = c_v[physical_modes, np.newaxis] * \
                                                          velocity[physical_modes, :] * lambd[physical_modes, np.newaxis]
        return conductivity_per_mode / (volume * phonons.n_k_points)
    
    
    def calculate_conductivity_with_evects(self):
        phonons = self.phonons
        velocity = self._keep_only_physical(phonons.velocity.real.reshape((phonons.n_phonons, 3)))
        scattering_matrix = -1 * self._scattering_matrix_without_diagonal
        gamma = self._keep_only_physical(self.phonons.bandwidth.reshape((self.n_phonons)))
        _scattering_matrix = scattering_matrix + np.diag(gamma)
        evals, evects = np.linalg.eig(_scattering_matrix)

        neg_diag = (_scattering_matrix.diagonal() < 0).sum()
        logging.info('negative on diagonal : ' + str(neg_diag))
        logging.info('negative eigenvals : ' + str((evals < 0).sum()))

        # TODO: find a better way to filter states
        new_physical_states = np.argwhere(evals >= 0)[0, 0]
        reduced_evects = evects[new_physical_states:, new_physical_states:]
        reduced_evals = evals[new_physical_states:]
        reduced_scattering_inverse = np.zeros_like(_scattering_matrix)
        reduced_scattering_inverse[new_physical_states:, new_physical_states:] = reduced_evects.dot(np.diag(1/reduced_evals)).dot(np.linalg.inv(reduced_evects))
        scattering_inverse = reduced_scattering_inverse
        # e, v = np.linalg.eig(a)
        # a = v.dot(np.diag(e)).dot(np.linalg.inv(v))
    
        lambd = scattering_inverse.dot(velocity[:, :])
        physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)
    
        volume = np.linalg.det(phonons.atoms.cell)
        c_v = self._keep_only_physical(phonons.heat_capacity.reshape((phonons.n_phonons)))
        conductivity_per_mode = np.zeros((phonons.n_phonons, 3, 3))
        conductivity_per_mode[physical_modes, :, :] = c_v[:, np.newaxis, np.newaxis] * \
                                                      velocity[:, :, np.newaxis] * lambd[:, np.newaxis, :]
        return conductivity_per_mode / (volume * phonons.n_k_points)
    
    
    def calculate_conductivity_sc(self):
        phonons = self.phonons
        finite_size_method = self.finite_length_method
        physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)
        velocity = phonons.velocity.real.reshape ((phonons.n_k_points, phonons.n_modes, 3))
        velocity = velocity.reshape((phonons.n_phonons, 3))

        if finite_size_method == 'matthiessen':
            lambd_n = self.calculate_sc_mfp(matthiessen_length=self.length)
        else:
            lambd_n = self.calculate_sc_mfp()
        if finite_size_method == 'caltech':
            for alpha in range(3):
                lambd_n[:, alpha] = mfp_caltech(lambd_n[:, alpha], velocity[:, alpha], self.length[alpha], physical_modes)

        conductivity_per_mode = calculate_conductivity_per_mode(phonons.heat_capacity.reshape((phonons.n_phonons)),
                                                                velocity, lambd_n, physical_modes, phonons.n_phonons)
    
        volume = np.linalg.det (phonons.atoms.cell)
        return conductivity_per_mode / (volume * phonons.n_k_points)
    
    
    def calculate_sc_mfp(self, matthiessen_length=None):
        tolerance = self.tolerance
        n_iterations = self.n_iterations
        phonons = self.phonons
        if n_iterations is None:
            n_iterations = MAX_ITERATIONS_SC
        velocity = phonons.velocity.real.reshape ((phonons.n_k_points, phonons.n_modes, 3))
        velocity = velocity.reshape((phonons.n_phonons, 3))
        physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)
        gamma = phonons.bandwidth.reshape(phonons.n_phonons)
        lambd_0 = mfp_matthiessen(gamma, velocity, matthiessen_length, physical_modes)
        if n_iterations == 0:
            return lambd_0
        else:
            lambd_n = np.zeros_like(lambd_0)
            avg_conductivity = None
            n_iteration = 0
            scattering_matrix = self._scattering_matrix_without_diagonal
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
            return lambd_n
    
    
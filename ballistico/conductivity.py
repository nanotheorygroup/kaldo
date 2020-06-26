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
hbar = 1 / (KELVINTOTHZ * 2 * np.pi)
kb = 1 / KELVINTOJOULE

def calculate_conductivity_per_mode(heat_capacity, velocity, mfp, physical_modes, n_phonons):
    conductivity_per_mode = np.zeros((n_phonons, 3, 3))
    physical_modes = physical_modes.reshape(n_phonons)
    velocity = velocity.reshape((n_phonons, 3))
    conductivity_per_mode[physical_modes, :, :] = \
        heat_capacity[physical_modes, np.newaxis, np.newaxis] * velocity[physical_modes, :, np.newaxis] * \
        mfp[physical_modes, np.newaxis, :]
    return conductivity_per_mode * 1e22


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

        """The conductivity object is responsible for mean free path and conductivity calculations.

        Parameters
        ----------
        phonons : Phonons
            contains all the information about the calculated phononic properties of the system
        method : 'rta', 'sc', 'qhgk', 'inverse'
            specifies the method used to calculate the conductivity.
        tolerance : int
            in the self consistent conductivity calculation, it specifies the difference in W/m/K between n
            and n+1 step, to set as exit/convergence condition.
        n_iterations : int
            specifies the max number of iterations to set as exit condition in the self consistent conductivity
            calculation
        tolerance : int
            in the self consistent conductivity calculation, it specifies the difference in W/m/K between n
            and n+1 step, to set as exit/convergence condition.
        length: (float, float, float)
            specifies the length to use in x, y, z to calculate the finite size conductivity. 0 or None values
            corresponds to the infinity length limit.
        finite_length_method : 'matthiessen', 'ms', 'caltech'
            specifies how to calculate the finite size conductivity. 'ms' is the Mckelvey-Schockley method.
        storage : 'formatted', 'hdf5', 'numpy', 'memory'
            defines the type of storage used for the simulation.

        Returns
        -------
        Conductivity
            An instance of the `Conductivity` class.

        Examples
        --------
        Here's an example to calculate the inverse conductivity on the phonons object and tracing over the phonons modes
        
        ```
        Conductivity(phonons=phonons, method='inverse', storage='memory').conductivity.sum(axis=0))
        ```
        """


        self.phonons = kwargs.pop('phonons')
        self.method = kwargs.pop('method', 'rta')
        self.storage = kwargs.pop('storage', 'numpy')
        if self.method == 'rta':
            self.n_iterations = 0
        else:
            self.n_iterations = kwargs.pop('n_iterations', None)
        self.length = kwargs.pop('length', np.array([None, None, None]))
        self.finite_length_method = kwargs.pop('finite_length_method', 'ms')
        self.tolerance = kwargs.pop('tolerance', None)
        self.folder = self.phonons.folder
        self.kpts = self.phonons.kpts
        self.n_k_points = self.phonons.n_k_points
        self.n_modes = self.phonons.n_modes
        self.n_phonons = self.phonons.n_phonons
        self.temperature = self.phonons.temperature
        self.is_classic = self.phonons.is_classic
        self.third_bandwidth = self.phonons.third_bandwidth
        self.diffusivity_bandwidth = self.phonons.diffusivity_bandwidth
        self.diffusivity_threshold = self.phonons.diffusivity_threshold
        self.store_format = {}
        for observable in DEFAULT_STORE_FORMATS:
            self.store_format[observable] = DEFAULT_STORE_FORMATS[observable] \
                if self.storage == 'formatted' else self.storage


    @lazy_property(label='<diffusivity_bandwidth>/<diffusivity_threshold>/<temperature>/<statistics>/<third_bandwidth>/<method>/<length>/<finite_length_method>')
    def conductivity(self):
        """
        Calculate the thermal conductivity per mode in W/m/K
        Returns
        -------
        conductivity : np array
            (n_k_points, n_modes, 3, 3) float
        """
        method = self.method
        other_avail_methods = ['rta', 'sc', 'inverse', 'relaxon']
        if (method == 'qhgk'):
            cond = self.calculate_conductivity_qhgk()
        elif method in other_avail_methods:
            lambd = self.mean_free_path
            conductivity_per_mode = calculate_conductivity_per_mode(self.phonons.heat_capacity.reshape((self.n_phonons)),
                                                                    self.phonons.velocity, lambd, self.phonons.physical_mode,
                                                                    self.n_phonons)

            volume = np.linalg.det(self.phonons.atoms.cell)
            cond = conductivity_per_mode / (volume * self.n_k_points)
        else:
            logging.error('Conductivity method not implemented')

        # folder = get_folder_from_label(phonons, '<temperature>/<statistics>/<third_bandwidth>')
        # save('cond', folder + '/' + method, cond.reshape(phonons.n_k_points, phonons.n_modes, 3, 3), \
        #      format=phonons.store_format['conductivity'])
        sum = (cond.imag).sum()
        if sum > 1e-3:
            logging.warning('The conductivity has an immaginary part. Sum(Im(k)) = ' + str(sum))
        logging.info('Conductivity calculated')
        return cond.real


    @lazy_property(label='<diffusivity_bandwidth>/<diffusivity_threshold>/<temperature>/<statistics>/<third_bandwidth>/<method>/<length>/<finite_length_method>')
    def mean_free_path(self):
        """
        Calculate the mean_free_path per mode in A
        Returns
        -------
        mfp : np array
            (n_k_points, n_modes) float
        """
        method = self.method

        if (method == 'qhgk'):
            logging.error('Mean free path not available for ' + str(method))
        elif method == 'rta':
            cond = self.calculate_mfp_sc()
        elif method == 'sc':
            cond = self.calculate_mfp_sc()
        elif (method == 'inverse'):
            cond = self.calculate_mfp_inverse()
        elif (method == 'relaxons'):
            cond = self.calculate_mfp_relaxons()
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


    def calculate_2d_heat_capacity(self):
        """
        Calculates the factor for the diffusivity which resembles the heat capacity.
        The array is returned in units of J/K.
        classical case: k_b
        quantum case: c_nm=hbar w_n w_m/T  * (n_n-n_m)/(w_m-w_n)
        Returns
        -------
        c_v : np.array
            (phonons.n_k_points,phonons.modes, phonons.n_modes) float
        """
        phonons = self.phonons
        frequencies = phonons.frequency
        c_v = np.zeros((phonons.n_k_points, phonons.n_modes, phonons.n_modes))
        temperature = phonons.temperature* KELVINTOTHZ
        physical_modes = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
        heat_capacity = phonons.heat_capacity
        if (phonons.is_classic):
            c_v[:, :, :] = KELVINTOJOULE
        else:
            f_be = phonons.population
            c_v_omega = (f_be[:, :, np.newaxis] - f_be[:, np.newaxis,: ])
            diff_omega =(frequencies[:, :, np.newaxis] - frequencies[:, np.newaxis, :])
            mask_degeneracy = np.where(diff_omega == 0, True, False)

            # value to do the division
            diff_omega[mask_degeneracy] = 1
            divide_omega = -1 / diff_omega
            freq_sq = frequencies[:, :, np.newaxis] * frequencies[:, np.newaxis, :]

            #remember here f_n-f_m/ w_m-w_n index reversed
            c_v = contract('knm,knm,knm->knm', freq_sq, c_v_omega, divide_omega)
            c_v = KELVINTOJOULE * c_v / temperature

            #Degeneracy part: let us substitute the wrong elements
            heat_capacity_deg_2d = (heat_capacity[:, :, np.newaxis] + heat_capacity[:, np.newaxis, :]) / 2
            c_v = np.where(mask_degeneracy, heat_capacity_deg_2d, c_v)

            #Physical modes
            mask = physical_modes[:, :, np.newaxis] * physical_modes[:, np.newaxis, :]
            c_v = c_v * mask
        return c_v


    def calculate_conductivity_qhgk(self):
        """
        Calculates the conductivity of each mode using the :ref:'Quasi-Harmonic-Green-Kubo Model'.
        The tensor is returned individual modes along the first axis and has units of W/m/K.

        Returns
        -------
        conductivity_per_mode : np.array
            (n_phonons, 3, 3) W/m/K
        """
        phonons = self.phonons
        volume = np.linalg.det(phonons.atoms.cell)
        diffusivity = phonons._generalized_diffusivity
        heat_capacity =self.calculate_2d_heat_capacity()
        conductivity_per_mode = contract('knm,knmab->knab', heat_capacity, diffusivity)
        conductivity_per_mode = conductivity_per_mode.reshape((phonons.n_phonons, 3, 3))
        conductivity_per_mode = conductivity_per_mode / (volume * phonons.n_k_points)
        return conductivity_per_mode * 1e22
    
    
    def calculate_mfp_inverse(self):
        """
        This method calculates the inverse of the mean free path for each phonon.
        The matrix returns k vectors for each mode and has units of inverse Angstroms.

        Returns
        -------
        lambda : np array
            (n_k_points, n_modes)

        """
        length = self.length
        phonons = self.phonons
        finite_size_method = self.finite_length_method
        physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)
        velocity = phonons.velocity.real.reshape((phonons.n_phonons, 3))
        lambd = np.zeros_like(velocity)
        for alpha in range (3):
            scattering_matrix = - 1 * self._scattering_matrix_without_diagonal
            gamma = phonons.bandwidth.reshape(phonons.n_phonons)
            if finite_size_method == 'ms':
                if length is not None:
                    if length[alpha]:
                        gamma = gamma_with_matthiessen(gamma, velocity[:, alpha],
                                                                            length[alpha])

            scattering_matrix += np.diag(gamma[physical_modes])
            scattering_inverse = np.linalg.inv(scattering_matrix)
            lambd[physical_modes, alpha] = scattering_inverse.dot(velocity[physical_modes, alpha])
            if finite_size_method == 'caltech':
                if length is not None:
                    if length[alpha]:
                        lambd[:, alpha] = mfp_caltech(lambd[:, alpha], velocity[:, alpha], length[alpha], physical_modes)
            if finite_size_method == 'matthiessen':
                if (self.length[alpha] is not None) and (self.length[alpha] != 0):
                    lambd[physical_modes, alpha] = 1 / (
                                np.sign(velocity[physical_modes, alpha]) / lambd[physical_modes, alpha] + 1 /
                                np.array(self.length)[np.newaxis, alpha]) * np.sign(velocity[physical_modes, alpha])
                else:
                    lambd[physical_modes, alpha] = 1 / (
                                np.sign(velocity[physical_modes, alpha]) / lambd[physical_modes, alpha]) * np.sign(
                        velocity[physical_modes, alpha])

                lambd[velocity[:, alpha] == 0, alpha] = 0
        return lambd
    
    
    def calculate_mfp_relaxons(self):
        """
        This calculates the mean free path of relaxons. In materials where most scattering events conserve momentum
        :ref:'Relaxon Theory Section' (e.g. in two dimensional materials or three dimensional materials at extremely low
        temparatures), this quantity can be used to calculate thermal conductivity.

        Returns
	-------
        lambda : np array
            (n_k_points, n_modes, 3)
        """
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
        return lambd
    
    
    def calculate_mfp_sc(self):
        phonons = self.phonons
        finite_size_method = self.finite_length_method
        physical_modes = phonons.physical_mode.reshape(phonons.n_phonons)
        velocity = phonons.velocity.real.reshape ((phonons.n_k_points, phonons.n_modes, 3))
        velocity = velocity.reshape((phonons.n_phonons, 3))

        if finite_size_method == 'ms':
            lambd_n = self.calculate_sc_mfp(matthiessen_length=self.length)
        else:
            lambd_n = self.calculate_sc_mfp()
        if finite_size_method == 'caltech':
            for alpha in range(3):
                lambd_n[:, alpha] = mfp_caltech(lambd_n[:, alpha], velocity[:, alpha], self.length[alpha], physical_modes)
        if finite_size_method == 'matthiessen':
            mfp = lambd_n.copy()
            for alpha in range(3):
                if (self.length[alpha] is not None) and (self.length[alpha] != 0):
                    lambd_n[physical_modes, alpha] = 1 / (np.sign(velocity[physical_modes, alpha]) / mfp[physical_modes, alpha] + 1 / np.array(self.length)[np.newaxis, alpha]) * np.sign(velocity[physical_modes, alpha])
                else:
                    lambd_n[physical_modes, alpha] = 1 / (np.sign(velocity[physical_modes, alpha]) / mfp[physical_modes, alpha]) * np.sign(velocity[physical_modes, alpha])

                lambd_n[velocity[:, alpha]==0, alpha] = 0
        return lambd_n
    
    
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
    
    

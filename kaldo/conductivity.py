"""
kaldo
Anharmonic Lattice Dynamics
"""
from opt_einsum import contract
import numpy as np
from kaldo.phonons import Phonons
from kaldo.controllers.dirac_kernel import lorentz_delta, gaussian_delta, triangular_delta
from kaldo.storable import lazy_property, Storable
import kaldo.observables.harmonic_with_q_temp as hwqwt
from kaldo.helpers.logger import get_logger, log_size
import gc

logging = get_logger()


def calculate_conductivity_per_mode(heat_capacity, velocity, mfp, physical_mode, n_phonons):
    conductivity_per_mode = np.zeros((n_phonons, 3, 3))
    physical_mode = physical_mode.reshape(n_phonons)
    velocity = velocity.reshape((n_phonons, 3))
    conductivity_per_mode[physical_mode, :, :] = \
        heat_capacity[physical_mode, np.newaxis, np.newaxis] * velocity[physical_mode, :, np.newaxis] * \
        mfp[physical_mode, np.newaxis, :]
    return conductivity_per_mode * 1e22


def calculate_diffusivity(omega, sij_left, sij_right, diffusivity_bandwidth, physical_mode, curve,
                          is_diffusivity_including_antiresonant=False,
                          diffusivity_threshold=None):
    # TODO: cache this
    sigma = 2 * (diffusivity_bandwidth[:, np.newaxis] + diffusivity_bandwidth[np.newaxis, :])
    physical_mode = physical_mode.astype(bool)
    delta_energy = omega[:, np.newaxis] - omega[np.newaxis, :]
    kernel = curve(delta_energy, sigma)
    if diffusivity_threshold is not None:
        condition = (np.abs(delta_energy) < diffusivity_threshold * 2 * np.pi * diffusivity_bandwidth)
        kernel[np.invert(condition)] = 0
    if is_diffusivity_including_antiresonant:
        sum_energy = omega[:, np.newaxis] + omega[np.newaxis, :]
        kernel += curve(sum_energy, sigma)
    kernel = kernel * np.pi
    kernel[np.isnan(kernel)] = 0
    mu_unphysical = np.argwhere(np.invert(physical_mode)).T
    kernel[:, :] = kernel / omega[:, np.newaxis]
    kernel[:, :] = kernel[:, :] / omega[np.newaxis, :] / 4
    kernel[mu_unphysical, :] = 0
    kernel[:, mu_unphysical] = 0
    diffusivity = sij_left * kernel * sij_right
    return diffusivity


def mfp_matthiessen(gamma, velocity, length, physical_mode):
    lambd_0 = np.zeros_like(velocity)
    for alpha in range(3):
        if length is not None:
            if length[alpha] and length[alpha] != 0:
                gamma = gamma + 2 * abs(velocity[:, alpha]) / length[alpha]
        lambd_0[physical_mode, alpha] = 1 / gamma[physical_mode] * velocity[physical_mode, alpha]
    return lambd_0


class Conductivity(Storable):
    """ The conductivity object is responsible for mean free path and
    conductivity calculations. It takes a phonons object as a required argument.

    Parameters
    ----------
    phonons : Phonons
        Contains all the information about the calculated phononic properties of the system
    method : 'rta', 'sc', 'qhgk', 'inverse'
        Specifies the method used to calculate the conductivity.
        `rta` is relaxation time approximation; `sc` is self-consistent; `qhgk` is Quasi-Harmonic Green Kubo; `inverse` is inversion of the scattering matrix for mean free path.
    diffusivity_bandwidth : float, optional
        (QHGK) Specifies the bandwidth to use in the calculation of the flux operator in the Allen-Feldman model of the
        thermal conductivity in amorphous systems. Units: rad/ps
    diffusivity_threshold : float, optional
        (QHGK) This option is off by default. In such case the flux operator in the QHGK and AF models is calculated
    diffusivity_shape : string, optional
        (QHGK) Defines the algorithm to use to calculate the diffusivity. Available broadenings are `gauss`, `lorentz` and `triangle`.
        Default is `lorentz`.
    is_diffusivity_including_antiresonant : bool, optional
        (QHGK) Defines if you want to include or not anti-resonant terms in diffusivity calculations.
        Default is `False`.
    tolerance : int
        (Self-consistent) In the self consistent conductivity calculation, it specifies the difference in W/m/K between n
        and n+1 step, to set as exit/convergence condition.
    n_iterations : int
        (Self-consistent) Specifies the max number of iterations to set as exit condition in the self consistent conductivity
        calculation
    length: (3) tuple
        (Finite Size) Specifies the length to use in x, y, z to calculate the finite size conductivity. 0 or None values
        corresponds to the infinity length limit.
    finite_length_method : 'ms', 'ballistic'
        (Finite Size) Specifies how to calculate the finite size conductivity. 'ms' is the Mckelvey-Schockley method.
        'ballistic' is the ballistic limit.
    storage : 'formatted', 'hdf5', 'numpy', 'memory', optional
        Defines the type of storage used for the simulation.
        Default is `formatted`

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
    
    # Define storage formats for conductivity properties
    _store_formats = {
        'conductivity': 'formatted',
        'mean_free_path': 'formatted',
        '_generalized_diffusivity': 'numpy'
    }
    
    def __init__(self,
                 phonons: Phonons,
                 *,
                 method: str = 'rta',
                 diffusivity_bandwidth: float | None = None,
                 diffusivity_threshold: float | None = None,
                 diffusivity_shape: str = 'lorentz',
                 is_diffusivity_including_antiresonant: bool = False,
                 tolerance: int | None = None,
                 n_iterations: int | None = None,
                 length: tuple[int, int, int] = (None, None, None),
                 finite_length_method: str = 'ms',
                 storage: str = 'formatted',
                 **kwargs):
        self.phonons = phonons
        self.method = method
        self.storage = storage

        # force n_iterations to 0 if method is `rta`
        if n_iterations != 0 and self.method == 'rta':
            logging.warning("rta method is specified, but n_iterations is not specified to be 0. Please check your setup. Now, n_iterations is reset to 0.")
        self.n_iterations = 0 if self.method == 'rta' else n_iterations

        self.length = length
        self.finite_length_method = finite_length_method
        self.tolerance = tolerance

        # initalize info from phonons object
        self.folder = self.phonons.folder
        self.kpts = self.phonons.kpts
        self.n_k_points = self.phonons.n_k_points
        self.n_modes = self.phonons.n_modes
        self.n_phonons = self.phonons.n_phonons
        self.temperature = self.phonons.temperature
        self.is_classic = self.phonons.is_classic
        self.third_bandwidth = self.phonons.third_bandwidth
        self.include_isotopes = self.phonons.include_isotopes

        # initalize QHGK method
        self.diffusivity_bandwidth = diffusivity_bandwidth
        self.diffusivity_threshold = diffusivity_threshold
        self.is_diffusivity_including_antiresonant = is_diffusivity_including_antiresonant
        self.diffusivity_shape = diffusivity_shape

    def _get_folder_path_components(self, label):
        """Get folder path components for Conductivity-specific attributes."""
        components = []
        
        if '<diffusivity_bandwidth>' in label and self.diffusivity_bandwidth is not None:
            components.append('db_' + str(np.mean(self.diffusivity_bandwidth)))
            
        if '<diffusivity_threshold>' in label and self.diffusivity_threshold is not None:
            components.append('dt_' + str(self.diffusivity_threshold))
            
        if '<method>' in label:
            components.append(str(self.method))
            if (self.method == 'rta' or self.method == 'sc' or self.method == 'inverse') \
                    and (self.length is not None):
                if not (np.array(self.length) == np.array([None, None, None])).all() \
                    and not (np.array(self.length) == np.array([0, 0, 0])).all():
                    if '<length>' in label:
                        length_str = 'l'
                        for alpha in range(3):
                            if self.length[alpha] is not None:
                                length_str += '_' + str(self.length[alpha])
                            else:
                                length_str += '_0'
                        components.append(length_str)
                    if '<finite_length_method>' in label and self.finite_length_method is not None:
                        components.append('fs' + str(self.finite_length_method))
                        
        # Delegate to parent class (Phonons) for common attributes
        if hasattr(self, 'phonons'):
            parent_components = self.phonons._get_folder_path_components(label)
            components.extend(parent_components)
            
        return components

    def _load_formatted_property(self, property_name, name):
        """Override formatted loading for Conductivity-specific properties"""
        if property_name == 'mean_free_path':
            loaded = []
            for alpha in range(3):
                loaded.append(np.loadtxt(name + '_' + str(alpha) + '.dat', skiprows=1))
            return np.array(loaded).transpose(1, 2, 0)
        elif property_name == 'conductivity':
            loaded = []
            for alpha in range(3):
                for beta in range(3):
                    loaded.append(np.loadtxt(name + '_' + str(alpha) + '_' + str(beta) + '.dat', skiprows=1))
            return np.array(loaded).reshape((3, 3, self.n_phonons)).transpose(2, 0, 1)
        else:
            # Use default implementation for other properties
            return super()._load_formatted_property(property_name, name)
    
    def _save_formatted_property(self, property_name, name, data):
        """Override formatted saving for Conductivity-specific properties"""
        fmt = '%.18e'
        if property_name == 'mean_free_path':
            for alpha in range(3):
                np.savetxt(name + '_' + str(alpha) + '.dat', data[..., alpha], fmt=fmt,
                          header=str(data[..., 0].shape))
        elif property_name == 'conductivity':
            for alpha in range(3):
                for beta in range(3):
                    np.savetxt(name + '_' + str(alpha) + '_' + str(beta) + '.dat', data[..., alpha, beta], fmt=fmt,
                              header=str(data[..., 0, 0].shape))
        else:
            # Use default implementation for other properties
            super()._save_formatted_property(property_name, name, data)

    @lazy_property(
        label='<diffusivity_bandwidth>/<diffusivity_threshold>/<temperature>/<statistics>/<third_bandwidth>/<include_isotopes>/<method>/<length>/<finite_length_method>')
    def conductivity(self):
        """
        Calculate the thermal conductivity per mode in W/m/K

        Returns
        -------
        conductivity : np array
            (n_k_points, n_modes, 3, 3) float
        """
        method = self.method
        match method:
            case 'qhgk':
                cond, diff = self.calculate_conductivity_and_diffusivity_qhgk()
                self._diffusivity = diff
            case 'full':
                cond = self.calculate_conductivity_full()
            case 'rta' | 'sc' | 'inverse':
                lambd = self.mean_free_path
                conductivity_per_mode = calculate_conductivity_per_mode(
                    self.phonons.heat_capacity.reshape((self.n_phonons)),
                    self.phonons.velocity, lambd, self.phonons.physical_mode,
                    self.n_phonons)

                volume = np.abs(np.linalg.det(self.phonons.atoms.cell))
                cond = conductivity_per_mode / (volume * self.n_k_points)

            case _:
                logging.error(f'Conductivity method "{method}" not implemented')

        # folder = get_folder_from_label(phonons, '<temperature>/<statistics>/<third_bandwidth>')
        # save('cond', folder + '/' + method, cond.reshape(phonons.n_k_points, phonons.n_modes, 3, 3), \
        #      format=phonons.store_format['conductivity'])
        sum = (cond.imag).sum()
        if sum > 1e-3:
            logging.warning('The conductivity has an immaginary part. Sum(Im(k)) = ' + str(sum))
        logging.info('Conductivity calculated')
        return cond.real

    @lazy_property(
        label='<diffusivity_bandwidth>/<diffusivity_threshold>/<temperature>/<statistics>/<third_bandwidth>/<include_isotopes>/<method>/<length>/<finite_length_method>')
    def mean_free_path(self):
        """
        Calculate the mean_free_path per mode in A

        Returns
        -------
        mfp : np array
            (n_k_points, n_modes) float
        """
        method = self.method
        match method:
            case 'rta' | 'sc':
                mfp = self._calculate_mfp_sc()
            case 'inverse':
                mfp = self.calculate_mfp_inverse()
            case _:
                logging.error('Mean free path not available for ' + method)

        # folder = get_folder_from_label(phonons, '<temperature>/<statistics>/<third_bandwidth>')
        # save('cond', folder + '/' + method, cond.reshape(phonons.n_k_points, phonons.n_modes, 3, 3), \
        #      format=phonons.store_format['conductivity'])
        sum = (mfp.imag).sum()
        if sum > 1e-3:
            logging.warning('The conductivity has an immaginary part. Sum(Im(k)) = ' + str(sum))
        return mfp.real

    @property
    def diffusivity(self):
        """
        Calculate the diffusivity, for each k point in k_points and each mode.

        Returns
        -------
        diffusivity : np.array(n_k_points, n_modes)
            diffusivity in mm^2/s
        """
        try:
            return self._diffusivity
        except AttributeError:
            logging.info('You need to calculate the conductivity QHGK first.')

    def calculate_scattering_matrix(self, is_including_diagonal, is_rescaling_omega, is_rescaling_population):
        """Calculate the scattering matrix for phonons."""
        physical_mode = self.phonons.physical_mode.reshape(self.n_phonons)
        frequency = self.phonons.frequency.reshape(self.n_phonons)[physical_mode]

        gamma_tensor = -1 * self.phonons._ps_gamma_and_gamma_tensor[:, 2:]
        gamma_tensor = gamma_tensor[np.ix_(physical_mode, physical_mode)]

        n_physical = physical_mode.sum()
        log_size((n_physical, n_physical), float, name='_scattering_matrix')

        if is_rescaling_population:
            n = self.phonons.population.reshape(self.n_phonons)[physical_mode]
            n_sqrt = (n * (n + 1)) ** 0.5
            np.einsum('a,ab,b->ab', n_sqrt, gamma_tensor, 1 / n_sqrt, out=gamma_tensor)
            logging.info('Asymmetry of gamma_tensor: ' + str(np.abs(gamma_tensor - gamma_tensor.T).sum()))

        if is_including_diagonal:
            gamma = self.phonons.bandwidth.reshape(self.n_phonons)[physical_mode]
            np.fill_diagonal(gamma_tensor, gamma_tensor.diagonal() + gamma)

        if is_rescaling_omega:
            frequency_inv = 1 / frequency
            np.multiply(frequency_inv[:, np.newaxis], gamma_tensor, out=gamma_tensor)
            np.multiply(gamma_tensor, frequency[np.newaxis, :], out=gamma_tensor)

        return gamma_tensor


    def calculate_conductivity_and_diffusivity_qhgk(self):
        phonons = self.phonons
        omega = phonons.omega.reshape((phonons.n_k_points, phonons.n_modes))
        volume = np.abs(np.linalg.det(phonons.atoms.cell))
        q_points = phonons._reciprocal_grid.unitary_grid(is_wrapping=False)
        physical_mode = phonons.physical_mode
        conductivity_per_mode = np.zeros((self.phonons.n_k_points, self.phonons.n_modes, 3, 3), dtype=np.float32)
        diffusivity_with_axis = np.zeros_like(conductivity_per_mode)

        match self.diffusivity_shape:
            case 'lorentz':
                logging.info('Using Lorentzian diffusivity_shape')
                curve = lorentz_delta
            case 'gauss':
                logging.info('Using Gaussian diffusivity_shape')
                curve = gaussian_delta
            case 'triangle':
                logging.info('Using triangular diffusivity_shape')
                curve = triangular_delta
            case _:
                logging.error('Diffusivity shape not implemented')
                return None, None

        is_diffusivity_including_antiresonant = self.is_diffusivity_including_antiresonant

        if self.diffusivity_bandwidth is not None:
            logging.info('Using diffusivity bandwidth from input')
            diffusivity_bandwidth = self.diffusivity_bandwidth * np.ones((phonons.n_k_points, phonons.n_modes),
                                                                         dtype=np.float32)
        else:
            diffusivity_bandwidth = self.phonons.bandwidth.reshape((phonons.n_k_points, phonons.n_modes)) / 2.

        logging.info('Start calculation diffusivity')

        for k_index, q_point in enumerate(q_points):
            phonon = hwqwt.HarmonicWithQTemp(
                q_point=q_point,
                second=self.phonons.forceconstants.second,
                distance_threshold=self.phonons.forceconstants.distance_threshold,
                folder=self.folder,
                storage=self.storage,
                temperature=self.temperature,
                is_classic=self.is_classic,
                is_nw=phonons.is_nw,
                is_unfolding=phonons.is_unfolding,
                is_amorphous=phonons._is_amorphous
            )
            heat_capacity_2d = phonon.heat_capacity_2d
            if phonons.n_modes > 100:
                logging.info('calculating conductivity for q = ' + str(q_point))

            for alpha in range(3):
                if alpha == 0:
                    sij_left = phonon._sij_x
                elif alpha == 1:
                    sij_left = phonon._sij_y
                elif alpha == 2:
                    sij_left = phonon._sij_z

                for beta in range(3):
                    if beta == 0:
                        sij_right = phonon._sij_x
                    elif beta == 1:
                        sij_right = phonon._sij_y
                    elif beta == 2:
                        sij_right = phonon._sij_z

                    if not phonons._is_amorphous:
                        sij_right = np.conjugate(sij_right)

                    diffusivity = calculate_diffusivity(
                        omega[k_index], sij_left, sij_right,
                        diffusivity_bandwidth[k_index],
                        physical_mode[k_index],
                        curve,
                        is_diffusivity_including_antiresonant,
                        self.diffusivity_threshold
                    )

                    conductivity_per_mode[k_index, :, alpha, beta] = (
                            np.sum(heat_capacity_2d * diffusivity, axis=-1) / (volume * phonons.n_k_points)
                    ).real
                    diffusivity_with_axis[k_index, :, alpha, beta] = np.sum(diffusivity, axis=-1).real
                del sij_left, sij_right, diffusivity
            del phonon, heat_capacity_2d
            if k_index % 10 == 0:
                gc.collect()

        diffusivity = 1 / 3 * 1 / 100 * contract('knaa->kn', diffusivity_with_axis)
        return conductivity_per_mode.reshape((self.n_phonons, 3, 3)) * 1e22, diffusivity

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
        finite_length_method = self.finite_length_method
        physical_mode = phonons.physical_mode.reshape(phonons.n_phonons)
        velocity = phonons.velocity.real.reshape((phonons.n_phonons, 3))
        lambd = np.zeros_like(velocity, dtype=np.float32)  # Use float32 if precision allows

        for alpha in range(3):
            scattering_matrix = self.calculate_scattering_matrix(
                is_including_diagonal=False,
                is_rescaling_omega=True,
                is_rescaling_population=False
            )

            gamma = phonons.bandwidth.reshape(phonons.n_phonons)

            if finite_length_method == 'ms' and length is not None and length[alpha]:
                gamma += 2 * np.abs(velocity[:, alpha]) / length[alpha]
            gc.collect()

            np.add(scattering_matrix, np.diag(gamma[physical_mode]), out=scattering_matrix)
            scattering_inverse = np.linalg.inv(scattering_matrix)
            lambd[physical_mode, alpha] = scattering_inverse.dot(velocity[physical_mode, alpha])
            del scattering_matrix, scattering_inverse, gamma
            gc.collect()

            if finite_length_method == 'ballistic' and (self.length[alpha] is not None) and (self.length[alpha] != 0):
                velocity_alpha = velocity[physical_mode, alpha]
                gamma_inv = np.zeros_like(velocity_alpha)
                nonzero_velocity = velocity_alpha != 0
                gamma_inv[nonzero_velocity] = length[alpha] / (2 * np.abs(velocity_alpha[nonzero_velocity]))
                lambd[physical_mode, alpha] = np.diag(gamma_inv).dot(velocity_alpha)
                del gamma_inv, nonzero_velocity, velocity_alpha
                gc.collect()

        return lambd

    def calculate_lambda_tensor(self, alpha, scattering_inverse):
        # TODO: replace with same caching strategy as rest of code
        n_k_points = self.n_k_points
        third_bandwidth = self.phonons.third_bandwidth
        if self.phonons.is_classic:
            stat_label = 'c'
        else:
            stat_label = 'q'
        str_to_add = str(n_k_points) + '_' + str(alpha) + '_' + str(int(self.phonons.temperature)) + '_' + stat_label
        lamdb_filename = 'lamdb_' + '_' + str_to_add
        psi_filename = 'psi_' + str_to_add
        psi_inv_filename = 'psi_inv_' + str_to_add
        if third_bandwidth is not None:
            lamdb_filename = lamdb_filename + '_' + str(third_bandwidth)
            psi_filename = psi_filename + '_' + str(third_bandwidth)
            psi_inv_filename = psi_inv_filename + '_' + str(third_bandwidth)

        lamdb_filename = lamdb_filename + '.npy'
        psi_filename = psi_filename + '.npy'
        psi_inv_filename = psi_inv_filename + '.npy'

        try:
            self._lambd = np.load(lamdb_filename, allow_pickle=True)
            self._psi = np.load(psi_filename, allow_pickle=True)
            self._psi_inv = np.load(psi_inv_filename, allow_pickle=True)
        except FileNotFoundError as err:
            logging.info(err)
            n_phonons = self.n_phonons
            physical_mode = self.phonons.physical_mode.reshape(n_phonons)
            velocity = self.phonons.velocity.real.reshape((n_phonons, 3))[physical_mode, :]
            heat_capacity = self.phonons.heat_capacity.flatten()[physical_mode]
            sqr_heat_capacity = heat_capacity ** 0.5

            v_new = velocity[:, alpha]
            lambd_tensor = contract('m,m,mn,n->mn', sqr_heat_capacity,
                                    v_new,
                                    scattering_inverse,
                                    1 / sqr_heat_capacity)

            # evals and evect equations
            # lambd_tensor = psi.dot(np.diag(lambd)).dot(psi_inv)
            # lambd_tensor.dot(psi) = psi.dot(np.diag(lambd))

            self._lambd, self._psi = np.linalg.eig(lambd_tensor)
            self._psi_inv = np.linalg.inv(self._psi)
            np.save(lamdb_filename, self._lambd)
            np.save(psi_filename, self._psi)
            np.save(psi_inv_filename, self._psi_inv)

    def calculate_conductivity_full(self, is_using_gamma_tensor_evects=False):
        """
        This calculates the conductivity using the full solution of the space-dependent Boltzmann Transport Equation.

        Returns
        -------
        conductivity_per_mode : np array
            (n_k_points, n_modes, 3)
        """
        length = self.length
        n_phonons = self.n_phonons
        n_k_points = self.n_k_points
        volume = np.abs(np.linalg.det(self.phonons.atoms.cell))
        physical_mode = self.phonons.physical_mode.reshape(n_phonons)
        velocity = self.phonons.velocity.real.reshape((n_phonons, 3))[physical_mode, :]
        heat_capacity = self.phonons.heat_capacity.flatten()[physical_mode]
        gamma_tensor = self.calculate_scattering_matrix(is_including_diagonal=True,
                                                        is_rescaling_omega=False,
                                                        is_rescaling_population=True)

        neg_diag = (gamma_tensor.diagonal() < 0).sum()
        logging.info('negative on diagonal : ' + str(neg_diag))
        log_size(gamma_tensor.shape, name='scattering_inverse')
        if is_using_gamma_tensor_evects:
            evals, evects = np.linalg.eigh(gamma_tensor)
            logging.info('negative eigenvals : ' + str((evals < 0).sum()))
            new_physical_states = np.argwhere(evals >= 0)[0, 0]
            reduced_evects = evects[new_physical_states:, new_physical_states:]
            reduced_evals = evals[new_physical_states:]
            scattering_inverse = np.zeros_like(gamma_tensor)
            scattering_inverse[new_physical_states:, new_physical_states:] = reduced_evects.dot(
                np.diag(1 / reduced_evals)).dot((reduced_evects.T))
        else:
            scattering_inverse = np.linalg.inv(gamma_tensor)
        full_cond = np.zeros((n_phonons, 3, 3))
        for alpha in range(3):
            self.calculate_lambda_tensor(alpha, scattering_inverse)
            forward_states = self._lambd > 0
            lambd_p = self._lambd[forward_states]
            only_lambd_plus = self._lambd.copy()
            only_lambd_plus[self._lambd < 0] = 0
            lambd_tilde = only_lambd_plus
            new_lambd = np.zeros_like(lambd_tilde)
            # using average
            # exp_tilde[self._lambd>0] = (length[alpha] + lambd_p * (-1 + np.exp(-length[alpha] / (lambd_p)))) * lambd_p/length[alpha]
            if length is not None:
                if length[alpha]:
                    leng = np.zeros_like(self._lambd)
                    leng[:] = length[alpha]
                    leng[self._lambd < 0] = 0
                    new_lambd[self._lambd > 0] = (1 - np.exp(-length[alpha] / (lambd_p))) * lambd_p
                    # exp_tilde[lambd<0] = (1 - np.exp(-length[0] / (-lambd_m))) * lambd_m
                else:
                    new_lambd[self._lambd > 0] = lambd_p
            else:
                new_lambd[self._lambd > 0] = lambd_p
            lambd_tilde = new_lambd
            for beta in range(3):
                cond = 2 * contract('nl,l,lk,k,k->n',
                                    self._psi,
                                    lambd_tilde,
                                    self._psi_inv,
                                    heat_capacity,
                                    velocity[:, beta],
                                    )
                full_cond[physical_mode, alpha, beta] = cond
        return full_cond / (volume * n_k_points) * 1e22

    def _calculate_mfp_sc(self):
        # TODO: rewrite this method as vector-vector multiplications instead of using the full inversion
        # in order to scale to higher k points meshes
        finite_length_method = self.finite_length_method

        if finite_length_method == 'ms':
            lambd_n = self._calculate_sc_mfp_with_length(matthiessen_length=self.length)
        else:
            lambd_n = self._calculate_sc_mfp_with_length()
        return lambd_n

    def _calculate_sc_mfp_with_length(self, matthiessen_length=None, max_iterations_sc=50):
        tolerance = self.tolerance
        n_iterations = self.n_iterations
        phonons = self.phonons
        if n_iterations is None:
            n_iterations = max_iterations_sc
        velocity = phonons.velocity.real.reshape((phonons.n_k_points, phonons.n_modes, 3))
        velocity = velocity.reshape((phonons.n_phonons, 3))
        physical_mode = phonons.physical_mode.reshape(phonons.n_phonons)
        if n_iterations == 0:
            gamma = phonons.bandwidth.reshape(phonons.n_phonons)
            lambd_0 = mfp_matthiessen(gamma, velocity, matthiessen_length, physical_mode)
            return lambd_0
        else:
            scattering_matrix = -1 * self.calculate_scattering_matrix(is_including_diagonal=False,
                                                                      is_rescaling_omega=True,
                                                                      is_rescaling_population=False)
            gamma = phonons.bandwidth.reshape(phonons.n_phonons)
            lambd_0 = mfp_matthiessen(gamma, velocity, matthiessen_length, physical_mode)
            lambd_n = np.zeros_like(lambd_0)
            avg_conductivity = None
            n_iteration = 0
            for n_iteration in range(n_iterations):
                conductivity_per_mode = calculate_conductivity_per_mode(
                    phonons.heat_capacity.reshape((phonons.n_phonons)),
                    velocity, lambd_n, physical_mode, phonons.n_phonons)
                new_avg_conductivity = np.diag(np.sum(conductivity_per_mode, 0)).mean()
                if avg_conductivity:
                    if tolerance is not None:
                        if np.abs(avg_conductivity - new_avg_conductivity) < tolerance:
                            break
                avg_conductivity = new_avg_conductivity
                delta_lambd = 1 / phonons.bandwidth.reshape((phonons.n_phonons))[physical_mode, np.newaxis] \
                              * scattering_matrix.dot(lambd_n[physical_mode, :])
                lambd_n[physical_mode, :] = lambd_0[physical_mode, :] + delta_lambd[:, :]
            logging.info('Number of self-consistent iterations: ' + str(n_iteration))
            return lambd_n

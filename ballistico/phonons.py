"""
Ballistico
Anharmonic Lattice Dynamics
"""
from opt_einsum import contract
from ballistico.helpers.tools import is_calculated
from ballistico.helpers.tools import lazy_property
from ballistico.helpers.tools import apply_boundary_with_cell, q_vec_from_q_index
from scipy.linalg.lapack import dsyev
import numpy as np
import ase.units as units

KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J
THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
EVTOTENJOVERMOL = units.mol / (10 * units.J)

DELTA_DOS = 1
NUM_DOS = 100
FOLDER_NAME = 'ald-output'


class Phonons:
    def __init__(self, **kwargs):
        """The phonons object exposes all the phononic properties of a system,

        Parameters
        ----------
        finite_difference : FiniteDifference
            contains all the information about the system and the derivatives of the potential.
        is_classic : bool
            specifies if the system is classic, `True` or quantum, `False`
        temperature : float
            defines the temperature of the simulation.
        folder (optional) : string
            specifies where to store the data files. Default is `output`.
        kpts (optional) : (3) tuple
            defines the number of k points to use to create the k mesh. Default is [1, 1, 1].
        min_frequency (optional) : float
            ignores all phonons with frequency below `min_frequency` THz, Default is None..
        max_frequency (optional) : float
            ignores all phonons with frequency above `max_frequency` THz, Default is None.
        sigma_in (optional) : float or None
            defines the width of the energy conservation smearing in the phonons scattering calculation.
            If `None` the width is calculated dynamically. Otherwise the input value corresponds to the
            width in THz. Default is None.
        broadening_shape (optional) : string
            defines the algorithm to use to calculate the broadening. Available broadenings are `gauss` and `triangle`.
            Default is `gauss`.
        is_tf_backend (optional) : bool
            defines if the third order phonons scattering calculations should be performed on tensorflow (True) or
            numpy (False). Default is True.
        Returns
        -------
        Phonons
            An instance of the `Phonons` class.

        """
        self.finite_difference = kwargs.pop('finite_difference')
        if 'is_classic' in kwargs:
            self.is_classic = bool(kwargs['is_classic'])
        if 'temperature' in kwargs:
            self.temperature = float(kwargs['temperature'])
        self.folder = kwargs.pop('folder', FOLDER_NAME)
        self.kpts = kwargs.pop('kpts', (1, 1, 1))
        self.kpts = np.array(self.kpts)
        self.min_frequency = kwargs.pop('min_frequency', None)
        self.max_frequency = kwargs.pop('max_frequency', None)
        self.sigma_in = kwargs.pop('sigma_in', None)
        self.broadening_shape = kwargs.pop('broadening_shape', 'gauss')
        self.is_tf_backend = kwargs.pop('is_tf_backend', False)
        self.is_nw = kwargs.pop('is_nw', False)
        self.atoms = self.finite_difference.atoms
        self.supercell = np.array(self.finite_difference.supercell)
        self.n_k_points = int(np.prod(self.kpts))
        self.n_atoms = self.finite_difference.n_atoms
        self.n_modes = self.finite_difference.n_modes
        self.n_phonons = self.n_k_points * self.n_modes
        self.is_able_to_calculate = True



    @lazy_property(is_storing=True, is_reduced_path=True)
    def frequencies(self):
        """
        Calculate phonons frequencies
        Returns
        -------
        frequencies : np array
            (n_k_points, n_modes) frequencies in THz
        """
        frequencies = self.calculate_frequencies()
        return frequencies


    @lazy_property(is_storing=True, is_reduced_path=True)
    def velocities(self):
        """Calculates the velocities using Hellmann-Feynman theorem.
        Returns
        -------
        velocities : np array
            (n_k_points, n_unit_cell * 3, 3) velocities in 100m/s or A/ps
        """
        velocities = self.calculate_velocities()
        return velocities


    @lazy_property(is_storing=True, is_reduced_path=False)
    def occupations(self):
        occupations =  self.calculate_occupations()
        return occupations


    @lazy_property(is_storing=True, is_reduced_path=False)
    def heat_capacity(self):
        """Calculate the heat capacity for each k point in k_points and each mode.
        If classical, it returns the Boltzmann constant in W/m/K. If quantum it returns

        .. math::

            c_\\mu = k_B \\frac{\\nu_\\mu^2}{ \\tilde T^2} n_\\mu (n_\\mu + 1)

        where the frequency :math:`\\nu` and the temperature :math:`\\tilde T` are in THz.

        Returns
        -------
        c_v : np.array(n_k_points, n_modes)
            heat capacity in W/m/K for each k point and each mode
        """
        c_v = self.calculate_c_v()
        return c_v


    @lazy_property(is_storing=False, is_reduced_path=False)
    def rescaled_eigenvectors(self):
        n_atoms = self.n_atoms
        n_modes = self.n_modes
        masses = self.atoms.get_masses()
        rescaled_eigenvectors = self.eigenvectors[:, :, :].reshape(
            (self.n_k_points, n_atoms, 3, n_modes), order='C') / np.sqrt(
            masses[np.newaxis, :, np.newaxis, np.newaxis])
        rescaled_eigenvectors = rescaled_eigenvectors.reshape((self.n_k_points, n_modes, n_modes), order='C')
        return rescaled_eigenvectors


    @property
    def eigenvalues(self):
        """Calculates the eigenvalues of the dynamical matrix in Thz^2.

        Returns
        -------
        eigenvalues : np array
            (n_phonons) Eigenvalues of the dynamical matrix
        """
        eigenvalues = self._eigensystem[:, 0, :]
        return eigenvalues


    @property
    def eigenvectors(self):
        """Calculates the eigenvectors of the dynamical matrix.

        Returns
        -------
        eigenvectors : np array
            (n_phonons, n_phonons) Eigenvectors of the dynamical matrix
        """
        eigenvectors = self._eigensystem[:, 1:, :]
        return eigenvectors


    @property
    def gamma(self):
        gamma = self._ps_and_gamma[:, 1]
        return gamma


    @property
    def ps(self):
        ps = self._ps_and_gamma[:, 0]
        return ps


#################
# Private methods
#################

    @lazy_property(is_storing=False, is_reduced_path=False)
    def _main_q_mesh(self):
        q_mesh = q_vec_from_q_index(np.arange(self.n_k_points), self.kpts)
        return q_mesh


    @lazy_property(is_storing=True, is_reduced_path=True)
    def _dynmat_derivatives(self):
        dynmat_derivatives = self.calculate_dynmat_derivatives()
        return dynmat_derivatives


    @lazy_property(is_storing=True, is_reduced_path=True)
    def _eigensystem(self):
        eigensystem = self.calculate_eigensystem()
        return eigensystem


    @lazy_property(is_storing=False, is_reduced_path=True)
    def _physical_modes(self):
        physical_modes = np.ones_like(self.frequencies.reshape(self.n_phonons), dtype=bool)
        if self.min_frequency is not None:
            physical_modes = physical_modes & (self.frequencies.reshape(self.n_phonons) > self.min_frequency)
        if self.max_frequency is not None:
            physical_modes = physical_modes & (self.frequencies.reshape(self.n_phonons) < self.max_frequency)
        if self.is_nw:
            physical_modes[:4] = False
        else:
            physical_modes[:3] = False
        return physical_modes


    @lazy_property(is_storing=False, is_reduced_path=True)
    def _chi_k(self):
        chi = np.zeros((self.n_k_points, self.finite_difference.n_replicas), dtype=np.complex)
        for index_q in range(self.n_k_points):
            k_point = q_vec_from_q_index(index_q, self.kpts)
            chi[index_q] = self._chi(k_point)
        return chi


    @lazy_property(is_storing=False, is_reduced_path=True)
    def _omegas(self):
        return self.frequencies * 2 * np.pi


    @lazy_property(is_storing=True, is_reduced_path=True)
    def _velocities_af(self):
        velocities_AF = self.calculate_velocities_af()
        return velocities_AF


    @lazy_property(is_storing=True, is_reduced_path=True)
    def _sij(self):
        sij = self.calculate_sij()
        return sij


    @lazy_property(is_storing=True, is_reduced_path=False)
    def _ps_and_gamma(self):
        if is_calculated('ps_gamma_and_gamma_tensor', self):
            ps_and_gamma = self._ps_gamma_and_gamma_tensor[:, :2]
        else:
            ps_and_gamma = self._calculate_ps_and_gamma(is_gamma_tensor_enabled=False)
        return ps_and_gamma


    @lazy_property(is_storing=True, is_reduced_path=False)
    def _ps_gamma_and_gamma_tensor(self):
        ps_gamma_and_gamma_tensor = self._calculate_ps_and_gamma(is_gamma_tensor_enabled=True)
        return ps_gamma_and_gamma_tensor


    @lazy_property(is_storing=False, is_reduced_path=False)
    def _scattering_matrix_without_diagonal(self):
        frequencies = self._keep_only_physical(self.frequencies.reshape((self.n_phonons), order='C'))
        gamma_tensor = self._keep_only_physical(self._ps_gamma_and_gamma_tensor[:, 2:])
        scattering_matrix_without_diagonal = contract('a,ab,b->ab', 1 / frequencies, gamma_tensor, frequencies)
        return scattering_matrix_without_diagonal


    @lazy_property(is_storing=False, is_reduced_path=False)
    def _scattering_matrix(self):
        scattering_matrix = -1 * self._scattering_matrix_without_diagonal
        gamma = self._keep_only_physical(self.gamma.reshape((self.n_phonons), order='C'))
        scattering_matrix = scattering_matrix + np.diag(gamma)
        return scattering_matrix


    @property
    def _is_amorphous(self):
        is_amorphous = (self.kpts == (1, 1, 1)).all()
        return is_amorphous


    def _keep_only_physical(self, operator):
        physical_modes = self._physical_modes
        if operator.shape == (self.n_phonons, self.n_phonons):
            index = np.outer(physical_modes, physical_modes)
            return operator[index].reshape((physical_modes.sum(), physical_modes.sum()), order='C')
        else:
            return operator[physical_modes, ...]


    def _chi(self, qvec):
        dxij = self.finite_difference.list_of_replicas
        cell_inv = self.finite_difference.cell_inv
        chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
        return chi_k


    def _calculate_ps_and_gamma(self, is_gamma_tensor_enabled=True):
        print('Projection started')
        if self.is_tf_backend:
            try:
                from ballistico.anharmonic_tf import Anharmonic
            except ModuleNotFoundError as err:
                print(err)
                print('tensorflow>=2.0 is required to run accelerated routines. Please consider installing tensorflow>=2.0. More info here: https://www.tensorflow.org/install/pip')
                print('Using numpy engine instead.')
                from ballistico.anharmonic import Anharmonic
        else:
            from ballistico.anharmonic import Anharmonic

        anharmonic = Anharmonic(finite_difference=self.finite_difference,
                                frequencies=self.frequencies,
                                kpts=self.kpts,
                                rescaled_eigenvectors=self.rescaled_eigenvectors,
                                is_gamma_tensor_enabled=is_gamma_tensor_enabled,
                                chi_k=self._chi_k,
                                velocities=self.velocities,
                                physical_modes=self._physical_modes,
                                occupations=self.occupations,
                                sigma_in=self.sigma_in,
                                broadening_shape=self.broadening_shape
                                )
        if self._is_amorphous:
            ps_and_gamma = anharmonic.project_amorphous()
        else:
            ps_and_gamma = anharmonic.project_crystal()
        return ps_and_gamma


    def calculate_occupations(self):
        frequencies = self.frequencies
        temp = self.temperature * KELVINTOTHZ
        density = np.zeros_like(frequencies)
        physical_modes = self._physical_modes.reshape((self.n_k_points, self.n_modes))
        if self.is_classic is False:
            density[physical_modes] = 1. / (np.exp(frequencies[physical_modes] / temp) - 1.)
        else:
            density[physical_modes] = temp / frequencies[physical_modes]
        return density


    def calculate_c_v(self):
        frequencies = self.frequencies
        c_v = np.zeros_like (frequencies)
        physical_modes = self._physical_modes.reshape((self.n_k_points, self.n_modes))
        temperature = self.temperature * KELVINTOTHZ
        if (self.is_classic):
            c_v[physical_modes] = KELVINTOJOULE
        else:
            f_be = self.occupations
            c_v[physical_modes] = KELVINTOJOULE * f_be[physical_modes] * (f_be[physical_modes] + 1) * self.frequencies[physical_modes] ** 2 / \
                                  (temperature ** 2)
        return c_v


    def calculate_frequencies(self, q_points=None):
        is_main_mesh = True if q_points is None else False
        if not is_main_mesh:
            if q_points.shape == self._main_q_mesh.shape:
                if (q_points == self._main_q_mesh).all():
                    is_main_mesh = True
        if is_main_mesh:
            q_points = self._main_q_mesh
        else:
            q_points = apply_boundary_with_cell(q_points)
        eigenvals = self.calculate_eigensystem(q_points, only_eigenvals=True)
        frequencies = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
        return frequencies.real


    def calculate_dynmat_derivatives(self, q_points=None):
        is_main_mesh = True if q_points is None else False
        if not is_main_mesh:
            if q_points.shape == self._main_q_mesh.shape:
                if (q_points == self._main_q_mesh).all():
                    is_main_mesh = True
        if is_main_mesh:
            q_points = self._main_q_mesh
        else:
            q_points = apply_boundary_with_cell(q_points)
        atoms = self.atoms
        list_of_replicas = self.finite_difference.list_of_replicas
        replicated_cell = self.finite_difference.replicated_atoms.cell
        replicated_cell_inv = self.finite_difference.replicated_cell_inv
        dynmat = self.finite_difference.dynmat
        positions = self.finite_difference.atoms.positions

        n_unit_cell = atoms.positions.shape[0]
        n_k_points = q_points.shape[0]
        ddyn = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3, 3)).astype(np.complex)
        for index_k in range(n_k_points):
            qvec = q_points[index_k]
            if self._is_amorphous:
                dxij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                dxij = apply_boundary_with_cell(dxij, replicated_cell, replicated_cell_inv)
                dynmat_derivatives = contract('ija,ibjc->ibjca', dxij, dynmat[:, :, 0, :, :])
            else:
                list_of_replicas = list_of_replicas
                dxij = positions[:, np.newaxis, np.newaxis, :] - (
                        positions[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])
                dynmat_derivatives = contract('ilja,ibljc,l->ibjca', dxij, dynmat, self.chi(qvec))
            ddyn[index_k] = dynmat_derivatives.reshape((self.n_modes, self.n_modes, 3), order='C')
        return ddyn


    def calculate_sij(self, q_points=None, is_antisymmetrizing=False):
        is_main_mesh = True if q_points is None else False
        if not is_main_mesh:
            if q_points.shape == self._main_q_mesh.shape:
                if (q_points == self._main_q_mesh).all():
                    is_main_mesh = True
        if is_main_mesh:
            q_points = self._main_q_mesh
        else:
            q_points = apply_boundary_with_cell(q_points)
        if is_main_mesh:
            dynmat_derivatives = self._dynmat_derivatives
        else:
            dynmat_derivatives = self.calculate_dynmat_derivatives(q_points)
        eigenvects = self.calculate_eigensystem(q_points)[:, 1:, :]
        if is_antisymmetrizing:
            error = np.linalg.norm(dynmat_derivatives + dynmat_derivatives.swapaxes(0, 1)) / 2
            dynmat_derivatives = (dynmat_derivatives - dynmat_derivatives.swapaxes(0, 1)) / 2
            print('Symmetrization error: ' + str(error))
        if self._is_amorphous:
            sij = contract('kim,kija,kjn->kmna', eigenvects, dynmat_derivatives, eigenvects)
        else:
            sij = contract('kim,kija,kjn->kmna', eigenvects.conj(), dynmat_derivatives, eigenvects)
        return sij


    def calculate_velocities_af(self, q_points=None, is_antisymmetrizing=False):
        is_main_mesh = True if q_points is None else False
        if not is_main_mesh:
            if q_points.shape == self._main_q_mesh.shape:
                if (q_points == self._main_q_mesh).all():
                    is_main_mesh = True
        if is_main_mesh:
            q_points = self._main_q_mesh
        else:
            q_points = apply_boundary_with_cell(q_points)
        frequencies = self.calculate_frequencies(q_points)
        if is_main_mesh:
            sij = self._sij
        else:
            sij = self.calculate_sij(q_points, is_antisymmetrizing)
        velocities_AF = contract('kmna,kmn->kmna', sij,
                                 1 / (2 * np.pi * np.sqrt(frequencies[:, :, np.newaxis]) * np.sqrt(
                                     frequencies[:, np.newaxis, :]))) / 2
        return velocities_AF


    def calculate_velocities(self, q_points=None, is_antisymmetrizing=False):
        is_main_mesh = True if q_points is None else False
        if not is_main_mesh:
            if q_points.shape == self._main_q_mesh.shape:
                if (q_points == self._main_q_mesh).all():
                    is_main_mesh = True
        if is_main_mesh:
            q_points = self._main_q_mesh
        else:
            q_points = apply_boundary_with_cell(q_points)
        if is_main_mesh:
            velocities_AF = self._velocities_af
        else:
            velocities_AF = self.calculate_velocities_af(q_points, is_antisymmetrizing=is_antisymmetrizing)
        velocities = 1j * contract('kmma->kma', velocities_AF)
        return velocities.real


    def calculate_eigensystem(self, q_points=None, only_eigenvals=False):
        is_main_mesh = True if q_points is None else False
        if not is_main_mesh:
            if q_points.shape == self._main_q_mesh.shape:
                if (q_points == self._main_q_mesh).all():
                    is_main_mesh = True
        if is_main_mesh:
            q_points = self._main_q_mesh
        else:
            q_points = apply_boundary_with_cell(q_points)
        atoms = self.atoms
        n_unit_cell = atoms.positions.shape[0]
        n_k_points = q_points.shape[0]
        # Here we store the eigenvalues in the last column
        if self._is_amorphous:
            dtype = np.float
        else:
            dtype = np.complex
        if only_eigenvals:
            esystem = np.zeros((n_k_points, n_unit_cell * 3), dtype=dtype)
        else:
            esystem = np.zeros((n_k_points, n_unit_cell * 3 + 1, n_unit_cell * 3), dtype=dtype)
        for index_k in range(n_k_points):
            qvec = q_points[index_k]
            is_at_gamma = (qvec == (0, 0, 0)).all()
            dynmat = self.finite_difference.dynmat
            if is_at_gamma:
                dyn_s = contract('ialjb->iajb', dynmat)
            else:
                # TODO: the following espression could be done on the whole main_q_mesh
                dyn_s = contract('ialjb,l->iajb', dynmat, self.chi(qvec))
            dyn_s = dyn_s.reshape((self.n_modes, self.n_modes), order='C')
            if only_eigenvals:
                evals = np.linalg.eigvalsh(dyn_s)
                esystem[index_k] =  evals
            else:
                if is_at_gamma:
                    evals, evects = dsyev(dyn_s)[:2]
                else:
                    evals, evects = np.linalg.eigh(dyn_s)
                esystem[index_k] =  np.vstack((evals, evects))
        return esystem


    def chi(self, qvec):
        dxij = self.finite_difference.list_of_replicas
        cell_inv = self.finite_difference.cell_inv
        chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
        return chi_k

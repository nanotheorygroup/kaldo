from kaldo.grid import wrap_coordinates, Grid
from kaldo.observables.forceconstant import chi
from kaldo.observables.observable import Observable
import numpy as np
from ase import units
from opt_einsum import contract
from kaldo.helpers.storage import lazy_property
import tensorflow as tf
from scipy.linalg.lapack import zheev
from kaldo.helpers.logger import get_logger, log_size
# from numpy.linalg import eigh

logging = get_logger()

MIN_N_MODES_TO_STORE = 1000


class HarmonicWithQ(Observable):

    def __init__(self, q_point, second,
                 distance_threshold=None,
                 storage='numpy',
                 is_nw=False,
                 is_unfolding=False,
                 is_amorphous=False,
                 *kargs,
                 **kwargs):
        super().__init__(*kargs, **kwargs)
        # Input arguments
        self.q_point = q_point
        self.atoms = second.atoms
        self.n_modes = self.atoms.positions.shape[0] * 3
        self.supercell = second.supercell
        self.second = second
        self.distance_threshold = distance_threshold
        self.physical_mode = np.ones((1, self.n_modes), dtype=bool)
        # Arguments for specific physical assumptions
        self.is_amorphous = is_amorphous
        self.is_unfolding = is_unfolding
        self.is_nac = True if 'dielectric' in self.atoms.info else False
        self.is_nw = is_nw
        if (q_point == [0, 0, 0]).all():
            if self.is_nw:
                self.physical_mode[0, :4] = False
            else:
                self.physical_mode[0, :3] = False
        if self.n_modes > MIN_N_MODES_TO_STORE:
            self.storage = storage
        else:
            self.storage = 'memory'

    @lazy_property(label='<q_point>')
    def frequency(self):
        frequency = self.calculate_frequency()[np.newaxis, :]
        return frequency

    @lazy_property(label='<q_point>')
    def velocity(self):
        velocity = self.calculate_velocity()
        return velocity

    @lazy_property(label='<q_point>')
    def participation_ratio(self):
        participation_ratio = self.calculate_participation_ratio()
        return participation_ratio

    @lazy_property(label='<q_point>')
    def _dynmat_derivatives_x(self):
        if self.is_unfolding:
            _dynmat_derivatives = self.calculate_dynmat_derivatives_unfolded(direction=0)
        else:
            _dynmat_derivatives = self.calculate_dynmat_derivatives(direction=0)
        return _dynmat_derivatives

    @lazy_property(label='<q_point>')
    def _dynmat_derivatives_y(self):
        if self.is_unfolding:
            _dynmat_derivatives = self.calculate_dynmat_derivatives_unfolded(direction=1)
        else:
            _dynmat_derivatives = self.calculate_dynmat_derivatives(direction=1)
        return _dynmat_derivatives

    @lazy_property(label='<q_point>')
    def _dynmat_derivatives_z(self):
        if self.is_unfolding:
            _dynmat_derivatives = self.calculate_dynmat_derivatives_unfolded(direction=2)
        else:
            _dynmat_derivatives = self.calculate_dynmat_derivatives(direction=2)
        return _dynmat_derivatives

    @lazy_property(label='<q_point>')
    def _dynmat_fourier(self):
        dynmat_fourier = self.calculate_dynmat_fourier()
        return dynmat_fourier

    @lazy_property(label='<q_point>')
    def _eigensystem(self):
        if self.is_unfolding:
            _eigensystem = self.calculate_eigensystem_unfolded(only_eigenvals=False)
        else:
            _eigensystem = self.calculate_eigensystem(only_eigenvals=False)
        return _eigensystem

    @lazy_property(label='<q_point>')
    def _sij_x(self):
        _sij = self.calculate_sij(direction=0)
        return _sij

    @lazy_property(label='<q_point>')
    def _sij_y(self):
        _sij = self.calculate_sij(direction=1)
        return _sij

    @lazy_property(label='<q_point>')
    def _sij_z(self):
        _sij = self.calculate_sij(direction=2)
        return _sij

    def calculate_frequency(self):
        # TODO: replace calculate_eigensystem() with eigensystem
        if self.is_unfolding:
            eigenvals = self.calculate_eigensystem_unfolded(only_eigenvals=True)
        else:
            eigenvals = self.calculate_eigensystem(only_eigenvals=True)
        frequency = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
        return frequency.real

    def calculate_dynmat_derivatives(self, direction):
        q_point = self.q_point
        is_amorphous = self.is_amorphous
        distance_threshold = self.distance_threshold
        atoms = self.atoms
        list_of_replicas = self.second.list_of_replicas
        replicated_cell = self.second.replicated_atoms.cell
        replicated_cell_inv = self.second._replicated_cell_inv
        cell_inv = self.second.cell_inv
        dynmat = self.second.dynmat
        positions = self.atoms.positions
        n_unit_cell = atoms.positions.shape[0]
        n_modes = n_unit_cell * 3
        n_replicas = np.prod(self.supercell)
        shape = (1, n_unit_cell * 3, n_unit_cell * 3)
        dir = ['_x', '_y', '_z']
        type = complex if (not self.is_amorphous) else float
        log_size(shape, type, name='dynamical_matrix_derivative_' + dir[direction])
        if self.is_amorphous:
            distance = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distance = wrap_coordinates(distance, replicated_cell, replicated_cell_inv)
            dynmat_derivatives = contract('ij,ibjc->ibjc',
                                          tf.convert_to_tensor(distance[..., direction]),
                                          dynmat[0, :, :, 0, :, :],
                                          backend='tensorflow')
        else:
            distance = positions[:, np.newaxis, np.newaxis, :] - (
                    positions[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])

            if distance_threshold is not None:

                distance_to_wrap = positions[:, np.newaxis, np.newaxis, :] - (
                    self.second.replicated_atoms.positions.reshape(n_replicas, n_unit_cell, 3)[
                    np.newaxis, :, :, :])

                shape = (n_unit_cell, 3, n_unit_cell, 3)
                type = complex
                dynmat_derivatives = np.zeros(shape, dtype=type)
                for l in range(n_replicas):
                    wrapped_distance = wrap_coordinates(distance_to_wrap[:, l, :, :], replicated_cell,
                                                        replicated_cell_inv)
                    mask = (np.linalg.norm(wrapped_distance, axis=-1) < distance_threshold)
                    id_i, id_j = np.argwhere(mask).T
                    dynmat_derivatives[id_i, :, id_j, :] += contract('f,fbc->fbc', distance[id_i, l, id_j, direction], \
                                                                     dynmat.numpy()[0, id_i, :, 0, id_j, :] *
                                                                     chi(q_point, list_of_replicas, cell_inv)[l])
            else:
                dynmat_derivatives = contract('ilj,ibljc,l->ibjc',
                                              tf.convert_to_tensor(distance.astype(complex)[..., direction]),
                                              tf.cast(dynmat[0], tf.complex128),
                                              tf.convert_to_tensor(
                                                  chi(q_point, list_of_replicas, cell_inv).flatten().astype(
                                                      complex)),
                                              backend='tensorflow')
        dynmat_derivatives = tf.reshape(dynmat_derivatives, (n_modes, n_modes))
        if self.is_nac:
            dynmat_derivatives += self.nac_derivatives(direction=direction)
        return dynmat_derivatives

    def calculate_sij(self, direction):
        q_point = self.q_point
        shape = (3 * self.atoms.positions.shape[0], 3 * self.atoms.positions.shape[0])
        if self.is_amorphous and (self.q_point == np.array([0, 0, 0])).all():
            type = float
        else:
            type = complex
        eigenvects = self._eigensystem[1:, :]
        if direction == 0:
            dynmat_derivatives = self._dynmat_derivatives_x
        if direction == 1:
            dynmat_derivatives = self._dynmat_derivatives_y
        if direction == 2:
            dynmat_derivatives = self._dynmat_derivatives_z
        if self.atoms.positions.shape[0] > 500:
            # We want to print only for big systems
            logging.info('Flux operators for q = ' + str(q_point) + ', direction = ' + str(direction))
            dir = ['_x', '_y', '_z']
            log_size(shape, type, name='sij' + dir[direction])
        if self.is_amorphous and (self.q_point == np.array([0, 0, 0])).all():
            sij = tf.tensordot(eigenvects, dynmat_derivatives, (0, 1))
            sij = tf.tensordot(eigenvects, sij, (0, 1))
        else:
            eigenvects = tf.cast(eigenvects, tf.complex128)
            dynmat_derivatives = tf.cast(dynmat_derivatives, tf.complex128)
            sij = tf.tensordot(eigenvects, dynmat_derivatives, (0, 1))
            sij = tf.tensordot(tf.math.conj(eigenvects), sij, (0, 1))

            qstr = '_'.join(np.round(self.q_point, 4).astype(str))
            direction_dic = {0: 'x', 1: 'y', 2: 'z'}
            dstr = direction_dic[direction]
            if self.is_nac:
                np.save('ddyns/c_sevects_{}-{}'.format(qstr, dstr), eigenvects.numpy())
                np.save('ddyns/c_sij_{}-{}'.format(qstr, dstr), sij.numpy())
            else:
                np.save('ddyns/uc_sevects_{}-{}'.format(qstr, dstr), eigenvects.numpy())
                np.save('ddyns/uc_sij_{}-{}'.format(qstr, dstr), sij.numpy())
        return sij

    def calculate_velocity(self):
        frequency = self.frequency[0]
        velocity = np.zeros((self.n_modes, 3))
        inverse_sqrt_freq = tf.cast(tf.convert_to_tensor(1 / np.sqrt(frequency)), tf.complex128)
        if self.is_amorphous:
            inverse_sqrt_freq = tf.cast(inverse_sqrt_freq, tf.float64)
        for alpha in range(3):
            if alpha == 0:
                sij = self._sij_x
            if alpha == 1:
                sij = self._sij_y
            if alpha == 2:
                sij = self._sij_z
            velocity_AF = 1 / (2 * np.pi) * contract('mn,m,n->mn', sij,
                                                     inverse_sqrt_freq, inverse_sqrt_freq, backend='tensorflow') / 2
            velocity_AF = tf.where(tf.math.is_nan(tf.math.real(velocity_AF)), 0., velocity_AF)
            velocity[..., alpha] = contract('mm->m', velocity_AF.numpy().imag)

            qstr = '_'.join(np.round(self.q_point, 4).astype(str))
            direction_dic = {0: 'x', 1: 'y', 2: 'z'}
            dstr = direction_dic[alpha]
            if self.is_nac:
                np.save('ddyns/c_isf_{}'.format(qstr), inverse_sqrt_freq.numpy())
                np.save('ddyns/c_velaf_{}-{}'.format(qstr, dstr), velocity_AF)
                np.save('ddyns/c_vel_{}-{}'.format(qstr, dstr), velocity[..., alpha])
            else:
                np.save('ddyns/uc_isf_{}'.format(qstr), inverse_sqrt_freq.numpy())
                np.save('ddyns/uc_velaf_{}-{}'.format(qstr, dstr), velocity_AF)
                np.save('ddyns/uc_vel_{}-{}'.format(qstr, dstr), velocity[..., alpha])
        return velocity[np.newaxis, ...]

    def calculate_dynmat_fourier(self):
        q_point = self.q_point
        distance_threshold = self.distance_threshold
        atoms = self.atoms
        n_unit_cell = atoms.positions.shape[0]
        n_replicas = np.prod(self.supercell)
        dynmat = self.second.dynmat
        cell_inv = self.second.cell_inv
        replicated_cell_inv = self.second._replicated_cell_inv
        is_at_gamma = (q_point == (0, 0, 0)).all()
        list_of_replicas = self.second.list_of_replicas
        log_size((self.n_modes, self.n_modes), complex, name='dynmat_fourier')
        if distance_threshold is not None:
            shape = (n_unit_cell, 3, n_unit_cell, 3)
            type = complex
            dyn_s = np.zeros(shape, dtype=type)
            replicated_cell = self.second.replicated_atoms.cell

            for l in range(n_replicas):
                distance_to_wrap = atoms.positions[:, np.newaxis, :] - (
                    self.second.replicated_atoms.positions.reshape(n_replicas, n_unit_cell, 3)[np.newaxis, l, :, :])

                distance_to_wrap = wrap_coordinates(distance_to_wrap, replicated_cell, replicated_cell_inv)

                mask = np.linalg.norm(distance_to_wrap, axis=-1) < distance_threshold
                id_i, id_j = np.argwhere(mask).T
                dyn_s[id_i, :, id_j, :] += dynmat.numpy()[0, id_i, :, 0, id_j, :] * \
                                           chi(q_point, list_of_replicas, cell_inv)[l]
        else:
            if is_at_gamma:
                if self.is_amorphous:
                    dyn_s = dynmat[0]
                else:
                    dyn_s = contract('ialjb->iajb', dynmat[0], backend='tensorflow')
            else:
                dyn_s = contract('ialjb,l->iajb',
                                 tf.cast(dynmat[0], tf.complex128),
                                 tf.convert_to_tensor(chi(q_point, list_of_replicas, cell_inv).flatten()),
                                 backend='tensorflow')
        dyn_s = tf.reshape(dyn_s, (self.n_modes, self.n_modes))
        return dyn_s

    def calculate_eigensystem(self, only_eigenvals):
        dyn_s = self._dynmat_fourier
        if self.is_nac:
            dyn_lr = self.nac_dynmat(qpoint=None)
            dyn_lr += self.nac_dynmat(qpoint=self.q_point)
            if (self.q_point == np.array([0, 0, 0])).all():
                dyn_lr = tf.cast(dyn_lr, tf.float64)
            else:
                dyn_lr = tf.cast(dyn_lr, tf.complex128)
            dyn_s += dyn_lr

        if only_eigenvals:
            esystem = tf.linalg.eigvalsh(dyn_s)
        else:
            log_size(self._dynmat_fourier.shape, type=complex, name='eigensystem')
            esystem = tf.linalg.eigh(dyn_s)
            esystem = tf.concat(axis=0, values=(esystem[0][tf.newaxis, :], esystem[1]))
        return esystem

    def calculate_participation_ratio(self):
        n_atoms = self.n_modes // 3
        eigensystem = self._eigensystem[1:, :]
        eigenvectors = tf.transpose(eigensystem)
        eigenvectors = np.reshape(eigenvectors, (self.n_modes, n_atoms, 3))
        conjugate = tf.math.conj(eigenvectors)
        participation_ratio = tf.math.reduce_sum(eigenvectors*conjugate, axis=2)
        participation_ratio = tf.math.square(participation_ratio)
        participation_ratio = tf.math.reciprocal(tf.math.reduce_sum(participation_ratio, axis=1) * n_atoms)
        return participation_ratio

    def calculate_eigensystem_unfolded(self, only_eigenvals=False):
        # This algorithm should be the same as the ShengBTE version
        q_point = self.q_point
        supercell = self.second.supercell
        atoms = self.second.atoms
        cell = atoms.cell
        reciprocal_n = np.round(atoms.cell.reciprocal(), 12)  # round to avoid accumulation of error
        reciprocal_n /= reciprocal_n[0, 0] # Normalized reciprocal cell
        n_unit_cell = len(atoms)
        distances = atoms.positions[:, None, :] - atoms.positions[None, :, :]

        # Get Force constants
        fc_s = self.second.dynmat.numpy()
        fc_s = fc_s.reshape((n_unit_cell, 3, supercell[0], supercell[1], supercell[2], n_unit_cell, 3))
        supercell_positions = self.second.supercell_positions
        supercell_norms = 1 / 2 * np.linalg.norm(supercell_positions, axis=1) ** 2
        cell_replicas = self.second.supercell_replicas
        cell_positions = np.einsum('ia,ab->ib', cell_replicas, cell)
        cell_plus_distance = cell_positions[:, None, None, :] + distances[None, :, :, :]
        supercell_positions = self.second.supercell_positions
        supercell_cell_distances = np.einsum('La,inma->Linm', supercell_positions, cell_plus_distance)
        projection = supercell_cell_distances - supercell_norms[:, None, None, None]

        # Filter + Weights
        mask_distance = (projection <= 1e-6).all(axis=0)
        n_equivalent = (np.abs(projection) <= 1e-6).sum(axis=0)
        weight = 1 / n_equivalent
        coefficients = weight * mask_distance

        # Find contributing replicas
        mask_full = coefficients.any(axis=(-2, -1))
        coefficients = coefficients[mask_full]
        cell_replicas = cell_replicas[mask_full]
        cell_indices = cell_replicas % supercell

        # Calculate phase and combine with coefficient to normalize contributions from replicas
        # that may be represented more than once
        phase = np.exp(-2j * np.pi * np.einsum('a,ia->i', q_point, cell_replicas))
        prefactors = np.einsum('i,inm->inm', phase, coefficients)
        prefactors = prefactors.repeat(9, axis=0).reshape((-1, 3, 3, n_unit_cell, n_unit_cell))
        prefactors = prefactors.transpose((4, 2, 0, 3, 1))

        # Sum over each contribution after multiplying the force at each replica by the phase + coefficient
        dyn_s = prefactors * fc_s[:, :, cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2], :, :]
        dyn_s = np.transpose(dyn_s, axes=(3, 4, 2, 0, 1))
        dyn_s = dyn_s.sum(axis=2)
        dyn_s = dyn_s.reshape((n_unit_cell * 3, n_unit_cell * 3))

        # Apply correction for Born effective charges, if detected
        qstr = '_'.join(np.round(self.q_point, 4).astype(str))
        if self.is_nac:
            np.save('ddyns/c-dyns_{}'.format(qstr), dyn_s)
            np.save('ddyns/c-nacq_{}'.format(qstr), self.nac_dynmat(qpoint=self.q_point))
            np.save('ddyns/c-nacnq_{}'.format(qstr), self.nac_dynmat(qpoint=None))
            np.save('ddyns/c-zdyns_{}'.format(qstr), zheev(dyn_s)[1])
            dyn_s += self.nac_dynmat(qpoint=None)
            dyn_s += self.nac_dynmat(qpoint=self.q_point)
            np.save('ddyns/c-zdyns+nac_{}'.format(qstr), zheev(dyn_s)[1])
        else:
            np.save('ddyns/uc-dyns_{}'.format(qstr), dyn_s)
            np.save('ddyns/uc-zdyns_{}'.format(qstr), zheev(dyn_s)[1])
        # Diagonalize
        if only_eigenvals:
            omega2, eigenvect, info = zheev(dyn_s, compute_v=False)
            frequency = np.sign(omega2) * np.sqrt(np.abs(omega2))
            frequency = frequency[:] / np.pi / 2
            esystem = (frequency[:] * np.pi * 2) ** 2
        else:
            omega2, eigenvect, info = zheev(dyn_s)
            frequency = np.sign(omega2) * np.sqrt(np.abs(omega2))
            frequency = frequency[:] / np.pi / 2
            esystem = np.vstack(((frequency[:] * np.pi * 2) ** 2, eigenvect))
        return esystem

    def calculate_dynmat_derivatives_unfolded(self, direction):
        # This algorithm should be the same as the ShengBTE version
        q_point = self.q_point
        supercell = self.second.supercell
        atoms = self.second.atoms
        cell = atoms.cell
        reciprocal_n = np.round(atoms.cell.reciprocal(), 12)  # round to avoid accumulation of error
        reciprocal_n /= reciprocal_n[0, 0] # Normalized reciprocal cell
        n_unit_cell = len(atoms)
        distances = atoms.positions[:, None, :] - atoms.positions[None, :, :]

        # Get Force constants
        fc_s = self.second.dynmat.numpy()
        fc_s = fc_s.reshape((n_unit_cell, 3, supercell[0], supercell[1], supercell[2], n_unit_cell, 3))
        supercell_positions = self.second.supercell_positions
        supercell_norms = 1 / 2 * np.linalg.norm(supercell_positions, axis=1) ** 2
        cell_replicas = self.second.supercell_replicas
        cell_positions = np.einsum('ia,ab->ib', cell_replicas, cell)
        cell_plus_distance = cell_positions[:, None, None, :] + distances[None, :, :, :]
        supercell_positions = self.second.supercell_positions
        supercell_cell_distances = np.einsum('La,inma->Linm', supercell_positions, cell_plus_distance)
        projection = supercell_cell_distances - supercell_norms[:, None, None, None]

        # Filter + Weights
        mask_distance = (projection <= 1e-6).all(axis=0)
        n_equivalent = (np.abs(projection) <= 1e-6).sum(axis=0)
        weight = 1 / n_equivalent
        coefficients = weight * mask_distance

        # Find contributing replicas
        mask_full = coefficients.any(axis=(-2, -1))
        coefficients = coefficients[mask_full]
        cell_replicas = cell_replicas[mask_full]
        cell_positions = cell_positions[mask_full]
        cell_indices = cell_replicas % supercell

        # Calculate phase and combine with coefficient to normalize contributions from replicas
        # that may be represented more than once
        # NOTE: If you wanted to redo this to calculate all the directions at the same time, the first
        # prefactors line is the only place where direction is used.
        phase = np.exp(-2j * np.pi * np.einsum('a,ia->i', q_point, cell_replicas))
        prefactors = np.einsum('i,i,inm->inm', cell_positions[:, direction], phase, coefficients)
        prefactors = prefactors.repeat(9, axis=0).reshape((-1, 3, 3, n_unit_cell, n_unit_cell))
        prefactors = prefactors.transpose((4, 2, 0, 3, 1))

        # Sum over each contribution after multiplying the force at each replica by the phase + coefficient
        ddyn_s = prefactors * fc_s[:, :, cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2], :, :]
        ddyn_s = np.transpose(ddyn_s, axes=(3, 4, 2, 0, 1))
        ddyn_s = ddyn_s.sum(axis=2)
        ddyn_s = ddyn_s.reshape((n_unit_cell * 3, n_unit_cell * 3))

        # Apply correction for Born effective charges, if detected
        qstr = '_'.join(np.round(self.q_point, 4).astype(str))
        direction_dic = {0: 'x', 1: 'y', 2: 'z'}
        dstr = direction_dic[direction]
        if self.is_nac:
            np.save('ddyns/c-ddyns_{}-{}'.format(qstr, dstr), ddyn_s)
            np.save('ddyns/c-dnac_{}-{}'.format(qstr,dstr), self.nac_derivatives(direction=direction))
            #ddyn_s += 1j * self.nac_derivatives(direction=direction)/10 -- pick up here
            ddyn_s += self.nac_derivatives(direction=direction)
        else:
            np.save('ddyns/nc-ddyns_{}-{}'.format(qstr,dstr), ddyn_s)
        return ddyn_s

    def nac_dynmat(self, qpoint=None, gmax=None, Lambda=None):
        '''
        Calculate the non-analytic correction to the dynamical matrix.

        Parameters
        ----------
        qpoint : (float, float, float)
            Vector in reciprocal space to measure at. If none, the correction is simpler, using only the second half of
            the second if block here.
        gmax : float
            Maximum g-vector to consider
        Lambda : float
            Parameter for Ewald summation. 1/(4*Lambda) is the cutoff for the

        Returns
        -------
        correction_matrix
        '''
        # Constants, and system information
        RyBr_to_eVA = units.Rydberg / (units.Bohr ** 2)  # Rydberg / Bohr^2 to eV/A^2
        eV_to_10Jmol = units.mol / (10 * units.J)
        e2 = 2.  # square of electron charge in A.U.
        atoms = self.second.atoms
        natoms = len(atoms)
        if gmax==None:
            gmax = 14  # maximum reciprocal vector (same default value in ShengBTE/QE)
        if Lambda==None:
            Lambda = 1 # (2*np.pi*units.Bohr/np.linalg.norm(atoms.cell[0,:]))**2
        geg0 = 4 * Lambda * gmax
        omega_bohr = np.linalg.det(atoms.cell.array / units.Bohr) # Vol. in Bohr^3
        positions_n = atoms.positions.copy() / atoms.cell[0, :].max()  # Normalized positions
        distances_n = positions_n[:, None, :] - positions_n[None, :, :]  # distance in crystal coordinates
        reciprocal_n = np.round(np.linalg.pinv(atoms.cell), 12)  # round to avoid accumulation of error
        reciprocal_n /= np.abs(reciprocal_n[0, 0])  # Normalized reciprocal cell
        correction_matrix = tf.zeros([3, 3, natoms, natoms], dtype=tf.complex64)
        prefactor = 4 * np.pi * e2 / omega_bohr

        sqrt_mass = np.sqrt(self.atoms.get_masses().repeat(3, axis=0))
        mass_prefactor = np.reciprocal(np.einsum('i,j->ij', sqrt_mass, sqrt_mass))

        # Charge information
        epsilon = atoms.info['dielectric']  # in e^2/Bohr
        zeff = atoms.get_array('charges')  # in e

        # Charge sum rules
        # Using the "simple" algorithm from QE, we enforce that the sum of
        # charges for each polarization (e.g. xy, or yy) is zero
        zeff -= zeff.mean(axis=0)

        # 1. Construct grid of reciprocal unit cells
        # a. Find the number of replicas to make
        n_greplicas = 2 + 2 * np.sqrt(geg0) / np.linalg.norm(reciprocal_n, axis=0)
        # b. If it's low-dimensional, don't replicate in reciprocal space along axes without replicas in real space
        n_greplicas[np.array(self.second.supercell) == 1] = 1
        # c. Generate the grid of replicas
        g_grid = Grid(n_greplicas.astype(int))
        g_replicas = g_grid.grid(is_wrapping=True)  # minimium distance replicas
        # d. Transform the raw indices, to coordinates in reciprocal space
        g_positions = np.einsum('ib,ab->ia', g_replicas, reciprocal_n)
        if qpoint is not None:  # If we're measuring at finite q, shift the images' positions
            g_positions = g_positions + (qpoint @ reciprocal_n.T)
        g_positions = np.round(g_positions, 6)  # avoids bad behavior at ultra-small q-points TODO: remove this

        # 2. Filter cells that don't meet our Ewald cutoff criteria
        # a. setup mask
        geg = np.einsum('ia,ab,ib->i', g_positions, epsilon, g_positions, dtype=np.float128)
        # change_units_gmax = 16/np.pi**2
        cells_to_include = (geg > 0) * (geg / (4 * Lambda) < gmax)
        # b. apply mask
        geg = geg[cells_to_include]
        g_positions = g_positions[cells_to_include]
        g_replicas = g_replicas[cells_to_include] # for debugging - remove in production

        # 3. Calculate for each cell
        # a. exponential decay term based on distance in reciprocal space, and dielectric tensor
        decay = prefactor * np.exp(-1 * geg / (4 * Lambda)) / geg
        # b. effective charges at each G-vector
        zag = np.einsum('nab,ia->inb', zeff, g_positions)

        # 4. Calculate the actual correction as a product of the effective charges, exponential decay term, and phase factor
        # the phase factor is based on the distance of the G-vector and atomic positions
        # TODO: This "if-else" block could likely be replaced with the just the "if" block since the imaginary term I
        # think should be zero at Gamma, but we'd need to check that for sure.
        if qpoint is not None:
            phase = np.exp(1j * np.pi * np.einsum('ia,nma->inm', g_positions, distances_n))

            # The long range forces are the outer product of the effective charges, scaled by the phase term. We impose
            # Hermicity on cartesian axes by taking the average of M and M^T
            lr_correction = np.einsum('ina,inm,imb->inmab', zag, phase, zag)
            lr_correction += np.transpose(lr_correction, (0, 1, 2, 4, 3))
            lr_correction *= 0.5

            # Scale by exponential decay term, sum over G-vectors
            lr_correction = np.einsum('i,inmab->abnm', decay, lr_correction)

            # Apply the correction to each atom pair
            correction_matrix += lr_correction

        else:  # only the real part of the phase is taken at Gamma
            phase = np.cos(np.pi * np.einsum('ia,nma->inm', g_positions, distances_n))

            # Also, this part of the correction is only applied on "diagonal" choices of atoms. (e.g. 00, 11, 22 etc)
            # The long range forces are an outer product of the effective charges, scaled by the exponential term.
            # We impose Hermicity on cartesian axes by taking the average of M and M^T
            lr_correction = np.einsum('ina,inm,imb->inab', zag, phase, zag)
            lr_correction += np.transpose(lr_correction, (0, 1, 3, 2))
            lr_correction *= 0.5

            # Scale by exponential decay term, sum over G-vectors
            lr_correction = np.einsum('i,inab->abn', decay, lr_correction)

            # Apply the correction to the diagonals of the dynamical matrix
            correction_matrix = tf.linalg.set_diag(correction_matrix,
                                                   tf.linalg.diag_part(correction_matrix) - lr_correction)
        correction_matrix = tf.transpose(correction_matrix, perm=[2, 0, 3, 1])
        correction_matrix = tf.reshape(correction_matrix, shape=(natoms * 3, natoms * 3))
        correction_matrix *= mass_prefactor # 1/sqrt(mass_i * mass_j)
        correction_matrix *= RyBr_to_eVA * eV_to_10Jmol # Rydberg / Bohr^2 to 10J/mol A^2
        return correction_matrix

    def nac_derivatives(self, direction, Lambda=None, gmax=None):
        '''
        Calculate the non-analytic correction to the dynamical matrix.

        qpoint : (float, float, float)
            Vector in reciprocal space to measure at. If none, the correction is simpler, using only the second half of
            the second if block here.
        gmax : float
            Maximum g-vector to consider
        Lambda : float
            Parameter for Ewald summation. 1/(4*Lambda) is the cutoff for the
        Returns
        -------
        correction_matrix
        '''
        # Constants, and system information
        RyBr_to_eVA = units.Rydberg / (units.Bohr ** 2)  # Rydberg / Bohr^2 to eV/A^2
        eV_to_10Jmol = units.mol / (10 * units.J) # eV to 10J/mol
        atoms = self.second.atoms
        natoms = len(atoms)
        cell = atoms.cell
        e2 = 2.  # square of electron charge in A.U.

        # Begin calculated values
        if gmax==None:
            gmax = 14  # maximum reciprocal vector (same default value in ShengBTE/QE)
        if Lambda==None:
            Lambda = (2*np.pi*units.Bohr/np.linalg.norm(cell[0,:]))**2  # Ewald parameter
        geg0 = 4 * Lambda * gmax
        omega_bohr = np.linalg.det(atoms.cell.array / units.Bohr) # Vol. in Bohr^3
        positions_n = atoms.positions.copy() / atoms.cell[0, :].max()  # Normalized positions
        distances_n = positions_n[:, None, :] - positions_n[None, :, :]  # distance in crystal coordinates
        positions_bohr = atoms.positions.copy() / units.Bohr
        distances_bohr = positions_bohr[:, None, :] - positions_bohr[None, :, :]  # distance in Bohr TODO: remove this
        # reciprocal_n = np.round(np.linalg.pinv(atoms.cell), 12)  # round to avoid accumulation of error
        # reciprocal_n /= np.abs(reciprocal_n[0, 0])  # Normalized reciprocal cell
        reciprocal_n = 2 * np.pi * np.linalg.inv(atoms.cell / units.Bohr)   # round to avoid accumulation of error
        prefactor = 4 * np.pi * e2 / omega_bohr

        sqrt_mass = np.sqrt(self.atoms.get_masses().repeat(3, axis=0))
        mass_prefactor = np.reciprocal(np.einsum('i,j->ij', sqrt_mass, sqrt_mass))

        # Charge information
        epsilon = atoms.info['dielectric']  # in e^2/Bohr
        zeff = atoms.get_array('charges')  # in e

        # Charge sum rules
        # Using the "simple" algorithm from QE, we enforce that the sum of
        # charges for each polarization (e.g. xy, or yy) is zero
        zeff -= zeff.mean(axis=0)

        # 1. Construct grid of reciprocal unit cells
        # a. Find the number of replicas to make
        n_greplicas = 2 + 2 * np.sqrt(geg0) / np.linalg.norm(reciprocal_n, axis=0)
        # b. If it's low-dimensional, don't replicate in reciprocal space along axes without replicas in real space
        n_greplicas[np.array(self.second.supercell) == 1] = 1
        # c. Generate the grid of replicas
        g_grid = Grid(n_greplicas.astype(int))
        g_replicas = g_grid.grid(is_wrapping=True)  # minimium distance replicas
        # d. Transform the raw indices, to coordinates in recip/    [--=rocal space
        g_positions = np.einsum('ib,ab->ia', g_replicas, reciprocal_n)
        g_positions = g_positions + (self.q_point @ reciprocal_n.T)
        g_positions = np.round(g_positions, 6) # This removes strange behavior at very small-q but loses accuracy

        # 2. Filter cells that don't meet our Ewald cutoff criteria
        # a. setup mask
        geg = np.einsum('ia,ab,ib->i', g_positions, epsilon, g_positions)
        cells_to_include = (geg > 0) * (geg / (4 * Lambda) < gmax)
        # b. apply mask
        geg = geg[cells_to_include]
        g_positions = g_positions[cells_to_include]
        g_replicas = g_replicas[cells_to_include]  # for debugging - remove in production

        # 3. Calculate for each cell
        # a. exponential decay term based on distance in reciprocal space, and dielectric tensor
        decay = prefactor * np.exp(-1 * geg / (Lambda * 4)) / geg
        shengfac = 1 #np.pi * units.Bohr / np.abs(atoms.cell[0,0]) # factor t convert to sheng's units
        shengfac2 = 1 # shengfac ** 2
        shengdecay = np.exp(-1 * geg / (Lambda * 4), dtype=np.complex256) / (geg * shengfac2) # Sheng's decay factor

        # b. effective charges at each G-vector
        zag = np.einsum('nab,ia->inb', zeff, g_positions)
        shengzag = np.einsum('nab,ia->inb', zeff, g_positions * shengfac)

        # 4. Calculate the actual correction as a product of the effective charges, exponential decay term, and phase factor
        # the phase factor is based on the distance of the G-vector and atomic positions
        phase = np.exp(1j * np.einsum('ia,nma->inm', g_positions, distances_bohr))

        # exactly equivalent by the way
        '''
        # All directions at once code
        # Terms 1 + 2
        zag_zeff = np.einsum('ina,mcb->inmabc', zag, zeff)
        zbg_zeff = np.transpose(zag_zeff, (0, 2, 1, 4, 3, 5))
        # Term 3 (imaginary)
        zag_zbg_rij = 1j * np.einsum('ina,imb,nmc->inmabc', zag, zag, distances_n)
        # Term 4 (negative)
        dgeg = np.einsum('ab,ib->ib', epsilon + epsilon.T, g_positions)
        zag_zbg_dgeg = -1 * np.einsum('ina,imb,ic,i->inmabc', zag, zag, dgeg, (1/(4*Lambda) + 1/geg))

        # Combine terms!
        lr_correction = zag_zeff + zbg_zeff + zag_zbg_rij + zag_zbg_dge

        # Scale by exponential decay term
        lr_correction = np.einsum('i,inm,inmabc->nmabc', decay, phase, lr_correction)
        '''
        # Derivative terms in a single direction
        # Terms 1 + 2
        # Units: Ry/Bohr
        zag_zeff = np.einsum('ina,mb->inmab', zag, zeff[:, direction, :])
        zbg_zeff = np.transpose(zag_zeff, (0, 2, 1, 4, 3))
        shengzag_zeff = np.einsum('ina,mb->inmab', shengzag, zeff[:, direction, :])
        shengzbg_zeff = np.transpose(shengzag_zeff, (0, 2, 1, 4, 3))
        # Term 3 (imaginary)
        zag_zbg_rij = 1j * np.einsum('ina,imb,nm->inmab', zag, zag, distances_bohr[:, :, direction])
        shengzag_zbg_rij = 1j * np.einsum('ina,imb,nm->inmab', shengzag, shengzag, distances_bohr[:, :, direction])
        # Term 4 (negative)
        dgeg = np.einsum('ab,ib->ib', epsilon + epsilon.T, g_positions)[:, direction]
        geg_units = (np.pi * units.Bohr / np.abs(cell[0, 0])) ** 2
        zag_zbg_dgeg = -1 * np.einsum('ina,imb,i,i->inmab', zag, zag, dgeg,\
                                      (1/(4*Lambda) + 1/(geg)))
        # zag_zbg_dgeg /= np.abs(atoms.cell[0, 0])
        shengzag_zbg_dgeg = np.einsum('ina,imb,i,i->inmab', shengzag, shengzag, dgeg*shengfac, (1 / (4 * Lambda) + 1 / (geg*shengfac2)))

        # Combine derivative terms!
        lr_correction = zag_zeff + zbg_zeff + zag_zbg_rij + zag_zbg_dgeg
        sheng_correction = shengzag_zeff + shengzbg_zeff + shengzag_zbg_rij - shengzag_zbg_dgeg

        # Scale by exponential decay and phase terms, sum over G-vectors
        # Note: Einsum does not use the distributive property for complex number mult., so we have to do a second
        # multiplication operation when applying the phase factor.
        # lr_correction = np.einsum('i,inm,inmab->nmab', decay, phase, lr_correction) # old
        lr_correction = np.einsum('i,inmab->inmab', decay, lr_correction) # new
        lr_correction *= phase[:, :, :, None, None]
        lr_correction = lr_correction.sum(axis=0)
        #sheng_correction2 = np.einsum('i,inmab->inmab', shengdecay, sheng_correction)
        #sheng_correction3 = sheng_correction2 * phase[:,:,:,None, None]

        # Rotate, reshape, rescale, and, finally, return correction value
        correction_matrix = np.transpose(lr_correction, axes=(0, 2, 1, 3))
        correction_matrix = np.reshape(correction_matrix, (natoms * 3, natoms * 3))
        correction_matrix *= mass_prefactor # 1/sqrt(mass_i * mass_j)
        correction_matrix *= RyBr_to_eVA * eV_to_10Jmol # Rydberg / Bohr^2 to 10J/mol A^2
        return correction_matrix

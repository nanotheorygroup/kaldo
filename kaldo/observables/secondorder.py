from kaldo.observables.forceconstant import ForceConstant
from ase import Atoms
import os
import tensorflow as tf
import ase.io
import numpy as np
from kaldo.interfaces.eskm_io import import_from_files
from kaldo.grid import Grid
import kaldo.interfaces.shengbte_io as shengbte_io
from kaldo.controllers.displacement import calculate_second
import ase.units as units
from kaldo.helpers.logger import get_logger, log_size

logging = get_logger()

SECOND_ORDER_FILE = 'second.npy'


def acoustic_sum_rule(dynmat):
    n_unit = dynmat[0].shape[0]
    sumrulecorr = 0.
    for i in range(n_unit):
        off_diag_sum = np.sum(dynmat[0, i, :, :, :, :], axis=(-2, -3))
        dynmat[0, i, :, 0, i, :] -= off_diag_sum
        sumrulecorr += np.sum(off_diag_sum)
    logging.info('error sum rule: ' + str(sumrulecorr))
    return dynmat


class SecondOrder(ForceConstant):
    def __init__(self, *kargs, **kwargs):
        ForceConstant.__init__(self, *kargs, **kwargs)
        try:
            self.is_acoustic_sum = kwargs['is_acoustic_sum']
        except KeyError:
            self.is_acoustic_sum = False

        self.value = kwargs['value']
        if self.is_acoustic_sum:
            self.value = acoustic_sum_rule(self.value)
        self.n_modes = self.atoms.positions.shape[0] * 3
        self._list_of_replicas = None
        self.storage = 'numpy'
        self.gmax = kwargs.pop('gmax', 14.)

    @classmethod
    def from_supercell(cls, atoms, grid_type, supercell=None, value=None, is_acoustic_sum=False, folder='kALDo'):
        if value is not None and is_acoustic_sum is not None:
            value = acoustic_sum_rule(value)
        ifc = super(SecondOrder, cls).from_supercell(atoms, supercell, grid_type, value, folder)
        return ifc

    @classmethod
    def load(cls, folder, supercell=(1, 1, 1), format='numpy', is_acoustic_sum=False):
        if format == 'numpy':
            if folder[-1] != '/':
                folder = folder + '/'
            replicated_atoms_file = 'replicated_atoms.xyz'
            config_file = folder + replicated_atoms_file
            replicated_atoms = ase.io.read(config_file, format='extxyz')

            n_replicas = np.prod(supercell)
            n_total_atoms = replicated_atoms.positions.shape[0]
            n_unit_atoms = int(n_total_atoms / n_replicas)
            unit_symbols = []
            unit_positions = []
            for i in range(n_unit_atoms):
                unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                unit_positions.append(replicated_atoms.positions[i])
            unit_cell = replicated_atoms.cell / supercell

            atoms = Atoms(unit_symbols,
                          positions=unit_positions,
                          cell=unit_cell,
                          pbc=[1, 1, 1])

            _second_order = np.load(folder + SECOND_ORDER_FILE, allow_pickle=True)
            second_order = SecondOrder(atoms=atoms,
                                       replicated_positions=replicated_atoms.positions,
                                       supercell=supercell,
                                       value=_second_order,
                                       is_acoustic_sum=is_acoustic_sum,
                                       folder=folder)

        elif format == 'eskm' or format == 'lammps':
            dynmat_file = str(folder) + "/Dyn.form"
            if format == 'eskm':
                config_file = str(folder) + "/CONFIG"
                replicated_atoms = ase.io.read(config_file, format='dlp4')
            elif format == 'lammps':
                config_file = str(folder) + "/replicated_atoms.xyz"
                replicated_atoms = ase.io.read(config_file, format='extxyz')
            n_replicas = np.prod(supercell)
            n_total_atoms = replicated_atoms.positions.shape[0]
            n_unit_atoms = int(n_total_atoms / n_replicas)
            unit_symbols = []
            unit_positions = []
            for i in range(n_unit_atoms):
                unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                unit_positions.append(replicated_atoms.positions[i])
            unit_cell = replicated_atoms.cell / supercell

            atoms = Atoms(unit_symbols,
                          positions=unit_positions,
                          cell=unit_cell,
                          pbc=[1, 1, 1])

            _second_order, _ = import_from_files(replicated_atoms=replicated_atoms,
                                                 dynmat_file=dynmat_file,
                                                 supercell=supercell)
            second_order = SecondOrder(atoms=atoms,
                                       replicated_positions=replicated_atoms.positions,
                                       supercell=supercell,
                                       value=_second_order,
                                       is_acoustic_sum=is_acoustic_sum,
                                       folder=folder)
        elif format == 'shengbte' or format == 'shengbte-qe':

            config_file = folder + '/' + 'CONTROL'
            try:
                atoms, supercell = shengbte_io.import_control_file(config_file)
            except FileNotFoundError as err:
                config_file = folder + '/' + 'POSCAR'
                logging.info('\nTrying to open POSCAR')
                atoms = ase.io.read(config_file)

            # Create a finite difference object
            # TODO: we need to read the grid type here
            is_qe_input = (format == 'shengbte-qe')
            n_replicas = np.prod(supercell)
            n_unit_atoms = atoms.positions.shape[0]
            if is_qe_input:
                filename = folder + '/espresso.ifc2'
                second_order, supercell, charges = shengbte_io.read_second_order_qe_matrix(filename)
                atoms.info['dielectric'] = charges[0, :, :]
                atoms.set_array('charges', charges[1:, :, :], shape=(3, 3))
                second_order = second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
                second_order = second_order.transpose(3, 4, 2, 0, 1)
                grid_type = 'F'
            else:
                second_order = shengbte_io.read_second_order_matrix(folder, supercell)
                second_order = second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
                grid_type = 'C'
            second_order = SecondOrder.from_supercell(atoms=atoms,
                                                      grid_type=grid_type,
                                                      supercell=supercell,
                                                      value=second_order[np.newaxis, ...],
                                                      is_acoustic_sum=True,
                                                      folder=folder)

        elif format == 'hiphive':
            filename = 'atom_prim.xyz'
            # TODO: add replicated filename in example
            replicated_filename = 'replicated_atoms.xyz'
            try:
                import kaldo.interfaces.hiphive_io as hiphive_io
            except ImportError:
                logging.error('In order to use hiphive along with kaldo, hiphive is required. \
                      Please consider installing hihphive. More info can be found at: \
                      https://hiphive.materialsmodeling.org/')

            atom_prime_file = str(folder) + '/' + filename
            replicated_atom_prime_file = str(folder) + '/' + replicated_filename
            # TODO: Make this independent of replicated file
            atoms = ase.io.read(atom_prime_file)
            try:
                replicated_atoms = ase.io.read(replicated_atom_prime_file)
            except FileNotFoundError:
                logging.warning('Replicated atoms file not found. Please check if the file exists. Using the unit cell atoms instead.')
                replicated_atoms = atoms * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])
            # Create a finite difference object
            if 'model2.fcs' in os.listdir(str(folder)):
                _second_order = hiphive_io.import_second_from_hiphive(folder, np.prod(supercell),
                                                                      atoms.positions.shape[0])
                second_order = SecondOrder(atoms=atoms,
                                           replicated_positions=replicated_atoms.positions,
                                           supercell=supercell,
                                           value=_second_order,
                                           folder=folder)

        # Newly added by me!!!!
        elif format == 'sscha':
            filename = 'atom_prim.xyz'
            replicated_filename = 'replicated_atoms.xyz'
            try:
                from hiphive import ForceConstants as HFC
            except ImportError:
                logging.error('In order to use hiphive along with kaldo, hiphive is required. \
                      Please consider installing hihphive. More info can be found at: \
                      https://hiphive.materialsmodeling.org/')
                return None
            atom_prime_file = str(folder) + '/' + filename
            replicated_atom_prime_file = str(folder) + '/' + replicated_filename
            atoms = ase.io.read(atom_prime_file)
            replicated_atoms = ase.io.read(replicated_atom_prime_file)
            if 'second.npy' in os.listdir(str(folder)):
                second_hiphive_file = str(folder) + '/second.npy'
                fcs2 = HFC.from_arrays(supercell=supercell,fc2_array=np.load(second_hiphive_file))
                n_replicas = np.prod(supercell)
                n_atoms = atoms.positions.shape[0]
                _second_order = fcs2.get_fc_array(2).transpose(0, 2, 1, 3)
                _second_order = _second_order.reshape((n_replicas, n_atoms, 3,
                                                      n_replicas, n_atoms, 3))
                _second_order = _second_order[0, np.newaxis]
                second_order = SecondOrder(atoms=atoms,
                                          replicated_positions=replicated_positions,
                                          supercell=supercell,
                                          value=_second_order,
                                          folder=folder)


        else:
            raise ValueError
        return second_order

    @property
    def supercell_replicas(self):
        try:
            return self._supercell_replicas
        except AttributeError:
            self._supercell_replicas = self.calculate_super_replicas()
            return self._supercell_replicas

    @property
    def supercell_positions(self):
        try:
            return self._supercell_positions
        except AttributeError:
            self._supercell_positions = self.calculate_supercell_positions()
            return self._supercell_positions

    @property
    def dynmat(self):
        try:
            return self._dynmat
        except AttributeError:
            self._dynmat = self.calculate_dynmat()
            return self._dynmat

    def calculate(self, calculator, delta_shift=1e-3, is_storing=True, is_verbose=False):
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        atoms.set_calculator(calculator)
        replicated_atoms.set_calculator(calculator)

        if is_storing:
            try:
                self.value = SecondOrder.load(folder=self.folder, supercell=self.supercell, format='numpy',
                                              is_acoustic_sum=self.is_acoustic_sum).value

            except FileNotFoundError:
                logging.info('Second order not found. Calculating.')
                self.value = calculate_second(atoms, replicated_atoms, delta_shift, is_verbose)
                self.save('second')
                ase.io.write(self.folder + '/replicated_atoms.xyz', self.replicated_atoms, 'extxyz')
            else:
                logging.info('Reading stored second')
        else:
            self.value = calculate_second(atoms, replicated_atoms, delta_shift, is_verbose)
        if self.is_acoustic_sum:
            self.value = acoustic_sum_rule(self.value)

    def calculate_dynmat(self):
        mass = self.atoms.get_masses()
        shape = self.value.shape
        log_size(shape, float, name='dynmat')
        dynmat = self.value * 1 / np.sqrt(mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        dynmat = dynmat * 1 / np.sqrt(mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])
        evtotenjovermol = units.mol / (10 * units.J)
        return tf.convert_to_tensor(dynmat * evtotenjovermol)

    def calculate_super_replicas(self):
        scell = self.supercell
        n_replicas = np.prod(scell)
        atoms = self.atoms
        cell = atoms.cell
        n_unit_cell = atoms.positions.shape[0]
        replicated_positions = self.replicated_atoms.positions.reshape((n_replicas, n_unit_cell, 3))

        list_of_index = np.round((replicated_positions - self.atoms.positions).dot(
            np.linalg.inv(atoms.cell))).astype(int)
        list_of_index = list_of_index[:, 0, :]

        tt = []
        rreplica = []
        for ix2 in [-1, 0, 1]:
            for iy2 in [-1, 0, 1]:
                for iz2 in [-1, 0, 1]:
                    for f in range(list_of_index.shape[0]):
                        scell_id = np.array([ix2 * scell[0], iy2 * scell[1], iz2 * scell[2]])
                        replica_id = list_of_index[f]
                        t = replica_id + scell_id
                        replica_position = np.tensordot(t, cell, (-1, 0))
                        tt.append(t)
                        rreplica.append(replica_position)

        tt = np.array(tt)
        return tt

    def calculate_supercell_positions(self):
        supercell = self.supercell
        atoms = self.atoms
        cell = atoms.cell
        replicated_cell = cell * supercell
        sc_r_pos = np.zeros((3 ** 3, 3))
        ir = 0
        for ix2 in [-1, 0, 1]:
            for iy2 in [-1, 0, 1]:
                for iz2 in [-1, 0, 1]:
                    for i in np.arange(3):
                        sc_r_pos[ir, i] = np.dot(replicated_cell[:, i], np.array([ix2, iy2, iz2]))
                    ir = ir + 1
        return sc_r_pos

    def calculate_nonanalytical_corrections(self, kpoints=None):
        '''    Calculates the nonanalytical contributions to the dynamical matrix according to
        the forumla developed by X. Gonze et. al. PRB 50. 13035 (1994).
        The implementation here is a pythonic version of the loops in rigid.f90 found
        in the Quantum Espresso source code. (/<your QE home>/PHonon/PH/rigid.f90)

        Parameters
        -------
        BEC (optional) : np.array(n_unit, 3, 3, )
            The 2-dimensional charge tensor on every atom

        Returns
        -------
        dyn_g : np.array(n_unit, 3, 3, )
            The ionic contribution to the dynamical matrix in
            10 J / mol based on the Born effective charges found on the
            atoms. Alternatively, users can specify arbitrary charges.
        '''
        atoms = self.atoms
        natoms = len(atoms)

        # Prepare cell and position data
        cell_a = atoms.cell.copy()  # in Angstrom
        positions_n = atoms.positions.copy() / cell_a[0, 0]  # normalized
        distances_n = positions_n[:, None, :] - positions_n[None, :, :]  # normalized
        reciprocal = cell_a.reciprocal()  # "gcell" in qe
        reciprocal_n = reciprocal / reciprocal[0, 0]  # normalized!
        omega_bohr = np.linalg.det(cell_a.array / units.Bohr)  # in Bohr^3

        # Constants in Atomic Units
        e2 = 2  # electron charge squared (atomic units)
        gmax = 14  # maximum reciprocal vector
        alpha = 1.0  # Ewald parameter
        geg0 = 4 * alpha * gmax
        prefactor = 4 * np.pi * e2 / omega_bohr  # "fac" in rigid.f90

        # Charge information
        epsilon = atoms.info['dielectric']
        zeff = atoms.get_array('charges')  # in e

        # Construct grid of reciprocal unit cells
        # Find the number of replicas to make
        n_greplicas = np.sqrt(geg0) / np.linalg.norm(reciprocal_n, axis=1) + 1
        # Generate the grid of replicas
        g_grid = Grid(n_greplicas.astype(int) * 2)
        g_replicas = g_grid.grid(is_wrapping=True)
        # Transform the raw indices, to coordinates in reciprocal space
        g_positions = np.einsum('ij,jk->ik', g_replicas, reciprocal_n, dtype=np.float128)
        # Calculate GEG (outer product scaled by epsilon)
        # g_supercell_norms = np.linalg.norm(g_positions, axis=1)
        geg = np.einsum('ij,jl,il->i', g_positions, epsilon, g_positions, dtype=np.float128)
        cells_to_include = (geg > 0) * (geg / (4 * alpha) < gmax)

        # Filter
        # g_replicas_gamma = g_replicas[cells_to_include]
        geg_gamma = geg[cells_to_include]
        g_positions_gamma = g_positions[cells_to_include]

        # Exponential factor for the replicas
        facgd = prefactor * np.exp(-1 * geg_gamma / (alpha * 4)) / geg_gamma
        # Effective charge at each site
        zag = np.einsum('abi,jb->jai', zeff, g_positions_gamma)
        # Real part of the exponential distance factor (sine neglected since they're all real at gamma)
        cosine = np.cos(2 * np.pi * np.einsum('ja,nma->jnm', g_positions_gamma, distances_n))

        # Symmetrize the long range correction (Imposing Hermicity) which is the outer product of the effective charges
        # scaled by the cosine term. Then take the average of M and M^T
        long_range_vector = np.einsum('ijk,ika->ija', cosine, zag)
        long_range_correction = np.einsum('ijk,ijl->ijkl', long_range_vector, zag) / 2
        long_range_correction += np.transpose(long_range_correction, (0, 1, 3, 2))

        # Apply exponential factor and sum over G-vectors
        # Arange axes of correction tensor to be compatible with the diagonals of the dynamical matrix
        long_range_correction = np.einsum('i,ijkl->klj', facgd, long_range_correction)

        # Fake dynamical matrix
        dynamical_matrix_nac = tf.zeros([3, 3, natoms, natoms], dtype=tf.complex64)
        dynamical_matrix_nac = tf.linalg.set_diag(dynamical_matrix_nac,
                                              tf.linalg.diag_part(dynamical_matrix_nac) - long_range_correction)

        # Measure geg at q-points
        scaled_kpoints = kpoints @ reciprocal_n
        g_plus_k = g_positions[:, None, :] + scaled_kpoints[None, :, :]
        g_plus_k = g_plus_k.reshape(-1, 3)  # List of G+K vectors
        geg = np.einsum('ij,jl,il->i', g_plus_k, epsilon, g_plus_k)
        cells_to_include = (geg > 0) * (geg / (4 * alpha) < gmax)

        # Filter G+K vectors
        geg = geg[cells_to_include]
        g_plus_k = g_plus_k[cells_to_include]

        # Prefactors for the GEG
        facgd = prefactor * np.exp(-1 * geg / (alpha * 4)) / geg
        zag = np.einsum('abi,jb->jai', zeff, g_plus_k)
        exponential = np.exp(-1j * 2 * np.pi * np.einsum('ja,nma->jnm', g_plus_k, distances_n))

        # Perform outer product on the effective charges at each G vector. Sum over the G-vectors (summing the first index)
        # to get long range correction matrices for each atom pair. Finally, scale it by the exponential prefactor
        long_range_correction = np.einsum('ija,ijk,ijb->ijkab', zag, exponential, zag)
        long_range_correction = np.einsum('i,ijkab->abjk', facgd, long_range_correction)

        dynamical_matrix_nac += long_range_correction
        return dynamical_matrix_nac

'''
    def calculate_nonanalytical_corrections_Sheng(self, k_points, bec=None):
        Calculates the nonanalytical contributions to the dynamical matrix according to
        the forumla developed by X. Gonze et. al. PRB 50. 13035 (1994).
        The implementation here is a pythonic version of the loops in rigid.f90 found
        in the Quantum Espresso source code. (/<your QE home>/PHonon/PH/rigid.f90)

        Parameters
        -------
        BEC (optional) : np.array(n_unit, 3, 3, )
            The 2-dimensional charge tensor on every atom

        Returns
        -------
        dyn_g : np.array(n_unit, 3, 3, )
            The ionic contribution to the dynamical matrix in
            10 J / mol based on the Born effective charges found on the
            atoms. Alternatively, users can specify arbitrary charges.

        
        e2 = 2.
        gmax = self.gmax
        atoms = self.atoms
        try:
            epsilon = atoms.info['dielectric']
        except KeyError:
            return np.zeros()
        positions = atoms.positions / units.Bohr
        epsilon = atoms.info['dielectric']

        zeff = atoms.get_array('charges')
        cell = atoms.cell.copy()
        distances = positions[:, None, :] - positions[None, :, :]
        prefactor = 4 * np.pi * e2 / np.linalg.det(cell)

        # Convert cell to bohr, get scaled reciprocal cell
        cell.array = cell.array / units.Bohr
        reciprocal = cell.reciprocal()
        rV = 2 * np.pi / np.abs(cell[:, 0] @ reciprocal[:, 0])
        gcell = reciprocal * rV

        # Ewald Parameter
        alpha = ( 2 * np.pi / np.linalg.norm(cell[0, :]) ) ** 2

        # Construct grid of reciprocal unit cells
        geg_0 = gmax * 4 * alpha
        g_replicas = np.sqrt(geg_0) / np.linalg.norm(gcell, axis=1)+1
        g_grid = Grid( g_replicas.astype(int) * 2)
        g_supercell_replicas = g_grid.grid(is_wrapping=True)
        g_supercell_positions = g_supercell_replicas @ gcell

        # geg = outerproduct of g-vectors * epsilon (aka g.T @ eps @ g )
        gxg = g_supercell_positions[:,:,None] @ g_supercell_positions[:,None]
        gxg *= epsilon
        geg = gxg.sum(axis=(-2, -1)) / (4 * alpha)

        # Ignore gcells that don't meet criteria
        indices = (geg>0) & (geg<gmax)
        geg = geg[indices]
        g_supercell_replicas = g_supercell_replicas[indices]
        g_supercell_positions = g_supercell_positions[indices]

        # Calculate contributions
        # Shapes:
        # exp_g = (Ng, )
        # gdotZ = (Na, Ng, 3, ) -- transpose --> (Ng, 3, Na, )
        # grij = (Ng, Na, Na, )
        # expgrij = (Ng, Na, Na, )
        exp_g = np.exp(-1 * geg) / geg
        gdotZ = g_supercell_positions @ zeff
        gdotZ = np.transpose(gdotZ, axes=(1, 2, 0))
        gdotZt = np.transpose(gdotZ, axes=(0, 2, 1))
        grij = g_supercell_positions[:, None, None, None, :] @ distances[None, :, :, :, None]
        expgrij = np.exp(-1j * grij.squeeze())

        # Matrix multiplication explanation
        # gdotZ @ egrij (Ng, 3, Na) @ (Ng, Na, Na)
        # This is treated like Ng stacks of (3, jat) @ (jat, iat) -> (3, iat)
        coeffs = exp_g[:, None, None] * (gdotZ @ egrij)
        coeffs = np.transpose(coeffs, axes=(0, 2, 1)) # (g, iat, 3, )

        # Final Summation
        # dyn_g = (Ng, Na, alpha, 1) @ (ng, na, 1, beta)
        #                   -> (ng, na, alpha, beta)
        dyn_g = -1 * np.sum( gdotZt[:, :, :, None] @ coeffs[:, :, None, :], axis=0, ) # sum over gvecs

        for iat in range(self.atoms.positions.shape[0]):
            self.dynmat[:, iat, :, 0, iat, :] += np.sum(dyn_g[:, iat, :, :], axis=0)[None, :, :, :]
        return dyn_g

        # over k-vectors
        # eTe = (beta, alpha) + (alpha, beta, )
        # eTedotg = (Ng, alpha, ) . (alpha, beta, ) -> (Ng, alpha, )
        # egkrij = (Nk, Ng, Na, Na, )
        #k_grid =
        gplusk = g_supercell_positions + kvec
        gkxgk = gplusk[:, :, None] @ gplusk[:, None]
        gkegk = (gkxgk * epsilon).sum(axis=(-2, -1))
        exp_gk = np.exp(-1 * gkegk)
        gkdotZ = gplusk @ zeff
        gkdotZ = np.transpose(gkdotZ, axes=(1, 0, 2))
        eTedotg = gplusk @ (epsilon + epsilon.T)
        gkrij = gplusk[:, None, None, None, :] @ distances[None, :, :, :, None]
        egkrij = np.exp(1j * gkrij.squeeze())

        # Final Summation
        # (Nk, Ng, ) * (Nk, Ng, Na, 3, None) @ (Nk, Ng, Na, None, 3,)
        #       ----> (Nk, Ng, Na,  Na,  3,     3,  )
        #       ----> (k,  g,  iat, jat, alpha, beta)
        dyn_gk = exp_gk[:, None, None, None, None] *\
                 ( gkdotZ[:, :, None, :, None] @ gkdotZ[:, None, :, None, :] ) *\
                 egkrij[:, :, :, None, None]
        dyn_gk = dyn_gk.sum(axis=0) # (Na, Na, 3, 3, )

        ddyn_gk = exp_gk[:, None, None, None, None, None] * \
                  ( gkdotZ )
        # for gvec in gcells:
        #     geg = gvec.T @ epsilon @ gvec
        #     geg_normalized = geg/(4 * alpha)
        #     if geg>0 and geg_normalized < gmax:
        #         expg = np.exp( -geg_normalized ) / geg
        #         for na in range(nunit):
        #             zig = gvec @ zeff[na, :, :]
        #             auxi = np.zeros(3)
        #             for nb in range(nunit):
        #                 gr = gvec @ distances[na, nb, :]
        #                 zjg = gvec @ zeff[nb, :, :]
        #                 auxi += zjg * np.exp(gr)
                    # for alpha in range(3):
                    #     for beta in range(3):
                    #         dyn_g(:, alpha, beta) -= expg * zig(alpha) * auxi(beta)
'''
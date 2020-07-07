from kaldo.forceconstant import ForceConstant
from opt_einsum import contract
from ase import Atoms
import os
import ase.io
import numpy as np
from kaldo.interface.eskm_io import import_from_files
import kaldo.interface.shengbte_io as shengbte_io
import ase.units as units
from kaldo.helpers.logger import get_logger
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
    def __init__(self, atoms, replicated_positions, supercell=None, force_constant=None, is_acoustic_sum=False):
        if is_acoustic_sum:
            force_constant = acoustic_sum_rule(force_constant)
        ForceConstant.__init__(self, atoms, replicated_positions, supercell, force_constant)
        self._list_of_replicas = None


    @classmethod
    def from_supercell(cls, atoms, grid_type, supercell=None, force_constant=None, is_acoustic_sum=False):
        if force_constant is not None and is_acoustic_sum is not None:
            force_constant = acoustic_sum_rule(force_constant)
        ifc = super(SecondOrder, cls).from_supercell(atoms, supercell, grid_type, force_constant)
        return ifc


    def dynmat(self, mass):
        dynmat = self.value
        dynmat = contract('mialjb,i,j->mialjb', dynmat, 1 / np.sqrt(mass), 1 / np.sqrt(mass))
        evtotenjovermol = units.mol / (10 * units.J)
        return dynmat * evtotenjovermol


    def get_band_structure(self, path_kc, modes=False):
        atoms = self.atoms
        n_atoms = atoms.positions.shape[0]
        supercell = self.supercell
        C_N = self.value.reshape((3 * n_atoms, np.prod(self.supercell), 3 * n_atoms)).swapaxes(1, 0)

        # Displace all atoms in the unit cell by default
        indices = np.arange(len(atoms))
        N_c = supercell

        D_N = C_N.copy()

        # Add mass prefactor
        m_a = atoms.get_masses()
        m_inv_x = np.repeat(m_a[indices] ** -0.5, 3)
        M_inv = np.outer(m_inv_x, m_inv_x)
        for D in D_N:
            D *= M_inv
        # Lattice vectors relevative to the reference cell
        R_cN = np.indices(N_c).reshape(3, -1)
        N_c = np.array(N_c)[:, np.newaxis]
        R_cN += N_c // 2
        R_cN %= N_c
        R_cN -= N_c // 2
        lattice_vectors = R_cN

        # Lattice vectors -- ordered as illustrated in class docstring
        R_cN = lattice_vectors

        # Dynamical matrix in real-space

        # Lists for frequencies and modes along path
        omega_kl = []
        u_kl = []

        for q_c in path_kc:

            # Evaluate fourier sum
            phase_N = np.exp(-2.j * np.pi * np.dot(q_c, R_cN))
            D_q = np.sum(phase_N[:, np.newaxis, np.newaxis] * D_N, axis=0)

            if modes:
                omega2_l, u_xl = np.linalg.eigh(D_q, UPLO='U')
                # Sort eigenmodes according to eigenvalues (see below) and
                # multiply with mass prefactor
                u_lx = (m_inv_x[:, np.newaxis] *
                        u_xl[:, omega2_l.argsort()]).T.copy()
                u_kl.append(u_lx.reshape((-1, len(indices), 3)))
            else:
                omega2_l = np.linalg.eigvalsh(D_q, UPLO='U')

            # Sort eigenvalues in increasing order
            omega2_l.sort()
            # Use dtype=complex to handle negative eigenvalues
            omega_l = np.sqrt(omega2_l.astype(complex))

            # Take care of imaginary frequencies
            if not np.all(omega2_l >= 0.):
                indices = np.where(omega2_l < 0)[0]

                omega_l[indices] = -1 * np.sqrt(np.abs(omega2_l[indices].real))

            omega_kl.append(omega_l.real)

        # Conversion factor: sqrt(eV / Ang^2 / amu) -> eV
        s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
        omega_kl = s * np.asarray(omega_kl)

        if modes:
            return omega_kl, np.asarray(u_kl)

        return omega_kl

    @classmethod
    def load(cls, folder, supercell=(1, 1, 1), format='eskm', is_acoustic_sum=False):
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

            _second_order = np.load(folder + SECOND_ORDER_FILE)
            second_order = SecondOrder(atoms, replicated_atoms.positions, supercell, _second_order,
                                                         is_acoustic_sum=is_acoustic_sum)

        elif format == 'eskm':
            config_file = str(folder) + "/CONFIG"
            dynmat_file = str(folder) + "/Dyn.form"

            replicated_atoms = ase.io.read(config_file, format='dlp4')
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
            second_order = SecondOrder(atoms, replicated_atoms.positions, supercell, _second_order,
                                                         is_acoustic_sum=is_acoustic_sum)
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
                second_order, supercell = shengbte_io.read_second_order_qe_matrix(filename)
                second_order = second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
                second_order = second_order.transpose(3, 4, 2, 0, 1)
                grid_type = 'F'
            else:
                second_order = shengbte_io.read_second_order_matrix(folder, supercell)
                second_order = second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
                grid_type = 'C'
            second_order = SecondOrder.from_supercell(atoms,
                                                                        grid_type=grid_type,
                                                                        supercell=supercell,
                                                                        force_constant=second_order[np.newaxis, ...],
                                                                        is_acoustic_sum=True)



        elif format == 'hiphive':
            filename = 'atom_prim.xyz'
            # TODO: add replicated filename in example
            replicated_filename = 'replicated_atoms.xyz'
            try:
                import kaldo.interface.hiphive_io as hiphive_io
            except ImportError:
                logging.error('In order to use hiphive along with kaldo, hiphive is required. \
                      Please consider installing hihphive. More info can be found at: \
                      https://hiphive.materialsmodeling.org/')

            atom_prime_file = str(folder) + '/' + filename
            replicated_atom_prime_file = str(folder) + '/' + replicated_filename
            # TODO: Make this independent of replicated file
            atoms = ase.io.read(atom_prime_file)
            replicated_atoms = ase.io.read(replicated_atom_prime_file)

            # Create a finite difference object
            if 'model2.fcs' in os.listdir(str(folder)):
                _second_order = hiphive_io.import_second_from_hiphive(folder, np.prod(supercell),
                                                                      atoms.positions.shape[0])
                second_order = SecondOrder(atoms, replicated_atoms.positions,
                                                             supercell,
                                                             _second_order)


        else:
            raise ValueError
        return second_order



    def __str__(self):
        return 'second'
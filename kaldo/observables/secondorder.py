from kaldo.observables.forceconstant import ForceConstant
from ase import Atoms
import os
import tensorflow as tf
import ase.io
import numpy as np
from kaldo.interfaces.eskm_io import import_from_files
import kaldo.interfaces.shengbte_io as shengbte_io
from kaldo.controllers.displacement import calculate_second
import ase.units as units
from kaldo.helpers.logger import get_logger, log_size
from pathlib import Path # NEW !
from ase.geometry import get_distances # NEW !
import sys # NEW !

logging = get_logger()

SECOND_ORDER_FILE = 'second.npy'


def parse_tdep_forceconstant(
        fc_file="infile.forceconstants",
        primitive="infile.ucposcar",
        supercell="infile.ssposcar",
        fortran=True,
        two_dim=True,
        symmetrize=False,
        reduce_fc=True,
        eps=1e-13,
        tol=1e-5,
        format="vasp", ):
    """Parse the the TDEP force constants into the phonopy format

    Returns:
        Force constants in (3*N_sc, 3*N_sc) shape
    """
    if isinstance(primitive, Atoms):
        uc = primitive
    elif Path(primitive).exists():
        uc = ase.io.read(primitive, format=format)
    else:
        raise RuntimeError("primitive cell missing")

    if isinstance(supercell, Atoms):
        sc = supercell
    elif Path(supercell).exists():
        sc = ase.io.read(supercell, format=format)
    else:
        raise RuntimeError("supercell missing")

    uc.wrap(eps=tol)
    sc.wrap(eps=tol)
    n_uc = len(uc)
    n_sc = len(sc)

    force_constants = np.zeros((len(uc), len(sc), 3, 3))

    with open(fc_file) as fo:
        n_atoms = int(next(fo).split()[0])
        cutoff = float(next(fo).split()[0])

        # make sure n_atoms coincides with number of atoms in supercell
        assert n_atoms == n_uc, f"n_atoms == {n_atoms}, should be {n_uc}"

        print(f".. Number of atoms:   {n_atoms}")
        print(rf".. Real space cutoff: {cutoff:.3f} \AA")

        for i1 in range(n_atoms):
            n_neighbors = int(next(fo).split()[0])
            for _ in range(n_neighbors):
                # neighbour index
                i2 = int(next(fo).split()[0]) - 1
                # lattice vector
                lp = np.array(next(fo).split(), dtype=float)
                phi = np.array([next(fo).split() for _ in range(3)], dtype=float)
                r_target = uc.positions[i2] + np.dot(lp, uc.cell[:])
                for ii, r1 in enumerate(sc.positions):
                    r_diff = np.abs(r_target - r1)
                    sc_temp = sc.get_cell(complete=True)
                    r_diff = np.linalg.solve(sc_temp.T, r_diff.T).T % 1.0
                    r_diff -= np.floor(r_diff + eps)
                    if np.sum(r_diff) < tol:
                        force_constants[i1, ii, :, :] += phi

    if not reduce_fc or two_dim:
        force_constants = remap_force_constants(
            force_constants, uc, sc, symmetrize=symmetrize
        )

    if two_dim:
        return force_constants.swapaxes(2, 1).reshape(2 * (3 * n_sc,))

    return force_constants

def progressbar(it, prefix="progress", size=35, file=sys.stdout, len_it=None, n_bars=200, verbose=True,):
    """a simple progress bar to decorate an iterator

    Args:
        it (iterator): show progressbar for this iterator
        prefix (str): prefix for the progressbar
        size (int): size of the progress bar
        file (file): file to write to
        start_count (int): length of iterable
        n_bars (int): show this many bars
        verbose (bool): show the bar
    """
    count = len_it or max(1, len(it))
    n = len(str(count)) + 1

    def show(jj):
        """show the progressbar"""
        x = int(size * jj / count)
        counter = "{:{}d}/{}".format(jj, n, count)
        bar = "{:17s} |{}{}| {}\r".format(
            f"[{prefix}]", "|" * x, " " * (size - x), counter
        )
        if verbose:
            file.write(bar)
            file.flush()

    show(0)

    divider = max(1, count // n_bars)

    ii = 0
    for ii, item in enumerate(it):
        yield item
        if not ii % divider:
            if hasattr(file, "isatty") and file.isatty():
                show(ii)

    show(ii + 1)
    if hasattr(file, "isatty") and file.isatty():
        if verbose:
            file.write("\n")
            file.flush()

def remap_force_constants(
        force_constants,
        primitive,
        supercell,
        new_supercell=None,
        reduce_fc=False,
        two_dim=False,
        symmetrize=True,
        tol=1e-5,
        eps=1e-13,):
    """remap force constants [N_prim, N_sc, 3, 3] to [N_sc, N_sc, 3, 3]

    Parameters
    ----------
    force_constants: np.ndarray
        force constants in [N_prim, N_sc, 3, 3] shape
    primitive: ase.atoms.Atoms
        primitive cell for reference
    supercell: ase.atoms.Atoms
        supercell for reference
    new_supercell: ase.atoms.Atoms, optional
        supercell to map to (default)
    reduce_fc: bool
        return in [N_prim, N_sc, 3, 3]  shape
    two_dim: bool
        return in [3*N_sc, 3*N_sc] shape
    symmetrize: bool
        make force constants symmetric
    tol: float
        tolerance to discern pairs
    eps: float
        finite zero
    """

    if new_supercell is None:
        new_supercell = supercell.copy()

    # find positions of primitive cell within the (reference) supercell
    primitive_cell = primitive.cell.copy()
    primitive.cell = supercell.cell

    primitive.wrap(eps=tol)
    supercell.wrap(eps=tol)

    n_sc_new = len(new_supercell)

    # make a list of all pairs for each atom in primitive cell
    sc_r = np.zeros((force_constants.shape[0], force_constants.shape[1], 3))
    for aa, a1 in enumerate(primitive):
        diff = supercell.positions - a1.position
        p2s = np.where(np.linalg.norm(diff, axis=1) < tol)[0][0]
        # sc_r[aa] = supercell.get_distances(p2s, range(n_sc), mic=True, vector=True)
        # replace with post 3.18 ase routine:
        spos = supercell.positions
        sc_r[aa], _ = get_distances([spos[p2s]], spos, cell=supercell.cell, pbc=True)

    # find mapping from supercell to origin atom in primitive cell
    primitive.cell = primitive_cell
    map2prim = _map2prim(primitive, new_supercell)

    ref_struct_pos = new_supercell.get_scaled_positions(wrap=True)
    sc_temp = new_supercell.get_cell(complete=True)

    fc_out = np.zeros((n_sc_new, n_sc_new, 3, 3))
    for a1, r0 in enumerate(progressbar(new_supercell.positions)):
        uc_index = map2prim[a1]
        for sc_a2, sc_r2 in enumerate(sc_r[uc_index]):
            r_pair = r0 + sc_r2
            r_pair = np.linalg.solve(sc_temp.T, r_pair.T).T % 1.0
            for a2 in range(n_sc_new):
                r_diff = np.abs(r_pair - ref_struct_pos[a2])
                # Integer value is the equivalent of 0.0
                r_diff -= np.floor(r_diff + eps)
                if np.linalg.norm(r_diff) < tol:
                    fc_out[a1, a2, :, :] += force_constants[uc_index, sc_a2, :, :]

    if two_dim:
        fc_out = fc_out.swapaxes(1, 2).reshape(2 * (3 * fc_out.shape[1],))

        # symmetrize
        violation = np.linalg.norm(fc_out - fc_out.T)
        if violation > 1e-5:
            msg = f"Force constants are not symmetric by {violation:.2e}."
            warn(msg, level=1)
            if symmetrize:
                talk("Symmetrize force constants.")
                fc_out = 0.5 * (fc_out + fc_out.T)

        # sum rule 1
        violation = abs(fc_out.sum(axis=0)).mean()
        if violation > 1e-9:
            msg = f"Sum rule violated by {violation:.2e} (axis 1)."
            warn(msg, level=1)

        # sum rule 2
        violation = abs(fc_out.sum(axis=1)).mean()
        if violation > 1e-9:
            msg = f"Sum rule violated by {violation:.2e} (axis 2)."
            warn(msg, level=1)

        return fc_out

    if reduce_fc:
        p2s_map = np.zeros(len(primitive), dtype=int)

        primitive.cell = new_supercell.cell

        new_supercell.wrap(eps=tol)
        primitive.wrap(eps=tol)

        for aa, a1 in enumerate(primitive):
            diff = new_supercell.positions - a1.position
            p2s_map[aa] = np.where(np.linalg.norm(diff, axis=1) < tol)[0][0]

        primitive.cell = primitive_cell
        primitive.wrap(eps=tol)

        return reduce_force_constants(fc_out, p2s_map)

    return fc_out

def _map2prim(primitive, supercell, tol=1e-5):
    map2prim = []
    primitive = primitive.copy()
    supercell = supercell.copy()

    # represent new supercell in fractional coords of primitive cell
    supercell_with_prim_cell = supercell.copy()

    supercell_with_prim_cell.cell = primitive.cell.copy()

    primitive.wrap(eps=tol)
    supercell_with_prim_cell.wrap(eps=tol)

    # create list that maps atoms in supercell to atoms in primitive cell
    for a1 in supercell_with_prim_cell:
        diff = primitive.positions - a1.position
        map2prim.append(np.where(np.linalg.norm(diff, axis=1) < tol)[0][0])

    # make sure every atom in primitive was matched equally often
    _, counts = np.unique(map2prim, return_counts=True)
    assert counts.std() == 0, counts

    return map2prim

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
                second_order, supercell = shengbte_io.read_second_order_qe_matrix(filename)
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

        elif format == 'tdep':
            uc_filename = 'infile.ucposcar'
            replicated_filename = 'infile.ssposcar'
            atom_prime_file = str(folder) + '/' + uc_filename
            replicated_atom_prime_file = str(folder) + '/' + replicated_filename
            uc = ase.io.read(atom_prime_file, format='vasp')
            sc = ase.io.read(replicated_atom_prime_file, format='vasp')
            d2 = parse_tdep_forceconstant(fc_file=folder+"/infile.forceconstant", primitive=uc_filename, supercell=replicated_filename, reduce_fc=False)
            n_unit_atoms = uc.positions.shape[0]
            n_replicas = np.prod(supercell)
            d2 = d2.reshape((n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
            d2 = d2[0, np.newaxis]
            second_order = SecondOrder(atoms=uc, replicated_positions=sc.positions, supercell=supercell, value=d2,
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

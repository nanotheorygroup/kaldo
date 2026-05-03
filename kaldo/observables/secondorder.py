from kaldo.observables.forceconstant import ForceConstant
from ase import Atoms
import os
import tensorflow as tf
import ase.io
import numpy as np
from numpy.typing import ArrayLike
from kaldo.interfaces.eskm_io import import_from_files
import kaldo.interfaces.shengbte_io as shengbte_io
from kaldo.interfaces.tdep_io import parse_tdep_forceconstant
from kaldo.controllers.displacement import calculate_second
import ase.units as units
from kaldo.helpers.logger import get_logger, log_size
from kaldo.storable import Storable, lazy_property
from kaldo.grid import Grid, wrap_coordinates
from kaldo.observables import harmonic_with_q as hwq

logging = get_logger()


def acoustic_sum_rule(dynmat):
    n_unit = dynmat[0].shape[0]
    sumrulecorr = 0.0
    for i in range(n_unit):
        off_diag_sum = np.sum(dynmat[0, i, :, :, :, :], axis=(-2, -3))
        dynmat[0, i, :, 0, i, :] -= off_diag_sum
        sumrulecorr += np.sum(off_diag_sum)
    logging.info("error sum rule: " + str(sumrulecorr))
    return dynmat


def _gonze_get_supercell_matrix(second_order):
    if "gonze_nac_supercell_matrix" in second_order.atoms.info:
        matrix = np.array(second_order.atoms.info["gonze_nac_supercell_matrix"], dtype=int, copy=True)
    elif "nac_bvk_supercell_matrix" in second_order.atoms.info:
        matrix = np.array(second_order.atoms.info["nac_bvk_supercell_matrix"], dtype=int, copy=True)
    else:
        matrix = np.diag(np.array(second_order.supercell, dtype=int))
    if matrix.shape != (3, 3):
        raise ValueError("Gonze NAC supercell matrix must be 3x3.")
    if not np.array_equal(matrix, np.diag(np.diag(matrix))):
        raise NotImplementedError(
            "Gonze short-range FC reconstruction currently supports diagonal supercell matrices only."
        )
    return matrix


def _gonze_get_commensurate_qpoints_diagonal(supercell):
    if np.shape(supercell) == (3, 3):
        diagonal = np.diag(np.array(supercell, dtype=int))
    else:
        diagonal = np.array(supercell, dtype=int)
    if diagonal.shape != (3,):
        raise ValueError("Diagonal supercell must be length-3.")
    qpoints = Grid(tuple(int(x) for x in diagonal), order="C").unitary_grid(is_wrapping=False)
    return wrap_coordinates(qpoints, np.eye(3))


def _gonze_build_harmonic_helper(second_order, q_point, nac_debug=False, nac_debug_folder="debug", q_index=None):
    return hwq.HarmonicWithQ(
        q_point=np.array(q_point, dtype=float),
        second=second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=nac_debug,
        nac_debug_folder=nac_debug_folder,
        q_index=q_index,
    )


def _gonze_build_static_dataset(second_order, nac_debug=False, nac_debug_folder="debug"):
    helper = _gonze_build_harmonic_helper(
        second_order,
        q_point=np.zeros(3, dtype=float),
        nac_debug=nac_debug,
        nac_debug_folder=nac_debug_folder,
    )
    static_data = helper._build_gonze_static_data()
    static_data["supercell_matrix"] = _gonze_get_supercell_matrix(second_order)
    static_data["helper"] = helper
    return static_data


def _gonze_build_short_range_inputs(second_order, static_data):
    atoms = second_order.atoms
    n_atom = len(atoms)
    n_replicas = int(np.prod(second_order.supercell))
    supercell = np.array(second_order.supercell, dtype=float)
    grid_indices = second_order._direct_grid.grid(is_wrapping=False)
    primitive_scaled = atoms.get_scaled_positions(wrap=False)
    p2s_map = np.arange(n_atom, dtype=int) * n_replicas
    s2p_map = np.repeat(p2s_map, n_replicas)
    s2pp_map = np.repeat(np.arange(n_atom, dtype=int), n_replicas)
    p2p_map = np.column_stack((p2s_map, np.arange(n_atom, dtype=int)))
    svecs = []
    multi = np.zeros((n_atom * n_replicas, n_atom, 2), dtype=np.int64)
    for atom_j in range(n_atom):
        for replica_id, grid_index in enumerate(grid_indices):
            i_s = atom_j * n_replicas + replica_id
            super_scaled_j = (primitive_scaled[atom_j] + grid_index) / supercell
            for i_p in range(n_atom):
                primitive_scaled_i = primitive_scaled[i_p] / supercell
                candidates = []
                distances = []
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        for c in (-1, 0, 1):
                            shift = np.array([a, b, c], dtype=float)
                            vec = super_scaled_j - primitive_scaled_i + shift
                            cart = vec @ static_data["supercell_cell"]
                            candidates.append(vec)
                            distances.append(np.linalg.norm(cart))
                min_distance = min(distances)
                start = len(svecs)
                for vec, distance in zip(candidates, distances):
                    if abs(distance - min_distance) < 1e-8:
                        svecs.append(vec * supercell)
                multi[i_s, i_p, 0] = len(svecs) - start
                multi[i_s, i_p, 1] = start
    return {
        "svecs": np.array(svecs, dtype=float),
        "multi": multi,
        "p2s_map": p2s_map,
        "s2p_map": s2p_map,
        "s2pp_map": s2pp_map,
        "p2p_map": p2p_map,
    }


def _gonze_build_full_fc_compact(second_order):
    n_atom = len(second_order.atoms)
    n_replicas = int(np.prod(second_order.supercell))
    fc_full = np.array(second_order.value[0], dtype=float)
    fc_full = np.transpose(fc_full, (0, 2, 3, 1, 4))
    fc_full = fc_full.reshape(n_atom, n_replicas * n_atom, 3, 3)
    permutation = np.concatenate(
        [np.arange(atom_j, n_replicas * n_atom, n_atom, dtype=int) for atom_j in range(n_atom)]
    )
    return fc_full[:, permutation]


def _gonze_calculate_dd_total_mass_weighted(q_red, static_data, mapping, q_direction_cart="auto"):
    masses = static_data["masses"]
    q_cart = static_data["reciprocal_lattice"] @ q_red
    if isinstance(q_direction_cart, str) and q_direction_cart == "auto":
        q_direction_red = np.array([-0.5, 0.0, -0.5], dtype=float)
        q_direction_cart = static_data["reciprocal_lattice"] @ q_direction_red
    dd_recip = hwq._gonze_recip_dipole_dipole(
        np.zeros_like(static_data["dd_q0"]),
        static_data["G_list"],
        q_cart,
        q_direction_cart,
        static_data["born"],
        static_data["dielectric"],
        static_data["primitive_positions"],
        float(static_data["nac_factor"]),
        float(static_data["Lambda"]),
        float(static_data["q_direction_tolerance"]),
    )
    dd_real = hwq._gonze_real_dipole_dipole(
        q_red,
        mapping["svecs"],
        mapping["multi"],
        mapping["s2pp_map"],
        static_data["dielectric"],
        float(static_data["Lambda"]),
        static_data["supercell_cell"],
    )
    dd_real_q0 = hwq._gonze_real_dipole_dipole(
        np.zeros(3, dtype=float),
        mapping["svecs"],
        mapping["multi"],
        mapping["s2pp_map"],
        static_data["dielectric"],
        float(static_data["Lambda"]),
        static_data["supercell_cell"],
    ).sum(axis=2)
    dd_limiting_expanded = np.zeros_like(dd_recip)
    dd_drift_expanded = np.zeros_like(dd_recip)
    for i in range(len(masses)):
        dd_limiting_expanded[i, :, i, :] = static_data["dd_limiting"]
        dd_drift_expanded[i, :, i, :] = (
            static_data["dd_q0"][i] + len(masses) * static_data["dd_limiting"] + dd_real_q0[i]
        )
    dd_total = dd_recip + float(static_data["nac_factor"]) * (
        dd_limiting_expanded + dd_real - dd_drift_expanded
    )
    conversion = units.mol / (10 * units.J)
    return hwq._gonze_mass_weight(dd_total * conversion, masses)


def _gonze_inverse_transform_dynmats_to_fc(dm_short_all_q, comm_q, mapping, masses):
    n_atom = len(masses)
    n_super = len(mapping["s2pp_map"])
    n_q = len(comm_q)
    conversion = units.mol / (10 * units.J)
    fc = np.zeros((n_atom, n_super, 3, 3), dtype=float)
    for p_i in range(n_atom):
        for s_j in range(n_super):
            p_j = mapping["s2pp_map"][s_j]
            multiplicity = int(mapping["multi"][s_j, p_i, 0])
            address = int(mapping["multi"][s_j, p_i, 1])
            pos = mapping["svecs"][address : address + multiplicity]
            phases = np.exp(-2j * np.pi * np.dot(comm_q, pos.T)).sum(axis=1) / multiplicity
            block_sum = np.zeros((3, 3), dtype=np.complex128)
            for q_index, coef in enumerate(phases):
                block_sum += dm_short_all_q[q_index, p_i * 3 : p_i * 3 + 3, p_j * 3 : p_j * 3 + 3] * coef
            fc[p_i, s_j] = (
                block_sum.real * np.sqrt(masses[p_i] * masses[p_j]) / n_q / conversion
            )
    return fc


def _calculate_gonze_short_range_force_constants(second_order, nac_debug=False, nac_debug_folder="debug"):
    static_data = _gonze_build_static_dataset(
        second_order,
        nac_debug=nac_debug,
        nac_debug_folder=nac_debug_folder,
    )
    mapping = _gonze_build_short_range_inputs(second_order, static_data)
    fc_full = _gonze_build_full_fc_compact(second_order)
    comm_q = _gonze_get_commensurate_qpoints_diagonal(static_data["supercell_matrix"])
    conversion = units.mol / (10 * units.J)
    dm_no_nac = []
    dd_total_mass_weighted = []
    dm_short = []
    for q_red in comm_q:
        dm_no_nac_q = hwq._gonze_short_range_dynamical_matrix(
            fc_full * conversion,
            q_red,
            mapping["svecs"],
            mapping["multi"],
            static_data["masses"],
            mapping["s2p_map"],
            mapping["p2s_map"],
        )
        dd_total_q = _gonze_calculate_dd_total_mass_weighted(
            q_red,
            static_data,
            mapping,
            q_direction_cart=None,
        )
        dm_short_q = dm_no_nac_q - dd_total_q
        dm_no_nac.append(dm_no_nac_q)
        dd_total_mass_weighted.append(dd_total_q)
        dm_short.append(dm_short_q)
    dm_no_nac = np.array(dm_no_nac, dtype=np.complex128)
    dd_total_mass_weighted = np.array(dd_total_mass_weighted, dtype=np.complex128)
    dm_short = np.array(dm_short, dtype=np.complex128)
    fc_short = _gonze_inverse_transform_dynmats_to_fc(
        dm_short,
        comm_q,
        mapping,
        static_data["masses"],
    )
    if nac_debug:
        helper = static_data["helper"]
        helper._gonze_save_debug(
            helper._gonze_debug_static_folder(),
            {
                "supercell_matrix": static_data["supercell_matrix"],
                "p2s_map": mapping["p2s_map"],
                "s2p_map": mapping["s2p_map"],
                "s2pp_map": mapping["s2pp_map"],
                "p2p_map": mapping["p2p_map"],
                "svecs": mapping["svecs"],
                "multi": mapping["multi"],
                "force_constants": fc_full,
                "commensurate_qpoints_red": comm_q,
                "commensurate_qpoints_bz_red": comm_q,
                "dm_no_nac_commensurate": dm_no_nac,
                "dd_total_mass_weighted_commensurate": dd_total_mass_weighted,
                "dm_short_commensurate": dm_short,
                "short_range_force_constants": fc_short,
            },
        )
    return fc_short


class SecondOrder(ForceConstant, Storable):
    _store_formats = {"gonze_short_range_force_constants": "numpy"}

    def __init__(self, value: ArrayLike, is_acoustic_sum: bool = False, *kargs, **kwargs):
        # apply acoustic sum rule before initialize in forceconstnat
        self.is_acoustic_sum = is_acoustic_sum
        if is_acoustic_sum:
            value = acoustic_sum_rule(value)

        super().__init__(value=value, *kargs, **kwargs)

        self.n_modes = self.atoms.positions.shape[0] * 3
        self._list_of_replicas = None  # TODO: why overwrite _list_of_replicas here?
        self.storage = "numpy"

    @lazy_property(label="", format="numpy")
    def gonze_short_range_force_constants(self):
        if "dielectric" not in self.atoms.info or "charges" not in self.atoms.arrays:
            raise ValueError(
                "Gonze short-range FC reconstruction requires atoms.info['dielectric'] and atoms.arrays['charges']."
            )
        return _calculate_gonze_short_range_force_constants(self, nac_debug=False)

    @classmethod
    def from_supercell(cls,
                       atoms: Atoms,
                       grid_type: str,
                       supercell: tuple[int, int, int] = None,
                       value: ArrayLike | None = None,
                       is_acoustic_sum: bool = False,
                       folder: str = "kALDo"):
        # acoustic sum rule will be applied later in SecondOrder.__init__ if applicable
        ifc = super().from_supercell(
            atoms=atoms,
            supercell=supercell,
            grid_type=grid_type,
            value=value,
            is_acoustic_sum=is_acoustic_sum,
            folder=folder)
        return ifc

    @classmethod
    def load(cls,
             folder: str,
             supercell: tuple[int, int, int] = (1, 1, 1),
             format: str = "numpy",
             is_acoustic_sum: bool = False):
        """
        Load second order force constants from a folder in the given format, used for library internally.

        To load force constants data, ``ForceConstants.from_folder`` is recommended.

        Parameters
        ----------
        folder : str
            Specifies where to load the data files.
        supercell : tuple[int, int, int]
            The supercell for the third order force constant matrix.
            Default: (1, 1, 1)
        format : str
            Format of the second order force constant information being loaded into SecondOrder object.
            Default: 'sparse'
        is_acoustic_sum : bool, optional
            If true, the acoustic sum rule is applied to the dynamical matrix.
            Default: False

        Returns
        -------
        second_order : SecondOrder object
            A new instance of the SecondOrder class
        """

        match format:
            case "numpy":
                replicated_atoms_file = "replicated_atoms.xyz"
                config_file = os.path.join(folder, replicated_atoms_file)
                replicated_atoms = ase.io.read(config_file, format="extxyz")

                n_replicas = np.prod(supercell)
                n_total_atoms = replicated_atoms.positions.shape[0]
                n_unit_atoms = int(n_total_atoms / n_replicas)
                unit_symbols = []
                unit_positions = []
                for i in range(n_unit_atoms):
                    unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                    unit_positions.append(replicated_atoms.positions[i])
                unit_cell = replicated_atoms.cell / supercell

                atoms = Atoms(unit_symbols, positions=unit_positions, cell=unit_cell, pbc=[1, 1, 1])

                _second_order = np.load(os.path.join(folder, "second.npy"), allow_pickle=True)
                second_order = SecondOrder(
                    atoms=atoms,
                    replicated_positions=replicated_atoms.positions,
                    supercell=supercell,
                    value=_second_order,
                    is_acoustic_sum=is_acoustic_sum,
                    folder=folder,
                )

            case "eskm" | "lammps":
                dynmat_file = os.path.join(folder, "Dyn.form")
                if format == "eskm":
                    config_file = os.path.join(folder, "CONFIG")
                    replicated_atoms = ase.io.read(config_file, format="dlp4")
                elif format == "lammps":
                    config_file = os.path.join(folder, "replicated_atoms.xyz")
                    replicated_atoms = ase.io.read(config_file, format="extxyz")
                n_replicas = np.prod(supercell)
                n_total_atoms = replicated_atoms.positions.shape[0]
                n_unit_atoms = int(n_total_atoms / n_replicas)
                unit_symbols = []
                unit_positions = []
                for i in range(n_unit_atoms):
                    unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                    unit_positions.append(replicated_atoms.positions[i])
                unit_cell = replicated_atoms.cell / supercell

                atoms = Atoms(unit_symbols, positions=unit_positions, cell=unit_cell, pbc=[1, 1, 1])

                _second_order, _ = import_from_files(
                    replicated_atoms=replicated_atoms, dynmat_file=dynmat_file, supercell=supercell
                )
                second_order = SecondOrder(
                    atoms=atoms,
                    replicated_positions=replicated_atoms.positions,
                    supercell=supercell,
                    value=_second_order,
                    is_acoustic_sum=is_acoustic_sum,
                    folder=folder,
                )

            case ("vasp-sheng" | "shengbte") | ("qe-sheng" | "shengbte-qe") | ("qe-d3q" | "shengbte-d3q") | "vasp-d3q":
                config_file = os.path.join(folder, "CONTROL")
                try:
                    atoms, supercell, charges = shengbte_io.import_control_file(config_file)
                    if charges is not None:
                        atoms.info['dielectric'] = charges[0, :, :]
                        atoms.set_array('charges', charges[1:, :, :], shape=(3, 3))
                except FileNotFoundError:
                    config_file = os.path.join(folder, "POSCAR")
                    logging.info("Trying to open POSCAR")
                    atoms = ase.io.read(config_file)

                # Create a finite difference object
                # TODO: we need to read the grid type here
                n_replicas = np.prod(supercell)
                n_unit_atoms = atoms.positions.shape[0]
                match format:
                    case ("qe-sheng" | "shengbte-qe") | ("qe-d3q" | "shengbte-d3q"):
                        # load QE second order force constant
                        filename = os.path.join(folder, "espresso.ifc2")
                        if not os.path.isfile(filename):
                            raise FileNotFoundError(f"File {filename} not found.")
                        _second_order, supercell, charges = shengbte_io.read_second_order_qe_matrix(filename)
                        if (not charges is None):
                            atoms.info['dielectric'] = charges[0, :, :]
                            atoms.set_array('charges', charges[1:, :, :], shape=(3, 3))
                        _second_order = _second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
                        grid_type = "F"
                    case _:
                        # load VASP second order force constant
                        filename = os.path.join(folder, "FORCE_CONSTANTS_2ND")
                        if not os.path.isfile(filename):
                            filename = os.path.join(folder, "FORCE_CONSTANTS")
                        if not os.path.isfile(filename):
                            raise FileNotFoundError(f"File {filename} not found.")
                        _second_order = shengbte_io.read_second_order_matrix(filename, supercell)
                        _second_order = _second_order.reshape((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
                        grid_type = "F"
                second_order = SecondOrder.from_supercell(
                    atoms=atoms,
                    grid_type=grid_type,
                    supercell=supercell,
                    value=_second_order[np.newaxis, ...],
                    is_acoustic_sum=True,
                    folder=folder,
                )

            case "hiphive":
                filename = "atom_prim.xyz"
                # TODO: add replicated filename in example
                replicated_filename = "replicated_atoms.xyz"
                try:
                    import kaldo.interfaces.hiphive_io as hiphive_io
                except ImportError:
                    logging.error(
                        "In order to use hiphive along with kaldo, hiphive is required. \
                        Please consider installing hihphive. More info can be found at: \
                        https://hiphive.materialsmodeling.org/"
                    )

                atom_prime_file = os.path.join(folder, filename)
                replicated_atom_prime_file = os.path.join(folder, replicated_filename)
                # TODO: Make this independent of replicated file
                atoms = ase.io.read(atom_prime_file)
                try:
                    replicated_atoms = ase.io.read(replicated_atom_prime_file)
                except FileNotFoundError:
                    logging.warning(
                        "Replicated atoms file not found. Please check if the file exists. Using the unit cell atoms instead."
                    )
                    replicated_atoms = atoms * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])
                # Create a finite difference object
                if "model2.fcs" in os.listdir(folder):
                    _second_order = hiphive_io.import_second_from_hiphive(
                        folder, np.prod(supercell), atoms.positions.shape[0]
                    )
                    second_order = SecondOrder(
                        atoms=atoms,
                        replicated_positions=replicated_atoms.positions,
                        supercell=supercell,
                        value=_second_order,
                        folder=folder,
                    )

            case "tdep":
                uc_filename = "infile.ucposcar"
                replicated_filename = "infile.ssposcar"
                atom_prime_file = os.path.join(folder, uc_filename)
                replicated_atom_prime_file = os.path.join(folder, replicated_filename)
                uc = ase.io.read(atom_prime_file, format="vasp")
                sc = ase.io.read(replicated_atom_prime_file, format="vasp")
                d2 = parse_tdep_forceconstant(
                    fc_file=os.path.join(folder, "infile.forceconstant"),
                    primitive=atom_prime_file,
                    supercell=replicated_atom_prime_file,
                    reduce_fc=False,
                )
                n_unit_atoms = uc.positions.shape[0]
                n_replicas = np.prod(supercell)
                d2 = d2.reshape((n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
                d2 = d2[0, np.newaxis]
                second_order = SecondOrder(
                    atoms=uc, replicated_positions=sc.positions, supercell=supercell, value=d2, folder=folder
                )

            case _:
                raise ValueError(f"{format} is not a valid format")

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
        replicated_atoms.calc = calculator

        if is_storing:
            try:
                self.value = SecondOrder.load(
                    folder=self.folder, supercell=self.supercell, format="numpy", is_acoustic_sum=self.is_acoustic_sum
                ).value

            except FileNotFoundError:
                logging.info("Second order not found. Calculating.")
                self.value = calculate_second(atoms, replicated_atoms, delta_shift, is_verbose)
                self.save("second")
                self.replicated_atoms.get_forces()
                ase.io.write(self.folder + "/replicated_atoms.xyz", self.replicated_atoms, "extxyz")
            else:
                logging.info("Reading stored second")
        else:
            self.value = calculate_second(atoms, replicated_atoms, delta_shift, is_verbose)
        if self.is_acoustic_sum:
            self.value = acoustic_sum_rule(self.value)

    def calculate_dynmat(self):
        evtotenjovermol = units.mol / (10 * units.J)
        mass = self.atoms.get_masses()
        shape = self.value.shape
        log_size(shape, float, name="dynmat")
        dynmat = self.value * 1 / np.sqrt(mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        dynmat /= np.sqrt(mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])
        return tf.convert_to_tensor(dynmat * evtotenjovermol)

    def calculate_super_replicas(self):
        scell = self.supercell
        n_replicas = np.prod(scell)
        atoms = self.atoms
        cell = atoms.cell
        n_unit_cell = atoms.positions.shape[0]
        replicated_positions = self.replicated_atoms.positions.reshape((n_replicas, n_unit_cell, 3))

        list_of_index = np.round((replicated_positions - self.atoms.positions).dot(np.linalg.inv(atoms.cell))).astype(
            int
        )
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
        sc_r_pos = np.zeros((3**3, 3))
        ir = 0
        for ix2 in [-1, 0, 1]:
            for iy2 in [-1, 0, 1]:
                for iz2 in [-1, 0, 1]:
                    for i in np.arange(3):
                        sc_r_pos[ir, i] = np.dot(replicated_cell[:, i], np.array([ix2, iy2, iz2]))
                    ir = ir + 1
        return sc_r_pos

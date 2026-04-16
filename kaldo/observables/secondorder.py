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


class SecondOrder(ForceConstant):
    def __init__(self, value: ArrayLike, is_acoustic_sum: bool = False, *kargs, **kwargs):
        # apply acoustic sum rule before initialize in forceconstnat
        self.is_acoustic_sum = is_acoustic_sum
        if is_acoustic_sum:
            value = acoustic_sum_rule(value)

        super().__init__(value=value, *kargs, **kwargs)

        self.n_modes = self.atoms.positions.shape[0] * 3
        self._list_of_replicas = None  # TODO: why overwrite _list_of_replicas here?
        self.storage = "numpy"

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
        Load second-order force constants from disk.

        Most users should prefer ``ForceConstants.from_folder(...)`` to construct
        force constants in one step. This lower-level classmethod is useful when
        you are already working with ``fc.second`` directly, or when you need to
        load second-order data from a non-standard workflow.

        Parameters
        ----------
        folder : str
            Directory containing the second-order force constant data and the
            associated replicated structure.
        supercell : tuple[int, int, int]
            Supercell used to build the stored second-order matrix.
            Default: (1, 1, 1)
        format : str
            Format of the stored second-order data.
            Default: ``"numpy"``
        is_acoustic_sum : bool, optional
            If True, apply the acoustic sum rule after loading.
            Default: False

        Returns
        -------
        second_order : SecondOrder object
            Loaded ``SecondOrder`` instance.
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
                        _second_order = _second_order.transpose(3, 4, 2, 0, 1)
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

    def calculate(self, calculator, delta_shift=1e-3, is_storing=True, is_verbose=False, n_workers=1,
                  scratch_dir=None, keep_scratch=False):
        """
        Calculate second-order force constants with finite differences.

        This is the method typically reached through ``fc.second.calculate(...)``.
        It can either load an existing ``second.npy`` from ``self.folder`` when
        ``is_storing`` is enabled, or compute the harmonic force constants from
        the current structure and calculator.

        Parameters
        ----------
        calculator : ASE Calculator instance, zero-arg factory, or CalculatorFactory
            One of three forms is accepted:

            1. A pure-Python ASE ``Calculator`` instance (e.g. ``LennardJones()``).
               Safe when the instance can be pickled, which generally requires
               that it holds no C-extension state, open file handles, or GPU
               context. Used as-is in serial and parallel runs.
            2. A zero-argument factory (a class or function) that constructs a
               fresh calculator when called, e.g.::

                   from ase.calculators.emt import EMT
                   calculator=EMT

               Each worker invokes the factory locally to obtain its own
               instance, which sidesteps pickling entirely.
            3. A ``CalculatorFactory`` that binds constructor arguments of a
               file-based or C++-backed calculator. Recommended for calculators
               like PyNEP whose instances cannot be pickled::

                   from kaldo.parallel import CalculatorFactory
                   from pynep.calculate import NEP
                   calculator=CalculatorFactory(NEP, args=('nep.txt',))

               The factory is picklable, so each worker rebuilds a fresh
               ``NEP('nep.txt')`` in its own process. Pass ``validate=False``
               to defer construction to workers when the calculator is slow
               to initialize or allocates GPU resources. ``functools.partial``
               also works and is retained for backward compatibility.

            When running with ``n_workers > 1``, prefer forms 2 or 3 unless
            you are certain the instance is picklable.
        delta_shift : float, optional
            Finite-difference displacement in Angstrom.
            Default: 1e-3
        is_storing : bool, optional
            If True, try to load an existing result from ``self.folder`` first
            and save newly computed data after the calculation.
            Default: True
        is_verbose : bool, optional
            If True, log per-atom progress information.
            Default: False
        n_workers : int or None, optional
            Number of worker processes used for the displaced-atom finite-
            difference tasks. ``1`` runs serially, ``None`` uses all available
            workers.
            Default: 1
        scratch_dir : str or None, optional
            Optional scratch directory for atom-by-atom intermediate files used
            for recovery of interrupted calculations.
            Default: None
        keep_scratch : bool, optional
            If True, keep scratch files after successful assembly.
            Default: False
        """
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms

        if is_storing:
            try:
                self.value = SecondOrder.load(
                    folder=self.folder, supercell=self.supercell, format="numpy", is_acoustic_sum=self.is_acoustic_sum
                ).value

            except FileNotFoundError:
                logging.info("Second order not found. Calculating.")
                self.value = calculate_second(
                    atoms,
                    replicated_atoms,
                    delta_shift,
                    is_verbose=is_verbose,
                    n_workers=n_workers,
                    calculator=calculator,
                    scratch_dir=scratch_dir,
                    keep_scratch=keep_scratch,
                )
                self.save("second")
                self.replicated_atoms.calc = calculator() if callable(calculator) else calculator
                self.replicated_atoms.get_forces()
                ase.io.write(self.folder + "/replicated_atoms.xyz", self.replicated_atoms, "extxyz")
            else:
                logging.info("Reading stored second")
        else:
            self.value = calculate_second(
                atoms,
                replicated_atoms,
                delta_shift,
                is_verbose=is_verbose,
                n_workers=n_workers,
                calculator=calculator,
                scratch_dir=scratch_dir,
                keep_scratch=keep_scratch,
            )
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

from kaldo.observables.forceconstant import ForceConstant
from ase import Atoms
import os
import ase.io
import numpy as np
from scipy.sparse import load_npz, save_npz
from sparse import COO
from kaldo.interfaces.eskm_io import import_from_files
from kaldo.interfaces.tdep_io import parse_tdep_third_forceconstant
import kaldo.interfaces.shengbte_io as shengbte_io
import ase.units as units
from kaldo.controllers.displacement import calculate_third
from kaldo.helpers.logger import get_logger

logging = get_logger()

REPLICATED_ATOMS_THIRD_FILE = 'replicated_atoms_third.xyz'
REPLICATED_ATOMS_FILE = 'replicated_atoms.xyz'
THIRD_ORDER_FILE_SPARSE = 'third.npz'
THIRD_ORDER_FILE = 'third.npy'


def detect_path(files: list[str], folder: str = ""):
    """return the path and the filename of the first existed file in the ``files`` list in ``folder``.
    Raise an error if none of the files in the list is found in ``folder``.
    """
    # file_list = list(map(lambda f: os.path.join(folder, f), files))
    results = list(filter(lambda f: os.path.isfile(os.path.join(folder, f)), files))
    if results:
        return os.path.join(folder, results[0]), results[0]
    else:
        raise ValueError(f"{' or '.join(files)} are not found.")


class ThirdOrder(ForceConstant):

    @classmethod
    def load(cls,
             folder: str,
             supercell: tuple[int, int, int] = (1, 1, 1),
             format: str = 'sparse',
             third_energy_threshold: float = 0.,
             chunk_size: int = 100000):
        """
        Load thrid order force constants from a folder in the given format, used for library internally.

        To load force constants data, ``ForceConstants.from_folder`` is recommended.

        Parameters
        ----------
        folder : str
            Specifies where to load the data files.
        supercell : tuple[int, int, int]
            The supercell for the third order force constant matrix.
            Default: (1, 1, 1)
        format : str
            Format of the third order force constant information being loaded into ForceConstant object.
            Default: 'sparse'
        third_energy_threshold : float, optional
            When importing sparse third order force constant matrices, energies below
            the threshold value in magnitude are ignored. Units: eV/A^3
            Default: `None`
        chunk_size : int, optional
            Number of entries to process per chunk when reading sparse third order files.
            Larger values use more memory but may be faster for very large files.
            Default: 100000

        Returns
        -------
        third_order : ThirdOrder object
            A new instance of the ThirdOrder class
        """

        match format:
            case 'sparse' | 'numpy':
                config_path, _ = detect_path([REPLICATED_ATOMS_THIRD_FILE, REPLICATED_ATOMS_FILE], folder)
                replicated_atoms = ase.io.read(config_path, format='extxyz')

                n_replicas = np.prod(supercell)
                n_total_atoms = replicated_atoms.positions.shape[0]
                n_unit_atoms = n_total_atoms // n_replicas
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

                _third_order = COO.from_scipy_sparse(load_npz(os.path.join(folder, THIRD_ORDER_FILE_SPARSE))) \
                    .reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
                third_order = ThirdOrder(atoms=atoms,
                                         replicated_positions=replicated_atoms.positions,
                                         supercell=supercell,
                                         value=_third_order,
                                         folder=folder)

            case 'eskm' | 'lammps':
                if format == 'eskm':
                    config_file = os.path.join(folder, "CONFIG")
                    replicated_atoms = ase.io.read(config_file, format='dlp4')
                elif format == 'lammps':
                    config_file = os.path.join(folder, "replicated_atoms.xyz")
                    replicated_atoms = ase.io.read(config_file, format='extxyz')

                third_file = os.path.join(folder, "THIRD")
                n_replicas = np.prod(supercell)
                n_total_atoms = replicated_atoms.positions.shape[0]
                n_unit_atoms = n_total_atoms // n_replicas
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

                out = import_from_files(replicated_atoms=replicated_atoms,
                                        third_file=third_file,
                                        supercell=supercell,
                                        third_energy_threshold=third_energy_threshold,
                                        chunk_size=chunk_size)
                third_order = ThirdOrder(atoms=atoms,
                                         replicated_positions=replicated_atoms.positions,
                                         supercell=supercell,
                                         value=out[1],
                                         folder=folder)

            case ("vasp" | "shengbte") | ("qe-vasp" | "shengbte-qe") | ("qe-d3q" | "shengbte-d3q") | "vasp-d3q":
                grid_type = 'F'
                config_path, config_file = detect_path(['CONTROL', 'POSCAR'], folder)
                match config_file:
                    case 'CONTROL':
                        atoms, _supercell, charges = shengbte_io.import_control_file(config_path)
                    case 'POSCAR':
                        logging.info('Trying to open POSCAR')
                        atoms = ase.io.read(config_path)

                match format:
                    case ("vasp" | "shengbte") | ("qe-vasp" | "shengbte-qe"):
                        # load VASP third order force constant
                        third_file = os.path.join(folder, 'FORCE_CONSTANTS_3RD')
                        third_order = shengbte_io.read_third_order_matrix(third_file, atoms, supercell, order='C')
                    case _:
                        # load d3q third order force constant
                        third_file = os.path.join(folder, 'FORCE_CONSTANTS_3RD_D3Q')
                        third_order = shengbte_io.read_third_d3q(third_file, atoms, supercell, order='C')
                third_order = ThirdOrder.from_supercell(atoms=atoms,
                                                        grid_type=grid_type,
                                                        supercell=supercell,
                                                        value=third_order,
                                                        folder=folder)

            case 'hiphive':
                filename = 'atom_prim.xyz'
                # TODO: add replicated filename in example
                replicated_filename = 'replicated_atoms.xyz'
                try:
                    import kaldo.interfaces.hiphive_io as hiphive_io
                except ImportError:
                    logging.error('In order to use hiphive along with kaldo, hiphive is required. \
                        Please consider installing hihphive. More info can be found at: \
                        https://hiphive.materialsmodeling.org/')

                atom_prime_file = os.path.join(folder, filename)
                replicated_atom_prime_file = os.path.join(folder, replicated_filename)
                # TODO: Make this independent of replicated file
                atoms = ase.io.read(atom_prime_file)
                if os.path.isfile(replicated_atom_prime_file):
                    replicated_atoms = ase.io.read(replicated_atom_prime_file)
                else:
                    logging.warning('Replicated atoms file not found. Please check if the file exists. Use the unit cell atoms instead.')
                    replicated_atoms = atoms * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])

                if 'model3.fcs' in os.listdir(str(folder)):
                    # Derive constants used for third-order reshape
                    supercell = np.array(supercell)
                    n_prim = atoms.copy().get_masses().shape[0]
                    n_sc = np.prod(supercell)
                    pbc_conditions = replicated_atoms.get_pbc()
                    dim = len(pbc_conditions[pbc_conditions == True])
                    _third_order = hiphive_io.import_third_from_hiphive(atoms, supercell, folder)
                    _third_order = _third_order[0].reshape(n_prim * dim, n_sc * n_prim * dim,
                                                           n_sc * n_prim * dim)
                    third_order = cls(atoms=atoms,
                                      replicated_positions=replicated_atoms.positions,
                                      supercell=supercell,
                                      value=_third_order,
                                      folder=folder)

            case 'tdep':
                uc = ase.io.read(os.path.join(folder, 'infile.ucposcar'), format='vasp')
                sc = ase.io.read(os.path.join(folder, 'infile.ssposcar'), format='vasp')

                third_ifcs = parse_tdep_third_forceconstant(
                    fc_filename=os.path.join(folder, 'infile.forceconstant_thirdorder'),
                    primitive=os.path.join(folder, 'infile.ucposcar'),
                    supercell=supercell,
                )

                third_order = cls(atoms=uc,
                                  replicated_positions=sc.positions,
                                  supercell=supercell,
                                  value=third_ifcs,
                                  folder=folder)

            case _:
                logging.error('Third order format not recognized: ' + str(format))
                raise ValueError

        return third_order


    def save(self, filename='THIRD', format='sparse', min_force=1e-6):
        folder = self.folder
        filename = folder + '/' + filename
        n_atoms = self.atoms.positions.shape[0]
        match format:
            case 'eskm':
                logging.info('Exporting third in eskm format')
                n_replicas = self.n_replicas
                n_replicated_atoms = n_atoms * n_replicas
                tenjovermoltoev = 10 * units.J / units.mol
                third = self.value.reshape((n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3)) / tenjovermoltoev
                with open(filename, 'w') as out_file:
                    for i in range(n_atoms):
                        for alpha in range(3):
                            for j in range(n_replicated_atoms):
                                for beta in range(3):
                                    value = third[i, alpha, j, beta].todense()
                                    mask = np.argwhere(np.linalg.norm(value, axis=1) > min_force)
                                    if mask.any():
                                        for k in mask:
                                            k = k[0]
                                            out_file.write("{:5d} ".format(i + 1))
                                            out_file.write("{:5d} ".format(alpha + 1))
                                            out_file.write("{:5d} ".format(j + 1))
                                            out_file.write("{:5d} ".format(beta + 1))
                                            out_file.write("{:5d} ".format(k + 1))
                                            for gamma in range(3):
                                                out_file.write(' {:16.6f}'.format(third[i, alpha, j, beta, k, gamma]))
                                            out_file.write('\n')
                logging.info('Done exporting third.')
            case 'sparse' | 'numpy':
                config_file = folder + REPLICATED_ATOMS_THIRD_FILE
                ase.io.write(config_file, self.replicated_atoms, format='extxyz')

                save_npz(folder + '/' + THIRD_ORDER_FILE_SPARSE, self.value.reshape((n_atoms * 3 * self.n_replicas *
                                                                            n_atoms * 3, self.n_replicas *
                                                                            n_atoms * 3)).to_scipy_sparse())
            case _:
                super(ThirdOrder, self).save(filename, format)



    def calculate(self, calculator, delta_shift=1e-4, distance_threshold=None, is_storing=True, is_verbose=False):
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        replicated_atoms.calc = calculator
        if is_storing:
            try:
                self.value = ThirdOrder.load(folder=self.folder, supercell=self.supercell).value

            except FileNotFoundError:
                logging.info('Third order not found. Calculating.')
                self.value = calculate_third(atoms,
                                             replicated_atoms,
                                             delta_shift,
                                             distance_threshold=distance_threshold,
                                             is_verbose=is_verbose)
                self.save('third')
                ase.io.write(self.folder + '/' + REPLICATED_ATOMS_THIRD_FILE, self.replicated_atoms, 'extxyz')
            else:
                logging.info('Reading stored third')
        else:
            self.value = calculate_third(atoms,
                                         replicated_atoms,
                                         delta_shift,
                                         distance_threshold=distance_threshold,
                                         is_verbose=is_verbose)
            if is_storing:
                self.save('third')
                ase.io.write(self.folder + '/' + REPLICATED_ATOMS_THIRD_FILE, self.replicated_atoms, 'extxyz')




    def __str__(self):
        return 'third'

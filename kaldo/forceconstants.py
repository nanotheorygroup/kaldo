"""
kaldo
Anharmonic Lattice Dynamics
"""
import numpy as np
from sparse import COO
from kaldo.grid import wrap_coordinates
from kaldo.observables.secondorder import SecondOrder, parse_tdep_forceconstant
from kaldo.observables.thirdorder import ThirdOrder
from kaldo.helpers.logger import get_logger
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import ase.units as units
from ase.geometry import find_mic
from ase.io import read
from sklearn.metrics import mean_squared_error
logging = get_logger()

MAIN_FOLDER = 'displacement'


class ForceConstants:
    """
    A ForceConstants class object is used to create or load the second or third order force constant matrices as well as
    store information related to the geometry of the system.

    Parameters
    ----------
    atoms: Tabulated xyz files or ASE Atoms object
        The atoms to work on.
    supercell: (3) tuple, optional
        Size of supercell given by the number of repetitions (l, m, n) of
        the small unit cell in each direction.
        Default: (1, 1, 1)
    third_supercell: tuple, optional
        Same as supercell, but for the third order force constant matrix.
        If not provided, it's copied from supercell.
        Default: `self.supercell`
    folder: str, optional
        Name to be used for the displacement information folder.
        Default: 'displacement'
    distance_threshold: float, optional
        If the distance between two atoms exceeds threshold, the interatomic
        force is ignored.
        Default: `None`

    Attributes
    ----------
    n_atoms: int
        Number of atoms in the unit cell
    n_modes: int
        The number of possible vibrational modes in the system from a lattice dynamics perspective. Equivalent to
        3*n_atoms where the factor of 3 comes from the 3 Cartesian directions.
    n_replicas: int
        The number of repeated unit cells represented in the system. Equivalent to np.prod(supercell).
    n_replicated_atoms: int
        The number of atoms represented in the system. Equivalent to n_atoms * np.prod(supercell)
    cell_inv: np.array(3, 3)
        A 3x3 matrix which satisfies AB=I where A is the matrix of cell vectors, I is the identity matrix, and B is the
        cell_inv matrix.


    """
    def __init__(self,
                 atoms,
                 supercell: tuple[int, int, int] | None = None,
                 third_supercell: tuple[int, int, int] | None = None,
                 folder: str | None = MAIN_FOLDER,
                 distance_threshold: float | None = None):

        # Store the user defined information to the object
        self.atoms = atoms
        self.supercell = supercell
        self.n_atoms = atoms.positions.shape[0]
        self.n_modes = self.n_atoms * 3
        self.n_replicas = np.prod(supercell)
        self.n_replicated_atoms = self.n_replicas * self.n_atoms
        self.cell_inv = np.linalg.inv(atoms.cell)
        self.folder = folder
        self.distance_threshold = distance_threshold
        self._list_of_replicas = None

        # TODO: we should probably remove the following initialization
        # * by default do not initialize second order 
        # * some options here:
        # * 1. allow user to use this function
        # * 2. directly remove this part
        # * 3. now allow user to use supercell and third_supercell, but warn that it is not encouraged
        if (supercell is not None) and (folder is not None):
            logging.warning("Directly use ForceConstants to load data from folder is discouraged with limit \
                            functionalities. Please use ForceConstants.from_folder for more options.")
            self.second = SecondOrder.from_supercell(atoms,
                                                     supercell=self.supercell,
                                                     grid_type='C',
                                                     is_acoustic_sum=False,
                                                     folder=folder)
            if third_supercell is None:
                third_supercell = supercell
            self.third = ThirdOrder.from_supercell(atoms,
                                                   supercell=third_supercell,
                                                   grid_type='C',
                                                   folder=folder)

        if distance_threshold is not None:
            logging.info('Using folded IFC matrices.')

    @classmethod
    def from_folder(cls,
                    folder: str,
                    supercell: tuple[int, int, int] = (1, 1, 1),
                    format: str = 'numpy',
                    third_energy_threshold: float = 0.,
                    third_supercell: tuple[int, int, int] | None = None,
                    is_acoustic_sum: bool = False,
                    only_second: bool = False,
                    distance_threshold: float | None = None):
        """
        Create a finite difference object from a folder

        The folder should contain the a set of files whose names and contents are dependent on the "format" parameter.
        Below is the list required for each format (also found in the api_forceconstants documentation if you prefer
        to read it with nicer formatting and explanations).

        numpy: replicated_atoms.xyz, second.npy, third.npz
        eskm: CONFIG, replicated_atoms.xyz, Dyn.form, THIRD
        lammps: replicated_atoms.xyz, Dyn.form, THIRD
        shengbte: CONTROL, POSCAR, FORCE_CONSTANTS_2ND/FORCE_CONSTANTS, FORCE_CONSTANTS_3RD
        shengbte-qe: CONTROL, POSCAR, espresso.ifc2, FORCE_CONSTANTS_3RD
        hiphive: atom_prim.xyz, replicated_atoms.xyz, model2.fcs, model3.fcs

        Parameters
        ----------
        folder : str
            Chosen folder to load in system information.
        supercell : (int, int, int), optional
            Number of unit cells in each cartesian direction replicated to form the input structure.
            Default is (1, 1, 1)
        format : 'numpy', 'eskm', 'lammps', 'shengbte', 'shengbte-qe', 'hiphive'
            Format of force constant information being loaded into ForceConstants object.
            Default is 'numpy'
        third_energy_threshold : float, optional
            When importing sparse third order force constant matrices, energies below
            the threshold value in magnitude are ignored. Units: ev/A^3
            Default is `None`
        distance_threshold : float, optional
            When calculating force constants, contributions from atoms further than the
            distance threshold will be ignored.
        third_supercell : (int, int, int), optional
            Takes in the unit cell for the third order force constant matrix.
            Default is self.supercell
        is_acoustic_sum : Bool, optional
            If true, the acoustic sum rule is applied to the dynamical matrix.
            Default is False

        Returns
        -------
        forceconstants: ForceConstants object
            A new instance of the ForceConstants class
        """
        second_order = SecondOrder.load(folder=folder, supercell=supercell, format=format,
                                        is_acoustic_sum=is_acoustic_sum)
        atoms = second_order.atoms

        # initialize forceconstants object, without initializing second and third
        forceconstants = cls(atoms=atoms,
                             supercell=supercell,
                             third_supercell=third_supercell,
                             folder=None,
                             distance_threshold=distance_threshold)
        # overwrite folder
        forceconstants.folder = folder

        # initialize second order force
        forceconstants.second = second_order

        # initialize third order force
        if not only_second:
            if format == 'numpy':
                third_format = 'sparse'
            else:
                third_format = format
            if third_supercell is None:
                third_supercell = supercell
            third_order = ThirdOrder.load(folder=folder, supercell=third_supercell, format=third_format,
                                          third_energy_threshold=third_energy_threshold)

            forceconstants.third = third_order

        return forceconstants

    def unfold_third_order(self, reduced_third=None, distance_threshold=None):
        """
        This method extrapolates a third order force constant matrix from a unit
        cell into a matrix for a larger supercell.

        Parameters
        ----------
        reduced_third : array, optional
            The third order force constant matrix.
            Default is `self.third`
        distance_threshold : float, optional
            When calculating force constants, contributions from atoms further than
            the distance threshold will be ignored.
            Default is `self.distance_threshold`
        """
        logging.info('Unfolding third order matrix')
        if distance_threshold is None:
            if self.distance_threshold is not None:
                distance_threshold = self.distance_threshold
            else:
                raise ValueError('Please specify a distance threshold in Angstrom')

        logging.info('Distance threshold: ' + str(distance_threshold) + ' A')
        if (self.atoms.cell[0, 0] / 2 < distance_threshold) | \
                (self.atoms.cell[1, 1] / 2 < distance_threshold) | \
                (self.atoms.cell[2, 2] / 2 < distance_threshold):
            logging.warning('The cell size should be at least twice the distance threshold')
        if reduced_third is None:
            reduced_third = self.third.value
        n_unit_atoms = self.n_atoms
        atoms = self.atoms
        n_replicas = self.n_replicas
        replicated_cell_inv = np.linalg.inv(self.third.replicated_atoms.cell)

        reduced_third = reduced_third.reshape(
            (n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
        replicated_positions = self.third.replicated_atoms.positions.reshape((n_replicas, n_unit_atoms, 3))
        dxij_reduced = wrap_coordinates(atoms.positions[:, np.newaxis, np.newaxis, :]
                                        - replicated_positions[np.newaxis, :, :, :], self.third.replicated_atoms.cell,
                                        replicated_cell_inv)
        indices = np.argwhere(np.linalg.norm(dxij_reduced, axis=-1) < distance_threshold)

        coords = []
        values = []
        for index in indices:
            for l in range(n_replicas):
                for j in range(n_unit_atoms):
                    dx2 = dxij_reduced[index[0], l, j]

                    is_storing = (np.linalg.norm(dx2) < distance_threshold)
                    if is_storing:
                        for alpha in range(3):
                            for beta in range(3):
                                for gamma in range(3):
                                    coords.append([index[0], alpha, index[1], index[2], beta, l, j, gamma])
                                    values.append(reduced_third[index[0], alpha, 0, index[2], beta, 0, j, gamma])

        logging.info('Created unfolded third order')

        shape = (n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3)
        expanded_third = COO(np.array(coords).T, np.array(values), shape)
        expanded_third = expanded_third.reshape(
            (n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
        return expanded_third


    def elastic_prop(self):
        """
        Return the stiffness tensor (aka elastic modulus tensor) of the system in GPa. This describes the stress-strain
        relationship of the material and can sometimes be used as a loose predictor for thermal conductivity. Requires
        the dynamical matrix to be loaded or calculated.

        Parameters
        ----------
        None

        Returns
        -------
        C_ijkl : np.array(3, 3, 3, 3)
            Elasticity tensor in GPa
        """
        # Notation of the variable comes from this paper:
        # Theory of the elastic constants of graphite and graphene, DOI 10.1002/pssb.200879604

        # Intake key parameters
        atoms = self.atoms
        masses = atoms.get_masses()
        volume = atoms.get_volume()
        list_of_replicas = self.second.list_of_replicas

        # rest of the code for elastic tensor assumes C-type grid
        # temp fix: flip the array from F grid to C grid if the grid type is F
        if self.second._direct_grid.order == 'F':
            list_of_replicas = np.flip(list_of_replicas, axis=1)

        dynmat = self.second.dynmat[0]  # units THz^2
        positions = self.atoms.positions
        n_unit = atoms.positions.shape[0]

        distance = positions[:, np.newaxis, np.newaxis, :] - (
                    positions[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])

        # first order term of the expansion of dynamical matrix
        d1 = np.einsum('iljx,ibljc->ibjcx', distance.astype(complex), dynmat.numpy().astype(complex))

        # second order term of the expansion of dynamical matrix
        d2 = -1 * np.einsum(
            'iljx,iljy,ibljc->ibjcxy',
            distance.astype(complex),
            distance.astype(complex),
            dynmat.numpy().astype(complex))

        # Compute Gamma tensor as eq.6
        h0 = HarmonicWithQ(np.array([0, 0, 0]), self.second, storage='numpy')
        # optical eigenvectors
        e_mu = np.array(h0._eigensystem[1:, :]).reshape(
            (n_unit, 3, 3 * (n_unit)))
        # optical eigenfrequencies
        w_mu = np.abs(np.array(h0._eigensystem[0, :])) ** (0.5)  # optical frequencies (w/(2*pi) = f) in THz
        gamma = np.einsum('iav,jbv,v->iajb', e_mu[:, :, 3:], e_mu[:, :, 3:], 1 / w_mu[3:] ** 2)

        # Compute component square braket (`b`) and round bracket (`r`) terms, keep the real component only

        # square braket term, eq.4, $[ij, kl] = b_{ijkl} = 1/(2 v_c) \sum_{n,m} \sqrt{M_n} \sqrt{M_m} D^{nm}_{ij,kl}^{(2)}$
        b = (1/(2*volume))*np.einsum('n,m,nimjkl->ijkl', masses**(0.5), masses**(0.5), d2).real

        # include mass in first order term
        d1r = np.einsum('nhmij,m->nhmij', d1, masses**(0.5))

        # round bracket term, eq.5, mass is included in d1r
        r = -1 * (1/volume) * np.einsum('nhmij,nhrp,rpskl->ijkl', d1r, gamma, d1r).real

        # Compute elastic constants C_{ij,kl} as eq.3
        cijkl = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        cijkl[i, j, k, l] = b[i, k, j, l] + b[j, k, i, l] - b[i, j, k, l] + r[i, j, k, l]

        evtotenjovermol = units.mol / (10 * units.J)
        # units._e = 1.602×10−19J
        # units.Angstorm = 1.0 = 1e-10 m
        # (units.Angstrom) ** 3 = 1e-30 m / 1e9 from Pa to GPa
        # give raises to 1e-21
        evperang3togpa = units._e / (units.Angstrom * 1e-21)

        # Denote parameter for irreducible Cij in the unit of GPa
        return evperang3togpa * cijkl / evtotenjovermol


    @staticmethod
    def _calculate_displacement(atoms, initial_structure):
        disp = atoms.positions - initial_structure.positions
        return find_mic(disp.reshape(-1, 3), atoms.cell)[0].reshape(initial_structure.positions.shape)


    @staticmethod
    def _calculate_harmonic_force(disp, second_order_fc):
        force_harmonic_vec = -np.dot(second_order_fc, disp.flatten())
        return force_harmonic_vec.reshape(disp.shape)


    @staticmethod
    def _calculate_sigma(md_forces, harmonic_forces):
        return np.sqrt(mean_squared_error(md_forces, harmonic_forces)) / np.std(md_forces)


    @staticmethod
    def sigma2_tdep_MD(fc_file: str = 'infile.forceconstant',
                       primitive_file: str = 'infile.ucposcar',
                       supercell_file: str = 'infile.ssposcar',
                       md_run: str = 'dump.xyz') -> float:
        """
        Calculate the sigma2 value using TDEP and MD data.

        Parameters
        ----------
        fc_file : str, optional
            Path to the force constant file. Default is 'infile.forceconstant'.
        primitive_file : str, optional
            Path to the primitive cell file. Default is 'infile.ucposcar'.
        supercell_file : str, optional
            Path to the supercell file. Default is 'infile.ssposcar'.
        md_run : str, optional
            Path to the MD trajectory file. Default is 'dump.xyz'.

        Returns
        -------
        float
            The average sigma2 value.

        """
        initial_structure = read(supercell_file, format="vasp")
        second_order_fc = parse_tdep_forceconstant(
            fc_file=fc_file,
            primitive=primitive_file,
            supercell=supercell_file,
            symmetrize=True,
            two_dim=True
        )
        full_MD_traj = read(md_run, index=":")
        displacements = [ForceConstants._calculate_displacement(atoms, initial_structure) for atoms in full_MD_traj]
        force_harmonic = [ForceConstants._calculate_harmonic_force(disp, second_order_fc) for disp in displacements]
        sigma_values = [ForceConstants._calculate_sigma(atoms.get_forces(), harm_force)
                        for atoms, harm_force in zip(full_MD_traj, force_harmonic)]

        return np.mean(sigma_values)


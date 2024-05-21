"""
kaldo
Anharmonic Lattice Dynamics
"""
import numpy as np
from sparse import COO
import tensorflow as tf
from kaldo.grid import wrap_coordinates
from kaldo.observables.secondorder import SecondOrder
from kaldo.observables.thirdorder import ThirdOrder
from kaldo.helpers.logger import get_logger
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import ase.units as units
logging = get_logger()

MAIN_FOLDER = 'displacement'


class ForceConstants:
    """
    Class for constructing the finite difference object to calculate
    the second/third order force constant matrices after providing the
    unit cell geometry and calculator information.

    Parameters
    ----------

    atoms: Tabulated xyz files or ASE Atoms object
        The atoms to work on.
    supercell: (3) tuple, optional
        Size of supercell given by the number of repetitions (l, m, n) of
        the small unit cell in each direction.
        Defaults to (1, 1, 1)
    third_supercell: tuple, optional
        Same as supercell, but for the third order force constant matrix.
        If not provided, it's copied from supercell.
        Defaults to `self.supercell`
    folder: str, optional
        Name to be used for the displacement information folder.
        Defaults to 'displacement'
    distance_threshold: float, optional
        If the distance between two atoms exceeds threshold, the interatomic
        force is ignored.
        Defaults to `None`
    """

    def __init__(self,
                 atoms,
                 supercell=(1, 1, 1),
                 third_supercell=None,
                 folder=MAIN_FOLDER,
                 distance_threshold=None):

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
    def from_folder(cls, folder, supercell=(1, 1, 1), format='numpy', third_energy_threshold=0., third_supercell=None,
                    is_acoustic_sum=False, only_second=False, distance_threshold=None):
        """
        Create a finite difference object from a folder

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

        Inputs
        ------
        numpy: replicated_atoms.xyz, second.npy, third.npz
        eskm: CONFIG, replicated_atoms.xyz, Dyn.form, THIRD
        lammps: replicated_atoms.xyz, Dyn.form, THIRD
        shengbte: CONTROL, POSCAR, FORCE_CONSTANTS_2ND/FORCE_CONSTANTS, FORCE_CONSTANTS_3RD
        shengbte-qe: CONTROL, POSCAR, espresso.ifc2, FORCE_CONSTANTS_3RD
        hiphive: atom_prim.xyz, replicated_atoms.xyz, model2.fcs, model3.fcs


        Returns
        -------
        ForceConstants object
        """
        second_order = SecondOrder.load(folder=folder, supercell=supercell, format=format,
                                        is_acoustic_sum=is_acoustic_sum)
        atoms = second_order.atoms
        # Create a finite difference object
        forceconstants = {'atoms': atoms,
                          'supercell': supercell,
                          'folder': folder}
        forceconstants = cls(**forceconstants)
        forceconstants.second = second_order
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
        forceconstants.distance_threshold = distance_threshold
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
            Default is self.distance_threshold
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

        # Intake key parameters
        atoms = self.atoms
        M = atoms.get_masses()
        lat = np.array(atoms.cell[:])
        V = np.abs(np.linalg.det(lat))
        list_of_replicas = self.second.list_of_replicas
        h0 = HarmonicWithQ(np.array([0, 0, 0]), self.second, storage='numpy')
        dynmat = self.second.dynmat[0]  # units THz^2
        positions = self.atoms.positions
        n_unit = atoms.positions.shape[0]
        e_mu = np.array(h0._eigensystem[1:, :]).reshape(
            (n_unit, 3, 3 * (n_unit)))
        w_mu = np.abs(np.array(h0._eigensystem[0, :])) ** (0.5)  # optical frequencies (w/(2*pi) = f) in THz
        distance = positions[:, np.newaxis, np.newaxis, :] - (
                    positions[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])
        d1 = np.einsum('iljx,ibljc->ibjcx',
                       tf.convert_to_tensor(distance.astype(complex)),
                       tf.cast(dynmat, tf.complex128))  # THz^2*Ang (it should be multiplied by i?)
        d2 = -1 * np.einsum('iljx,iljy,ibljc->ibjcxy',
                            tf.convert_to_tensor(distance.astype(complex)),
                            tf.convert_to_tensor(distance.astype(complex)),
                            tf.cast(dynmat, tf.complex128))  # THz^2*Ang^2
        Gamma = np.einsum('iav,jbv,v->iajb', e_mu[:, :, 3:], e_mu[:, :, 3:], 1 / w_mu[3:] ** 2)  # Gamma tensor from paper
        
        # Compute component b and r, keep the real component only
        b = (1/(2*V))*np.einsum('n,m,nimjkl->ijkl', M**(0.5), M**(0.5), d2).real
        d1r = np.einsum('nhmij,m->nhmij', d1, M**(0.5))
        r = -1 * (1/V) * np.einsum('nhmij,nhrp,rpskl->ijkl', d1r, Gamma, d1r).real
        Cijkl = np.zeros((3, 3, 3, 3))
        evtotenjovermol = units.mol / (10 * units.J)
        evperang3togpa = 160.21766208
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        Cijkl[i, j, k, l] = b[i, k, j, l] + b[j, k, i, l] - b[i, j, k, l] + r[i, j, k, l]
        
        # Denote parameter for irreducible Cij in the unit of GPa
        return evperang3togpa * Cijkl / evtotenjovermol

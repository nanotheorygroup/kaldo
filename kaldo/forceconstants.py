"""
kaldo
Anharmonic Lattice Dynamics
"""
import numpy as np
from sparse import COO
from kaldo.grid import wrap_coordinates
from kaldo.observables.secondorder import SecondOrder
from kaldo.observables.thirdorder import ThirdOrder
from kaldo.helpers.logger import get_logger
logging = get_logger()

MAIN_FOLDER = 'displacement'


class ForceConstants:
    """
    The ForceConstants class creates objects that store the system information and load (or calculate) force constant
    matrices (IFCs) to be used by an instance of the Phonon class.

    Parameters
    ----------

    atoms: ase.Atoms
        The atoms to study. This is required to be an ASE Atoms object, and if you would like to calculate the IFCs in
        kALDo, it should have a calculator object set.
    supercell: (int, int, int)
        Size of supercell given by the number of repetitions (l, m, n) of the unit cell in each direction. You should be
        able to calculate this by dividing the size of the supercell by the size of the unit cell in orthorhombic cells,
        but may be more complicated for other cell types.
        Default: (1, 1, 1)
    third_supercell: (int, int, int), optional
        This argument should be used if you are loading force constants from an external source in the event that the
        second and third order forces were calculated on different supercells. If not provided, we assume those
        supercells were equivalent. This argument is most likely to be used for those with ab initio calculations, where
        third order force constants are regularly calculated on smaller supercells to save computational cost.
        Default: self.supercell
    folder: str, optional
        Name to be used for the displacement information folder, in the event you would like to save to a custom
        location. You can adjust the default value on line 13 of this file.
        Default: "displacement"
    distance_threshold: float, optional
        If the distance between two atoms exceeds threshold, the forces will be ignored when calculating both harmonic
        (e.g. frequency, velocity) and anharmonic (e.g. lifetimes) phonon properties. This is useful for systems with
        coulomb forces or other long range interactions.
        Default: None
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
        Initializes a ForceConstants object from data in the folder provided. The folder should contain a file with the
        configuration of the atoms (e.g. xyz, cif, pdb) as well as the second and third order force constants.

        Two requirements:

        1. The format of all three files need to be consistent to use this method e.g. if you use a LAMMPS data file for
        the dynamical matrix, the third order forces constants need to be in LAMMPS format as well.

        2. Files should be named following the table in :ref:`input-files`.

        Parameters
        ----------
        folder : str
            Path to the folder containing your configuration file, second and third order force constants.
        format : {"numpy", "eskm", "lammps", "shengbte", "shengbte-qe", "hiphive"}
            Format of force constant information being loaded into ForceConstants object. Numpy is the default format
            because of it's memory-efficient storage and ability to load matrices into an array quickly. If possible,
            we recommend converting your data to numpy arrays using kALDo's "save" functions found in the second and
            third order classes.
            Default: 'numpy'
        supercell : (int, int, int)
            Size of supercell given by the number of repetitions (l, m, n) of the unit cell in each direction. You
            should be able to calculate this by dividing the size of the supercell by the size of the unit cell in
            orthorhombic cells, but may be more complicated for other cell types.
            Default: (1, 1, 1)
        third_supercell : (int, int, int), optional
            This argument should be used if you are loading force constants from an external source in the event that the
            second and third order forces were calculated on different supercells. If not provided, we assume those
            supercells were equivalent. This argument is most likely to be used for those with ab initio calculations, where
            third order force constants are regularly calculated on smaller supercells to save computational cost.
            Default: self.supercell
        third_energy_threshold : float, optional
            Setting this argument to a number will filter out third order (anharmonic) force constants with energies
            smaller than the threshold. This is useful for systems with long range interactions or cases where the third
            order matrix is very large. This option is currently only available for matrices in the ESKM and LAMMPS
            format.
            Units: eV/A^3
            Default: None
        distance_threshold : float, optional
            When calculating force constants, contributions from atoms further than the distance threshold will be
            ignored. This has no effect when loading precalculated force constants.
            Units: A
        is_acoustic_sum : Bool, optional
            If true, the acoustic sum rule is applied to the dynamical matrix.
            Default: False

        Returns
        -------
        ForceConstants instance
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
        This method extrapolates a third order force constant matrix from a unit cell into a matrix for the full
        supercell. You only need to set the reduced_third argument if you are using a different value than the one set
        in the ForceConstants object.

        Pass it back to the ForceConstants object to use it.

        Parameters
        ----------

        reduced_third : array, optional
            The third order force constant matrix of the unit cell.
            Default: `self.third`
        distance_threshold : float, optional
            When calculating force constants, contributions from atoms further than
            the distance threshold will be ignored.
            Default: self.distance_threshold

        Examples
        --------

        >>> unfolded_third = forceconstants.unfold_third_order()
        >>> newthird = ThirdOrder(atoms=forceconstants.atoms,
                                  supercell=forceconstants.supercell,
                                  value=unfolded_third)

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

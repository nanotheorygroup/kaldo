"""
kaldo
Anharmonic Lattice Dynamics
"""
import numpy as np
from ase import Atoms
from kaldo.grid import SupercellGrid, TranslationSupport
from sparse import COO
from kaldo.helpers.logger import get_logger

# Re-exported for back-compat: these readers historically lived here.
from kaldo.interfaces.vasp_io import _split_index, read_second_order_matrix  # noqa: F401
from kaldo.interfaces.qe_io import read_second_order_qe_matrix, read_third_d3q  # noqa: F401

logging = get_logger()


def _resolve_lattice_translation(cell_position, cell_inv, source):
    """Convert a Cartesian offset to its literal primitive-lattice vector.

    ShengBTE writes cell offsets in Cartesian Angstrom.  Retaining the exact
    integer vector, rather than reducing it modulo the supercell, is essential
    for off-commensurate IFC3 phases.
    """
    frac = cell_position.dot(cell_inv)
    if np.max(np.abs(frac - np.round(frac))) > 1e-3:
        raise ValueError(
            f"{source}: cell offset {cell_position} is not a primitive-lattice vector."
        )
    return frac.round().astype(np.int64)


def read_third_order_matrix(
    third_file: str,
    atoms: Atoms,
    supercell: tuple[int, int, int] | np.ndarray,
    order: str = "C",
    return_support: bool = False,
):
    """Read ShengBTE IFC3 without folding literal file translations.

    The returned sparse tensor follows ``(i,a,Rj,j,b,Rk,k,c)``. Distinct
    translations in the same periodic class remain distinct support entries;
    repeated records at the exact same tensor coordinate are additive.

    Parameters
    ----------
    third_file : str
        Path to the ShengBTE ``FORCE_CONSTANTS_3RD``-style file.
    atoms : ase.Atoms
        Primitive structure whose row-vector cell converts Cartesian file
        offsets into integer lattice translations.
    supercell : tuple of int or ndarray
        Diagonal repetitions or the exact 3 by 3 integer supercell matrix.
        It defines periodic equivalence but does not truncate file support.
    order : {"C", "F"}
        Ordering used for the compact periodic topology.
    return_support : bool
        When true, return ``(tensor, TranslationSupport)`` so the caller can
        preserve the literal translation axes. The compatibility return is
        the sparse tensor alone.

    Returns
    -------
    sparse.COO or tuple
        Rank-8 IFC3 tensor, optionally paired with its literal translation
        support.  Numerical IFC units are preserved exactly as stored.
    """
    n_unit_atoms = atoms.positions.shape[0]
    supercell_matrix = np.asarray(supercell)
    if supercell_matrix.shape == (3,):
        supercell_matrix = np.diag(supercell_matrix)
    elif supercell_matrix.shape != (3, 3):
        raise ValueError("supercell must have shape (3,) or (3, 3)")
    current_grid = SupercellGrid(supercell_matrix, order=order)
    cell_inv = np.linalg.inv(np.array(atoms.cell))
    translations = []
    translation_ids = {}
    records = []

    def translation_id(translation):
        key = tuple(int(value) for value in translation)
        if key not in translation_ids:
            translation_ids[key] = len(translations)
            translations.append(translation)
        return translation_ids[key]

    with open(third_file, 'r') as file:
        first_line = file.readline()
        n_third = int(first_line.strip())
        for i in range(n_third):
            # skip two lines
            file.readline()
            file.readline()

            # next two lines are the positions of the second and third cell
            second_cell_position = np.array([float(x) for x in file.readline().split()])
            second_translation = _resolve_lattice_translation(
                second_cell_position, cell_inv, third_file
            )

            third_cell_position = np.array([float(x) for x in file.readline().split()])
            third_translation = _resolve_lattice_translation(
                third_cell_position, cell_inv, third_file
            )

            # index to atom
            atom_i, atom_j, atom_k = (
                np.array([int(x) for x in file.readline().split()]) - 1
            )
            if np.any(np.asarray((atom_i, atom_j, atom_k)) < 0) or np.any(
                np.asarray((atom_i, atom_j, atom_k)) >= n_unit_atoms
            ):
                raise ValueError(
                    f"{third_file}: FC3 atom index is outside the primitive cell"
                )
            second_id = translation_id(second_translation)
            third_id = translation_id(third_translation)

            # for x,y,z directions with 3 atoms
            # FC3 assigns each quartet's block directly into its slot (ShengBTE-format writers emit
            # unique (atom, cell) slots here).
            for _ in range(27):
                values = np.array([float(x) for x in file.readline().split()])
                directions = values[:3]
                if not np.allclose(directions, np.rint(directions)):
                    raise ValueError(
                        f"{third_file}: FC3 Cartesian indices must be integers"
                    )
                alpha, beta, gamma = np.rint(directions).astype(int) - 1
                if min(alpha, beta, gamma) < 0 or max(alpha, beta, gamma) > 2:
                    raise ValueError(
                        f"{third_file}: FC3 Cartesian index is outside 1..3"
                    )
                if values[3] != 0.0:
                    records.append(
                        (
                            atom_i,
                            alpha,
                            second_id,
                            atom_j,
                            beta,
                            third_id,
                            atom_k,
                            gamma,
                            values[3],
                        )
                    )

    # The support order is independent of file record order so equivalent
    # files produce identical tensor axes and cache identities.
    sorted_old_ids = sorted(
        range(len(translations)),
        key=lambda index: (
            tuple(translations[index]) != (0, 0, 0),
            tuple(int(value) for value in translations[index]),
        ),
    )
    old_to_new = {old: new for new, old in enumerate(sorted_old_ids)}
    translations = np.asarray(
        [translations[index] for index in sorted_old_ids], dtype=np.int64
    ).reshape((-1, 3))
    support = TranslationSupport(translations, current_grid, provenance="file")
    shape = (
        n_unit_atoms,
        3,
        support.size,
        n_unit_atoms,
        3,
        support.size,
        n_unit_atoms,
        3,
    )
    if records:
        coordinates = np.asarray([record[:-1] for record in records], dtype=np.int64).T
        coordinates[2] = np.asarray([old_to_new[index] for index in coordinates[2]])
        coordinates[5] = np.asarray([old_to_new[index] for index in coordinates[5]])
        data = np.asarray([record[-1] for record in records], dtype=np.float64)
        third_order = COO(coordinates, data, shape=shape, has_duplicates=True)
        duplicate_entries = len(data) - third_order.nnz
        if duplicate_entries:
            logging.info(
                "%s: accumulated %d repeated nonzero FC3 tensor entries.",
                third_file,
                duplicate_entries,
            )
    else:
        third_order = COO(np.empty((8, 0), dtype=np.int64), np.empty(0), shape=shape)
    repeated_classes = support.size - np.unique(support.class_ids).size
    if repeated_classes:
        logging.info(
            "%s: retained %d literal FC3 translations sharing periodic classes.",
            third_file,
            repeated_classes,
        )
    return (third_order, support) if return_support else third_order


def import_control_file(control_file):
    positions = []
    latt_vecs = []
    eps_vecs = []
    bec_vecs = []
    lfactor = 1
    masses = None
    with open(control_file, "r") as fo:
        lines = fo.readlines()
    for line in lines:
        if 'lattvec' in line:
            value = line.split('=')[1]
            latt_vecs.append(np.array([float(x.rstrip(',')) for x in value.split()]))
        if 'elements' in line and not ('nelements' in line):
            value = line.split('=')[1]
            # TODO: only one species at the moment
            value = value.replace('"', '\'')
            value = value.replace(" ", '')
            value = value.replace("\n", '')
            value = value.replace(',', '')
            value = value.replace("''", '\t')
            value = value.replace("'", '')
            elements = value.split("\t")
        if 'types' in line:
            value = line.split('=')[1]
            types = np.array([int(x.rstrip(',')) for x in value.split()])
        if 'positions' in line:
            value = line.split('=')[1]
            positions.append(np.array([float(x.rstrip(',')) for x in value.split()]))
        if 'lfactor' in line:
            lfactor = float(line.split('=')[1].split(',')[0])
        # TODO: only one species/mass at the moment
        if 'masses' in line:
            value = line.split('=')[1]
            masses=np.array([float(x.rstrip(',')) for x in value.split()])
            #masses = float(line.split('=')[1].split(',')[0])
        if 'scell' in line:
            value = line.split('=')[1]
            supercell = np.array([int(x.rstrip(',')) for x in value.split()])
        if 'epsilon' in line:
            value = line.split('=')[1]
            eps_vecs.append(np.array([float(x.rstrip(',')) for x in value.split()]))
        if 'born' in line:
            value = line.split('=')[1]
            bec_vecs.append(np.array([float(x.rstrip(',')) for x in value.split()]))
    # l factor is in nanometer
    cell = np.array(latt_vecs) * lfactor * 10
    positions = np.array(positions).dot(cell)
    list_of_elem = []
    if masses is None:
        for i in range(len(types)):
            list_of_elem.append(elements[types[i] - 1])

        atoms = Atoms(list_of_elem,
                      positions=positions,
                      cell=cell,
                      pbc=[1, 1, 1])
    else:
        list_of_masses = []
        for i in range(len(types)):
            list_of_elem.append(elements[types[i] - 1])
            list_of_masses.append(masses[types[i] - 1])
        atoms = Atoms(list_of_elem,
                      positions=positions,
                      cell=cell,
                      masses=list_of_masses,
                      pbc=[1, 1, 1])

    logging.info('Atoms object created.')
    if len(eps_vecs) == 0:
        charges = None
    else:
        charges = np.zeros((len(atoms)+1, 3, 3))
        charges[0, ...] = np.array(eps_vecs)
        charges[1:, ...] = np.array(bec_vecs).reshape((len(atoms), 3, 3))
        logging.info('Charge data found in CONTROL file.')
    return atoms, supercell, charges



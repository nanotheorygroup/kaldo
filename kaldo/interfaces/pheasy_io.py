"""
kaldo
Anharmonic Lattice Dynamics

I/O for force constants extracted by pheasy (https://gitlab.com/cplin/pheasy),
read from an unmodified pheasy working directory after an IFC fit. See the
``api_forceconstants`` documentation for the folder contract. kaldo never
imports pheasy (BSD vs GPL); this module only parses pheasy's output files.

pheasy always writes IFC2 as phonopy-style text (``FORCE_CONSTANTS``, eV/A^2,
compact header by default, full with pheasy's ``--full_ifc``) and IFC3 as
ShengBTE text (``FORCE_CONSTANTS_3RD``, eV/A^3); with ``--hdf5`` it also
writes ``fc2.hdf5``/``fc3.hdf5`` (dataset keys ``fc2``/``fc3``). IFC4 goes to
the ShengBTE/FourPhonon ``FORCE_CONSTANTS_4TH`` text file (eV/A^4). Born
effective charges and the dielectric tensor live in pheasy's ``born.fmt``.

pheasy enumerates supercell atoms atom-major with the cell image minor and
the first lattice vector fastest (pheasy/structure/atoms.py,
``create_supercell``/``set_smap``/``set_pmap``)::

    s = atom * n_cells + c,    c = ox + oy * d0 + oz * d0 * d1

kaldo stores replicas on a ``Grid``; every pheasy flat index is decomposed
explicitly and mapped through ``Grid(supercell, order='C')``, never assuming
the two enumerations coincide.
"""
import os

import ase.io
import numpy as np

import kaldo.interfaces.shengbte_io as shengbte_io
from kaldo.grid import Grid
from kaldo.helpers.logger import get_logger

logging = get_logger()

STRUCTURE_FILES = ('POSCAR', 'cell.in')
SECOND_ORDER_FILES = ('FORCE_CONSTANTS', 'fc2.hdf5')
THIRD_ORDER_FILES = ('FORCE_CONSTANTS_3RD', 'fc3.hdf5')
FOURTH_ORDER_FILE = 'FORCE_CONSTANTS_4TH'
BORN_FILE = 'born.fmt'


def read_pheasy_structure(folder):
    """Read the primitive cell of a pheasy run (``POSCAR`` or ``cell.in``)."""
    poscar = os.path.join(str(folder), STRUCTURE_FILES[0])
    espresso = os.path.join(str(folder), STRUCTURE_FILES[1])
    if os.path.isfile(poscar):
        return ase.io.read(poscar, format='vasp')
    if os.path.isfile(espresso):
        return ase.io.read(espresso, format='espresso-in')
    raise FileNotFoundError(f"No primitive cell found in {folder}: expected one of {STRUCTURE_FILES} "
                            "(pheasy's --pcell file, kept under its default name).")


def read_born_charges(folder, n_atoms):
    """Read pheasy's ``born.fmt``.

    Returns an array of shape ``(n_atoms + 1, 3, 3)`` in the same convention
    the ShengBTE CONTROL reader uses: index 0 is the dielectric tensor, the
    rest are the Born effective charges of each primitive atom.
    """
    filename = os.path.join(str(folder), BORN_FILE)
    born_info = np.loadtxt(filename)
    expected_rows = 3 * (n_atoms + 1)
    if born_info.ndim != 2 or born_info.shape[1] != 3 or born_info.shape[0] != expected_rows:
        raise ValueError(f"{filename}: expected {expected_rows} rows of 3 floats (3x3 dielectric tensor followed "
                         f"by one 3x3 Born tensor per atom), got shape {born_info.shape}.")
    return born_info.reshape((n_atoms + 1, 3, 3))


def _pheasy_cell_to_kaldo_id(supercell):
    """Map pheasy's flat cell index c to the ``Grid(supercell, 'C')`` replica id.

    pheasy: ``c = ox + oy * d0 + oz * d0 * d1`` (first lattice vector fastest).
    Returns an int array of length ``prod(supercell)``.
    """
    d0, d1, d2 = (int(x) for x in supercell)
    grid = Grid((d0, d1, d2), order='C')
    n_cells = d0 * d1 * d2
    mapping = np.empty(n_cells, dtype=np.int64)
    for c in range(n_cells):
        ox = c % d0
        oy = (c // d0) % d1
        oz = c // (d0 * d1)
        ids = grid.grid_index_to_id(np.array([ox, oy, oz]), is_wrapping=False)
        mapping[c] = int(ids[0])
    return mapping


def check_pheasy_supercell_order(folder, atoms, supercell):
    """Warn if pheasy's written supercell deviates from its expected atom order.

    Reads ``SPOSCAR``/``supercell.in`` when present and compares against the
    atom-major, x-fastest supercell reconstructed from the primitive cell.
    Never raises and never reorders anything; a mismatch logs a warning.
    """
    sposcar = os.path.join(str(folder), 'SPOSCAR')
    supercell_in = os.path.join(str(folder), 'supercell.in')
    try:
        if os.path.isfile(sposcar):
            written = ase.io.read(sposcar, format='vasp')
        elif os.path.isfile(supercell_in):
            written = ase.io.read(supercell_in, format='espresso-in')
        else:
            return
    except Exception as error:
        # pheasy's own supercell.in omits the &CONTROL/&SYSTEM namelists a standalone
        # pw.x input needs (pheasy/structure/atoms.py:write_pw_in only ever emits
        # CELL_PARAMETERS/ATOMIC_POSITIONS), so ASE's espresso-in reader can fail on it.
        # This check is an optional cross-check; degrade to a warning, never raise.
        logging.warning(f"pheasy supercell file in {folder} could not be read ({error}); "
                        "skipping the ordering check.")
        return
    d0, d1, d2 = (int(x) for x in supercell)
    n_cells = d0 * d1 * d2
    scaled = atoms.get_scaled_positions()
    expected = np.zeros((len(atoms) * n_cells, 3))
    for i in range(len(atoms)):
        for c in range(n_cells):
            ox = c % d0
            oy = (c // d0) % d1
            oz = c // (d0 * d1)
            expected[i * n_cells + c] = (scaled[i] + np.array([ox, oy, oz])) / np.array([d0, d1, d2])
    if len(written) != expected.shape[0]:
        logging.warning(f"pheasy supercell file in {folder} has {len(written)} atoms, expected "
                        f"{expected.shape[0]} for supercell {(d0, d1, d2)}; skipping the ordering check.")
        return
    if not np.array_equal(written.numbers, np.repeat(atoms.numbers, n_cells)):
        logging.warning(f"pheasy supercell file in {folder} has a species order inconsistent with the "
                        "atom-major convention reconstructed from the primitive cell; skipping the "
                        "ordering check.")
        return
    diff = written.get_scaled_positions() - expected
    diff -= np.round(diff)
    if not np.allclose(diff, 0.0, atol=1e-5):
        logging.warning("pheasy supercell file atom order does not match pheasy's atom-major convention; "
                        "imported force constants may be misassigned. Check the pheasy run inputs.")


def _decode_row_atoms(row_labels, n_unit_atoms, n_cells):
    """Decode compact FORCE_CONSTANTS row labels into unit-cell atom indices.

    pheasy labels compact rows with the supercell index of each unit-cell atom
    (``i * n_cells + 1``, 1-based); plain phonopy compact files use
    ``1..n_unit_atoms``. Returns a dict label -> unit atom, or raises.
    """
    labels = np.array(sorted(set(int(x) for x in row_labels)))
    pmap_style = np.arange(n_unit_atoms) * n_cells + 1
    direct_style = np.arange(n_unit_atoms) + 1
    if np.array_equal(labels, pmap_style):
        return {int(label): int(label - 1) // n_cells for label in labels}
    if np.array_equal(labels, direct_style):
        return {int(label): int(label) - 1 for label in labels}
    raise ValueError(f"FORCE_CONSTANTS: cannot interpret row indices {labels[:6].tolist()}... as compact rows for "
                     f"{n_unit_atoms} unit-cell atoms and {n_cells} cells; the file layout is not recognized.")


def _read_fc2_text(filename, n_unit_atoms, n_cells, supercell):
    """Parse a pheasy FORCE_CONSTANTS file into (rows, cols, blocks, is_full).

    For full-format files, rows outside the central cell are redundant under
    translational symmetry; their 3 value lines are consumed but not stored,
    so ``rows``/``cols``/``blocks`` only ever hold the rows the caller needs.
    """
    n_sc = n_unit_atoms * n_cells
    with open(filename, 'r') as file:
        header = file.readline().split()
        if len(header) < 2:
            raise ValueError(f"{filename}: malformed header {header!r}.")
        nat_a, nat_b = int(header[0]), int(header[1])
        if nat_b != n_sc or nat_a not in (n_unit_atoms, n_sc):
            raise ValueError(f"{filename}: header ({nat_a}, {nat_b}) is inconsistent with {n_unit_atoms} unit-cell "
                             f"atoms and supercell {tuple(int(x) for x in supercell)} (expected "
                             f"({n_unit_atoms} or {n_sc}, {n_sc})); check the supercell argument against "
                             "pheasy's --dim.")
        # for a (1, 1, 1) supercell n_uc == n_sc, so compact and full coincide and either path is correct
        is_full = nat_a == n_sc
        n_blocks = nat_a * nat_b
        n_kept = n_unit_atoms * nat_b if is_full else n_blocks
        rows = np.empty(n_kept, dtype=np.int64)
        cols = np.empty(n_kept, dtype=np.int64)
        blocks = np.empty((n_kept, 3, 3))
        kept = 0
        for n in range(n_blocks):
            index_line = file.readline().split()
            if len(index_line) < 2:
                raise ValueError(f"{filename}: unexpected end of file at block {n}.")
            row, col = int(index_line[0]), int(index_line[1])
            if is_full and (row - 1) % n_cells != 0:
                # rows outside the central cell are redundant under translational symmetry
                for _ in range(3):
                    file.readline()
                continue
            rows[kept], cols[kept] = row, col
            for alpha in range(3):
                blocks[kept, alpha] = [float(x) for x in file.readline().split()]
            kept += 1
    return rows[:kept], cols[:kept], blocks[:kept], is_full


def read_pheasy_second(folder, atoms, supercell):
    """Read pheasy IFC2 into kaldo's ``(1, n_uc, 3, n_rep, n_uc, 3)`` layout, eV/A^2.

    Prefers the always-written ``FORCE_CONSTANTS`` text file; falls back to
    ``fc2.hdf5`` (dataset ``fc2``). Handles pheasy's compact and full forms.
    """
    n_unit_atoms = atoms.positions.shape[0]
    d0, d1, d2 = (int(x) for x in supercell)
    n_cells = d0 * d1 * d2
    n_sc = n_unit_atoms * n_cells
    cell_map = _pheasy_cell_to_kaldo_id((d0, d1, d2))
    value = np.zeros((1, n_unit_atoms, 3, n_cells, n_unit_atoms, 3))

    text_path = os.path.join(str(folder), SECOND_ORDER_FILES[0])
    hdf5_path = os.path.join(str(folder), SECOND_ORDER_FILES[1])
    if os.path.isfile(text_path):
        logging.info(f"Reading pheasy second order from {text_path}")
        rows, cols, blocks, is_full = _read_fc2_text(text_path, n_unit_atoms, n_cells, supercell)
        row_to_unit = None if is_full else _decode_row_atoms(rows, n_unit_atoms, n_cells)
        for n in range(rows.shape[0]):
            row, col = int(rows[n]), int(cols[n])
            if is_full:
                i_uc = (row - 1) // n_cells
            else:
                i_uc = row_to_unit[row]
            j_uc = (col - 1) // n_cells
            rep = cell_map[(col - 1) % n_cells]
            value[0, i_uc, :, rep, j_uc, :] = blocks[n]
    elif os.path.isfile(hdf5_path):
        import h5py
        logging.info(f"Reading pheasy second order from {hdf5_path}")
        with h5py.File(hdf5_path, 'r') as fd:
            if 'fc2' not in fd:
                raise ValueError(f"{hdf5_path}: dataset 'fc2' not found.")
            ifc2 = np.array(fd['fc2'], dtype=np.float64)
        if ifc2.shape == (n_sc, n_sc, 3, 3):
            ifc2 = ifc2[np.arange(n_unit_atoms) * n_cells]
        elif ifc2.shape != (n_unit_atoms, n_sc, 3, 3):
            raise ValueError(f"{hdf5_path}: fc2 shape {ifc2.shape} is inconsistent with supercell "
                             f"{(d0, d1, d2)} (expected ({n_unit_atoms}, {n_sc}, 3, 3) or "
                             f"({n_sc}, {n_sc}, 3, 3)); check the supercell argument against pheasy's --dim.")
        for i_uc in range(n_unit_atoms):
            for col in range(n_sc):
                j_uc = col // n_cells
                rep = cell_map[col % n_cells]
                value[0, i_uc, :, rep, j_uc, :] = ifc2[i_uc, col]
    else:
        raise FileNotFoundError(f"No pheasy second order file found in {folder}: expected one of "
                                f"{SECOND_ORDER_FILES}.")
    return value.astype(np.float64)


def read_pheasy_third(folder, atoms, supercell):
    """Read pheasy IFC3, eV/A^3, into the dense ``(n_uc*3, n_rep*n_uc*3, n_rep*n_uc*3)`` layout.

    Prefers the always-written ShengBTE ``FORCE_CONSTANTS_3RD`` (self-describing:
    Cartesian cell offsets plus 1-based unit-cell atom indices, parsed by the
    existing ShengBTE reader); falls back to ``fc3.hdf5`` (dataset ``fc3``).
    """
    text_path = os.path.join(str(folder), THIRD_ORDER_FILES[0])
    hdf5_path = os.path.join(str(folder), THIRD_ORDER_FILES[1])
    if os.path.isfile(text_path):
        logging.info(f"Reading pheasy third order from {text_path}")
        return shengbte_io.read_third_order_matrix(text_path, atoms, supercell, order='C')
    if os.path.isfile(hdf5_path):
        logging.info(f"Reading pheasy third order from {hdf5_path}")
        return _read_fc3_hdf5(hdf5_path, atoms, supercell)
    raise FileNotFoundError(f"No pheasy third order file found in {folder}: expected one of {THIRD_ORDER_FILES}.")


def _read_fc3_hdf5(filename, atoms, supercell):
    import h5py
    n_unit_atoms = atoms.positions.shape[0]
    d0, d1, d2 = (int(x) for x in supercell)
    n_cells = d0 * d1 * d2
    n_sc = n_unit_atoms * n_cells
    with h5py.File(filename, 'r') as fd:
        if 'fc3' not in fd:
            raise ValueError(f"{filename}: dataset 'fc3' not found.")
        fc3 = np.array(fd['fc3'], dtype=np.float64)
    if fc3.shape == (n_sc, n_sc, n_sc, 3, 3, 3):
        fc3 = fc3[np.arange(n_unit_atoms) * n_cells]
    elif fc3.shape != (n_unit_atoms, n_sc, n_sc, 3, 3, 3):
        raise ValueError(f"{filename}: fc3 shape {fc3.shape} is inconsistent with supercell {(d0, d1, d2)} "
                         f"(expected ({n_unit_atoms}, {n_sc}, {n_sc}, 3, 3, 3) or ({n_sc}, {n_sc}, {n_sc}, 3, 3, 3)); "
                         "check the supercell argument against pheasy's --dim.")
    cell_map = _pheasy_cell_to_kaldo_id((d0, d1, d2))
    unit_of_col = np.arange(n_sc) // n_cells
    rep_of_col = cell_map[np.arange(n_sc) % n_cells]
    third = np.zeros((n_unit_atoms, 3, n_cells, n_unit_atoms, 3, n_cells, n_unit_atoms, 3))
    for i_uc in range(n_unit_atoms):
        for col_j in range(n_sc):
            for col_k in range(n_sc):
                block = fc3[i_uc, col_j, col_k]
                if not block.any():
                    continue
                third[i_uc, :, rep_of_col[col_j], unit_of_col[col_j], :,
                      rep_of_col[col_k], unit_of_col[col_k], :] = block
    return third.reshape((n_unit_atoms * 3, n_cells * n_unit_atoms * 3, n_cells * n_unit_atoms * 3))


def read_pheasy_fourth(folder, atoms, supercell):
    """Read pheasy IFC4 (``FORCE_CONSTANTS_4TH``, eV/A^4) as a rank-11 sparse COO tensor.

    ``fc4.hdf5`` is intentionally not supported: it stores the dense rank-8
    tensor, O(n_sc^3) memory, while the text file is sparse and always written.
    """
    path = os.path.join(str(folder), FOURTH_ORDER_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No pheasy fourth order file found in {folder}: expected {FOURTH_ORDER_FILE} "
                                "(written when fitting with order >= 4; fc4.hdf5 is not supported).")
    logging.info(f"Reading pheasy fourth order from {path}")
    return shengbte_io.read_fourth_order_matrix(path, atoms, supercell, order='C')

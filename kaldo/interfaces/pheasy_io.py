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
    if os.path.isfile(sposcar):
        written = ase.io.read(sposcar, format='vasp')
    elif os.path.isfile(supercell_in):
        written = ase.io.read(supercell_in, format='espresso-in')
    else:
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
    diff = written.get_scaled_positions() - expected
    diff -= np.round(diff)
    if not np.allclose(diff, 0.0, atol=1e-5):
        logging.warning("pheasy supercell file atom order does not match pheasy's atom-major convention; "
                        "imported force constants may be misassigned. Check the pheasy run inputs.")

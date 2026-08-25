"""VASP force-constant readers."""

import numpy as np


def _split_index(index, nx, ny, nz):
    """Split index into ix, iy, iz, iatom. A helper function for read_second_order_matrix.
    index = (((iatom * nz + iz) * ny + iy) * nx + ix), assuming index is zero-indexing. """
    tmp1, ix = divmod(index, nx)
    tmp2, iy = divmod(tmp1, ny)
    iatom, iz = divmod(tmp2, nz)
    return ix, iy, iz, iatom


def read_second_order_matrix(filename, supercell):
    """Read second order force constants from a file in VASP format.

    Parameters
    ----------
    filename : str
        The path to file of second order force constants in VASP format.

    supercell : [t1, t2, t3]
        The size of the supercell as t1 * t2 * t3.

    Returns
    -------
        second_order : np.ndarray(i_at, alpha, t1, t2, t3, j_at, beta)
            The array contains second order force constants.
            alpha and beta are directional indexes in x,y,z.
            t1, t2, t3 is the index to the supercell for j-th atom.
            i_at and j_at are the indexes to the atoms as in unit cell.
    """
    with open(filename, 'r') as file:
        first_row = file.readline()
        n_rows = int(first_row.strip().split()[0])
        n_replicas = np.prod(supercell)
        n_unit_atoms = int(n_rows / n_replicas)

        second_order = np.zeros((n_unit_atoms, 3, supercell[0],
                                 supercell[1], supercell[2], n_unit_atoms, 3))

        line = file.readline()
        while line:
            try:
                # VASP uses one-indexing
                i, j = np.array([int(x) for x in line.split()])
                # convert to zero-indexing
                i -= 1
                j -= 1
            except ValueError as err:
                raise ValueError(f"malformed index line in second order file: {line!r}") from err

            # i_ix, i_iy, i_iz, i_iatom, j_ix, j_iy, j_iz, j_iatom, alpha are zero-indexing
            i_ix, i_iy, i_iz, i_iatom = _split_index(i, supercell[0], supercell[1], supercell[2])
            j_ix, j_iy, j_iz, j_iatom = _split_index(j, supercell[0], supercell[1], supercell[2])
            for alpha in range(3):
                if (i_ix == 0) and (i_iy == 0) and (i_iz == 0):
                    line = file.readline()
                    second_order[i_iatom, alpha, j_ix, j_iy, j_iz, j_iatom, :] = \
                        np.array([float(x) for x in line.split()])
                else:
                    file.readline()
            line = file.readline()
    return second_order

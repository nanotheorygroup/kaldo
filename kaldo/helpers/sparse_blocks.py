"""Vectorized operations on sparse 3x3 atom-block matrices in CSR/BSR order.

A block matrix is described by an atom-row pointer ``indptr``, sorted column
atoms ``indices`` per row, and ``data`` of shape ``(n_blocks, 3, 3)``. Work on
``data`` runs in bounded batches of ``PAIR_BATCH`` blocks.
"""

import numpy as np


PAIR_BATCH = 65536


def reverse_block_slots(indptr, indices, displacement=None, *, image_quantum=1e-2):
    """Return the slot of every stored block's reverse block, or -1 if it is absent.

    Block ``p`` couples row atom ``i`` to column atom ``j``; its reverse couples
    ``j`` to ``i``. When an atom pair is stored more than once (several
    periodic images), ``displacement`` identifies the images: the reverse of
    image ``r`` is image ``-r``. Displacements are compared on an
    ``image_quantum`` grid in Angstrom, far finer than the lattice vectors that
    separate distinct images.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    columns = np.asarray(indices, dtype=np.int64)
    n_atoms = len(indptr) - 1
    rows = np.repeat(np.arange(n_atoms, dtype=np.int64), np.diff(indptr))
    keys = rows * n_atoms + columns
    reverse_keys = columns * n_atoms + rows
    del rows
    if not len(keys):
        return np.empty(0, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if not np.any(sorted_keys[1:] == sorted_keys[:-1]):
        position = np.minimum(np.searchsorted(sorted_keys, reverse_keys), len(keys) - 1)
        return np.where(sorted_keys[position] == reverse_keys, order[position], -1)
    if displacement is None:
        raise ValueError("repeated atom pairs require displacements to identify periodic images")
    image = np.rint(np.asarray(displacement, dtype=np.float64) / image_quantum).astype(np.int64)
    forward = np.column_stack((keys, image))
    backward = np.column_stack((reverse_keys, -image))
    combined, inverse = np.unique(np.concatenate((forward, backward)), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    if len(np.unique(inverse[:len(keys)])) != len(keys):
        raise ValueError("two stored blocks share an atom pair and a periodic image")
    slot_of = np.full(len(combined), -1, dtype=np.int64)
    slot_of[inverse[:len(keys)]] = np.arange(len(keys))
    return slot_of[inverse[len(keys):]]


def block_row_sums(data, indptr):
    """Sum 3x3 blocks over each CSR row; every row must be nonempty."""
    return np.add.reduceat(data, indptr[:-1], axis=0)


def block_column_sums(data, columns, n_atoms):
    """Sum 3x3 blocks over each column atom, in float64."""
    sums = np.zeros((n_atoms, 3, 3), dtype=np.float64)
    for start in range(0, len(columns), PAIR_BATCH):
        stop = min(start + PAIR_BATCH, len(columns))
        np.add.at(sums, columns[start:stop], data[start:stop])
    return sums


def symmetrize_blocks(data, reverse):
    """Average each block with its reverse's transpose, in place; onsite blocks map to themselves."""
    for start in range(0, len(reverse), PAIR_BATCH):
        slots = np.arange(start, min(start + PAIR_BATCH, len(reverse)))
        first = slots[slots <= reverse[slots]]
        second = reverse[first]
        average = 0.5 * (data[first] + data[second].transpose(0, 2, 1))
        data[first] = average
        data[second] = average.transpose(0, 2, 1)

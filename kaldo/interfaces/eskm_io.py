"""
kaldo
Anharmonic Lattice Dynamics
"""

import ase.units as units
from sparse import COO
from ase import Atoms
import pandas as pd
import numpy as np

from kaldo.helpers.logger import get_logger
from kaldo.helpers.tools import count_rows

logging = get_logger()


def import_from_files(
    replicated_atoms, dynmat_file=None, third_file=None, supercell=(1, 1, 1), third_energy_threshold=0.0
):
    # TODO: split this method into two pieces
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

    second_order = None
    third_order = None

    if dynmat_file:
        logging.info("Reading dynamical matrix")
        second_dl = import_second(atoms, replicas=supercell, filename=dynmat_file)
        second_order = second_dl

    if third_file:
        try:
            logging.info("Reading sparse third order")
            third_dl = import_sparse_third(
                atoms=atoms, supercell=supercell, filename=third_file, third_energy_threshold=third_energy_threshold
            )

        except UnicodeDecodeError:
            if third_energy_threshold != 0:
                raise ValueError("Third threshold not supported for dense third")
            logging.info("Reading dense third order")
            third_dl = import_dense_third(atoms, supercell=supercell, filename=third_file)
            logging.info("Third order matrix stored.")
        third_dl = third_dl[:n_unit_atoms]
        third_shape = (n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3)
        third_dl = third_dl.reshape(third_shape)
        third_order = third_dl

    return second_order, third_order


def import_second(atoms, replicas=(1, 1, 1), filename="Dyn.form"):
    replicas = np.array(replicas)
    n_unit_cell = atoms.positions.shape[0]
    dyn_mat = import_dynamical_matrix(n_unit_cell, replicas, filename)
    mass = np.sqrt(atoms.get_masses())
    dyn_mat = dyn_mat * mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    dyn_mat = dyn_mat * mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    return dyn_mat


def import_dynamical_matrix(n_atoms, supercell=(1, 1, 1), filename="Dyn.form"):
    supercell = np.array(supercell)
    dynamical_matrix_frame = pd.read_csv(filename, header=None, sep=r"\s+")
    dynamical_matrix = dynamical_matrix_frame.values
    n_replicas = np.prod(supercell)
    if dynamical_matrix.size == n_replicas * (n_atoms * 3) ** 2:
        dynamical_matrix = dynamical_matrix.reshape((n_atoms, 3, n_replicas, n_atoms, 3))
    elif dynamical_matrix.size == (n_replicas * n_atoms * 3) ** 2:
        dynamical_matrix = dynamical_matrix.reshape((n_replicas, n_atoms, 3, n_replicas, n_atoms, 3))[0]
    elif dynamical_matrix.size == (n_atoms * 3) ** 2:
        dynamical_matrix = dynamical_matrix.reshape((n_atoms, 3, 1, n_atoms, 3))
    else:
        logging.error("Impossible to read calculate_dynmat with size " + str(dynamical_matrix.size))
    tenjovermoltoev = 10 * units.J / units.mol
    return dynamical_matrix * tenjovermoltoev


def import_sparse_third(atoms, supercell=(1, 1, 1), filename="THIRD", third_energy_threshold=0.0, chunk_size=100000):
    supercell = np.array(supercell)
    n_replicas = np.prod(supercell)
    n_atoms = atoms.get_positions().shape[0]
    n_replicated_atoms = n_atoms * n_replicas
    tenjovermoltoev = 10 * units.J / units.mol

    # Load entire file at once
    lines = np.loadtxt(filename)
    logging.info("Loaded third order file. Processing sparse input.")

    if third_energy_threshold > 0.0:
        # Keep original method for threshold case (not optimized as requested)
        n_rows = len(lines)
        array_size = min(n_rows * 3, n_atoms * 3 * (n_replicated_atoms * 3) ** 2)
        coords = np.zeros((array_size, 6), dtype=int)
        values = np.zeros((array_size))
        alphas = np.arange(3)
        index_in_unit_cell = 0

        above_threshold = np.abs(lines[:, -3:]) > third_energy_threshold
        to_write = np.where((lines[:, 0] - 1 < n_atoms) & (above_threshold.any(axis=1)))
        parsed_coords = lines[to_write][:, :-3] - 1
        parsed_values = lines[to_write][:, -3:]

        for i, (write, coords_to_write, values_to_write) in enumerate(
            zip(above_threshold[to_write], parsed_coords, parsed_values)
        ):
            if i % 1000000 == 0:
                logging.info("Reading third order with threadhold: " + str(np.round(i / n_rows, 2) * 100) + "%")

            for alpha in alphas[write]:
                coords[index_in_unit_cell, :-1] = coords_to_write[np.newaxis, :]
                coords[index_in_unit_cell, -1] = alpha
                values[index_in_unit_cell] = tenjovermoltoev * values_to_write[alpha]
                index_in_unit_cell += 1

        logging.info("read " + str(3 * i) + " interactions")

        coords = coords[:index_in_unit_cell].T
        values = values[:index_in_unit_cell]
        sparse_third = COO(coords, values, shape=(n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3))
        return sparse_third
    
    # Optimized path for reading entire file (third_energy_threshold == 0)
    logging.info("Using optimized chunked sparse import")
    
    import gc
    
    n_interactions = 0
    all_sparse_chunks = []
    # Use provided chunk_size parameter
    
    # Pre-allocate arrays for current chunk
    coords_chunk = np.empty((chunk_size, 6), dtype=np.int32)
    values_chunk = np.empty(chunk_size, dtype=np.float64)
    chunk_idx = 0
    
    def process_chunk():
        nonlocal chunk_idx
        if chunk_idx == 0:
            return None
            
        # Create COO from filled portion of arrays
        valid_coords = coords_chunk[:chunk_idx].T  # Transpose for COO format
        valid_values = values_chunk[:chunk_idx].copy()
        
        chunk_sparse = COO(valid_coords, valid_values,
                          shape=(n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3))
        
        chunk_idx = 0  # Reset for next chunk
        gc.collect()   # Force garbage collection
        return chunk_sparse
    
    # Use buffered reading for better I/O performance
    with open(filename, 'r', buffering=65536) as f:
        for line in f:
            parts = line.split()  # Avoid strip() if not needed
            if len(parts) < 8:
                continue
                
            # Fast integer parsing with error handling
            try:
                atom1_idx = int(parts[0]) - 1
                if atom1_idx >= n_atoms:
                    continue
                    
                n_interactions += 1
                
                # Parse coordinates once
                base_coords = [
                    atom1_idx,
                    int(parts[1]) - 1,
                    int(parts[2]) - 1,
                    int(parts[3]) - 1,
                    int(parts[4]) - 1
                ]
                
                # Parse values once with multiplication
                val0 = float(parts[5]) * tenjovermoltoev
                val1 = float(parts[6]) * tenjovermoltoev
                val2 = float(parts[7]) * tenjovermoltoev
                
                # Process non-zero values directly into pre-allocated arrays
                if val0 != 0.0:
                    coords_chunk[chunk_idx] = base_coords + [0]
                    values_chunk[chunk_idx] = val0
                    chunk_idx += 1
                    
                if val1 != 0.0 and chunk_idx < chunk_size:
                    coords_chunk[chunk_idx] = base_coords + [1]
                    values_chunk[chunk_idx] = val1
                    chunk_idx += 1
                    
                if val2 != 0.0 and chunk_idx < chunk_size:
                    coords_chunk[chunk_idx] = base_coords + [2]
                    values_chunk[chunk_idx] = val2
                    chunk_idx += 1
                
                # Process chunk when full
                if chunk_idx >= chunk_size:
                    chunk_sparse = process_chunk()
                    if chunk_sparse is not None:
                        all_sparse_chunks.append(chunk_sparse)
                        logging.info(f"Processed chunk {len(all_sparse_chunks)} with {chunk_size} entries")
                        
            except (ValueError, IndexError):
                continue  # Skip malformed lines
    
    # Process final chunk
    final_chunk = process_chunk()
    if final_chunk is not None:
        all_sparse_chunks.append(final_chunk)
        logging.info(f"Processed final chunk {len(all_sparse_chunks)} with {chunk_idx} entries")
    
    # Combine chunks
    if all_sparse_chunks:
        logging.info(f"Combining {len(all_sparse_chunks)} chunks...")
        if len(all_sparse_chunks) == 1:
            sparse_third = all_sparse_chunks[0]
        else:
            sparse_third = sum(all_sparse_chunks)
        logging.info(f"Read {n_interactions} interactions with {sparse_third.nnz} non-zero entries")
    else:
        # Handle empty case
        empty_coords = np.empty((6, 0), dtype=np.int32)
        empty_values = np.empty(0, dtype=np.float64)
        sparse_third = COO(empty_coords, empty_values,
                          shape=(n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3))
        logging.info("No valid interactions found")
    
    return sparse_third


def import_dense_third(atoms, supercell, filename, is_reduced=True):
    supercell = np.array(supercell)
    n_replicas = np.prod(supercell)
    n_atoms = atoms.get_positions().shape[0]
    if is_reduced:
        total_rows = (n_atoms * 3) * (n_atoms * n_replicas * 3) ** 2
        third = np.fromfile(filename, dtype=float, count=total_rows)
        third = third.reshape((n_atoms, 3, n_atoms * n_replicas, 3, n_atoms * n_replicas, 3))
    else:
        total_rows = (n_atoms * n_replicas * 3) ** 3
        third = np.fromfile(filename, dtype=float, count=total_rows)
        third = third.reshape((n_atoms * n_replicas, 3, n_atoms * n_replicas, 3, n_atoms * n_replicas, 3))
    return third
